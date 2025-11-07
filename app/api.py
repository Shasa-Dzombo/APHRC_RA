"""
Unified Research API with Session Management
"""
from fastapi import FastAPI, HTTPException, Depends
from typing import List, Dict, Optional, Any
from uuid import UUID, uuid4
from datetime import datetime, timedelta
import logging
import json

from model.models import (
    ProjectRequest, SessionRequest, ResearchQuestionResponse, 
    SubQuestionMappingResponse, DataGapResponse, LiteratureResponse,
    ResearchAnalysisResponse, LiteratureSearchRequest, ProjectInfo,
    QuestionSelectionRequest, QuestionSelectionResponse, SelectedQuestionsListResponse,
    SubQuestionAnalysisRequest, SubQuestionAnswer, SubQuestionAnswersResponse,
    DatabaseSchemaResponse, TableDetailsResponse, ResearchQuestion, SubQuestionMap,
    DataGap, LiteratureReference, ResearchSource, SessionInfo, SessionResponse,
    LLMQuestionParseResult
)
from utils.database_utils import parse_database_schema, get_table_details
from utils.formatting_utils import format_answer_content
from utils.research_utils import fetch_google_scholar, fetch_crossref, fetch_semantic_scholar
from config.llm_factory import get_llm
from prompts.research_prompts import PROMPT_ANSWER_GENERATION

# Configure comprehensive logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('research_api.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Unified Research API", version="1.0.0")

# --- Session Management ---
sessions: Dict[UUID, SessionInfo] = {}

async def get_active_session(session_id: UUID) -> SessionInfo:
    """Get active session or raise 404"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    session = sessions[session_id]
    if datetime.now() > session.expires_at:
        raise HTTPException(status_code=400, detail="Session expired")
    return session

# --- Database Exploration Endpoints ---
@app.get("/database/overview", response_model=DatabaseSchemaResponse)
async def get_database_overview():
    """Get an overview of all available tables and their descriptions"""
    try:
        logger.info("Fetching database overview")
        schema = parse_database_schema()
        return schema
    except Exception as e:
        logger.error(f"Error getting database overview: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/database/table/{table_name}", response_model=TableDetailsResponse)
async def get_table_info(table_name: str):
    """Get detailed information about a specific table including columns and relationships"""
    try:
        logger.info(f"Fetching table details for: {table_name}")
        details = await get_table_details(table_name)
        if not details:
            raise HTTPException(status_code=404, detail="Table not found")
        return details
    except ValueError as ve:
        logger.warning(f"Table not found: {table_name} - {str(ve)}")
        raise HTTPException(status_code=404, detail=str(ve))
    except Exception as e:
        logger.error(f"Error getting table details for {table_name}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving table details: {str(e)}")

# --- Research Workflow Endpoints ---
@app.post("/research/start", response_model=SessionResponse)
async def start_research(source: ResearchSource):
    """Start a new research session with either a topic or dataset focus"""
    try:
        logger.info(f"Starting new research session - Type: {source.source_type}, Title: {source.title}")
        
        session_id = uuid.uuid4()
        session = SessionInfo(
            session_id=session_id,
            research_type=source.source_type,
            source_id=source.table_name if source.source_type == "dataset" else None,
            current_step="started",
            data={
                "source": source.dict(),
                "main_questions": [],
                "sub_questions": [], 
                "mappings": [],
                "data_gaps": [],
                "literature": {},
                "sub_question_answers": []
            },
            created_at=datetime.now(),
            expires_at=datetime.now() + timedelta(hours=24)
        )
        sessions[session.session_id] = session
        
        logger.info(f"Session created successfully - ID: {session.session_id}")
        return SessionResponse(
            session_id=session.session_id,
            expires_at=session.expires_at,
            message=f"Research session started successfully - Type: {source.source_type}"
        )
        
    except Exception as e:
        logger.error(f"Failed to start research session: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to start research session: {str(e)}"
        )


@app.post("/research/generate-questions")
async def generate_questions(
    session_request: SessionRequest,
    session: SessionInfo = Depends(get_active_session)
):
    """Generate main research questions and sub-questions"""
    try:
        logger.info(f"Generating research questions for session {session_request.session_id}")
        
        # Get project info from session
        project_data = session.data.get("project", {})
        project_info = ProjectInfo(**project_data)
        
        # Use LLM to generate questions (simplified version)
        llm = get_llm()
        prompt = f"""
        Generate research questions for the following project:
        Title: {project_info.title}
        Description: {project_info.description}
        Area of Study: {project_info.area_of_study}
        Geography: {project_info.geography}
        
        Please provide 3-5 main research questions and 2-3 sub-questions for each main question.
        """
        
        response = llm.invoke(prompt)
        questions_data = parse_llm_question_response(response.content)
        
        # Store questions in session
        session.data.update(questions_data)
        session.current_step = "questions_generated"
        
        logger.info(f"Generated {len(questions_data.get('main_questions', []))} main questions")
        return questions_data
        
    except Exception as e:
        logger.error(f"Failed to generate research questions: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate research questions: {str(e)}"
        )

# --- Core Analysis Endpoints (from first API) ---
@app.post("/analyze-subquestions", response_model=List[SubQuestionMappingResponse])
async def analyze_subquestions(
    request: SubQuestionAnalysisRequest,
    session: SessionInfo = Depends(get_active_session)
):
    """Analyze data requirements and analysis approaches for sub-questions"""
    try:
        logger.info(f"Analyzing sub-questions for session {request.session_id}")
        
        # Validate session has questions
        if not session.data.get("main_questions") or not session.data.get("sub_questions"):
            raise HTTPException(
                status_code=400, 
                detail="No questions found. Please generate questions first."
            )
        
        # Filter sub-questions for specified main questions
        main_questions = session.data["main_questions"]
        sub_questions = session.data["sub_questions"]
        
        filtered_subs = [
            sq for sq in sub_questions 
            if any(mq["id"] == sq["parent_question_id"] for mq in main_questions if mq["id"] in request.main_question_ids)
        ]
        
        if not filtered_subs:
            raise HTTPException(
                status_code=400,
                detail="No sub-questions found for the specified main question IDs"
            )
        
        # Analyze each sub-question
        llm = get_llm()
        mappings = []
        
        for sub_q in filtered_subs:
            prompt = f"""
            Analyze the following research sub-question:
            "{sub_q['text']}"
            
            Provide:
            1. Data requirements (what data/variables are needed)
            2. Analysis approach (how to analyze the data)
            
            Format as JSON with keys: data_requirements, analysis_approach
            """
            
            response = llm.invoke(prompt)
            analysis_data = parse_analysis_response(response.content)
            
            mapping = SubQuestionMap(
                sub_question_id=sub_q["id"],
                sub_question=sub_q["text"],
                data_requirements=analysis_data.get("data_requirements", ""),
                analysis_approach=analysis_data.get("analysis_approach", "")
            )
            mappings.append(mapping)
        
        # Store mappings in session
        session.data["mappings"] = [mapping.dict() for mapping in mappings]
        session.data["selected_main_question_ids"] = request.main_question_ids
        session.current_step = "subquestions_analyzed"
        
        logger.info(f"Analyzed {len(mappings)} sub-questions")
        return mappings
        
    except Exception as e:
        logger.error(f"Failed to analyze sub-questions: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to analyze sub-questions: {str(e)}"
        )

@app.post("/analyze-selected-subquestions", response_model=SubQuestionAnswersResponse)
async def analyze_selected_subquestions(
    session_request: SessionRequest,
    session: SessionInfo = Depends(get_active_session)
):
    """Generate comprehensive answers for analyzed sub-questions"""
    try:
        logger.info(f"Generating answers for sub-questions in session {session_request.session_id}")
        
        # Check if we have analyzed sub-questions
        mappings = session.data.get("mappings", [])
        if not mappings:
            raise HTTPException(
                status_code=400,
                detail="No analyzed sub-questions found. Please run analyze-subquestions first."
            )
        
        # Generate answers for each mapped sub-question
        llm = get_llm()
        answers = []
        
        for mapping in mappings:
            # Use the enhanced answer generation prompt
            prompt = PROMPT_ANSWER_GENERATION.format(
                sub_question=mapping.sub_question,
                data_requirements=mapping.data_requirements,
                analysis_approach=mapping.analysis_approach
            )
            
            # Get parent question for context
            parent_question = None
            for mq in session.data.get("main_questions", []):
                sq_list = [sq for sq in session.data.get("sub_questions", []) 
                          if sq.get("parent_question_id") == mq.get("id")]
                if any(sq.get("id") == mapping.sub_question_id for sq in sq_list):
                    parent_question = mq.get("text")
                    break

            # Add research context with both high-level and specific context
            source_type = session.research_type
            source_data = session.data.get("source", {})
            
            if source_type == "dataset":
                context = (f"Dataset Context: {source_data.get('table_name')} - {source_data.get('description')}\n"
                          f"Main Research Question: {parent_question}\n"
                          f"Specific Sub-Question to Answer: {mapping.sub_question}\n\n")
            else:
                context = (f"Research Context: {source_data.get('title')} - {source_data.get('description')}\n"
                          f"Main Research Question: {parent_question}\n"
                          f"Specific Sub-Question to Answer: {mapping.sub_question}\n\n")
            
            # Build prompt with explicit focus instructions
            focused_prompt = f"""IMPORTANT: Your task is to analyze and answer ONLY this specific sub-question:
"{mapping.sub_question}"

This sub-question is part of the larger research question:
"{parent_question}"

Ensure your answer:
1. ONLY addresses this specific sub-question
2. Does NOT discuss other questions or topics
3. Is specific to the exact relationship or factor being asked about
4. Uses the provided data requirements and analysis approach

{prompt}"""
            
            full_prompt = context + focused_prompt
            
            logger.info("\n=== Processing Sub-Question ===")
            logger.info("Sub-Question: %s", mapping.sub_question)
            logger.info("Context: %s", context.strip())
            
            response = llm.invoke(full_prompt)
            logger.info("\n=== Raw LLM Response ===\n%s\n=== End Raw Response ===\n", response.content)
            
            # Format the response content
            formatted_content = format_answer_content(response.content)
            logger.info("\n=== Formatted Response ===\n%s\n=== End Formatted Response ===\n", formatted_content)
            
            answer = SubQuestionAnswer(
                sub_question_id=mapping["sub_question_id"],
                sub_question_text=mapping["sub_question"],
                answer=response.content,
                confidence_score=0.8,  # Could be calculated based on response quality
                sources_used=["AI analysis based on research context"]
            )
            answers.append(answer)
            
            # Log each section separately for verification
            sections = formatted_content.split("\n\n")
            logger.info("\n=== Answer Sections ===")
            for section in sections:
                if "**" in section:  # Only log actual sections
                    logger.info("\n%s", section)
        
        # Store answers in session
        session.data["sub_question_answers"] = [answer.dict() for answer in answers]
        session.current_step = "answers_generated"
        
        logger.info(f"Generated answers for {len(answers)} sub-questions")
        
        return SubQuestionAnswersResponse(
            session_id=session_request.session_id,
            answers=answers,
            total_answered=len(answers),
            processing_summary=f"Generated comprehensive answers for {len(answers)} sub-questions based on data requirements and analysis approaches."
        )
        
    except Exception as e:
        logger.error(f"Failed to generate sub-question answers: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate sub-question answers: {str(e)}"
        )

@app.post("/identify-data-gaps", response_model=List[DataGapResponse])
async def identify_data_gaps(
    session_request: SessionRequest,
    session: SessionInfo = Depends(get_active_session)
):
    """Identify data gaps for analyzed sub-questions"""
    try:
        logger.info(f"Identifying data gaps for session {session_request.session_id}")
        
        # Check if we have analyzed sub-questions
        mappings = session.data.get("mappings", [])
        if not mappings:
            raise HTTPException(
                status_code=400,
                detail="No analyzed sub-questions found. Please run analyze-subquestions first."
            )
        
        # Get database schema for comparison
        db_schema = await parse_database_schema()
        available_variables = extract_available_variables(db_schema)
        
        data_gaps = []
        
        for mapping in mappings:
            # Extract required variables from data requirements
            required_vars = extract_required_variables(mapping["data_requirements"])
            missing_vars = [var for var in required_vars if var not in available_variables]
            
            if missing_vars:
                for var in missing_vars:
                    gap = DataGap(
                        missing_variable=var,
                        gap_description=f"Required variable '{var}' not found in available database",
                        suggested_sources=suggest_data_sources([var]),
                        sub_question_id=mapping["sub_question_id"]
                    )
                    data_gaps.append(gap)
        
        # Store gaps in session
        session.data["data_gaps"] = [gap.dict() for gap in data_gaps]
        session.current_step = "gaps_identified"
        
        logger.info(f"Identified {len(data_gaps)} data gaps")
        return data_gaps
        
    except Exception as e:
        logger.error(f"Failed to identify data gaps: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to identify data gaps: {str(e)}"
        )

# --- Literature Search Endpoints ---
@app.post("/research/literature/search")
async def search_literature(
    session_request: SessionRequest,
    session: SessionInfo = Depends(get_active_session)
):
    """Search literature based on research questions"""
    try:
        logger.info(f"Starting literature search for session {session_request.session_id}")
        
        if not session.data.get("main_questions"):
            raise HTTPException(status_code=400, detail="Generate questions first")
        
        main_questions = session.data["main_questions"]
        all_results = []
        
        for question in main_questions:
            question_text = question.get("text", question.get("question", ""))
            if question_text:
                # Search multiple sources
                google_results = await fetch_google_scholar(question_text)
                crossref_results = await fetch_crossref(question_text)
                ss_results = await fetch_semantic_scholar(question_text)
                
                # Add source information
                for result in google_results:
                    result["source"] = "Google Scholar"
                    result["question_text"] = question_text
                    
                for result in crossref_results:
                    result["source"] = "CrossRef" 
                    result["question_text"] = question_text

                for result in ss_results:
                    result["source"] = "Semantic Scholar"
                    result["question_text"] = question_text
                    
                all_results.extend(google_results + crossref_results +ss_results)
        
        session.data["literature"] = all_results
        session.current_step = "literature_searched"
        
        logger.info(f"Literature search completed. Found {len(all_results)} results")
        return all_results
        
    except Exception as e:
        logger.error(f"Literature search failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Literature search failed: {str(e)}"
        )

# --- Utility Functions ---
def parse_llm_question_response(response_text: str) -> LLMQuestionParseResult:
    """Parse LLM response into structured question format"""
    # Simplified parsing - in practice, you'd want more robust parsing
    lines = [line.strip() for line in response_text.split('\n') if line.strip()]
    
    main_questions = []
    sub_questions_by_main = {}
    current_main_id = None
    
    for line in lines:
        if line.lower().startswith(('main', 'primary', 'research question')):
            # Main question
            main_id = str(len(main_questions) + 1)
            current_main_id = main_id
            question_text = line.split(':', 1)[-1].strip()
            main_questions.append({
                "id": main_id,
                "text": question_text,
                "question_type": "main"
            })
            sub_questions_by_main[main_id] = []
        elif line.startswith('-') and current_main_id:
            # Sub-question
            sub_question_text = line[1:].strip()
            sub_questions_by_main[current_main_id].append(sub_question_text)
    
    return LLMQuestionParseResult(
        main_questions=main_questions,
        sub_questions=sub_questions_by_main,
        approach="Structured research questions with hierarchical organization"
    )

def parse_analysis_response(response_text: str) -> Dict:
    """Parse analysis response into structured data"""
    # Simple parsing - extract data requirements and analysis approach
    lines = [line.strip() for line in response_text.split('\n') if line.strip()]
    
    data_requirements = []
    analysis_approach = []
    current_section = None
    
    for line in lines:
        lower_line = line.lower()
        if 'data' in lower_line and 'requirement' in lower_line:
            current_section = 'data'
        elif 'analysis' in lower_line and 'approach' in lower_line:
            current_section = 'analysis'
        elif current_section == 'data' and line:
            data_requirements.append(line)
        elif current_section == 'analysis' and line:
            analysis_approach.append(line)
    
    return {
        "data_requirements": ' '.join(data_requirements),
        "analysis_approach": ' '.join(analysis_approach)
    }

def extract_available_variables(db_schema: Dict) -> List[str]:
    """Extract available variables from database schema"""
    variables = []
    for table in db_schema.get("tables", []):
        for column in table.get("columns", []):
            variables.append(column["name"].lower())
    return variables

def extract_required_variables(data_requirements: str) -> List[str]:
    """Extract required variables from data requirements text"""
    # Simple extraction - look for common variable patterns
    words = data_requirements.lower().split()
    variables = []
    
    for word in words:
        if len(word) > 3 and word not in ['data', 'variable', 'requirement']:
            # Simple heuristic for variable names
            if word.isalpha():
                variables.append(word)
    
    return list(set(variables))

def suggest_data_sources(missing_variables: List[str]) -> str:
    """Suggest data sources for missing variables"""
    source_mapping = {
        "health": ["Health Management Information System", "Hospital Records"],
        "demographic": ["Census Data", "Demographic Health Survey"],
        "economic": ["World Bank Data", "Household Economic Survey"],
        "geographic": ["GIS Data", "Spatial Databases"]
    }
    
    suggestions = []
    for var in missing_variables:
        for key, sources in source_mapping.items():
            if key in var:
                suggestions.extend(sources)
    
    return ", ".join(set(suggestions)) if suggestions else "General research databases"

# --- Session Management Endpoints ---
@app.get("/research/session/{session_id}")
async def get_session_status(
    session_id: UUID,
    session: SessionInfo = Depends(get_active_session)
):
    """Get current session status and data"""
    return {
        "session_id": session.session_id,
        "research_type": session.research_type,
        "current_step": session.current_step,
        "created_at": session.created_at.isoformat(),
        "expires_at": session.expires_at.isoformat(),
        "data_summary": {
            "main_questions_count": len(session.data.get("main_questions", [])),
            "sub_questions_count": len(session.data.get("sub_questions", [])),
            "mappings_count": len(session.data.get("mappings", [])),
            "data_gaps_count": len(session.data.get("data_gaps", [])),
            "literature_count": len(session.data.get("literature", [])),
            "answers_count": len(session.data.get("sub_question_answers", []))
        }
    }

