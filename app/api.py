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
    DataGap, LiteratureReference, ResearchSource, SessionInfo, SessionResponse
)
from utils.database_utils import parse_database_schema, get_table_details
from utils.research_utils import fetch_google_scholar, fetch_crossref, fetch_webpage
from config.llm_factory import get_llm
from utils.parser_utils import parse_main_and_sub_questions, parse_subquestion_mappings
from utils.formatting_utils import format_answer_content
from prompts.research_prompts import (
    PROMPT_STEP2, 
    PROMPT_STEP3, 
    PROMPT_STEP4, 
    PROMPT_ANSWER_GENERATION
)

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
sessions = {}

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
        
        # Create session using Pydantic model defaults
        session = SessionInfo(
            research_type=source.source_type,
            source_id=source.table_name if source.source_type == "dataset" else None,
            data={
                "source": source.dict(),
                "main_questions": [],
                "sub_questions": [], 
                "mappings": [],
                "data_gaps": [],
                "literature": {},
                "sub_question_answers": []
            }
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
    """Generate research questions based on source type (topic or dataset)"""
    try:
        logger.info(f"Generating research questions for session {session_request.session_id}")
        
        # Get source info from session
        source_data = session.data.get("source", {})
        source_type = session.research_type
        
        # Use appropriate prompt based on research type
        llm = get_llm()
        
        if source_type == "dataset":
            logger.info(f"Dataset-driven research on table: {source_data.get('table_name')}")
            context = f"""
            Dataset Information:
            Table: {source_data.get('table_name')}
            Variables: {', '.join(source_data.get('variables', []))}
            Description: {source_data.get('description')}
            """
        else:
            logger.info(f"Topic-driven research on: {source_data.get('title')}")
            context = f"""
            Research Topic Information:
            Title: {source_data.get('title')}
            Description: {source_data.get('description')}
            Area of Study: {source_data.get('area_of_study', '')}
            Geography: {source_data.get('geography', '')}
            """
        
        # Use the enhanced prompt from research_prompts.py
        prompt = context + PROMPT_STEP2
        
        logger.info("Sending prompt to LLM")
        response = llm.invoke(prompt)
        logger.info(f"Raw LLM response: {response.content}")
        
        logger.info("Parsing questions from LLM response")
        parsed_questions = parse_main_and_sub_questions(response.content)
        logger.info(f"Parsed questions: {json.dumps(parsed_questions, indent=2)}")
        
        # Convert parsed questions to our model format with proper IDs
        main_questions = []
        sub_questions = []
        
        for idx, main_q_text in enumerate(parsed_questions["main_questions"]):
            main_id = str(uuid4())
            main_questions.append({
                "id": main_id,
                "text": main_q_text,
                "question_type": "main"
            })
            
            # Add sub-questions for this main question
            for sub_q in parsed_questions["sub_questions"]:
                if sub_q["main_question_index"] == idx:
                    sub_questions.append({
                        "id": str(uuid4()),
                        "text": sub_q["text"],
                        "question_type": "sub",
                        "parent_question_id": main_id
                    })
        
        questions_data = {
            "main_questions": main_questions,
            "sub_questions": sub_questions
        }
        
        # Store questions in session
        session.data.update(questions_data)
        session.current_step = "questions_generated"
        
        logger.info(f"Generated {len(main_questions)} main questions and {len(sub_questions)} sub-questions")
        return questions_data
        
    except Exception as e:
        logger.error(f"Failed to generate research questions: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate research questions: {str(e)}"
        )

# --- Core Analysis Endpoints ---
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
        
        # Get source context for analysis
        source_data = session.data.get("source", {})
        source_type = session.research_type
        
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
        
        # Generate analysis prompt using enhanced PROMPT_STEP3
        llm = get_llm()
        
        # Build the sub-questions list for the prompt
        sub_questions_text = ""
        for sub_q in filtered_subs:
            sub_questions_text += f"SUB-QUESTION: {sub_q['text']}\n\n"
        
        prompt = PROMPT_STEP3 + "\n\n" + sub_questions_text
        
        # Get LLM response
        logger.info("Sending prompt to LLM...")
        response = llm.invoke(prompt)
        
        # Log the actual generated analysis
        logger.info("\n=== Generated Analysis ===\n%s\n=== End Analysis ===\n", response.content)
        
        logger.info("Generated sub-questions text for analysis")
        logger.debug("Sub-questions text: %s", sub_questions_text)
        
        prompt = PROMPT_STEP3 + "\n\n" + sub_questions_text
        
        logger.info("Sending prompt to LLM for sub-question analysis")
        response = llm.invoke(prompt)
        logger.info("Received LLM response")
        logger.debug("Raw LLM response: %s", response.content)
        
        if not response.content.strip():
            logger.error("Received empty response from LLM")
            raise HTTPException(status_code=500, detail="No analysis generated by LLM")
            
        # Use the parser utility to parse the response
        logger.info("Parsing LLM response for sub-question mappings")
        parsed_mappings = parse_subquestion_mappings(response.content, filtered_subs)
        
        # Convert to SubQuestionMap objects
        mappings = []
        for parsed_mapping in parsed_mappings:
            mapping = SubQuestionMap(
                sub_question_id=parsed_mapping["sub_question_id"],
                sub_question=parsed_mapping["sub_question"],
                data_requirements=parsed_mapping["data_requirements"],
                analysis_approach=parsed_mapping["analysis_approach"]
            )
            mappings.append(mapping)
        
        # Store mappings in session
        session.data["mappings"] = [mapping.dict() for mapping in mappings]
        session.data["selected_main_question_ids"] = request.main_question_ids
        session.current_step = "subquestions_analyzed"
        
        # Log analysis results
        if mappings:
            logger.info("\n=== Analysis Results ===")
            for mapping in mappings:
                logger.info("\nSub-question: %s", mapping.sub_question)
                logger.info("Data Requirements:\n%s", mapping.data_requirements)
                logger.info("Analysis Approach:\n%s", mapping.analysis_approach)
            logger.info("=== End Results ===\n")
        else:
            logger.warning("No analysis mappings were generated")
            
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
        
        # Get research context
        source_data = session.data.get("source", {})
        source_type = session.research_type
        
        # Generate answers for each mapped sub-question using enhanced prompt
        llm = get_llm()
        answers = []
        
        for mapping in mappings:
            # Use the enhanced answer generation prompt
            prompt = PROMPT_ANSWER_GENERATION.format(
                sub_question=mapping['sub_question'],
                data_requirements=mapping['data_requirements'],
                analysis_approach=mapping['analysis_approach']
            )
            
            # Get parent question for context
            parent_question = None
            for mq in session.data.get("main_questions", []):
                sq_list = [sq for sq in session.data.get("sub_questions", []) 
                          if sq.get("parent_question_id") == mq.get("id")]
                if any(sq.get("id") == mapping["sub_question_id"] for sq in sq_list):
                    parent_question = mq.get("text")
                    break

            # Add research context with both high-level and specific context
            if source_type == "dataset":
                context = (f"Dataset Context: {source_data.get('table_name')} - {source_data.get('description')}\n"
                          f"Main Research Question: {parent_question}\n"
                          f"Specific Sub-Question to Answer: {mapping['sub_question']}\n\n")
            else:
                context = (f"Research Context: {source_data.get('title')} - {source_data.get('description')}\n"
                          f"Main Research Question: {parent_question}\n"
                          f"Specific Sub-Question to Answer: {mapping['sub_question']}\n\n")
            
            # Build prompt with explicit focus instructions
            focused_prompt = f"""IMPORTANT: Your task is to analyze and answer ONLY this specific sub-question:
"{mapping['sub_question']}"

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
            logger.info("Sub-Question: %s", mapping['sub_question'])
            logger.info("Context: %s", context.strip())
            
            response = llm.invoke(full_prompt)
            logger.info("\n=== Raw LLM Response ===\n%s\n=== End Raw Response ===\n", response.content)
            
            # Format the response content
            formatted_content = format_answer_content(response.content)
            logger.info("\n=== Formatted Response ===\n%s\n=== End Formatted Response ===\n", formatted_content)
            
            answer = SubQuestionAnswer(
                sub_question_id=mapping["sub_question_id"],
                sub_question_text=mapping["sub_question"],
                answer=formatted_content,
                confidence_score=calculate_confidence_score(formatted_content),
                sources_used=[f"{source_type.capitalize()} research analysis", "Evidence-based synthesis"]
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
            session_id=session.session_id,  # Use the actual session ID from the active session
            answers=answers,
            total_answered=len(answers),
            processing_summary=f"Generated comprehensive, evidence-based answers for {len(answers)} sub-questions using structured analytical frameworks."
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
        raw_mappings = session.data.get("mappings", [])
        if not raw_mappings:
            raise HTTPException(
                status_code=400,
                detail="No analyzed sub-questions found. Please run analyze-subquestions first."
            )
        
        # Convert dictionary mappings to SubQuestionMap objects
        mappings = [SubQuestionMap(**mapping) for mapping in raw_mappings]
        
        source_type = session.research_type
        source_data = session.data.get("source", {})
        
        # Build prompt for data gap analysis using enhanced PROMPT_STEP4
        llm = get_llm()
        
        # Prepare sub-questions and data requirements for the prompt
        gap_analysis_input = ""
        for mapping in mappings:
            gap_analysis_input += f"SUB-QUESTION: {mapping['sub_question']}\n"
            gap_analysis_input += f"DATA REQUIREMENTS: {mapping['data_requirements']}\n\n"
        
        # Add research context
        if source_type == "dataset":
            context = f"Dataset Context: {source_data.get('table_name')} - Available variables: {', '.join(source_data.get('variables', []))}\n\n"
        else:
            context = f"Research Context: {source_data.get('title')} - {source_data.get('area_of_study', '')} - {source_data.get('geography', '')}\n\n"
        
        prompt = context + PROMPT_STEP4 + "\n\n" + gap_analysis_input
        
        response = llm.invoke(prompt)
        
        # Parse the data gaps from the response
        data_gaps = parse_data_gaps_response(response.content, mappings)
        
        # Store gaps in session
        session.data["data_gaps"] = [gap.dict() for gap in data_gaps]
        session.current_step = "gaps_identified"
        
        logger.info(f"Identified {len(data_gaps)} data gaps for {source_type} research")
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
        
        source_data = session.data.get("source", {})
        main_questions = session.data["main_questions"]
        all_results = []
        
        for question in main_questions:
            question_text = question.get("text", question.get("question", ""))
            if question_text:
                # Add research context to search query
                if session.research_type == "dataset":
                    search_query = f"{question_text} {source_data.get('table_name')} dataset"
                else:
                    search_query = f"{question_text} {source_data.get('area_of_study', '')} {source_data.get('geography', '')}"
                
                # Search multiple sources
                google_results = await fetch_google_scholar(search_query)
                crossref_results = await fetch_crossref(search_query)
                
                # Add source information
                for result in google_results:
                    result["source"] = "Google Scholar"
                    result["question_text"] = question_text
                    
                for result in crossref_results:
                    result["source"] = "CrossRef" 
                    result["question_text"] = question_text
                    
                all_results.extend(google_results + crossref_results)
        
        session.data["literature"] = all_results
        session.current_step = "literature_searched"
        
        logger.info(f"Literature search completed. Found {len(all_results)} results for {session.research_type} research")
        return all_results
        
    except Exception as e:
        logger.error(f"Literature search failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Literature search failed: {str(e)}"
        )

# --- Utility Functions ---
def parse_data_gaps_response(response_text: str, mappings: List[SubQuestionMap]) -> List[DataGap]:
    """Parse the data gaps response from LLM using the expected format"""
    data_gaps = []
    lines = response_text.strip().split('\n')
    
    current_gap = {}
    
    for line in lines:
        line = line.strip()
        if line.startswith("MISSING VARIABLE:"):
            # Save previous gap if exists
            if current_gap and current_gap.get("missing_variable"):
                # Find the matching sub-question ID
                sub_question_id = find_matching_subquestion_id(current_gap.get("sub_question", ""), mappings)
                if sub_question_id:
                    gap = DataGap(
                        missing_variable=current_gap["missing_variable"],
                        gap_description=current_gap.get("gap_description", ""),
                        suggested_sources=current_gap.get("suggested_sources", ""),
                        sub_question_id=sub_question_id
                    )
                    data_gaps.append(gap)
            
            # Start new gap
            current_gap = {"missing_variable": line.replace("MISSING VARIABLE:", "").strip()}
            
        elif line.startswith("GAP DESCRIPTION:"):
            current_gap["gap_description"] = line.replace("GAP DESCRIPTION:", "").strip()
        elif line.startswith("SUGGESTED SOURCES:"):
            current_gap["suggested_sources"] = line.replace("SUGGESTED SOURCES:", "").strip()
        elif line.startswith("SUB-QUESTION:"):
            current_gap["sub_question"] = line.replace("SUB-QUESTION:", "").strip()
    
    # Add the last gap
    if current_gap and current_gap.get("missing_variable"):
        sub_question_id = find_matching_subquestion_id(current_gap.get("sub_question", ""), mappings)
        if sub_question_id:
            gap = DataGap(
                missing_variable=current_gap["missing_variable"],
                gap_description=current_gap.get("gap_description", ""),
                suggested_sources=current_gap.get("suggested_sources", ""),
                sub_question_id=sub_question_id
            )
            data_gaps.append(gap)
    
    return data_gaps

def find_matching_subquestion_id(sub_question_text: str, mappings: List[SubQuestionMap]) -> Optional[str]:
    """Find the sub-question ID that matches the given text"""
    if not sub_question_text:
        return None
        
    sub_question_text = sub_question_text.lower()
    
    # First try exact match
    for mapping in mappings:
        if sub_question_text == mapping.sub_question.lower():
            return mapping.sub_question_id
            
    # Then try partial matches
    for mapping in mappings:
        mapping_text = mapping.sub_question.lower()
        if sub_question_text in mapping_text or mapping_text in sub_question_text:
            return mapping.sub_question_id
            
    return None

def calculate_confidence_score(answer_text: str) -> float:
    """Calculate confidence score based on answer quality indicators"""
    # Simple heuristic based on answer length and structure
    score = 0.5  # Base score
    
    # Increase score for well-structured answers
    if "**DIRECT ANSWER:**" in answer_text:
        score += 0.2
    if "**DATA-DRIVEN INSIGHTS:**" in answer_text:
        score += 0.15
    if "**ANALYTICAL METHODOLOGY:**" in answer_text:
        score += 0.15
    
    # Adjust based on length (longer answers tend to be more comprehensive)
    word_count = len(answer_text.split())
    if word_count > 100:
        score += 0.1
    elif word_count < 50:
        score -= 0.1
    
    return min(1.0, max(0.1, score))

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
        "source_id": session.source_id,
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

@app.get("/")
async def root():
    """API root endpoint"""
    return {
        "message": "Unified Research API",
        "version": "1.0.0",
        "endpoints": {
            "database": ["/database/overview", "/database/table/{table_name}"],
            "research": [
                "/research/start",
                "/research/generate-questions", 
                "/analyze-subquestions",
                "/analyze-selected-subquestions",
                "/identify-data-gaps",
                "/research/literature/search"
            ],
            "session": ["/research/session/{session_id}"]
        }
    }