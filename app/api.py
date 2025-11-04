from fastapi import FastAPI, HTTPException, Depends
from typing import List, Dict, Optional
from uuid import UUID
from datetime import datetime
import logging
from utils.database_utils import parse_database_schema, get_table_details
from utils.research_utils import fetch_google_scholar, fetch_crossref, fetch_webpage
from config.llm_factory import get_llm
from app.models import (
    AIInsightResponse,
    SubQuestion,
    Question,
    QuestionResponse,
    SessionInfo,
    ResearchSource
)
from prompts import (
    DATASET_QUESTIONS_PROMPT,
    TOPIC_QUESTIONS_PROMPT,
    AI_INSIGHTS_PROMPT
)

# Configure API logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('research_api.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Import models from models.py

app = FastAPI()

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

# --- Step 1: Database Exploration ---
@app.get("/database/overview")
async def get_database_overview():
    """Get an overview of all available tables and their descriptions"""
    try:
        schema = parse_database_schema()
        return schema
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/database/table/{table_name}")
async def get_table_info(table_name: str):
    """Get detailed information about a specific table including columns and relationships"""
    try:
        # Using a different name for the function to avoid naming conflict
        details = await get_table_details(table_name)
        if not details:
            raise HTTPException(status_code=404, detail="Table not found")
        return details
    except ValueError as ve:
        # Handle specific case of table not found
        raise HTTPException(status_code=404, detail=str(ve))
    except Exception as e:
        # Log unexpected errors
        logger.error(f"Error getting table details for {table_name}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving table details: {str(e)}")

# --- Step 2: Start Research ---
@app.post("/research/start", response_model=SessionInfo)
async def start_research(source: ResearchSource):
    """Start a new research session with either a topic or dataset focus"""
    try:
        logger.info(f"Starting new research session - Type: {source.source_type}, Title: {source.title}")
        
        session = SessionInfo(
            research_type=source.source_type,
            source_id=source.table_name if source.source_type == "dataset" else None,
            current_step="started",
            data={
                "source": source.dict(),
            }
        )
        sessions[session.session_id] = session
        
        logger.info(f"Session created successfully - ID: {session.session_id}")
        return session
        
    except Exception as e:
        logger.error(f"Failed to start research session: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to start research session: {str(e)}"
        )

# --- Helper Functions ---
def parse_llm_question_response(response_text: str) -> Dict:
    """Parse the LLM response into structured question format"""
    lines = response_text.split('\n')
    main_questions = []
    current_main = None
    current_subs = []
    approach = ""
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        if line.upper().startswith("MAIN QUESTION"):
            # Save previous main question if exists
            if current_main:
                main_questions.append({
                    "question": current_main,
                    "sub_questions": current_subs.copy()
                })
            current_main = None
            current_subs = []
        elif line.upper().startswith("APPROACH:"):
            approach = line.replace("APPROACH:", "").strip()
        elif ":" in line and not line.startswith("-"):
            current_main = line.split(":", 1)[1].strip()
        elif line.startswith("-"):
            if current_main:  # Only add sub-questions if we have a main question
                sub_q = line.replace("-", "").strip()
                current_subs.append(sub_q)
                
    # Add the last main question if exists
    if current_main:
        main_questions.append({
            "question": current_main,
            "sub_questions": current_subs
        })
        
    # Structure the response
    structured_questions = []
    for idx, main in enumerate(main_questions):
        question = {
            "question": main["question"],
            "sub_questions": [
                {"question": sq, "main_question_index": idx}
                for sq in main["sub_questions"]
            ]
        }
        structured_questions.append(question)
        
    return {
        "main_questions": structured_questions,
        "approach": approach
    }

# --- Step 3: Generate Research Questions ---
@app.post("/research/questions/generate")
async def generate_research_questions(
    session_id: UUID,
    session: SessionInfo = Depends(get_active_session)
):
    """Generate research questions based on source type (topic or dataset)"""
    try:
        logger.info(f"Generating research questions for session {session_id}")
        llm = get_llm()
        source = session.data["source"]
        
        if session.research_type == "dataset":
            logger.info(f"Dataset-driven research on table: {source['table_name']}")
            table_details = get_table_details(source["table_name"])
            if not table_details:
                logger.error(f"Table not found: {source['table_name']}")
                raise HTTPException(status_code=404, detail="Table not found")
                
            prompt = DATASET_QUESTIONS_PROMPT.format(
                table_name=source['table_name'],
                variables=', '.join(source['variables']),
                description=source['description']
            )
        else:
            logger.info(f"Topic-driven research on: {source['title']}")
            prompt = TOPIC_QUESTIONS_PROMPT.format(
                title=source['title'],
                description=source['description'],
                area_of_study=source.get('area_of_study', ''),
                geography=source.get('geography', '')
            )
        
        logger.debug(f"Sending prompt to LLM: {prompt[:200]}...")
        response = llm.invoke(prompt)
        
        if hasattr(response, 'content'):
            response_text = response.content
        else:
            response_text = str(response)
            
        questions = parse_llm_question_response(response_text)
        logger.info(f"Generated {len(questions.get('sub_questions', []))} sub-questions")
        
        session.data["questions"] = questions
        session.current_step = "questions_generated"
        return questions
        
    except Exception as e:
        logger.error(f"Failed to generate research questions: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate research questions: {str(e)}"
        )

# --- Step 4: Analyze Data Gaps ---
@app.post("/research/analyze/gaps")
async def analyze_gaps(
    session_id: UUID,
    session: SessionInfo = Depends(get_active_session)
):
    """Analyze gaps in data and methodology for research questions"""
    try:
        logger.info(f"Analyzing data gaps for session {session_id}")
        
        if "questions" not in session.data:
            raise HTTPException(status_code=400, detail="Generate questions first")
        
        questions = session.data["questions"]
        
        # Ensure we have main questions
        main_questions = questions.get("main_questions", [])
        if not main_questions:
            raise HTTPException(status_code=400, detail="No research questions found")
            
        logger.info(f"Analyzing gaps for {len(main_questions)} questions")
        
        # Get data context based on research type
        if session.research_type == "dataset":
            available_data = {
                "type": "dataset",
                "variables": session.data["source"].get("variables", []),
                "table": session.data["source"].get("table_name", "")
            }
        else:
            available_data = {
                "type": "topic",
                "area": session.data["source"].get("area_of_study", ""),
                "geography": session.data["source"].get("geography", "")
            }
            
        # Analyze gaps for each main question and its sub-questions
        gaps = []
        for idx, main_q in enumerate(main_questions):
            main_question = main_q.get("question", "")
            sub_questions = main_q.get("sub_questions", [])
            
            # Analyze main question
            question_gaps = analyze_question_gaps(
                main_question,
                available_data,
                is_main=True,
                question_index=idx
            )
            gaps.extend(question_gaps)
            
            # Analyze each sub-question
            for sub_idx, sub_q in enumerate(sub_questions):
                sub_gaps = analyze_question_gaps(
                    sub_q.get("question", ""),
                    available_data,
                    is_main=False,
                    question_index=idx,
                    sub_question_index=sub_idx
                )
                gaps.extend(sub_gaps)
        
        # Store gaps in session
        session.data["data_gaps"] = gaps
        session.current_step = "gaps_analyzed"
        
        logger.info(f"Identified {len(gaps)} data gaps")
        return gaps
        
    except Exception as e:
        logger.error(f"Failed to analyze data gaps: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to analyze data gaps: {str(e)}"
        )

def analyze_question_gaps(
    question: str,
    available_data: Dict,
    is_main: bool = True,
    question_index: int = 0,
    sub_question_index: Optional[int] = None
) -> List[Dict]:
    """Analyze data gaps for a specific research question"""
    gaps = []
    
    # Extract key concepts and variables needed
    required_vars = extract_required_variables(question)
    
    # For dataset research, compare with available variables
    if available_data["type"] == "dataset":
        available_vars = set(available_data["variables"])
        missing_vars = required_vars - available_vars
        
        for var in missing_vars:
            gaps.append({
                "type": "missing_variable",
                "variable": var,
                "question_type": "main" if is_main else "sub",
                "question_index": question_index,
                "sub_question_index": sub_question_index,
                "suggested_sources": suggest_data_sources([var]),
                "description": f"Required variable '{var}' not available in dataset"
            })
    
    # For topic research, identify potential data requirements
    else:
        data_reqs = suggest_data_requirements(question)
        for req in data_reqs:
            gaps.append({
                "type": "data_requirement",
                "requirement": req,
                "question_type": "main" if is_main else "sub",
                "question_index": question_index,
                "sub_question_index": sub_question_index,
                "suggested_methodology": suggest_methodology(req),
                "description": f"Data required for analysis: {req}"
            })
    
    return gaps

def extract_required_variables(question: str) -> set:
    """Extract potential required variables from a question"""
    # Basic variable extraction - could be enhanced with NLP
    words = question.lower().split()
    variables = set()
    
    # Keywords that often precede variables
    indicators = ["rate", "level", "status", "number", "count", "percentage", "ratio",
                "frequency", "incidence", "prevalence", "measure", "score", "index"]
    
    for i, word in enumerate(words):
        # Check if word is or follows an indicator
        if (word in indicators or
            (i > 0 and words[i-1] in indicators)):
            variables.add(word)
            
        # Check for compound variables (e.g., "education_level")
        if "_" in word:
            variables.add(word)
    
    return variables

def suggest_data_requirements(question: str) -> List[str]:
    """Suggest data requirements for a research question"""
    requirements = []
    
    # Basic patterns to identify data requirements
    if "compare" in question.lower() or "difference" in question.lower():
        requirements.append("Comparative data across groups")
    if "time" in question.lower() or "period" in question.lower():
        requirements.append("Time series data")
    if "factor" in question.lower() or "influence" in question.lower():
        requirements.append("Multiple variable relationships")
    if "where" in question.lower() or "location" in question.lower():
        requirements.append("Geographic data")
    if "why" in question.lower() or "reason" in question.lower():
        requirements.append("Qualitative explanatory data")
    
    return requirements

def suggest_methodology(data_requirement: str) -> List[str]:
    """Suggest methodological approaches"""
    method_map = {
        "Comparative data": [
            "Cross-sectional analysis",
            "Comparative case studies",
            "Statistical hypothesis testing"
        ],
        "Time series data": [
            "Longitudinal analysis",
            "Time series modeling",
            "Trend analysis"
        ],
        "Multiple variable relationships": [
            "Regression analysis",
            "Factor analysis",
            "Path analysis"
        ],
        "Geographic data": [
            "Spatial analysis",
            "GIS mapping",
            "Cluster analysis"
        ],
        "Qualitative explanatory data": [
            "Thematic analysis",
            "Content analysis",
            "In-depth interviews"
        ]
    }
    
    # Find matching methodologies
    methods = []
    for key, value in method_map.items():
        if key.lower() in data_requirement.lower():
            methods.extend(value)
    
    return methods or ["General statistical analysis"]

# --- Step 5: AI Research Insights ---
@app.post("/research/analyze/ai-insights", 
          response_model=AIInsightResponse,
          tags=["Research Analysis"])
async def get_ai_research_insights(
    session_id: UUID,
    session: SessionInfo = Depends(get_active_session)
):
    """
    Provide AI-generated insights and preliminary answers for research questions.
    
    This endpoint:
    1. Uses identified data gaps to understand limitations
    2. Searches web sources for relevant information
    3. Generates preliminary answers and insights
    4. Provides citations and confidence levels
    5. Suggests areas for further research
    
    Returns a structured response with:
    - Timestamp of analysis
    - Important disclaimers
    - AI-generated insights
    - Data gap analysis
    - Methodology notes
    """
    try:
        logger.info(f"Generating AI research insights for session {session_id}")
        
        if "questions" not in session.data:
            raise HTTPException(status_code=400, detail="Generate questions first")
            
        if "data_gaps" not in session.data:
            # First analyze data gaps by calling the gaps endpoint
            questions = session.data["questions"]
            gaps_response = await analyze_gaps(session_id, session)
            session.data["data_gaps"] = gaps_response  # analyze_gaps already updates the session
            
        # Get the LLM instance with research-focused system prompt
        llm = get_llm(temperature=0.7)  # Slightly higher temperature for more comprehensive responses
        
        # Prepare context from questions and gaps
        main_questions = session.data["questions"].get("main_questions", [])
        data_gaps = session.data["data_gaps"]
        
            # Search the web for relevant information
        all_web_results = []
        for question in main_questions:
            # Make sure we're using the question text, not the dict
            question_text = question.get("question", "")
            if question_text:
                web_results = await fetch_webpage(
                    urls=[], # The tool will search automatically
                    query=question_text
                )
                if web_results:
                    all_web_results.extend(web_results)
        
        # Create comprehensive prompt for the LLM
        prompt = AI_INSIGHTS_PROMPT.format(
            main_questions=[q.get("question", "") for q in main_questions],
            data_gaps=data_gaps,
            web_results=all_web_results
        )
        
        response = await llm.invoke(prompt)
        
        # Structure the insights
        insights = {
            "timestamp": datetime.now().isoformat(),
            "disclaimer": """IMPORTANT: These insights are AI-generated based on available information and should be 
                        considered preliminary. All findings should be verified through rigorous research and peer-reviewed sources.""",
            "insights": response.content,
            "data_gaps": data_gaps,
            "methodology_note": """This analysis combines web-sourced information with AI interpretation. 
                               Statistics and claims should be independently verified."""
        }
        
        # Store in session for future reference
        session.data["ai_insights"] = insights
        
        logger.info(f"Successfully generated AI insights for session {session_id}")
        return insights
        
    except Exception as e:
        logger.error(f"Error generating AI insights: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error generating AI insights: {str(e)}"
        )
    # gaps = []
    # try:
    #     main_question = questions.get("main_question", "")
    #     sub_questions = questions.get("sub_questions", [])
        
    #     # Get only main questions
    #     main_questions = []
        
    #     # Handle new format with multiple main questions
    #     if isinstance(questions.get("main_questions"), list):
    #         for idx, main_q in enumerate(questions["main_questions"]):
    #             main_questions.append({
    #                 "question": main_q,
    #                 "type": "main",
    #                 "index": idx,
    #                 "sub_questions": questions.get("sub_questions", {}).get(idx, [])
    #             })
    #     else:
    #         # Legacy format support
    #         if questions.get("main_question"):
    #             main_questions.append({
    #                 "question": questions.get("main_question"),
    #                 "type": "main",
    #                 "index": 0,
    #                 "sub_questions": questions.get("sub_questions", [])
    #             })

    #     # Analyze gaps for each question
    #     if source_type == "dataset":
    #         available_vars = set(source_data.get("variables", []))
    #         for q_info in main_questions:
    #             required_vars = extract_required_variables(q_info["question"])
    #             missing_vars = required_vars - available_vars
    #             if missing_vars:
    #                 gap_entry = {
    #                     "question": q_info["question"],
    #                     "question_type": "main",
    #                     "question_index": q_info["index"],
    #                     "missing_variables": list(missing_vars),
    #                     "suggested_sources": suggest_data_sources(missing_vars),
    #                     "sub_questions": len(q_info["sub_questions"])
    #                 }
    #                 gaps.append(gap_entry)
    #     else:
    #         # For topic-based research, focus on methodological gaps
    #         for q_info in main_questions:
    #             gap_entry = {
    #                 "question": q_info["question"],
    #                 "question_type": "main",
    #                 "question_index": q_info["index"],
    #                 "data_requirements": suggest_data_requirements(q_info["question"]),
    #                 "methodology_gaps": suggest_methodology(q_info["question"]),
    #                 "sub_questions": len(q_info["sub_questions"])
    #             }
    #             gaps.append(gap_entry)
                
    #     return gaps
    # except Exception as e:
    #     logger.error(f"Error analyzing gaps: {e}")
    #     return []

def extract_required_variables(question: str) -> set:
    """Extract potential required variables from a question"""
    # Common health and demographic variables to look for
    common_vars = {
        "age", "gender", "location", "education", "income", "household",
        "health", "mortality", "morbidity", "vaccination", "treatment",
        "symptoms", "diagnosis", "outcome", "intervention"
    }
    
    # Look for these variables in the question
    found_vars = set()
    question_lower = question.lower()
    
    for var in common_vars:
        if var in question_lower:
            found_vars.add(var)
            
    # Add specific health condition variables if mentioned
    health_conditions = {
        "malaria", "hiv", "tuberculosis", "pneumonia", "diarrhea",
        "malnutrition", "anemia", "diabetes", "hypertension"
    }
    
    for condition in health_conditions:
        if condition in question_lower:
            found_vars.add(f"{condition}_status")
            found_vars.add(f"{condition}_treatment")
            
    return found_vars

def suggest_data_sources(variables: set) -> List[str]:
    """Suggest potential data sources for missing variables"""
    source_mapping = {
        "health": ["Health Management Information System (HMIS)", "Hospital Records"],
        "mortality": ["Civil Registration and Vital Statistics", "Health Facility Reports"],
        "education": ["Ministry of Education Database", "School Records"],
        "income": ["Household Economic Survey", "World Bank Development Indicators"],
        "location": ["Geographic Information System (GIS)", "Census Data"],
        "vaccination": ["Immunization Records", "Health Facility Data"],
        "treatment": ["Clinical Records", "Pharmacy Data"],
        "household": ["Demographic Health Survey", "Household Census"]
    }
    
    suggestions = []
    for var in variables:
        for key, sources in source_mapping.items():
            if key in var.lower():
                suggestions.extend(sources)
    
    return list(set(suggestions)) if suggestions else ["No specific sources identified"]

def suggest_data_requirements(question: str) -> List[str]:
    """Suggest data requirements for a research question"""
    requirements = []
    question_lower = question.lower()
    
    # Add demographic data requirements
    if any(term in question_lower for term in ["age", "gender", "population"]):
        requirements.append("Demographic data (age, gender, population statistics)")
        
    # Add temporal data requirements
    if any(term in question_lower for term in ["trend", "pattern", "over time", "duration"]):
        requirements.append("Time series data with regular intervals")
        
    # Add geographic data requirements
    if any(term in question_lower for term in ["region", "area", "location", "geographical"]):
        requirements.append("Geographic and spatial data")
        
    # Add health data requirements
    if any(term in question_lower for term in ["health", "disease", "condition", "symptoms"]):
        requirements.append("Clinical and health outcome data")
        
    # Add default requirement if none found
    if not requirements:
        requirements.append("Basic demographic and health indicators")
        
    return requirements

def suggest_methodology(question: str) -> List[str]:
    """Suggest methodological approaches"""
    methods = []
    question_lower = question.lower()
    
    # Suggest quantitative methods
    if any(term in question_lower for term in ["how many", "rate", "percentage", "level", "amount"]):
        methods.append("Quantitative analysis using statistical methods")
        methods.append("Descriptive and inferential statistical analysis")
        
    # Suggest qualitative methods
    if any(term in question_lower for term in ["why", "how", "experience", "perception"]):
        methods.append("Qualitative research through interviews or focus groups")
        methods.append("Thematic analysis of responses")
        
    # Suggest mixed methods
    if any(term in question_lower for term in ["impact", "effect", "influence", "relationship"]):
        methods.append("Mixed methods approach combining quantitative and qualitative data")
        methods.append("Triangulation of multiple data sources")
        
    # Add default method if none found
    if not methods:
        methods.append("Mixed methods approach with both quantitative and qualitative components")
        
    return methods

@app.post("/research/analyze/gaps")
async def analyze_data_gaps(
    session_id: UUID,
    session: SessionInfo = Depends(get_active_session)
):
    """Identify data gaps and analysis approach for research questions"""
    try:
        logger.info(f"Analyzing data gaps for session {session_id}")
        
        if "questions" not in session.data:
            logger.error("Questions not found in session data")
            raise HTTPException(status_code=400, detail="Generate questions first")
        
        questions = session.data["questions"]
        if not isinstance(questions, dict):
            logger.error("Invalid questions format in session data")
            raise HTTPException(status_code=500, detail="Invalid questions format")
            
        # Get main questions
        main_questions = questions.get("main_questions", [])
        if not main_questions:
            logger.error("No main questions found in session data")
            raise HTTPException(status_code=400, detail="No main questions found")
            
        logger.info(f"Analyzing gaps for {len(main_questions)} main questions")
        
        # Get research context
        source_data = session.data.get("source", {})
        source_type = source_data.get("source_type", "topic")
        
        # Initialize available data context
        available_data = {
            "type": source_type,
            "variables": source_data.get("variables", []) if source_type == "dataset" else [],
            "table": source_data.get("table_name", "") if source_type == "dataset" else ""
        }
        
        # Analyze gaps for each question
        all_gaps = []
        for idx, main_q in enumerate(main_questions):
            # Analyze main question
            main_gaps = analyze_question_gaps(
                main_q.get("question", ""),
                available_data,
                is_main=True,
                question_index=idx
            )
            all_gaps.extend(main_gaps)
            
            # Analyze sub-questions if any
            sub_questions = main_q.get("sub_questions", [])
            for sub_idx, sub_q in enumerate(sub_questions):
                sub_gaps = analyze_question_gaps(
                    sub_q.get("question", ""),
                    available_data,
                    is_main=False,
                    question_index=idx,
                    sub_question_index=sub_idx
                )
                all_gaps.extend(sub_gaps)
        
        session.data["gaps"] = all_gaps
        session.current_step = "gaps_analyzed"
        logger.info(f"Found {len(all_gaps)} potential gaps in research")
        
        return all_gaps
    except Exception as e:
        logger.error(f"Failed to analyze data gaps: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to analyze data gaps: {str(e)}"
        )

# --- Step 5: Literature Search ---
@app.post("/research/literature/search")
async def search_literature(
    session_id: UUID,
    session: SessionInfo = Depends(get_active_session)
):
    """Search literature based on research questions"""
    try:
        logger.info(f"Starting literature search for session {session_id}")
        
        if "questions" not in session.data:
            logger.error("No questions found in session data")
            raise HTTPException(status_code=400, detail="Generate questions first")
        
        questions = session.data["questions"]
        
        # Handle both new and legacy formats
        if isinstance(questions.get("main_questions"), list):
            main_questions = questions["main_questions"]
        else:
            main_questions = [questions.get("main_question")] if questions.get("main_question") else []
            
        if not main_questions:
            logger.error("No main questions found to search literature for")
            raise HTTPException(status_code=400, detail="No main questions found")
            
        logger.info(f"Searching literature for {len(main_questions)} main questions")
        
        # Search across multiple sources for each main question
        all_results = []
        for idx, question in enumerate(main_questions):
            logger.info(f"Searching for main question {idx + 1}: {question[:100]}...")
            
            # Search across multiple sources
            question_results = []
            google_results = await fetch_google_scholar(question)
            crossref_results = await fetch_crossref(question)
            
            # Add source and question index to results
            for result in google_results:
                result["question_index"] = idx
                result["source"] = "Google Scholar"
                question_results.append(result)
                
            for result in crossref_results:
                result["question_index"] = idx
                result["source"] = "CrossRef"
                question_results.append(result)
                
            all_results.extend(question_results)
            logger.info(f"Found {len(question_results)} results for question {idx + 1}")
        
        session.data["literature"] = all_results
        session.current_step = "literature_found"
        
        logger.info(f"Literature search completed. Total results: {len(all_results)}")
        return all_results
        
    except Exception as e:
        logger.error(f"Literature search failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Literature search failed: {str(e)}"
        )

# --- Utility Endpoints ---
@app.get("/research/session/{session_id}")
async def get_session_status(
    session_id: UUID,
    session: SessionInfo = Depends(get_active_session)
):
    """Get current session status and data"""
    return session

def parse_llm_question_response(response: str) -> Dict:
    """Parse the LLM's response into structured question data with multiple main questions.
    Handles various response formats and provides robust error handling.
    """
    try:
        logger.debug(f"Parsing LLM response: {response[:200]}...")
        result = {
            "main_questions": [],
            "sub_questions": {},  # Dictionary mapping main question index to its sub-questions
            "approach": ""
        }
        
        # Split response into sections
        sections = response.split("\n")
        current_main_index = None
        current_section = None
        
        for line in sections:
            line = line.strip()
            if not line:
                continue
                
            # Detect main question sections
            if line.lower().startswith("main question"):
                current_main_index = len(result["main_questions"])
                current_section = "main"
                continue
                
            # Detect sub-questions sections
            elif line.lower().startswith("sub-questions"):
                current_section = "sub"
                if current_main_index not in result["sub_questions"]:
                    result["sub_questions"][current_main_index] = []
                continue
                
            # Detect approach section
            elif line.lower().startswith("approach"):
                current_section = "approach"
                continue
                
            # Process content based on section
            if current_section == "main" and line and not line.lower().startswith("sub-questions"):
                question = line.lstrip("123456789.- ").strip()
                if question:
                    result["main_questions"].append(question)
                    
            elif current_section == "sub" and line.startswith("-"):
                if current_main_index is not None:
                    sub_q = line.lstrip("- ").strip()
                    if sub_q:
                        if current_main_index not in result["sub_questions"]:
                            result["sub_questions"][current_main_index] = []
                        result["sub_questions"][current_main_index].append(sub_q)
                        
            elif current_section == "approach":
                if result["approach"]:
                    result["approach"] += " " + line
                else:
                    result["approach"] = line.lstrip("123456789.- ").strip()
        
        # Validate parsed results
        if not result["main_questions"]:
            logger.warning("No main questions found in response")
            # Try alternative parsing
            parts = response.split("\n\n")
            for part in parts:
                if "?" in part and not part.startswith("-"):
                    result["main_questions"].append(part.strip())
        
        # Ensure we have exactly 5 main questions
        if len(result["main_questions"]) > 5:
            result["main_questions"] = result["main_questions"][:5]
        
        # Ensure each main question has exactly 3 sub-questions
        for idx in range(len(result["main_questions"])):
            if idx not in result["sub_questions"]:
                result["sub_questions"][idx] = []
            if len(result["sub_questions"][idx]) > 3:
                result["sub_questions"][idx] = result["sub_questions"][idx][:3]
        
        logger.info(f"Successfully parsed response: Found {len(result['main_questions'])} main questions")
        return result
        
    except Exception as e:
        logger.error(f"Failed to parse LLM response: {e}")
        # Return a safe default with error indication
        return {
            "main_question": "Error parsing response",
            "sub_questions": [],
            "approach": f"Parser error: {str(e)}"
        }