"""
Unified Pydantic Models for Research API
"""
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from uuid import UUID, uuid4

# --- Core Research Models ---
class ProjectInfo(BaseModel):
    title: str
    description: str
    area_of_study: Optional[str] = None
    geography: Optional[str] = None

    @validator("title", "description")
    def non_empty(cls, v):
        if not v.strip():
            raise ValueError("Cannot be empty")
        return v.strip()

class ResearchSource(BaseModel):
    source_type: str  # "dataset" or "topic"
    title: str
    description: str
    table_name: Optional[str] = None
    variables: Optional[List[str]] = []
    area_of_study: Optional[str] = None
    geography: Optional[str] = None

class ResearchQuestion(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    text: str
    question_type: str
    parent_question_id: Optional[str] = None

class SubQuestion(BaseModel):
    question: str
    main_question_index: int

class Question(BaseModel):
    question: str
    sub_questions: List[SubQuestion]

class QuestionResponse(BaseModel):
    main_questions: List[Question]
    approach: str

class SubQuestionMap(BaseModel):
    sub_question_id: str
    sub_question: str
    data_requirements: str
    analysis_approach: str
    sub_question: str
    data_requirements: str
    analysis_approach: str

class ResearchVariable(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    description: Optional[str] = None
    sub_question_id: str

class DataGap(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    missing_variable: str
    gap_description: str
    suggested_sources: str
    sub_question_id: str

class LiteratureReference(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    title: str
    authors: List[str] = []
    abstract: str = ""
    year: Optional[int] = None
    venue: str = ""
    url: str = ""
    relevance: float = 0.0
    source: str = ""
    sub_question_id: str = ""
    hierarchy_rank: Optional[int] = None
    is_primary: bool = False

class HierarchicalLiterature(BaseModel):
    """Hierarchical literature structure with primary and supporting papers"""
    sub_question_id: str
    sub_question_text: str
    primary_paper: Optional[LiteratureReference] = None
    supporting_papers: List[LiteratureReference] = []
    total_papers: int = 0
    max_relevance_score: float = 0.0

# --- Session Models ---
class SessionInfo(BaseModel):
    session_id: UUID = Field(default_factory=uuid4)
    research_type: str
    source_id: Optional[str] = None
    current_step: str = "started"
    data: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    expires_at: datetime = Field(default_factory=lambda: datetime.utcnow() + timedelta(hours=24))

    @validator("expires_at", pre=True, always=True)
    def set_expires_at(cls, v, values):
        if v is None and "created_at" in values:
            return values["created_at"] + timedelta(hours=24)
        return v

class SessionResponse(BaseModel):
    session_id: UUID
    expires_at: datetime
    message: str

# --- Database Schema Models ---
class DatabaseColumn(BaseModel):
    name: str
    type: str
    nullable: bool
    description: str
    primary_key: Optional[bool] = False
    foreign_key: Optional[str] = None

class DatabaseTable(BaseModel):
    name: str
    description: str
    columns: List[DatabaseColumn]
    column_count: int

class DatabaseSchemaResponse(BaseModel):
    database_name: str
    version: str
    description: str
    tables: List[DatabaseTable]
    total_tables: int
    last_updated: str

class TableDetailsResponse(BaseModel):
    table_name: str
    description: str
    columns: List[DatabaseColumn]
    total_columns: int
    primary_keys: List[str]
    foreign_keys: List[str]

# --- Request Models ---
class ProjectRequest(BaseModel):
    source_type: str  # "dataset" or "topic"
    title: str
    description: str
    area_of_study: Optional[str] = None
    geography: Optional[str] = None
    custom_sub_questions: Optional[List[str]] = []

class SessionRequest(BaseModel):
    session_id: UUID = Field(
        ...,  # ... means the field is required
        example="123e4567-e89b-12d3-a456-426614174000"  # Custom example for documentation
    )

class LiteratureSearchRequest(BaseModel):
    query: str
    limit: int = Field(default=10, ge=1, le=50)

class SubQuestionAnalysisRequest(BaseModel):
    session_id: UUID
    main_question_ids: List[str] = Field(..., description="List of main question IDs to analyze their sub-questions")

class QuestionSelectionRequest(BaseModel):
    session_id: UUID
    selected_main_question_ids: List[str] = Field(..., description="List of main question IDs to select")

# --- Response Models ---
class ResearchQuestionResponse(BaseModel):
    id: str
    text: str
    question_type: str
    parent_question_id: Optional[str] = None
    sub_questions: Optional[List['ResearchQuestionResponse']] = []

class AIInsightResponse(BaseModel):
    timestamp: str
    disclaimer: str
    insights: str
    data_gaps: List[Dict[str, Any]]
    methodology_note: str

class SubQuestionMappingResponse(BaseModel):
    sub_question_id: str
    sub_question: str
    data_requirements: str
    analysis_approach: str

class DataGapResponse(BaseModel):
    id: str
    missing_variable: str
    gap_description: str
    suggested_sources: str
    sub_question_id: str

class LiteratureResponse(BaseModel):
    id: str
    title: str
    authors: List[str]
    abstract: str
    year: Optional[int]
    venue: str
    url: str
    relevance: float
    source: str
    sub_question_id: str

class ResearchAnalysisResponse(BaseModel):
    main_questions: List[ResearchQuestionResponse]
    sub_questions: List[ResearchQuestionResponse]
    mappings: List[SubQuestionMappingResponse]
    data_gaps: List[DataGapResponse]
    literature: Dict[str, List[LiteratureResponse]]

# --- Question Selection Models ---
class QuestionSelectionResponse(BaseModel):
    session_id: UUID
    selected_questions: List[ResearchQuestionResponse]
    message: str

class SelectedQuestionsListResponse(BaseModel):
    session_id: UUID
    selected_main_questions: List[ResearchQuestionResponse]
    total_selected: int

# --- Sub-question Answering Models ---
class SubQuestionAnswer(BaseModel):
    sub_question_id: str
    sub_question_text: str
    answer: str
    confidence_score: Optional[float] = None
    sources_used: Optional[List[str]] = []

class SubQuestionAnswersResponse(BaseModel):
    session_id: UUID
    answers: List[SubQuestionAnswer]
    total_answered: int
    processing_summary: str

# --- Data Gap Analysis Models ---
class DataGapAnalysis(BaseModel):
    type: str
    variable: Optional[str] = None
    requirement: Optional[str] = None
    question_type: str
    question_index: int
    sub_question_index: Optional[int] = None
    suggested_sources: List[str]
    description: str
    suggested_methodology: Optional[List[str]] = None

# --- Session Status Models ---
class SessionStatusResponse(BaseModel):
    session_id: UUID
    research_type: str
    current_step: str
    created_at: str
    expires_at: str
    data_summary: Dict[str, Any]

# --- API Response Models ---
class DatabaseOverviewResponse(BaseModel):
    database_name: str
    tables: List[Dict[str, Any]]

class TableInfoResponse(BaseModel):
    table_name: str
    columns: List[Dict[str, Any]]
    relationships: List[Dict[str, Any]]

class ResearchStartResponse(BaseModel):
    session_id: UUID
    research_type: str
    current_step: str

class QuestionsGeneratedResponse(BaseModel):
    main_questions: List[Dict[str, Any]]
    sub_questions: Dict[str, List[str]]
    approach: str

# --- Error Models ---
class ErrorResponse(BaseModel):
    detail: str
    error_code: Optional[str] = None
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())

# --- Utility Models ---
class LLMQuestionParseResult(BaseModel):
    main_questions: List[Dict[str, Any]]
    sub_questions: Dict[str, List[str]]
    approach: str

class AnalysisParseResult(BaseModel):
    data_requirements: str
    analysis_approach: str

# Enable forward references
ResearchQuestionResponse.model_rebuild()

# Export all models
__all__ = [
    # Core models
    "ProjectInfo", "ResearchSource", "ResearchQuestion", "SubQuestion", "Question", 
    "QuestionResponse", "SubQuestionMap", "ResearchVariable", "DataGap", 
    "LiteratureReference", "HierarchicalLiterature",
    
    # Session models
    "SessionInfo", "SessionResponse",
    
    # Database models
    "DatabaseColumn", "DatabaseTable", "DatabaseSchemaResponse", "TableDetailsResponse",
    
    # Request models
    "ProjectRequest", "SessionRequest", "LiteratureSearchRequest", 
    "SubQuestionAnalysisRequest", "QuestionSelectionRequest",
    
    # Response models
    "ResearchQuestionResponse", "AIInsightResponse", "SubQuestionMappingResponse",
    "DataGapResponse", "LiteratureResponse", "ResearchAnalysisResponse",
    "QuestionSelectionResponse", "SelectedQuestionsListResponse",
    "SubQuestionAnswer", "SubQuestionAnswersResponse", "SessionStatusResponse",
    
    # Analysis models
    "DataGapAnalysis",
    
    # API response models
    "DatabaseOverviewResponse", "TableInfoResponse", "ResearchStartResponse", 
    "QuestionsGeneratedResponse",
    
    # Error models
    "ErrorResponse",
    
    # Utility models
    "LLMQuestionParseResult", "AnalysisParseResult"
]