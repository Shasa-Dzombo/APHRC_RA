# """
# Unified Pydantic request/response models
# """
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from datetime import datetime
import uuid

class ResearchQuestion(BaseModel):
    main_question: str
    sub_questions: List[str]
    related_variables: List[str]

class LiteratureSearchRequest(BaseModel):
    question: str
    search_depth: Optional[int] = 10

# # Core workflow models
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

class ResearchQuestion(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    text: str
    question_type: str
    parent_question_id: Optional[str] = None

class SubQuestionMap(BaseModel):
    sub_question_id: str
    sub_question: str
    data_requirements: str
    analysis_approach: str

class ResearchVariable(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: Optional[str] = None
    sub_question_id: str

class DataGap(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    missing_variable: str
    gap_description: str
    suggested_sources: str
    sub_question_id: str

class DataGapResponse(BaseModel):
    id: str
    missing_variable: str
    gap_description: str
    suggested_sources: str
    sub_question_id: str

# class LiteratureReference(BaseModel):
#     id: str = Field(default_factory=lambda: str(uuid.uuid4()))
#     title: str
#     authors: List[str] = []
#     abstract: str = ""
#     year: Optional[int] = None
#     venue: str = ""
#     url: str = ""
#     relevance: float = 0.0
#     source: str = ""
#     sub_question_id: str = ""
#     hierarchy_rank: Optional[int] = None  # 1 = primary/most relevant, 2+ = supporting
#     is_primary: bool = False  # True for highest scoring paper

# class HierarchicalLiterature(BaseModel):
#     """Hierarchical literature structure with primary and supporting papers"""
#     sub_question_id: str
#     sub_question_text: str
#     primary_paper: Optional[LiteratureReference] = None  # Highest relevance score
#     supporting_papers: List[LiteratureReference] = []  # Rest in descending order
#     total_papers: int = 0
#     max_relevance_score: float = 0.0

# Database Schema Models
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

# # Request Models
# class ProjectRequest(BaseModel):
#     title: str
#     description: str
#     area_of_study: Optional[str] = None
#     geography: Optional[str] = None
#     custom_sub_questions: Optional[List[str]] = []

# class SessionRequest(BaseModel):
#     session_id: str

# class LiteratureSearchRequest(BaseModel):
#     query: str
#     limit: int = Field(default=10, ge=1, le=50)

# # Response Models
# class ResearchQuestionResponse(BaseModel):
#     id: str
#     text: str
#     question_type: str
#     parent_question_id: Optional[str] = None
#     sub_questions: Optional[List['ResearchQuestionResponse']] = []

# class SessionResponse(BaseModel):
#     session_id: str
#     expires_at: datetime

# class SubQuestionMappingResponse(BaseModel):
#     sub_question_id: str
#     sub_question: str
#     data_requirements: str
#     analysis_approach: str

# class DataGapResponse(BaseModel):
#     id: str
#     missing_variable: str
#     gap_description: str
#     suggested_sources: str
#     sub_question_id: str

# class LiteratureResponse(BaseModel):
#     id: str
#     title: str
#     authors: List[str]
#     abstract: str
#     year: Optional[int]
#     venue: str
#     url: str
#     relevance: float
#     source: str
#     sub_question_id: str

# class ResearchAnalysisResponse(BaseModel):
#     main_questions: List[ResearchQuestionResponse]
#     sub_questions: List[ResearchQuestionResponse]
#     mappings: List[SubQuestionMappingResponse]
#     data_gaps: List[DataGapResponse]
#     literature: Dict[str, List[LiteratureResponse]]

# # Question selection models
# class QuestionSelectionRequest(BaseModel):
#     session_id: str
#     selected_main_question_ids: List[str] = Field(..., description="List of main question IDs to select")

# class SubQuestionAnalysisRequest(BaseModel):
#     session_id: str
#     main_question_ids: List[str] = Field(..., description="List of main question IDs to analyze their sub-questions")

# class QuestionSelectionResponse(BaseModel):
#     session_id: str
#     selected_questions: List[ResearchQuestionResponse]
#     message: str

# class SelectedQuestionsListResponse(BaseModel):
#     session_id: str
#     selected_main_questions: List[ResearchQuestionResponse]
#     total_selected: int

# # Sub-question answering models
# class SubQuestionAnswer(BaseModel):
#     sub_question_id: str
#     sub_question_text: str
#     answer: str
#     confidence_score: Optional[float] = None
#     sources_used: Optional[List[str]] = []

# class SubQuestionAnswersResponse(BaseModel):
#     session_id: str
#     answers: List[SubQuestionAnswer]
#     total_answered: int
#     processing_summary: str

# # Enable forward references
# ResearchQuestionResponse.model_rebuild()
