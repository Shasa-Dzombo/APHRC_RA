from uuid import UUID, uuid4
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from pydantic import BaseModel, Field, validator

class AIInsightResponse(BaseModel):
    """Response model for AI-generated research insights"""
    timestamp: str
    disclaimer: str
    insights: str
    data_gaps: List[Dict]
    methodology_note: str
    
class SubQuestion(BaseModel):
    question: str
    main_question_index: int

class Question(BaseModel):
    question: str
    sub_questions: List[SubQuestion]
    approach: Optional[str] = None

class QuestionResponse(BaseModel):
    main_questions: List[Question]
    approach: str

class SessionInfo(BaseModel):
    session_id: UUID = Field(default_factory=uuid4)
    research_type: str  # "topic" or "dataset"
    source_id: Optional[str] = None
    current_step: str
    data: Dict = {}
    created_at: datetime = Field(default_factory=datetime.now)
    expires_at: datetime = None

    def __init__(self, **data):
        super().__init__(**data)
        if not self.expires_at:
            self.expires_at = self.created_at + timedelta(hours=24)

class ResearchSource(BaseModel):
    source_type: str = Field(..., description="Type of research: 'topic' or 'dataset'")
    title: str = Field(..., description="Research topic or dataset title")
    description: str = Field(None, description="Detailed description of the research focus")
    table_name: Optional[str] = Field(None, description="Required for dataset-led research")
    variables: List[str] = Field(default_factory=list, description="Required variables for dataset research")

    @validator('table_name')
    def validate_table_name(cls, v, values):
        if values.get('source_type') == 'dataset' and not v:
            raise ValueError("table_name is required for dataset-led research")
        return v