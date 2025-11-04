from pydantic import BaseModel
from typing import List, Optional

class DatabaseRelationship(BaseModel):
    """Model for database table relationships"""
    source_table: str
    source_column: str
    target_table: str
    target_column: str

class DatabaseColumn(BaseModel):
    """Model for database column information"""
    name: str
    type: str
    nullable: bool = True
    description: str = ""
    primary_key: bool = False
    foreign_key: Optional[str] = None

class DatabaseTable(BaseModel):
    """Model for database table information"""
    name: str
    description: str = ""
    columns: List[DatabaseColumn]
    column_count: int
    relationships: List[DatabaseRelationship] = []

class DatabaseSchemaResponse(BaseModel):
    """Response model for database schema information"""
    database_name: str
    version: str
    description: str = ""
    tables: List[DatabaseTable]
    total_tables: int
    last_updated: str
    relationships: List[DatabaseRelationship] = []

class TableDetailsResponse(BaseModel):
    """Response model for detailed table information"""
    name: str
    description: str
    columns: List[DatabaseColumn]
    column_count: int
    primary_keys: List[str]
    foreign_keys: List[str]
    relationships: List[DatabaseRelationship] = []