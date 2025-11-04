"""
Database schema utilities for exploring available data
"""
import json
import os
from typing import Dict, List, Any, Tuple
from model.database_models import (
    DatabaseColumn, DatabaseTable, DatabaseSchemaResponse,
    TableDetailsResponse, DatabaseRelationship
)

def load_database_schema() -> Dict[str, Any]:
    """Load the database schema from the JSON file"""
    schema_path = os.path.join(os.path.dirname(__file__), "..", "database", "database_schema.json")
    
    try:
        with open(schema_path, 'r', encoding='utf-8') as f:
            schema = json.load(f)
        return schema
    except FileNotFoundError:
        raise FileNotFoundError(f"Database schema file not found at {schema_path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in database schema file: {e}")

def extract_relationships(tables_data: Dict[str, Any]) -> List[DatabaseRelationship]:
    """Extract relationships from table definitions"""
    relationships = []
    
    for table_name, table_info in tables_data.items():
        columns_data = table_info.get("columns", {})
        
        for col_name, col_info in columns_data.items():
            if col_info.get("foreign_key"):
                ref = col_info["foreign_key"]["references"].split(".")
                if len(ref) == 2:
                    target_table, target_column = ref
                    relationships.append(DatabaseRelationship(
                        source_table=table_name,
                        source_column=col_name,
                        target_table=target_table,
                        target_column=target_column
                    ))
    
    return relationships

def parse_database_schema() -> DatabaseSchemaResponse:
    """Parse the database schema into structured response format"""
    schema = load_database_schema()
    db_info = schema.get("database", {})
    
    tables = []
    tables_data = db_info.get("tables", {})
    all_relationships = extract_relationships(tables_data)

    # Group relationships by table
    table_relationships = {}
    for rel in all_relationships:
        if rel.source_table not in table_relationships:
            table_relationships[rel.source_table] = []
        table_relationships[rel.source_table].append(rel)
    
    for table_name, table_info in tables_data.items():
        columns = []
        columns_data = table_info.get("columns", {})
        
        for col_name, col_info in columns_data.items():
            # Skip constraint entries
            if col_name == "CONSTRAINT":
                continue
                
            column = DatabaseColumn(
                name=col_name,
                type=col_info.get("type", ""),
                nullable=col_info.get("nullable", True),
                description=col_info.get("description", ""),
                primary_key=col_info.get("primary_key", False),
                foreign_key=col_info.get("foreign_key", {}).get("references") if col_info.get("foreign_key") else None
            )
            columns.append(column)
        
        table = DatabaseTable(
            name=table_name,
            description=table_info.get("description", ""),
            columns=columns,
            column_count=len(columns),
            relationships=table_relationships.get(table_name, [])
        )
        tables.append(table)
    
    return DatabaseSchemaResponse(
        database_name=db_info.get("name", ""),
        version=db_info.get("version", ""),
        description=db_info.get("description", ""),
        tables=tables,
        total_tables=len(tables),
        last_updated=db_info.get("last_updated", ""),
        relationships=all_relationships
    )

async def get_table_details(table_name: str) -> TableDetailsResponse:
    """Get detailed information about a specific table"""
    schema = load_database_schema()
    tables_data = schema.get("database", {}).get("tables", {})
    
    if table_name not in tables_data:
        raise ValueError(f"Table '{table_name}' not found in database schema")
    
    table_info = tables_data[table_name]
    columns_data = table_info.get("columns", {})
    
    columns = []
    primary_keys = []
    foreign_keys = []
    
    for col_name, col_info in columns_data.items():
        # Skip constraint entries
        if col_name == "CONSTRAINT":
            continue
            
        column = DatabaseColumn(
            name=col_name,
            type=col_info.get("type", ""),
            nullable=col_info.get("nullable", True),
            description=col_info.get("description", ""),
            primary_key=col_info.get("primary_key", False),
            foreign_key=col_info.get("foreign_key", {}).get("references") if col_info.get("foreign_key") else None
        )
        columns.append(column)
        
        # Track keys
        if col_info.get("primary_key"):
            primary_keys.append(col_name)
        if col_info.get("foreign_key"):
            foreign_keys.append(f"{col_name} -> {col_info['foreign_key']['references']}")
    
    return TableDetailsResponse(
        name=table_name,
        description=table_info.get("description", ""),
        columns=columns,
        column_count=len(columns),
        primary_keys=primary_keys,
        foreign_keys=foreign_keys,
        relationships=[]  # Add relationships if needed
    )

def get_available_table_names() -> List[str]:
    """Get a list of all available table names"""
    schema = load_database_schema()
    tables_data = schema.get("database", {}).get("tables", {})
    return list(tables_data.keys())

def extract_database_keywords() -> Dict[str, List[str]]:
    """Extract meaningful keywords from database table names, column names, and descriptions"""
    schema = load_database_schema()
    tables_data = schema.get("database", {}).get("tables", {})
    
    keywords = {
        "table_names": [],
        "column_names": [],
        "descriptions": []
    }
    
    for table_name, table_info in tables_data.items():
        # Extract words from table names
        table_words = table_name.replace('_', ' ').split()
        keywords["table_names"].extend([word.lower() for word in table_words if len(word) > 2])
        
        # Extract words from table description
        table_desc = table_info.get("description", "")
        if table_desc:
            desc_words = table_desc.lower().replace(',', ' ').replace('.', ' ').split()
            keywords["descriptions"].extend([word for word in desc_words if len(word) > 3])
        
        # Extract words from column names and descriptions
        columns_data = table_info.get("columns", {})
        for col_name, col_info in columns_data.items():
            if col_name == "CONSTRAINT":
                continue
                
            # Column name words
            col_words = col_name.replace('_', ' ').split()
            keywords["column_names"].extend([word.lower() for word in col_words if len(word) > 2])
            
            # Column description words
            col_desc = col_info.get("description", "")
            if col_desc:
                desc_words = col_desc.lower().replace(',', ' ').replace('.', ' ').split()
                keywords["descriptions"].extend([word for word in desc_words if len(word) > 3])
    
    # Remove duplicates and common words
    common_words = {'for', 'the', 'and', 'with', 'type', 'data', 'null', 'not', 'default', 'reference', 'identifier', 'timestamp', 'character', 'varying'}
    
    for category in keywords:
        keywords[category] = list(set(keywords[category]) - common_words)
        keywords[category].sort()
    
    return keywords

def find_relevant_tables_by_research_context(research_keywords: List[str]) -> Dict[str, Any]:
    """Find relevant tables and columns based on research keywords using database vocabulary"""
    schema = load_database_schema()
    tables_data = schema.get("database", {}).get("tables", {})
    
    # Get all database keywords
    db_keywords = extract_database_keywords()
    all_db_words = set(db_keywords["table_names"] + db_keywords["column_names"] + db_keywords["descriptions"])
    
    # Find matches between research keywords and database keywords
    research_words = set([word.lower().strip('.,!?') for word in research_keywords if len(word) > 3])
    matching_keywords = research_words.intersection(all_db_words)
    
    relevant_tables = {}
    relevant_columns = {}
    
    # Search for matching keywords in tables and columns
    for keyword in matching_keywords:
        # Search in table names and descriptions
        for table_name, table_info in tables_data.items():
            table_match = False
            
            # Check table name
            if keyword in table_name.lower():
                if keyword not in relevant_tables:
                    relevant_tables[keyword] = []
                relevant_tables[keyword].append(table_name)
                table_match = True
            
            # Check table description
            table_desc = table_info.get("description", "").lower()
            if keyword in table_desc and not table_match:
                if keyword not in relevant_tables:
                    relevant_tables[keyword] = []
                relevant_tables[keyword].append(table_name)
            
            # Check columns in this table
            columns_data = table_info.get("columns", {})
            matching_cols = []
            
            for col_name, col_info in columns_data.items():
                if col_name == "CONSTRAINT":
                    continue
                    
                # Check column name
                if keyword in col_name.lower():
                    matching_cols.append(col_name)
                
                # Check column description
                col_desc = col_info.get("description", "").lower()
                if keyword in col_desc and col_name not in matching_cols:
                    matching_cols.append(col_name)
            
            if matching_cols:
                if keyword not in relevant_columns:
                    relevant_columns[keyword] = {}
                relevant_columns[keyword][table_name] = matching_cols
    
    return {
        "database_keywords_found": list(matching_keywords),
        "research_keywords_searched": list(research_words),
        "relevant_tables": relevant_tables,
        "relevant_columns": relevant_columns,
        "total_matches": len(matching_keywords)
    }

def get_database_summary() -> Dict[str, Any]:
    """Get a summary of the database with key statistics and available keywords"""
    schema = load_database_schema()
    db_info = schema.get("database", {})
    tables_data = db_info.get("tables", {})
    
    # Calculate statistics
    total_tables = len(tables_data)
    total_columns = 0
    
    for table_info in tables_data.values():
        columns_count = len([col for col in table_info.get("columns", {}).keys() if col != "CONSTRAINT"])
        total_columns += columns_count
    
    # Get available keywords
    db_keywords = extract_database_keywords()
    
    return {
        "database_name": db_info.get("name", ""),
        "version": db_info.get("version", ""),
        "description": db_info.get("description", ""),
        "statistics": {
            "total_tables": total_tables,
            "total_columns": total_columns,
            "avg_columns_per_table": round(total_columns / total_tables, 1) if total_tables > 0 else 0
        },
        "available_keywords": {
            "from_table_names": db_keywords["table_names"][:10],  # First 10
            "from_column_names": db_keywords["column_names"][:15],  # First 15
            "from_descriptions": db_keywords["descriptions"][:20]  # First 20
        },
        "table_names": list(tables_data.keys())
    }
