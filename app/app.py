"""
FastAPI Application Factory
Main entry point for the AI Research Agent API

This module sets up the FastAPI application and integrates the research workflow
endpoints from research_api.py.

The research workflow consists of four main steps:
1. Database exploration
2. Research initialization (topic or dataset based)
3. Question generation and gap analysis
4. Literature search

Each research session maintains state through a unique session ID.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_app() -> FastAPI:
    """Create and configure the FastAPI application"""
    # Initialize FastAPI app
    app = FastAPI(
        title="AI Research Agent API",
        description="Advanced research assistant for topic and dataset-driven research",
        version="1.0.0"
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Import research workflow endpoints
    from app.api import (
        get_database_overview,
        get_table_details,
        start_research,
        generate_research_questions,
        analyze_data_gaps,
        get_ai_research_insights,
        search_literature,
        get_session_status
    )

    # Register research workflow endpoints
    app.get("/database/overview")(get_database_overview)
    app.get("/database/table/{table_name}")(get_table_details)
    app.post("/research/start")(start_research)
    app.post("/research/questions/generate")(generate_research_questions)
    app.post("/research/analyze/ai-insights")(get_ai_research_insights) 
    app.post("/research/analyze/gaps")(analyze_data_gaps) 
    app.post("/research/literature/search")(search_literature)
    app.get("/research/session/{session_id}")(get_session_status)

    @app.get("/")
    async def root():
        """Root endpoint with API information"""
        return {
            "message": "🔬 AI Research Agent API",
            "version": "1.0.0",
            "docs": "/docs",
            "endpoints": {
                "database": {
                    "overview": "/database/overview",
                    "table_details": "/database/table/{table_name}"
                },
                "research": {
                    "start": "/research/start",
                    "questions": "/research/questions/generate",
                    "ai_insights": "/research/analyze/ai-insights",
                    "gaps": "/research/analyze/gaps",
                    "literature": "/research/literature/search",
                    "session": "/research/session/{session_id}"
                }
            }
        }

    # Startup logging
    logger.info("🚀 AI Research Agent API starting up")
    logger.info("📚 API documentation available at /docs")

    return app

# Create the application instance
app = create_app()