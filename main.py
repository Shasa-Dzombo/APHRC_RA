"""
Research Agent API - Main Entry Point
"""
import uvicorn
from config.config import get_settings

def main():
    """Main entry point for the Research Agent API"""
    settings = get_settings()
    
    uvicorn.run(
        "app.app:create_app",
        host=settings.host,
        port=settings.port,
        reload=settings.reload,
        factory=settings.factory
    )

if __name__ == "__main__":
    main()
