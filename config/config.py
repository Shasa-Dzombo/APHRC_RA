"""
Settings and comprehensive logging configuration
"""
import os
import logging
from functools import lru_cache
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Settings:
    """Application settings"""
    def __init__(self):
        self.app_name: str = "AI Research Agent API"
        self.version: str = "1.0.0"
        self.debug: bool = os.getenv("DEBUG", "False").lower() == "true"
        self.OLLAMA_HOST: str = os.getenv("OLLAMA_HOST", "")
        
        # Supabase settings
        self.supabase_url: str = os.getenv("SUPABASE_URL", "")
        self.supabase_key: str = os.getenv("SUPABASE_CLIENT_KEY", "")
        
        # Debug: Print Supabase config (remove in production)
        if self.debug:
            print(f"Supabase URL loaded: {'Yes' if self.supabase_url else 'No'}")
            print(f"Supabase Key loaded: {'Yes' if self.supabase_key else 'No'}")
           
        
        # Server settings
        self.host: str = os.getenv("HOST", "localhost")
        self.port: int = int(os.getenv("PORT", "8000"))
        self.reload: bool = os.getenv("RELOAD", "True").lower() == "true"
        self.factory: bool = True
        self.environment: str = os.getenv("ENVIRONMENT", "development")
        
        # CORS settings
        self.allowed_origins: list = [
            "http://localhost:3000",
            "http://localhost:8080", 
            "http://localhost:8501",  # Streamlit default port
            "http://127.0.0.1:3000",
            "http://127.0.0.1:8080",
            "http://127.0.0.1:8501"
        ]
        
        # Add custom origins from environment
        custom_origins = os.getenv("ALLOWED_ORIGINS", "")
        if custom_origins:
            self.allowed_origins.extend([origin.strip() for origin in custom_origins.split(",")])
        
        # Security settings
        self.allowed_hosts: list = ["localhost", "127.0.0.1", "0.0.0.0"]
        custom_hosts = os.getenv("ALLOWED_HOSTS", "")
        if custom_hosts:
            self.allowed_hosts.extend([host.strip() for host in custom_hosts.split(",")])
        
        # Session settings
        self.session_expiry_hours: int = int(os.getenv("SESSION_EXPIRY_HOURS", "24"))
        
        # Literature search settings
        self.default_search_limit: int = int(os.getenv("DEFAULT_SEARCH_LIMIT", "10"))
        self.max_search_limit: int = int(os.getenv("MAX_SEARCH_LIMIT", "50"))

@lru_cache()
def get_settings():
    """Get cached settings instance"""
    return Settings()

def setup_logging():
    """Setup comprehensive logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('research_agent.log')
        ]
    )
    
    # Configure specific loggers
    logging.getLogger("uvicorn").setLevel(logging.INFO)
    logging.getLogger("fastapi").setLevel(logging.INFO)
