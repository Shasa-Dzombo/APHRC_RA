"""
LLM configuration using Ollama
"""
import os
from ollama import Client
from dotenv import load_dotenv
import logging

# Import research prompts
from prompts.research_prompts import PROMPT_STEP2, PROMPT_STEP3, PROMPT_STEP4, PROMPT_ANSWER_GENERATION

def get_llm(model: str = None, temperature: float = 0.1, host: str = None):
    """
    Returns an Ollama client instance with the given settings.
    
    Args:
        model: The Ollama model to use (defaults to OLLAMA_MODEL from .env)
        temperature: Temperature setting for response generation (0.0 to 1.0)
        host: Ollama server host (defaults to OLLAMA_HOST from .env)
    
    Returns:
        OllamaClient: Configured Ollama client wrapper
    """
    # Load environment variables
    load_dotenv()
    
    # Get configuration from environment or use defaults
    ollama_host = host or os.getenv("OLLAMA_HOST", "https://api.ollama.vizanalyticx.com")
    ollama_model = model or os.getenv("OLLAMA_MODEL", "mistral:latest")
    timeout = int(os.getenv("OLLAMA_TIMEOUT", "180"))  # Default 3 minutes, configurable
    print(f"Using Ollama model: {ollama_model} at {ollama_host} with temperature {temperature} and timeout {timeout}s")
    
    # Initialize the Ollama client with configurable timeout
    client = Client(host=ollama_host, timeout=timeout)
    
    # Return wrapped client with our interface
    return OllamaClient(client, ollama_model, temperature)


def get_system_prompt(task_type: str = "question_generation") -> str:
    """
    Get the appropriate system prompt for the given task type
    
    Args:
        task_type: The type of task (question_generation, analysis, data_gaps, answer_generation)
    
    Returns:
        The appropriate system prompt string
    """
    prompt_map = {
        "question_generation": PROMPT_STEP2,
        "analysis": PROMPT_STEP3,
        "data_gaps": PROMPT_STEP4,
        "answer_generation": PROMPT_ANSWER_GENERATION
    }
    
    return prompt_map.get(task_type, PROMPT_STEP2)


class OllamaClient:
    """
    Wrapper class to provide a consistent interface for Ollama
    """
    def __init__(self, client, model, temperature):
        self.client = client
        self.model = model
        self.temperature = temperature
    
    def invoke(self, prompt: str, system_prompt: str = None):
        """
        Invoke the Ollama model with a prompt
        
        Args:
            prompt: The user prompt/question
            system_prompt: System instructions for the model (defaults to question generation prompt)
        
        Returns:
            Response object with .content attribute
        """
        # Use question generation prompt as default if no system prompt provided
        if system_prompt is None:
            system_prompt = get_system_prompt("question_generation")
        
        # Retry logic with exponential backoff for better reliability
        import time
        max_retries = 3
        base_delay = 2  # seconds
        
        for attempt in range(max_retries):
            try:
                logging.info(f"Attempting Ollama call (attempt {attempt + 1}/{max_retries})...")
                
                response = self.client.chat(
                    model=self.model,
                    messages=[{
                        "role": "system",
                        "content": str(system_prompt)
                    }, {
                        "role": "user",
                        "content": str(prompt)
                    }],
                    options={
                        "temperature": float(self.temperature)
                    }
                )
                
                logging.info("Ollama call successful!")
                # Return response in expected format
                return OllamaResponse(response['message']['content'])
                
            except Exception as e:
                logging.warning(f"Ollama call attempt {attempt + 1} failed: {e}")
                
                # If this is the last attempt, raise the exception
                if attempt == max_retries - 1:
                    logging.error(f"All {max_retries} Ollama call attempts failed. Final error: {e}")
                    raise
                
                # Wait before retrying (exponential backoff)
                delay = base_delay * (2 ** attempt)
                logging.info(f"Waiting {delay} seconds before retry...")
                time.sleep(delay)


class OllamaResponse:
    """
    Response wrapper to match expected interface
    """
    def __init__(self, content):
        self.content = content
