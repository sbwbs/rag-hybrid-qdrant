import os
from dotenv import load_dotenv

class Config:
    """Configuration management for the application"""
    
    def __init__(self):
        # Load environment variables
        load_dotenv()
        
        # API keys
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.qdrant_api_key = os.getenv("QDRANT_API_KEY")
        
        # API endpoints
        self.qdrant_url = os.getenv("QDRANT_URL")
        
        # Search configuration
        self.collection_name = os.getenv("COLLECTION_NAME", "hybrid_rfp_rag")
        self.llm_model = os.getenv("LLM_MODEL", "gpt-4o")
        self.search_top_k = int(os.getenv("SEARCH_TOP_K", "5"))  # Default to 5 results
        
        # Validate configuration
        self.validate()
    
    def validate(self):
        """Validate that all required configuration is present"""
        required_vars = ["openai_api_key", "qdrant_api_key", "qdrant_url"]
        missing = [var for var in required_vars if not getattr(self, var)]
        
        if missing:
            raise ValueError(f"Missing required configuration: {', '.join(missing)}") 
        