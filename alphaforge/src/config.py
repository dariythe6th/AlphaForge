# src/config.py
import os
from typing import Dict, Any

class Settings:
    # API Keys
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")
    
    # Model Configurations
    LLM_ULTRA = "gpt-4"
    LLM_FAST = "gpt-3.5-turbo"
    
    # MCP Server URLs
    MCP_DATA_SERVER = "http://localhost:8001"
    MCP_ALGORITHM_SERVER = "http://localhost:8002"
    
    # Database Connections
    REDIS_URL = "redis://localhost:6379"
    NEO4J_URL = "bolt://localhost:7687"
    NEO4J_USER = "neo4j"
    NEO4J_PASSWORD = "password"
    
    # Performance Settings
    MAX_CONCURRENT_AGENTS = 5
    REQUEST_TIMEOUT = 300
    CACHE_TTL = 3600
    
    # Evaluation Thresholds
    CONFIDENCE_THRESHOLD = 0.7
    PERFORMANCE_THRESHOLD = 0.8
    EVOLUTION_TRIGGER = 100  # Analyses before evolution

settings = Settings()