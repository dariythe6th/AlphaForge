# src/config.py
import os
from typing import Dict, Any
from dotenv import load_dotenv

load_dotenv()


class Settings:
    # API Keys - Load from environment
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    NEWSAPI_KEY = os.getenv("NEWSAPI_KEY", "")

    # Model Configurations
    LLM_ULTRA = "gpt-4"
    LLM_FAST = "gpt-3.5-turbo"

    # MCP Server URLs
    MCP_DATA_SERVER = "http://localhost:8001"
    MCP_ALGORITHM_SERVER = "http://localhost:8002"

    # Database Connections - Load from environment
    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
    NEO4J_URL = os.getenv("NEO4J_URL", "bolt://localhost:7687")
    NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")

    # Performance Settings
    MAX_CONCURRENT_AGENTS = 5
    REQUEST_TIMEOUT = 300
    CACHE_TTL = 3600

    # Evaluation Thresholds
    CONFIDENCE_THRESHOLD = 0.7
    PERFORMANCE_THRESHOLD = 0.8
    EVOLUTION_TRIGGER = 100

    # Model Paths
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    SENTIMENT_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"


settings = Settings()