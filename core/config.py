# core/config.py
"""
Configuration settings for Multimodel Agent API
Uses Groq for LLM and Zilliz Cloud (Milvus) for vector storage
"""
import os
from dotenv import load_dotenv
from pydantic_settings import BaseSettings

# Load environment variables
load_dotenv()

class Settings(BaseSettings):
    """Application settings"""
    
    # ============================================================================
    # API Settings
    # ============================================================================
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Multimodel Agent API"
    VERSION: str = "1.0.0"
    
    # API Key Authentication (comma-separated)
    API_KEYS: str = "dev-key-12345"
    
    # ============================================================================
    # Groq Settings (Primary LLM - Free)
    # ============================================================================
    GROQ_API_KEY: str = ""
    # GROQ_BASE_URL: str = "https://api.groq.com/openai/v1"
    GROQ_BASE_URL: str = "https://api.groq.com"
    GROQ_MODEL: str = "meta-llama/llama-4-scout-17b-16e-instruct"
    GROQ_FAST_MODEL: str = "llama-3.1-8b-instant"
    
    # ============================================================================
    # MongoDB Settings
    # ============================================================================
    MONGODB_URL: str = "mongodb://localhost:27017"
    DATABASE_NAME: str = "multimodel_agent"
    MONGODB_COLLECTION: str = "documents"
    
    # ============================================================================
    # Zilliz Cloud (Milvus) Settings
    # ============================================================================
    MILVUS_URI: str = ""  # Your Zilliz Cloud endpoint
    MILVUS_USER: str = ""  # Empty for API key auth
    MILVUS_PASSWORD: str = ""  # Your Zilliz API token
    MILVUS_COLLECTION_NAME: str = "rag_langchain"
    MILVUS_DIMENSION: int = 384
    MILVUS_METRIC_TYPE: str = "COSINE"
    
    # ============================================================================
    # Embedding Settings (Local - sentence-transformers)
    # ============================================================================
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DIMENSION: int = 384
    
    # ============================================================================
    # Document Processing
    # ============================================================================
    CHUNK_SIZE: int = 5000
    CHUNK_OVERLAP: int = 200
    MAX_UPLOAD_SIZE: int = 100 * 1024 * 1024  # 100MB
    UPLOAD_DIR: str = "uploads"
    ALLOWED_EXTENSIONS: list = [".pdf", ".doc", ".docx", ".ppt", ".pptx", ".xls", ".xlsx", ".txt"]
    
    # ============================================================================
    # RAG Settings
    # ============================================================================
    RETRIEVAL_TOP_K: int = 5
    RERANK_TOP_K: int = 3
    
    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "ignore"

# Global settings instance
settings = Settings()

# Legacy exports for compatibility with your existing code
GROQ_API_KEY = settings.GROQ_API_KEY
GROQ_BASE_URL = settings.GROQ_BASE_URL
GROQ_MODEL = settings.GROQ_MODEL

MILVUS_URI = settings.MILVUS_URI
MILVUS_USER = settings.MILVUS_USER
MILVUS_PASSWORD = settings.MILVUS_PASSWORD
MILVUS_COLLECTION_NAME = settings.MILVUS_COLLECTION_NAME
MILVUS_DIMENSION = settings.MILVUS_DIMENSION
MILVUS_METRIC_TYPE = settings.MILVUS_METRIC_TYPE

# Validate critical settings
if not settings.GROQ_API_KEY:
    print("⚠️  WARNING: GROQ_API_KEY not set in .env file")
    print("   Get your free key from: https://console.groq.com/keys")

if not settings.MILVUS_URI or not settings.MILVUS_PASSWORD:
    print("⚠️  WARNING: MILVUS_URI or MILVUS_PASSWORD not set")
    print("   Get free Zilliz Cloud account from: https://cloud.zilliz.com/")
