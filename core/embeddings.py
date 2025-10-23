# core/embeddings.py
"""
Embedding models with OpenAI and HuggingFace support
"""
import logging
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from core.config import settings

logger = logging.getLogger(__name__)

# Global cache for embedding model
_embedding_model = None


def get_embeddings_model(provider: str = None, force_reload: bool = False):
    """
    Get embedding model with caching support.
    
    Args:
        provider: "openai" or "huggingface" (defaults to settings.EMBEDDING_PROVIDER)
        force_reload: Force reload model (bypass cache)
    
    Returns:
        Embedding model instance
    """
    global _embedding_model
    
    provider = provider or getattr(settings, 'EMBEDDING_PROVIDER', 'openai')
    
    # Return cached model if available
    if _embedding_model is not None and not force_reload:
        logger.debug(f"♻️ Using cached {provider} embedding model")
        return _embedding_model
    
    # Create new model
    if provider == "openai":
        _embedding_model = _get_openai_embeddings()
    elif provider == "huggingface":
        _embedding_model = _get_huggingface_embeddings()
    else:
        logger.error(f"Unknown embedding provider: {provider}. Defaulting to OpenAI")
        _embedding_model = _get_openai_embeddings()
    
    return _embedding_model


def _get_openai_embeddings():
    """Create OpenAI embedding model (1536 dimensions)"""
    logger.info("🔄 Loading OpenAI embeddings: text-embedding-3-small")
    
    return OpenAIEmbeddings(
        model=getattr(settings, 'OPENAI_EMBEDDING_MODEL', 'text-embedding-3-small'),
        openai_api_key=settings.OPENAI_API_KEY,
        chunk_size=1000  # Max texts per API call
    )


def _get_huggingface_embeddings():
    """Create HuggingFace embedding model (384 dimensions)"""
    logger.info("🔄 Loading HuggingFace embeddings: all-MiniLM-L6-v2")
    
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )


def get_embedding_dimension(provider: str = None) -> int:
    """
    Get embedding dimension for the specified provider.
    
    Args:
        provider: "openai" or "huggingface"
    
    Returns:
        Embedding dimension
    """
    provider = provider or getattr(settings, 'EMBEDDING_PROVIDER', 'openai')
    
    dimensions = {
        "openai": 1536,  # text-embedding-3-small
        "huggingface": 384  # all-MiniLM-L6-v2
    }
    
    return dimensions.get(provider, 1536)
