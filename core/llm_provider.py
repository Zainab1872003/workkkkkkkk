# core/llm_provider.py
"""
Unified LLM provider interface supporting both OpenAI and Groq
"""
import logging
from typing import Optional
from agno.models.openai import OpenAIChat
from agno.models.groq import Groq
from core.config import settings

logger = logging.getLogger(__name__)


def get_llm_model(
    provider: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None
):
    """
    Get LLM model based on provider configuration.
    
    Args:
        provider: "openai" or "groq" (defaults to settings.LLM_PROVIDER)
        temperature: Override default temperature
        max_tokens: Override default max tokens
    
    Returns:
        Configured LLM model instance
    """
    provider = provider or settings.LLM_PROVIDER
    
    if provider == "openai":
        return _get_openai_model(temperature, max_tokens)
    elif provider == "groq":
        return _get_groq_model(temperature)
    else:
        logger.error(f"Unknown provider: {provider}. Defaulting to OpenAI")
        return _get_openai_model(temperature, max_tokens)


def _get_openai_model(temperature: Optional[float] = None, max_tokens: Optional[int] = None):
    """Create OpenAI model instance"""
    
    model_config = {
        "id": settings.OPENAI_MODEL,
        "api_key": settings.OPENAI_API_KEY,
        "temperature": temperature if temperature is not None else settings.OPENAI_TEMPERATURE,
    }
    
    if max_tokens:
        model_config["max_tokens"] = max_tokens
    elif settings.OPENAI_MAX_TOKENS:
        model_config["max_tokens"] = settings.OPENAI_MAX_TOKENS
    
    logger.info(f"🤖 Using OpenAI: {settings.OPENAI_MODEL}")
    return OpenAIChat(**model_config)


def _get_groq_model(temperature: Optional[float] = None):
    """Create Groq model instance"""
    
    logger.info(f"🤖 Using Groq: {settings.GROQ_MODEL}")
    return Groq(
        id=settings.GROQ_MODEL,
        api_key=settings.GROQ_API_KEY,
        temperature=temperature if temperature is not None else 0.0
    )


def get_model_info() -> dict:
    """Get current model configuration info"""
    return {
        "provider": settings.LLM_PROVIDER,
        "model": settings.OPENAI_MODEL if settings.LLM_PROVIDER == "openai" else settings.GROQ_MODEL,
        "temperature": settings.OPENAI_TEMPERATURE if settings.LLM_PROVIDER == "openai" else 0.0,
        "max_tokens": settings.OPENAI_MAX_TOKENS if settings.LLM_PROVIDER == "openai" else "N/A"
    }
