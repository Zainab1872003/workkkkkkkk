# middleware/api_key_auth.py
"""
API Key authentication middleware
Validates x-api-key header for protected endpoints
"""
from typing import Optional
from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader
from core.config import settings

# Define API key header
api_key_header = APIKeyHeader(name="x-api-key", auto_error=False)

def get_api_key(api_key: Optional[str] = Security(api_key_header)) -> str:
    """
    Dependency to validate API key from header
    
    Args:
        api_key: API key from x-api-key header
        
    Returns:
        str: Valid API key
        
    Raises:
        HTTPException: If API key is missing or invalid
    """
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API key. Include 'x-api-key' header in your request.",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    
    # Get valid keys from settings
    valid_keys = [key.strip() for key in settings.API_KEYS.split(",") if key.strip()]
    
    if not valid_keys:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="API keys not configured on server"
        )
    
    if api_key not in valid_keys:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    
    return api_key

def verify_api_key(api_key: str) -> bool:
    """
    Verify API key manually (for use in route functions)
    
    Args:
        api_key: API key to verify
        
    Returns:
        bool: True if valid
        
    Raises:
        HTTPException: If invalid
    """
    valid_keys = [key.strip() for key in settings.API_KEYS.split(",") if key.strip()]
    
    if api_key not in valid_keys:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )
    
    return True
