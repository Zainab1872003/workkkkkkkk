# middleware/__init__.py
"""Middleware module initialization"""
from .api_key_auth import get_api_key, verify_api_key

__all__ = ["get_api_key", "verify_api_key"]
