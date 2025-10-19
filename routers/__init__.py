# routers/__init__.py
"""Routers module initialization"""
from .health import router as health_router

__all__ = ["health_router"]
