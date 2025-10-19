# routers/health.py
"""
Health check and readiness endpoints
No authentication required per OpenAPI spec
"""
from fastapi import APIRouter, HTTPException, status
from datetime import datetime
from typing import Dict

router = APIRouter(tags=["Health"])

# Global state for service readiness
service_state = {
    "is_uploading": False,
    "last_check": None,
    "start_time": datetime.now()
}

@router.get(
    "/",
    summary="Health Check",
    description="Basic health check endpoint",
    response_model=Dict[str, str]
)
async def root() -> Dict[str, str]:
    """
    Basic health check endpoint
    
    Returns:
        dict: Service status
    """
    return {"response": "Multimodel Agent"}

@router.get(
    "/health-check",
    summary="Health Check",
    description="Detailed health check endpoint",
    response_model=Dict[str, str]
)
async def health_check() -> Dict[str, str]:
    """
    Detailed health check endpoint
    
    Returns:
        dict: Service status with timestamp
    """
    service_state["last_check"] = datetime.now().isoformat()
    uptime = (datetime.now() - service_state["start_time"]).total_seconds()
    
    return {
        "response": "Multimodel Agent",
        "status": "healthy",
        "timestamp": service_state["last_check"],
        "uptime_seconds": str(int(uptime))
    }

@router.get(
    "/readiness",
    summary="Check Service Readiness",
    description="Check if the service is ready to accept requests",
    responses={
        200: {
            "description": "Service is ready",
            "content": {
                "application/json": {
                    "example": {"status": "Service is ready"}
                }
            }
        },
        503: {
            "description": "Service is busy",
            "content": {
                "application/json": {
                    "example": {"detail": "Service is busy processing an upload"}
                }
            }
        }
    }
)
async def readiness_check() -> Dict[str, str]:
    """
    Check if service is ready to accept new requests
    
    Returns:
        dict: Readiness status
        
    Raises:
        HTTPException: 503 if service is busy
    """
    if service_state["is_uploading"]:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service is busy processing an upload"
        )
    
    return {
        "status": "Service is ready",
        "timestamp": datetime.now().isoformat()
    }

# Helper functions for other routers to use
def set_upload_status(is_uploading: bool):
    """Set upload status"""
    service_state["is_uploading"] = is_uploading

def get_upload_status() -> bool:
    """Get upload status"""
    return service_state["is_uploading"]
