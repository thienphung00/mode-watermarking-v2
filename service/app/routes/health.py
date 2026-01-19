"""
Health check route.

GET /health
"""
from fastapi import APIRouter

from service.app.schemas import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """
    Health check endpoint.
    
    Returns:
        Health status
    """
    return HealthResponse(status="ok")

