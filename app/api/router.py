import logging

from fastapi import APIRouter, Depends, HTTPException

from app.dependencies.auth import require_token
from app.schemas.schemas import (
    CompareRequest,
    CompareResponse,
    ErrorResponse,
)
from app.services.comparator_service import ImageComparatorService
from app.services.redis_service import get_redis_service

router = APIRouter()
logger = logging.getLogger(__name__)

_shared_service = ImageComparatorService()


def get_service() -> ImageComparatorService:
    return _shared_service


@router.get("/ping-pong")
async def ping_pong():
    return {"message": "pong"}


@router.get("/health")
async def health_check():
    redis_service = get_redis_service()
    redis_health = redis_service.health_check()

    return {
        "status": "healthy",
        "redis": redis_health,
    }


@router.post(
    "/compare",
    response_model=CompareResponse,
    responses={400: {"model": ErrorResponse}},
)
async def compare_images(
    body: CompareRequest,
    service: ImageComparatorService = Depends(get_service),
    _token: dict = Depends(require_token),
):
    try:
        result = await service.compare(
            body.imagem1,
            body.imagem2,
            algorithm=body.algorithm,
            threshold=body.threshold,
        )
        return result

    except HTTPException:
        raise
    except Exception:
        logger.exception("Erro interno ao comparar imagens")
        raise HTTPException(status_code=500, detail="Internal server error")
