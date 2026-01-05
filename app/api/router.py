import logging
from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException

from app.dependencies.auth import require_token
from app.schemas.schemas import (
    CompareRequest,
    CompareResponse,
    ErrorResponse,
    MassTestRequest,
    MassTestResponse,
)
from app.services.comparator_service import ImageComparatorService
from app.services.mass_test_service import MassTestService
from app.services.redis_service import get_redis_service

router = APIRouter()
logger = logging.getLogger(__name__)

_shared_service = ImageComparatorService()


def get_service() -> ImageComparatorService:
    return _shared_service


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
    token_obj: Dict[str, Any] = Depends(require_token),
):
    try:
        result = await service.compare(
            body.url1, body.url2, algorithm=body.algorithm, threshold=body.threshold
        )
        return result

    except HTTPException:
        raise
    except Exception:
        logger.exception("Erro interno ao comparar imagens")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/mass_test", response_model=MassTestResponse)
async def mass_test(
    body: MassTestRequest,
    service: ImageComparatorService = Depends(get_service),
    token_obj: Dict[str, Any] = Depends(require_token),
):
    try:
        mass_svc = MassTestService(service)
        pairs = [(p.url1, p.url2) for p in body.pairs]
        res = await mass_svc.run(
            pairs, concurrency=body.concurrency, total=body.total, timeout=body.timeout
        )
        metrics = res.get("metrics", {})
        return {
            "metrics": {
                "total": metrics.get("total", 0),
                "successes": metrics.get("successes", 0),
                "failures": metrics.get("failures", 0),
                "elapsed_seconds": metrics.get("elapsed_seconds", 0.0),
                "throughput_rps": metrics.get("throughput_rps", 0.0),
                "p50": metrics.get("p50"),
                "p95": metrics.get("p95"),
                "p99": metrics.get("p99"),
                "mean": metrics.get("mean"),
            },
            "sample_latencies": res.get("sample_latencies"),
        }
    except Exception as e:
        logger.exception("Erro no mass_test")
        raise HTTPException(status_code=500, detail=str(e))
