from fastapi import APIRouter, Depends, HTTPException
import logging

from app.schemas.schemas import CompareResponse, ErrorResponse, CompareRequest
from app.services.comparator import ImageComparatorService

router = APIRouter()
logger = logging.getLogger(__name__)

_shared_service = ImageComparatorService()


def get_service() -> ImageComparatorService:
    return _shared_service


@router.post("/compare", response_model=CompareResponse, responses={400: {"model": ErrorResponse}})
async def compare_images(
    body: CompareRequest,
    service: ImageComparatorService = Depends(get_service),
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
