from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse
from typing import Callable
import asyncio

from app.schemas.schemas import CompareResponse, ErrorResponse, CompareRequest
from app.services.comparator import ImageComparatorService
from app.core.config import get_settings

router = APIRouter()

# reuse a mesma instância do service (poderia vir de um DI container)
_shared_service = ImageComparatorService()


def get_service() -> ImageComparatorService:
    return _shared_service


@router.post("/compare", response_model=CompareResponse, responses={400: {"model": ErrorResponse}})
async def compare_images(
    body: CompareRequest,
    service: ImageComparatorService = Depends(get_service),
):
    """Controller: recebe URLs no body JSON e delega ao Service.

    O controller NÃO realiza downloads — a responsabilidade é do Service.
    """
    settings = get_settings()

    try:
        # Delegate fully to service which will download and compare
        result = await service.compare(body.url1, body.url2, algorithm=body.algorithm, threshold=body.threshold)
        return JSONResponse(status_code=200, content=result)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
