from pydantic import BaseModel, Field
from typing import Dict, Any


class CompareDetail(BaseModel):
    similarity: float = Field(..., ge=0.0, le=1.0)
    distance: int


class CompareResponse(BaseModel):
    are_same: bool
    similarity: float = Field(..., ge=0.0, le=1.0)
    distance: int
    algorithm: str
    time: float
    cache_hit: bool = False
    details: Dict[str, CompareDetail] | None = None


class ErrorResponse(BaseModel):
    error: str
    detail: Any | None = None


class CompareRequest(BaseModel):
    url1: str = Field(..., description="URL da primeira imagem")
    url2: str = Field(..., description="URL da segunda imagem")
    algorithm: str | None = Field(None, description="Algoritmo: phash|dhash")
    threshold: float | None = Field(None, ge=0.0, le=1.0)
