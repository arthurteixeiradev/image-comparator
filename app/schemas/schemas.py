from typing import Any

from pydantic import BaseModel, Field


class CompareResponse(BaseModel):
    isEqual: bool
    message: str


class ErrorResponse(BaseModel):
    error: str
    detail: Any | None = None


class CompareRequest(BaseModel):
    imagem1: str = Field(..., description="URL da primeira imagem")
    imagem2: str = Field(..., description="URL da segunda imagem")
    algorithm: str | None = Field(None, description="Algoritmo: phash|dhash")
    threshold: float | None = Field(None, ge=0.0, le=1.0)
