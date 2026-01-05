from typing import Any, Dict

from pydantic import BaseModel, Field


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


class Pair(BaseModel):
    url1: str
    url2: str


class MassTestRequest(BaseModel):
    pairs: list[Pair]
    concurrency: int = Field(10, ge=1)
    total: int | None = Field(
        None,
        ge=1,
        description="Total de requisições (repetirá pares) - se None, usa len(pairs)",
    )
    timeout: float = Field(30.0, gt=0.0)


class MassTestMetrics(BaseModel):
    total: int
    successes: int
    failures: int
    elapsed_seconds: float
    throughput_rps: float
    p50: float | None
    p95: float | None
    p99: float | None
    mean: float | None


class MassTestResponse(BaseModel):
    metrics: MassTestMetrics
    sample_latencies: list[float] | None = None
