import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.router import router as api_router
from app.core.config import get_settings
from app.middleware.logging_middleware import RequestLoggingMiddleware
from app.services.redis_service import get_redis_service
from app.services.token_service import get_token_service, reset_token_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger = logging.getLogger("comparador.startup")

    redis_service = get_redis_service()
    if redis_service.is_connected:
        health = redis_service.health_check()
        logger.info(f"Redis pronto - latência: {health.get('latency_ms')}ms")
    else:
        logger.warning("Redis não disponível - usando apenas cache em memória")

    yield

    logger.info("Encerrando conexões...")

    token_service = get_token_service()
    await token_service.close()
    reset_token_service()

    redis_service.disconnect()
    logger.info("Aplicação encerrada")


def create_app() -> FastAPI:
    settings = get_settings()

    logging.basicConfig(
        level=settings.LOG_LEVEL,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    app = FastAPI(
        title=settings.PROJECT_NAME,
        version=settings.PROJECT_VERSION,
        lifespan=lifespan,
    )

    app.add_middleware(RequestLoggingMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(api_router, prefix="/api")

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
