import time
import logging
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger("comparador.middleware")


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Exemplo simples de middleware que registra tempo e rota.

    Não realiza lógica de negócio, apenas observabilidade.
    """

    async def dispatch(self, request: Request, call_next) -> Response:  # type: ignore[override]
        start = time.time()
        response = await call_next(request)
        elapsed = (time.time() - start) * 1000.0
        logger.info(f"{request.method} {request.url.path} completed_in={elapsed:.2f}ms status={response.status_code}")
        return response
