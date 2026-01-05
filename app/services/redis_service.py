from __future__ import annotations

import logging
import pickle
from typing import Any, Optional, Union

import redis
from redis.exceptions import ConnectionError, RedisError, TimeoutError

from app.core.config import get_settings

logger = logging.getLogger("comparador.redis")


class RedisService:
    _instance: Optional["RedisService"] = None
    _client: Optional[redis.Redis] = None
    _is_connected: bool = False

    def __new__(cls) -> "RedisService":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self) -> None:
        settings = get_settings()

        if not settings.REDIS_ENABLED:
            logger.info("Redis desabilitado via configuração")
            self._is_connected = False
            return

        try:
            if settings.REDIS_URL:
                self._client = redis.from_url(
                    settings.REDIS_URL,
                    decode_responses=False,
                    socket_connect_timeout=settings.REDIS_CONNECT_TIMEOUT,
                    socket_timeout=settings.REDIS_SOCKET_TIMEOUT,
                    retry_on_timeout=True,
                    health_check_interval=30,
                )
            else:
                self._client = redis.Redis(
                    host=settings.REDIS_HOST,
                    port=settings.REDIS_PORT,
                    db=settings.REDIS_DB,
                    password=settings.REDIS_PASSWORD or None,
                    decode_responses=False,
                    socket_connect_timeout=settings.REDIS_CONNECT_TIMEOUT,
                    socket_timeout=settings.REDIS_SOCKET_TIMEOUT,
                    retry_on_timeout=True,
                    health_check_interval=30,
                )

            self._client.ping()
            self._is_connected = True
            logger.info("Redis conectado com sucesso")

        except (ConnectionError, TimeoutError) as e:
            logger.warning(f"Redis não disponível: {e}")
            self._client = None
            self._is_connected = False
        except Exception as e:
            logger.error(f"Erro ao conectar Redis: {e}")
            self._client = None
            self._is_connected = False

    @property
    def is_connected(self) -> bool:
        return self._is_connected and self._client is not None

    @property
    def client(self) -> Optional[redis.Redis]:
        return self._client

    def get(self, key: str) -> Optional[bytes]:
        if not self.is_connected or self._client is None:
            return None

        try:
            result = self._client.get(key)
            return result if isinstance(result, bytes) else None
        except RedisError as e:
            logger.error(f"Erro ao ler chave '{key}': {e}")
            return None

    def get_str(self, key: str) -> Optional[str]:
        value = self.get(key)
        if value is not None:
            return value.decode("utf-8")
        return None

    def get_object(self, key: str) -> Optional[Any]:
        value = self.get(key)
        if value is not None:
            try:
                return pickle.loads(value)
            except Exception as e:
                logger.error(f"Erro ao deserializar objeto '{key}': {e}")
        return None

    def set(
        self, key: str, value: Union[str, bytes], ttl: Optional[int] = None
    ) -> bool:
        if not self.is_connected or self._client is None:
            return False

        try:
            if ttl:
                self._client.setex(key, ttl, value)
            else:
                self._client.set(key, value)
            return True
        except RedisError as e:
            logger.error(f"Erro ao definir chave '{key}': {e}")
            return False

    def set_object(self, key: str, obj: Any, ttl: Optional[int] = None) -> bool:
        try:
            serialized = pickle.dumps(obj)
            return self.set(key, serialized, ttl)
        except Exception as e:
            logger.error(f"Erro ao serializar objeto '{key}': {e}")
            return False

    def delete(self, key: str) -> bool:
        if not self.is_connected or self._client is None:
            return False

        try:
            result = self._client.delete(key)
            return bool(result)
        except RedisError as e:
            logger.error(f"Erro ao deletar chave '{key}': {e}")
            return False

    def delete_pattern(self, pattern: str) -> int:
        if not self.is_connected or self._client is None:
            return 0

        try:
            keys: list[str] = []
            for key in self._client.scan_iter(match=pattern, count=100):
                keys.append(key.decode("utf-8") if isinstance(key, bytes) else key)
            if keys:
                result = self._client.unlink(*keys)
                return int(result) if result else 0
            return 0
        except RedisError as e:
            logger.error(f"Erro ao deletar padrão '{pattern}': {e}")
            return 0

    def ping(self) -> bool:
        if self._client is None:
            return False

        try:
            result = self._client.ping()
            return bool(result)
        except RedisError:
            self._is_connected = False
            return False

    def disconnect(self) -> None:
        if self._client:
            try:
                self._client.close()
                logger.info("Redis desconectado")
            except Exception as e:
                logger.error(f"Erro ao desconectar Redis: {e}")
            finally:
                self._client = None
                self._is_connected = False

    def health_check(self) -> dict:
        import time

        from app.core.config import get_settings

        settings = get_settings()

        if not self.is_connected or self._client is None:
            return {
                "status": "disconnected",
                "latency_ms": None,
                "server_info": None,
                "token_cache_ttl": settings.REDIS_TOKEN_CACHE_TTL,
                "comparator_cache_ttl": settings.REDIS_COMPARATOR_CACHE_TTL,
            }

        try:
            start = time.perf_counter()
            self._client.ping()
            latency = (time.perf_counter() - start) * 1000

            info: dict[str, Any] = self._client.info("server")  # type: ignore[assignment]

            return {
                "status": "connected",
                "latency_ms": round(latency, 2),
                "token_cache_ttl": settings.REDIS_TOKEN_CACHE_TTL,
                "comparator_cache_ttl": settings.REDIS_COMPARATOR_CACHE_TTL,
                "server_info": {
                    "redis_version": info.get("redis_version"),
                    "uptime_days": info.get("uptime_in_days"),
                },
            }
        except RedisError as e:
            return {
                "status": "error",
                "error": str(e),
                "latency_ms": None,
                "server_info": None,
            }


_redis_service: Optional[RedisService] = None


def get_redis_service() -> RedisService:
    global _redis_service
    if _redis_service is None:
        _redis_service = RedisService()
    return _redis_service
