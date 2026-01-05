from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

from app.clients.vtrina_client import AppError, VtrinaClient
from app.core.config import get_settings
from app.services.redis_service import RedisService, get_redis_service

logger = logging.getLogger("comparador.token_service")


@dataclass
class TokenServiceConfig:
    cache_ttl_seconds: int = 3600
    redis_key_prefix: str = "token_"


class TokenService:
    _instance: Optional["TokenService"] = None

    def __init__(
        self,
        redis_service: RedisService,
        vtrina_client: VtrinaClient,
        config: Optional[TokenServiceConfig] = None,
    ) -> None:
        self._redis = redis_service
        self._vtrina = vtrina_client
        self._config = config or TokenServiceConfig()

    def _cache_key(self, token: str) -> str:
        return f"{self._config.redis_key_prefix}{token}"

    def get(self, token: str) -> Optional[Dict[str, Any]]:
        if not token or not token.strip():
            return None

        key = self._cache_key(token)

        try:
            cached_str = None
            if hasattr(self._redis, "get_str"):
                cached_str = self._redis.get_str(key)
            else:
                raw = self._redis.get(key)
                if raw:
                    cached_str = (
                        raw.decode("utf-8")
                        if isinstance(raw, (bytes, bytearray))
                        else str(raw)
                    )

            if cached_str:
                try:
                    return json.loads(cached_str)
                except Exception:
                    logger.warning(
                        "TokenService: failed to parse cached token - evicting"
                    )
                    try:
                        if hasattr(self._redis, "delete"):
                            self._redis.delete(key)
                        elif hasattr(self._redis, "delete_pattern"):
                            self._redis.delete_pattern(key)
                    except Exception:
                        logger.exception(
                            "TokenService: failed to evict malformed cache"
                        )
        except Exception:
            logger.exception("TokenService: error reading from Redis")

        try:
            token_obj = self._vtrina.get_token(token)
        except AppError:
            logger.warning("TokenService: Vtrina client returned error resolving token")
            return None
        except Exception:
            logger.exception("TokenService: unexpected error when calling Vtrina")
            return None

        if token_obj is not None:
            try:
                serialized = json.dumps(token_obj)
                if hasattr(self._redis, "set"):
                    self._redis.set(key, serialized, ttl=self._config.cache_ttl_seconds)
                elif hasattr(self._redis, "set_object"):
                    self._redis.set_object(
                        key, token_obj, ttl=self._config.cache_ttl_seconds
                    )
            except Exception:
                logger.exception("TokenService: failed to cache token")

        return token_obj

    def clear_cache_for_token(self, token: str) -> bool:
        if not token or not token.strip():
            return False

        key = self._cache_key(token)

        try:
            if hasattr(self._redis, "delete"):
                return bool(self._redis.delete(key))
            if hasattr(self._redis, "delete_pattern"):
                return bool(self._redis.delete_pattern(key))
            if hasattr(self._redis, "set"):
                self._redis.set(key, "", ttl=1)
                return True
        except Exception:
            logger.exception("TokenService: failed to clear cache")
        return False


_token_service_instance: Optional[TokenService] = None


def get_token_service() -> TokenService:
    global _token_service_instance

    if _token_service_instance is None:
        redis_srv = get_redis_service()
        settings = get_settings()
        base = settings.VTRINA_BASE_URL
        vtrina_cli = VtrinaClient(base_url=base)
        cfg = TokenServiceConfig(cache_ttl_seconds=settings.REDIS_TOKEN_CACHE_TTL)
        _token_service_instance = TokenService(redis_srv, vtrina_cli, config=cfg)

    return _token_service_instance


def reset_token_service() -> None:
    global _token_service_instance
    _token_service_instance = None
