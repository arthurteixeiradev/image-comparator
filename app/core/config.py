from functools import lru_cache
from typing import Optional

try:
    from pydantic import BaseSettings
except Exception:
    from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    PROJECT_NAME: str = "Comparador API"

    PROJECT_VERSION: str = "0.1.0"

    DEBUG: bool = True

    LOG_LEVEL: str = "INFO"

    CORS_ORIGINS: list[str] = ["*"]

    MAX_IMAGE_SIZE: int = 5 * 1024 * 1024

    REDIS_URL: Optional[str] = None

    VTRINA_BASE_URL: Optional[str] = None

    REDIS_HOST: str = "localhost"

    REDIS_PORT: int = 6379

    REDIS_PASSWORD: Optional[str] = None

    REDIS_DB: int = 0

    REDIS_ENABLED: bool = True

    REDIS_CONNECT_TIMEOUT: int = 5

    REDIS_SOCKET_TIMEOUT: int = 5

    REDIS_TOKEN_CACHE_TTL: int = 3600

    REDIS_COMPARATOR_CACHE_TTL: int = 300

    REDIS_CACHE_TTL: Optional[int] = None

    MEMORY_CACHE_SIZE: int = 1000

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True
        extra = "ignore"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
