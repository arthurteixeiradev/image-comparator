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

    # URL completa do Redis
    REDIS_URL: Optional[str] = None
    
    # Configurações alternativas usadas se REDIS_URL não for definido
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_PASSWORD: Optional[str] = None
    REDIS_DB: int = 0
    
    # Habilitar/desabilitar Redis, útil para testes ou fallback
    REDIS_ENABLED: bool = True
    
    # Timeouts (em segundos)
    REDIS_CONNECT_TIMEOUT: int = 5
    REDIS_SOCKET_TIMEOUT: int = 5
    
    # TTL padrão para cache em segundos
    # 30 dias = 2592000 segundos
    REDIS_CACHE_TTL: int = 86400 * 30
    
    # Tamanho máximo do cache em memória
    MEMORY_CACHE_SIZE: int = 1000

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        # Permite que variáveis de ambiente sobrescrevam valores do .env
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    return Settings()
