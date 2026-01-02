from functools import lru_cache

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

    class Config:
        env_file = ".env"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
