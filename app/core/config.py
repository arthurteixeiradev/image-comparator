from functools import lru_cache
try:
    # pydantic v1
    from pydantic import BaseSettings  # type: ignore
except Exception:
    # pydantic v2+ separates settings into pydantic-settings
    from pydantic_settings import BaseSettings  # type: ignore


class Settings(BaseSettings):
    PROJECT_NAME: str = "Comparador API"
    PROJECT_VERSION: str = "0.1.0"

    # Runtime
    DEBUG: bool = True

    # Limits
    MAX_IMAGE_SIZE: int = 5 * 1024 * 1024

    class Config:
        env_file = ".env"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
