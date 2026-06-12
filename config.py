"""Application configuration loaded from environment variables."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Centralized application settings sourced from `.env`."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    openrouter_api_key: str = ""
    openrouter_model: str = "openai/gpt-oss-120b:free"
    openrouter_base_url: str = "https://openrouter.ai/api/v1"

    database_url: str = "postgresql+psycopg://postgres:12345@localhost:5432/arabic_support"

    log_level: str = "INFO"
    request_timeout: int = 30
    max_retries: int = 3

    api_key: str = ""
    rate_limit_per_minute: int = 30

    data_dir: str = "data"
    knowledge_base_dir: str = "data/knowledge_base"
    logs_dir: str = "logs"
    trajectories_dir: str = "logs/trajectories"


@lru_cache
def get_settings() -> Settings:
    """Return a cached `Settings` instance.

    Returns:
        Settings: The application settings singleton.
    """
    return Settings()


settings = get_settings()
