from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration settings."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    MODEL_LIST_URL: str = "https://api.eliza.yandex.net/models"  # Example default
    MODEL_LIST_AUTH_TOKEN: Optional[str] = None # Optional OAuth token for fetching the model list
    LOG_LEVEL: str = "INFO" # Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL, NONE)


settings = Settings()
