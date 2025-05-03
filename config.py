from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration settings."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    MODEL_LIST_URL: str = "https://api.eliza.yandex.net/models"  # Example default


settings = Settings()
