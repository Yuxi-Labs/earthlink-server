"""Application configuration."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Database
    database_url: str = "postgresql+asyncpg://earthlink:earthlink@localhost:5432/earthlink"

    # Redis
    redis_url: str = "redis://localhost:6379/0"

    # Ollama (LLM Gateway)
    ollama_url: str = "http://localhost:11434"

    # External APIs
    wikipedia_api_url: str = "https://en.wikipedia.org/api/rest_v1"

    # Voice APIs
    xai_api_key: str = ""
    openai_api_key: str = ""

    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    log_level: str = "INFO"

    # Vector Database
    qdrant_url: str = "http://localhost:6333"


settings = Settings()
