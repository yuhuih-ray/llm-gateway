from functools import lru_cache

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    tenant_request_capacity: float = Field(default=20, ge=1, allow_inf_nan=False)
    tenant_request_refill: float = Field(default=2, gt=0, allow_inf_nan=False)
    gemini_request_capacity: float = Field(default=60, ge=1, allow_inf_nan=False)
    gemini_request_refill: float = Field(default=5, gt=0, allow_inf_nan=False)
    database_url: str = ""
    usage_retention_months: int = Field(default=12, ge=1)
    redis_url: str = "redis://localhost:6379/0"
    jwt_secret: SecretStr | None = None
    gemini_api_key: SecretStr | None = None


@lru_cache
def get_settings() -> Settings:
    return Settings()
