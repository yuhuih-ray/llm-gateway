from functools import lru_cache
from ipaddress import ip_network

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    fake_delay_ms: int = Field(default=0, ge=0)
    server_timing: bool = False
    max_request_bytes: int = Field(default=1_048_576, ge=1)
    tenant_request_capacity: float = Field(default=20, ge=1, allow_inf_nan=False)
    tenant_request_refill: float = Field(default=2, gt=0, allow_inf_nan=False)
    gemini_request_capacity: float = Field(default=60, ge=1, allow_inf_nan=False)
    gemini_request_refill: float = Field(default=5, gt=0, allow_inf_nan=False)
    tenant_token_capacity: float = Field(default=100_000, ge=1, allow_inf_nan=False)
    tenant_token_refill: float = Field(default=1_000, gt=0, allow_inf_nan=False)
    trusted_proxies: list[str] = Field(default_factory=list)
    database_url: str = ""
    usage_retention_months: int = Field(default=12, ge=1)
    cache_redis_url: str = "redis://localhost:6380/0"
    redis_url: str = "redis://localhost:6379/0"
    jwt_secret: SecretStr | None = None
    gemini_api_key: SecretStr | None = None

    @field_validator("trusted_proxies")
    @classmethod
    def validate_proxies(cls, values: list[str]) -> list[str]:
        return [str(ip_network(value, strict=False)) for value in values]


@lru_cache
def get_settings() -> Settings:
    return Settings()
