"""
Central typed configuration. Every tunable value lives in .env —
nothing is hardcoded anywhere else in the codebase.

Usage: from app.config import settings
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Anthropic ─────────────────────────────────────────────────────────────
    anthropic_api_key: str = ""

    # ── Models ────────────────────────────────────────────────────────────────
    primary_model: str = "claude-sonnet-4-6"
    fallback_model: str = "claude-haiku-4-5"
    max_output_tokens: int = 4096

    # ── OAuth2 ────────────────────────────────────────────────────────────────
    oauth_jwt_secret: str = "dev-secret-change-in-production"
    oauth_jwt_algorithm: str = "HS256"
    oauth_token_expiry_seconds: int = 3600

    # ── Cost-based routing ────────────────────────────────────────────────────
    cost_route_threshold_usd: float = 0.05

    # ── Per-key budget enforcement ────────────────────────────────────────────
    daily_soft_budget_usd: float = 5.00
    daily_hard_budget_usd: float = 10.00

    # ── Rate limiting ─────────────────────────────────────────────────────────
    rate_limit_requests_per_minute: int = 20
    rate_limit_burst_size: int = 5

    # ── Resilience ────────────────────────────────────────────────────────────
    request_timeout_seconds: int = 30
    retry_max_attempts: int = 3
    retry_backoff_base_seconds: float = 1.0
    circuit_breaker_failure_threshold: int = 5
    circuit_breaker_recovery_seconds: int = 30

    # ── Security ──────────────────────────────────────────────────────────────
    max_prompt_length_chars: int = 8000
    prompt_injection_check_enabled: bool = True

    # ── App ───────────────────────────────────────────────────────────────────
    app_env: str = "development"
    log_level: str = "INFO"

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"


settings = Settings()
