import asyncio

import anthropic

from app.adapters.base import ModelAdapter, ModelResult
from app.observability.logger import get_logger
from app.resilience.circuit_breaker import circuit_registry
from app.resilience.retry import build_retry_decorator
from app.config import settings

log = get_logger(__name__)


class ServiceUnavailableError(Exception):
    """Raised when both primary and fallback adapters have failed."""

    pass


async def _call_with_retry_and_timeout(
    adapter: ModelAdapter,
    caller_context: str,
    user_prompt: str,
    temperature: float,
    max_tokens: int,
) -> ModelResult:
    """
    Call one adapter with retry (exponential backoff) and timeout.
    Raises the last exception if all retries are exhausted.
    """
    retry_decorator = build_retry_decorator()

    @retry_decorator
    async def _attempt() -> ModelResult:
        return await asyncio.wait_for(
            adapter.generate(
                caller_context=caller_context,
                user_prompt=user_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            ),
            timeout=settings.request_timeout_seconds,
        )

    return await _attempt()


async def execute_with_fallback(
    primary_adapter: ModelAdapter,
    fallback_adapter: ModelAdapter,
    caller_context: str,
    user_prompt: str,
    temperature: float,
    max_tokens: int,
) -> tuple[ModelResult, bool]:
    """
    Execute a generation request with full resilience:
        1. Check primary circuit breaker — skip if open
        2. Try primary adapter with retry + timeout
        3. On failure — record failure, check fallback circuit breaker
        4. Try fallback adapter with retry + timeout
        5. On both failing — raise ServiceUnavailableError

    Returns (ModelResult, fallback_used: bool).
    """
    primary_cb = circuit_registry.get(primary_adapter.model_name)
    fallback_cb = circuit_registry.get(fallback_adapter.model_name)

    # ── Try primary ───────────────────────────────────────────────────────────
    if not primary_cb.is_open():
        try:
            result = await _call_with_retry_and_timeout(
                adapter=primary_adapter,
                caller_context=caller_context,
                user_prompt=user_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            primary_cb.record_success()
            log.info("resilience_primary_succeeded", model=primary_adapter.model_name)
            return result, False

        except (
            anthropic.RateLimitError,
            anthropic.APITimeoutError,
            anthropic.APIStatusError,
            asyncio.TimeoutError,
        ) as e:
            primary_cb.record_failure()
            log.warning(
                "resilience_primary_failed",
                model=primary_adapter.model_name,
                error=str(e),
                circuit_state=primary_cb.state.value,
            )
    else:
        log.warning("resilience_primary_circuit_open", model=primary_adapter.model_name)

    # ── Try fallback ──────────────────────────────────────────────────────────
    if not fallback_cb.is_open():
        try:
            result = await _call_with_retry_and_timeout(
                adapter=fallback_adapter,
                caller_context=caller_context,
                user_prompt=user_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            fallback_cb.record_success()
            log.warning("resilience_fallback_succeeded", model=fallback_adapter.model_name)
            return result, True

        except (
            anthropic.RateLimitError,
            anthropic.APITimeoutError,
            anthropic.APIStatusError,
            asyncio.TimeoutError,
        ) as e:
            fallback_cb.record_failure()
            log.error(
                "resilience_fallback_failed",
                model=fallback_adapter.model_name,
                error=str(e),
                circuit_state=fallback_cb.state.value,
            )
    else:
        log.error("resilience_fallback_circuit_open", model=fallback_adapter.model_name)

    # ── Both failed ───────────────────────────────────────────────────────────
    raise ServiceUnavailableError(
        "Both primary and fallback adapters are unavailable. "
        "Check circuit breaker status and Anthropic API health."
    )
