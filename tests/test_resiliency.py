from unittest.mock import AsyncMock, MagicMock

import anthropic
import pytest

from app.adapters.base import ModelResult
from app.resilience.circuit_breaker import CircuitBreaker, CircuitBreakerRegistry, CircuitState
from app.resilience.fallback import ServiceUnavailableError, execute_with_fallback


# ── CircuitBreaker tests ───────────────────────────────────────────────────────


class TestCircuitBreaker:
    def test_starts_closed(self):
        """Circuit starts in closed state — calls go through."""
        cb = CircuitBreaker(adapter_name="test-model")
        assert cb.state == CircuitState.CLOSED
        assert not cb.is_open()

    def test_opens_after_threshold_failures(self, monkeypatch):
        """Circuit opens after hitting the failure threshold."""
        from app.config import settings

        monkeypatch.setattr(settings, "circuit_breaker_failure_threshold", 3)

        cb = CircuitBreaker(adapter_name="test-model")
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.CLOSED  # not yet

        cb.record_failure()
        assert cb.state == CircuitState.OPEN  # threshold hit
        assert cb.is_open()

    def test_success_resets_circuit(self, monkeypatch):
        """A success after failures resets back to closed."""
        from app.config import settings

        monkeypatch.setattr(settings, "circuit_breaker_failure_threshold", 2)

        cb = CircuitBreaker(adapter_name="test-model")
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.OPEN

        cb.record_success()
        assert cb.state == CircuitState.CLOSED
        assert cb.failure_count == 0

    def test_transitions_to_half_open_after_recovery_window(self, monkeypatch):
        """After the recovery window, circuit transitions OPEN → HALF_OPEN."""
        from app.config import settings

        monkeypatch.setattr(settings, "circuit_breaker_failure_threshold", 1)
        monkeypatch.setattr(settings, "circuit_breaker_recovery_seconds", 0)

        cb = CircuitBreaker(adapter_name="test-model")
        cb.record_failure()
        assert cb.state == CircuitState.OPEN

        # Recovery window is 0 seconds — should immediately allow through
        assert not cb.is_open()
        assert cb.state == CircuitState.HALF_OPEN

    def test_registry_returns_same_instance(self):
        """Registry returns the same CircuitBreaker for the same adapter name."""
        registry = CircuitBreakerRegistry()
        cb1 = registry.get("my-model")
        cb2 = registry.get("my-model")
        assert cb1 is cb2


# ── execute_with_fallback tests ───────────────────────────────────────────────


def _make_adapter(model_name: str, result: ModelResult | Exception):
    """Build a mock adapter that returns a result or raises an exception."""
    adapter = MagicMock()
    adapter.model_name = model_name
    if isinstance(result, Exception):
        adapter.generate = AsyncMock(side_effect=result)
    else:
        adapter.generate = AsyncMock(return_value=result)
    return adapter


def _make_result(model_name: str) -> ModelResult:
    return ModelResult(
        text="Test response",
        model_name=model_name,
        input_tokens=50,
        output_tokens=30,
        cost_usd=0.0001,
    )


class TestExecuteWithFallback:
    @pytest.mark.asyncio
    async def test_primary_succeeds_no_fallback(self):
        """When primary succeeds, fallback is not called."""
        primary = _make_adapter("sonnet", _make_result("sonnet"))
        fallback = _make_adapter("haiku", _make_result("haiku"))

        result, fallback_used = await execute_with_fallback(
            primary_adapter=primary,
            fallback_adapter=fallback,
            caller_context="",
            user_prompt="test",
            temperature=0.7,
            max_tokens=100,
        )

        assert not fallback_used
        assert result.model_name == "sonnet"
        fallback.generate.assert_not_called()

    @pytest.mark.asyncio
    async def test_primary_fails_fallback_used(self, monkeypatch):
        """When primary raises a transient error, fallback is tried."""
        from app.config import settings

        monkeypatch.setattr(settings, "retry_max_attempts", 1)

        primary = _make_adapter(
            "sonnet",
            anthropic.RateLimitError(message="rate limited", response=MagicMock(), body={}),
        )
        fallback = _make_adapter("haiku", _make_result("haiku"))

        result, fallback_used = await execute_with_fallback(
            primary_adapter=primary,
            fallback_adapter=fallback,
            caller_context="",
            user_prompt="test",
            temperature=0.7,
            max_tokens=100,
        )

        assert fallback_used
        assert result.model_name == "haiku"

    @pytest.mark.asyncio
    async def test_both_fail_raises_service_unavailable(self, monkeypatch):
        """When both adapters fail, ServiceUnavailableError is raised."""
        from app.config import settings

        monkeypatch.setattr(settings, "retry_max_attempts", 1)

        error = anthropic.RateLimitError(message="rate limited", response=MagicMock(), body={})
        primary = _make_adapter("sonnet", error)
        fallback = _make_adapter("haiku", error)

        with pytest.raises(ServiceUnavailableError):
            await execute_with_fallback(
                primary_adapter=primary,
                fallback_adapter=fallback,
                caller_context="",
                user_prompt="test",
                temperature=0.7,
                max_tokens=100,
            )

    @pytest.mark.asyncio
    async def test_open_circuit_skips_primary(self, monkeypatch):
        """When primary circuit is open, it is skipped and fallback is tried directly."""
        from app.config import settings
        from app.resilience.circuit_breaker import circuit_registry

        monkeypatch.setattr(settings, "retry_max_attempts", 1)

        primary = _make_adapter("sonnet-open", _make_result("sonnet-open"))
        fallback = _make_adapter("haiku-open", _make_result("haiku-open"))

        # Force primary circuit open
        cb = circuit_registry.get("sonnet-open")
        monkeypatch.setattr(settings, "circuit_breaker_failure_threshold", 1)
        cb.record_failure()
        assert cb.is_open()

        result, fallback_used = await execute_with_fallback(
            primary_adapter=primary,
            fallback_adapter=fallback,
            caller_context="",
            user_prompt="test",
            temperature=0.7,
            max_tokens=100,
        )

        assert fallback_used
        primary.generate.assert_not_called()  # skipped due to open circuit
        assert result.model_name == "haiku-open"
