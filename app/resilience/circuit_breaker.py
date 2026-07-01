import time
from dataclasses import dataclass, field
from enum import Enum

from app.config import settings
from app.observability.logger import get_logger

log = get_logger(__name__)


class CircuitState(Enum):
    CLOSED = "closed"  # normal — calls go through
    OPEN = "open"  # failing fast — calls rejected immediately
    HALF_OPEN = "half_open"  # testing recovery — one call allowed through


@dataclass
class CircuitBreaker:
    """
    Per-adapter circuit breaker.
    Tracks consecutive failures and opens the circuit when the threshold
    is exceeded, preventing cascading failures.
    """

    adapter_name: str
    failure_count: int = field(default=0, init=False)
    state: CircuitState = field(default=CircuitState.CLOSED, init=False)
    last_failure_time: float = field(default=0.0, init=False)

    def is_open(self) -> bool:
        """
        Returns True if the circuit is open (calls should be rejected).
        Automatically transitions OPEN → HALF_OPEN after the recovery window.
        """
        if self.state == CircuitState.CLOSED:
            return False

        if self.state == CircuitState.OPEN:
            elapsed = time.time() - self.last_failure_time
            if elapsed >= settings.circuit_breaker_recovery_seconds:
                # Recovery window passed — allow one test request through
                self.state = CircuitState.HALF_OPEN
                log.info(
                    "circuit_breaker_half_open",
                    adapter=self.adapter_name,
                    elapsed=round(elapsed, 1),
                )
                return False
            return True

        # HALF_OPEN — let the request through (caller must call record_success
        # or record_failure based on the outcome)
        return False

    def record_success(self) -> None:
        """Call after a successful API response — resets the circuit."""
        if self.state != CircuitState.CLOSED:
            log.info(
                "circuit_breaker_closed", adapter=self.adapter_name, previous_state=self.state.value
            )
        self.failure_count = 0
        self.state = CircuitState.CLOSED

    def record_failure(self) -> None:
        """Call after a failed API response — may open the circuit."""
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.failure_count >= settings.circuit_breaker_failure_threshold:
            if self.state != CircuitState.OPEN:
                log.warning(
                    "circuit_breaker_opened",
                    adapter=self.adapter_name,
                    failures=self.failure_count,
                    threshold=settings.circuit_breaker_failure_threshold,
                )
            self.state = CircuitState.OPEN


class CircuitBreakerRegistry:
    """
    Holds one CircuitBreaker per adapter name.
    Single shared instance — imported by the fallback orchestrator.
    """

    def __init__(self) -> None:
        self._breakers: dict[str, CircuitBreaker] = {}

    def get(self, adapter_name: str) -> CircuitBreaker:
        if adapter_name not in self._breakers:
            self._breakers[adapter_name] = CircuitBreaker(adapter_name=adapter_name)
        return self._breakers[adapter_name]


# Single shared instance
circuit_registry = CircuitBreakerRegistry()
