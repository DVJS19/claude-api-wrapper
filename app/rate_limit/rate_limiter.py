import time
from dataclasses import dataclass, field

from app.config import settings
from app.observability.logger import get_logger

log = get_logger(__name__)


class RateLimitExceededError(Exception):
    """Raised when a client exceeds their request rate limit."""

    def __init__(self, client_id: str, retry_after_seconds: float):
        self.client_id = client_id
        self.retry_after_seconds = retry_after_seconds
        super().__init__(
            f"Rate limit exceeded for client '{client_id}'. "
            f"Retry after {retry_after_seconds:.1f} seconds."
        )


@dataclass
class TokenBucket:
    """
    Token bucket for one API key.

    Tokens refill continuously at requests_per_minute / 60 per second.
    Burst allows short spikes above the sustained rate.
    """

    client_id: str
    capacity: float  # max tokens (burst size)
    refill_rate_per_sec: float  # tokens added per second
    tokens: float = field(init=False)
    last_refill_time: float = field(default_factory=time.time, init=False)

    def __post_init__(self) -> None:
        self.tokens = self.capacity  # start full

    def _refill(self) -> None:
        """Add tokens based on elapsed time since last refill."""
        now = time.time()
        elapsed = now - self.last_refill_time
        gained = elapsed * self.refill_rate_per_sec

        self.tokens = min(self.capacity, self.tokens + gained)
        self.last_refill_time = now

    def consume(self) -> None:
        """
        Consume one token.
        Raises RateLimitExceededError if the bucket is empty.
        """
        self._refill()

        if self.tokens < 1.0:
            # Calculate how long until one token is available
            retry_after = (1.0 - self.tokens) / self.refill_rate_per_sec
            log.warning(
                "rate_limit_exceeded",
                client_id=self.client_id,
                tokens_remaining=round(self.tokens, 3),
                retry_after=round(retry_after, 1),
            )
            raise RateLimitExceededError(
                client_id=self.client_id,
                retry_after_seconds=retry_after,
            )

        self.tokens -= 1.0
        log.info(
            "rate_limit_consumed",
            client_id=self.client_id,
            tokens_remaining=round(self.tokens, 3),
        )


class RateLimiter:
    """
    Per-API-key rate limiter.
    Creates a TokenBucket for each client on first use.
    In-memory — resets on server restart.
    """

    def __init__(self) -> None:
        self._buckets: dict[str, TokenBucket] = {}

    def _get_bucket(self, client_id: str) -> TokenBucket:
        if client_id not in self._buckets:
            self._buckets[client_id] = TokenBucket(
                client_id=client_id,
                capacity=float(settings.rate_limit_burst_size),
                refill_rate_per_sec=(settings.rate_limit_requests_per_minute / 60.0),
            )
        return self._buckets[client_id]

    def check(self, client_id: str) -> None:
        """
        Check and consume one rate limit token for client_id.
        Raises RateLimitExceededError if the bucket is empty.
        """
        self._get_bucket(client_id).consume()

    def get_tokens_remaining(self, client_id: str) -> float:
        """Return current token count — used for response headers."""
        bucket = self._get_bucket(client_id)
        bucket._refill()
        return round(bucket.tokens, 3)


# Single shared instance
rate_limiter = RateLimiter()
