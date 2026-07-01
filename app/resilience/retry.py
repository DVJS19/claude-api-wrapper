import anthropic
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.config import settings
from app.observability.logger import get_logger

log = get_logger(__name__)


def _is_transient(exc: BaseException) -> bool:
    """
    Returns True for errors worth retrying — transient infrastructure failures.
    Returns False for permanent errors where retrying wastes money and time.
    """
    if isinstance(exc, anthropic.RateLimitError):
        return True
    if isinstance(exc, anthropic.APITimeoutError):
        return True
    # 5xx server errors are transient; 4xx client errors are permanent
    if isinstance(exc, anthropic.APIStatusError):
        return exc.status_code >= 500
    return False


def build_retry_decorator():
    """
    Build a tenacity retry decorator using values from settings.
    Called once per adapter instance — not at module load time —
    so settings are fully initialised when this runs.
    """
    return retry(
        retry=retry_if_exception_type(
            (anthropic.RateLimitError, anthropic.APITimeoutError, anthropic.APIStatusError)
        ),
        wait=wait_exponential(
            multiplier=settings.retry_backoff_base_seconds,
            min=settings.retry_backoff_base_seconds,
            max=30.0,
        ),
        stop=stop_after_attempt(settings.retry_max_attempts),
        reraise=True,  # re-raise the original exception after all attempts exhausted
    )
