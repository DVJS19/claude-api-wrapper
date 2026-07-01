import re

from app.config import settings
from app.observability.logger import get_logger

log = get_logger(__name__)

# Known injection patterns — phrases commonly used to override system prompts.
# Case-insensitive matching.
INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?previous\s+instructions",
    r"ignore\s+(all\s+)?prior\s+instructions",
    r"disregard\s+(all\s+)?previous\s+instructions",
    r"forget\s+(all\s+)?(your\s+)?instructions",
    r"you\s+are\s+now\s+\w+",  # "you are now DAN / an evil AI / etc"
    r"act\s+as\s+(if\s+you\s+are\s+)?(?!a\s+helpful)",  # "act as [something other than helpful]"
    r"pretend\s+you\s+(have\s+no|are\s+not)",
    r"reveal\s+(your\s+)?(system\s+prompt|instructions|api\s+key)",
    r"(print|show|display|output)\s+(your\s+)?(system\s+prompt|instructions)",
    r"jailbreak",
    r"do\s+anything\s+now",  # DAN variant
    r"developer\s+mode",
    r"sudo\s+mode",
    r"bypass\s+(all\s+)?(safety|filter|restriction)",
]

# Compiled once at module load — not per request
_COMPILED_PATTERNS = [re.compile(pattern, re.IGNORECASE) for pattern in INJECTION_PATTERNS]


class PromptInjectionError(ValueError):
    """Raised when a prompt injection attempt is detected."""

    pass


def check_prompt_injection(prompt: str, client_id: str = "unknown") -> None:
    """
    Scan a prompt for injection attempts.
    Raises PromptInjectionError if any pattern matches.
    Does nothing if injection checking is disabled in settings.
    """
    if not settings.prompt_injection_check_enabled:
        return

    prompt_lower = prompt.lower()

    for pattern in _COMPILED_PATTERNS:
        match = pattern.search(prompt_lower)
        if match:
            log.warning(
                "prompt_injection_detected",
                client_id=client_id,
                pattern=pattern.pattern,
                matched=match.group(0),
            )
            raise PromptInjectionError(
                "Request contains content that appears to be a prompt injection attempt "
                "and cannot be processed."
            )
