import re
from dataclasses import dataclass
from enum import Enum

from app.observability.logger import get_logger

log = get_logger(__name__)


class OutputStatus(Enum):
    OK = "ok"  # normal response
    NO_INFO = "no_information"  # model has no relevant information
    REFUSED = "refused"  # model declined the request


@dataclass
class ValidatedOutput:
    text: str
    status: OutputStatus


# Phrases indicating the model has no relevant information.
# Checked in the first 300 characters of the response to avoid false positives.
NO_INFO_SIGNALS = [
    r"i\s+(don't|do not|doesn't|does not)\s+have\s+(any\s+)?(information|data|details)",
    r"i\s+(cannot|can't|could not|couldn't)\s+find",
    r"i\s+(don't|do not)\s+have\s+access\s+to",
    r"no\s+(information|data|details)\s+(is\s+|are\s+)?(available|found|provided)",
    r"i\s+was\s+not\s+(provided|given)\s+(any\s+)?(information|context|data)",
    r"there\s+(is|are)\s+no\s+(information|data|details)",
    r"i\s+(don't|do not)\s+see\s+any\s+(contract|document|text|content)",
]

# Phrases indicating the model refused the request.
REFUSAL_SIGNALS = [
    r"i\s+(cannot|can't|am\s+not\s+able\s+to)\s+help\s+with\s+that",
    r"i\s+(cannot|can't|am\s+not\s+able\s+to)\s+assist\s+with",
    r"i\s+(won't|will\s+not)\s+(help|assist)\s+with",
    r"that\s+(request\s+)?(is|falls)\s+(outside|beyond)",
    r"i\s+(must|have\s+to)\s+decline",
]

_COMPILED_NO_INFO = [re.compile(p, re.IGNORECASE) for p in NO_INFO_SIGNALS]
_COMPILED_REFUSALS = [re.compile(p, re.IGNORECASE) for p in REFUSAL_SIGNALS]

# Controlled messages returned instead of uncertain/hallucinated content
NO_INFORMATION_MESSAGE = (
    "No relevant information was found to answer this question. "
    "Please provide more context or rephrase your request."
)

REFUSED_MESSAGE = (
    "This request cannot be processed as it falls outside the scope of what this service supports."
)

# How many characters from the start of the response to check for signals.
# Checking the full response causes too many false positives.
SIGNAL_CHECK_WINDOW = 300


def validate_output(
    text: str,
    client_id: str = "unknown",
) -> ValidatedOutput:
    """
    Validate model output before returning to the caller.

    Detects uncertainty and refusal signals — returns controlled messages
    instead of potentially hallucinated or mismatched content.
    """
    if not text or not text.strip():
        log.warning("output_empty", client_id=client_id)
        return ValidatedOutput(text=NO_INFORMATION_MESSAGE, status=OutputStatus.NO_INFO)

    # Check only the beginning of the response for signals
    check_window = text[:SIGNAL_CHECK_WINDOW].lower()

    # Check for refusal first — higher priority than no_info
    for pattern in _COMPILED_REFUSALS:
        if pattern.search(check_window):
            log.info("output_refusal_detected", client_id=client_id)
            return ValidatedOutput(text=REFUSED_MESSAGE, status=OutputStatus.REFUSED)

    # Check for no-information signals
    for pattern in _COMPILED_NO_INFO:
        if pattern.search(check_window):
            log.info("output_no_information_detected", client_id=client_id)
            return ValidatedOutput(
                text=NO_INFORMATION_MESSAGE,
                status=OutputStatus.NO_INFO,
            )

    return ValidatedOutput(text=text, status=OutputStatus.OK)
