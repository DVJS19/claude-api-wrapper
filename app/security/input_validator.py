import re
import unicodedata

from app.observability.logger import get_logger

log = get_logger(__name__)


class InputValidationError(ValueError):
    """Raised when input fails content sanitisation."""

    pass


def sanitise_prompt(prompt: str) -> str:
    """
    Sanitise a prompt string before sending to the model.

    Removes control characters and null bytes that could cause unexpected
    model behaviour or log injection. Normalises Unicode to NFC form.
    Returns the cleaned prompt.
    """
    # Remove null bytes — these can cause issues in downstream processing
    prompt = prompt.replace("\x00", "")

    # Remove ASCII control characters except tab, newline, carriage return
    # (which are legitimate in prompts)
    prompt = re.sub(r"[\x01-\x08\x0b\x0c\x0e-\x1f\x7f]", "", prompt)

    # Normalise Unicode to NFC — prevents homoglyph attacks
    # (where visually identical characters have different code points)
    prompt = unicodedata.normalize("NFC", prompt)

    # Collapse more than 3 consecutive newlines — legitimate prompts don't need more
    prompt = re.sub(r"\n{4,}", "\n\n\n", prompt)

    return prompt.strip()


def check_repetition(prompt: str) -> None:
    """
    Detect excessive repetition — a signal of token-stuffing attacks.
    Raises InputValidationError if the same phrase repeats too many times.
    """
    words = prompt.lower().split()
    if len(words) < 20:
        return  # too short to meaningfully analyse

    # Count unique words — very low ratio signals repetitive content
    unique_ratio = len(set(words)) / len(words)
    if unique_ratio < 0.1:
        raise InputValidationError("Prompt contains excessive repetition and cannot be processed.")


def validate_and_sanitise(prompt: str, client_id: str = "unknown") -> str:
    """
    Full input validation pipeline: sanitise then check for abuse patterns.
    Returns the cleaned prompt ready to send to the model.
    """
    cleaned = sanitise_prompt(prompt)
    check_repetition(cleaned)

    if len(cleaned) < 1:
        raise InputValidationError("Prompt is empty after sanitisation.")

    log.info(
        "input_validated", client_id=client_id, original_len=len(prompt), cleaned_len=len(cleaned)
    )
    return cleaned
