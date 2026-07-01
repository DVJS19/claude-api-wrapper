from typing import Optional

from pydantic import BaseModel, Field, field_validator

from app.config import settings


class GenerateRequest(BaseModel):
    """
    Request schema for POST /generate.

    Pydantic validates this BEFORE any application code runs — malformed
    requests (wrong types, missing fields, oversized prompts) are rejected
    with a 422 automatically by FastAPI.
    """

    prompt: str = Field(
        ...,
        min_length=1,
        description="The user prompt to send to Claude.",
    )

    system_context: Optional[str] = Field(
        default=None,
        max_length=2000,
        description=(
            "Optional additional context/instructions appended to the service's "
            "base system prompt. This does NOT replace the base system prompt — "
            "it is added alongside it. Use this for task-specific framing, "
            "not to override safety behavior."
        ),
    )

    temperature: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Sampling temperature 0.0-1.0. Defaults to a safe value if omitted.",
    )

    max_tokens: Optional[int] = Field(
        default=None,
        gt=0,
        description="Max output tokens. Defaults to settings.max_output_tokens if omitted.",
    )

    @field_validator("prompt")
    @classmethod
    def validate_prompt_length(cls, v: str) -> str:
        """
        Enforce max prompt length from settings — not hardcoded.
        Using a validator (not Field max_length) so this always reads
        the current settings value, regardless of import order.
        """
        if len(v) > settings.max_prompt_length_chars:
            raise ValueError(
                f"Prompt exceeds max length of {settings.max_prompt_length_chars} characters "
                f"(got {len(v)})"
            )
        return v

    @field_validator("prompt")
    @classmethod
    def validate_prompt_not_blank(cls, v: str) -> str:
        """Reject whitespace-only prompts — min_length=1 alone allows ' '."""
        if not v.strip():
            raise ValueError("Prompt cannot be blank or whitespace-only")
        return v
