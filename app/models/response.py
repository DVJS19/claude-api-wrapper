from pydantic import BaseModel, Field


class UsageInfo(BaseModel):
    """Token usage and cost for one request — grouped for clarity and extensibility."""

    input_tokens: int = Field(..., description="Tokens consumed by prompt + system prompt.")
    output_tokens: int = Field(..., description="Tokens generated in the response.")
    total_tokens: int = Field(..., description="input_tokens + output_tokens.")
    estimated_cost_usd: float = Field(..., description="Estimated cost in USD for this request.")


class GenerateResponse(BaseModel):
    """
    Response schema for POST /generate.

    Every field is guaranteed present and correctly typed — callers can rely
    on this contract without defensive null-checking.
    """

    text: str = Field(..., description="The generated response text.")

    model_used: str = Field(
        ...,
        description="Which model actually served this request (primary or fallback).",
    )

    fallback_used: bool = Field(
        ...,
        description="True if the fallback model was used instead of the primary.",
    )

    output_status: str = Field(
        default="ok",
        description=(
            "ok — normal response. "
            "no_information — model had no relevant data, controlled message returned. "
            "refused — model declined the request, controlled message returned."
        ),
    )

    usage: UsageInfo = Field(..., description="Token and cost accounting for this request.")

    request_id: str = Field(
        ..., description="Unique ID for this request — use for support/debugging."
    )
