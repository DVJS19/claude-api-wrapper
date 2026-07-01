import anthropic

from app.adapters.base import ModelAdapter, ModelResult, compose_system_prompt
from app.config import settings


class HaikuAdapter(ModelAdapter):
    """
    Adapter for claude-haiku-4-5 — the fallback model.
    Lower cost, faster, used when cost routing or failure-based
    fallback selects it instead of the primary model.
    """

    model_name = settings.fallback_model

    # Haiku 4.5 pricing — update here if rates change, nowhere else
    cost_per_input_token_usd = 0.000001  # $1.00 / 1M input tokens
    cost_per_output_token_usd = 0.000005  # $5.00 / 1M output tokens

    def __init__(self) -> None:
        self._client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)

    async def generate(
        self,
        caller_context: str,
        user_prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> ModelResult:
        final_system_prompt = compose_system_prompt(caller_context)

        response = await self._client.messages.create(
            model=self.model_name,
            max_tokens=max_tokens,
            temperature=temperature,
            system=final_system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )

        text = response.content[0].text if response.content else ""

        cost = (
            response.usage.input_tokens * self.cost_per_input_token_usd
            + response.usage.output_tokens * self.cost_per_output_token_usd
        )

        return ModelResult(
            text=text,
            model_name=self.model_name,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            cost_usd=round(cost, 6),
        )
