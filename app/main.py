import uuid
import os
from contextlib import asynccontextmanager
from fastapi import Depends, FastAPI, HTTPException, status
from app.resilience.fallback import ServiceUnavailableError, execute_with_fallback
from fastapi.security import OAuth2PasswordRequestForm

from app.adapters.selector import selector
from app.auth.oauth import (
    TokenResponse,
    get_current_client_id,
    issue_token,
    seed_demo_client,
)
from app.config import settings
from app.models.request import GenerateRequest
from app.models.response import GenerateResponse, UsageInfo
from app.observability.logger import get_logger, setup_logging

log = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: seed demo client. Shutdown: log."""
    setup_logging()
    log.info("startup", env=settings.app_env)

    # Push API key into os.environ so the Anthropic SDK finds it.
    # pydantic-settings loads .env into settings but not into os.environ.
    os.environ["ANTHROPIC_API_KEY"] = settings.anthropic_api_key

    log.info("startup", env=settings.app_env)

    # Seed one demo client so the API is immediately testable.
    # In production, clients are registered out-of-band via admin tooling.
    seed_demo_client()
    log.info("oauth_demo_client_ready")

    yield

    log.info("shutdown")


app = FastAPI(
    title="Claude API Wrapper",
    description="Production-hardened wrapper around the Anthropic API",
    version="0.2.0",
    lifespan=lifespan,
)


# ── Public endpoints ───────────────────────────────────────────────────────────


@app.get("/health")
async def health():
    return {"status": "ok", "env": settings.app_env}


@app.post("/oauth/token", response_model=TokenResponse)
async def oauth_token(form_data: OAuth2PasswordRequestForm = Depends()):
    """
    OAuth2 Client Credentials grant.
    username = client_id, password = client_secret.
    Returns a short-lived JWT for use on all protected endpoints.
    """
    return issue_token(form_data.username, form_data.password)


# ── Protected endpoints ────────────────────────────────────────────────────────


@app.post("/generate", response_model=GenerateResponse)
async def generate(
    request: GenerateRequest,
    client_id: str = Depends(get_current_client_id),
) -> GenerateResponse:
    """
    Generate a response from Claude.

    What happens in order:
    1. FastAPI validates the JWT (get_current_client_id dependency)
    2. Pydantic validates the request body (GenerateRequest)
    3. Adapter selector estimates cost and picks primary or fallback model
    4. compose_system_prompt() builds the final system prompt:
           BASE_SYSTEM_PROMPT (always, not caller-editable)
           + request.system_context (optional caller addition, appended only)
    5. Adapter calls the Anthropic API
    6. Pydantic validates the response (GenerateResponse)
    7. Structured log records client_id, model used, tokens, cost, selection reason
    """
    request_id = str(uuid.uuid4())

    log.info(
        "generate_request_received",
        client_id=client_id,
        request_id=request_id,
        prompt_len=len(request.prompt),
        has_system_context=request.system_context is not None,
    )

    # Resolve optional fields to defaults
    temperature = request.temperature if request.temperature is not None else 0.7
    max_tokens = (
        request.max_tokens if request.max_tokens is not None else settings.max_output_tokens
    )

    # Select adapter — cost-based routing
    # will extend this with per-key budget enforcement
    selection = selector.select_adapter(
        caller_context=request.system_context or "",
        user_prompt=request.prompt,
        max_tokens=max_tokens,
    )

    # Determine which adapter is primary vs fallback for this request.
    # The selector already picked one based on cost — the other is the fallback
    # for failure-triggered routing.
    is_primary_selected = selection.adapter.model_name == settings.primary_model
    primary_adapter  = selection.adapter          if is_primary_selected else selector._fallback
    fallback_adapter = selector._fallback         if is_primary_selected else selector._primary

    try:
        result, failure_fallback_used = await execute_with_fallback(
            primary_adapter=primary_adapter,
            fallback_adapter=fallback_adapter,
            caller_context=request.system_context or "",
            user_prompt=request.prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    except ServiceUnavailableError as e:
        log.error("generate_both_adapters_failed",
                  client_id=client_id, request_id=request_id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service temporarily unavailable. Please retry shortly.",
        )

    # fallback_used is True if cost routing OR failure routing chose the fallback
    fallback_used = (selection.adapter.model_name != settings.primary_model) or failure_fallback_used

    log.info(
        "generate_request_completed",
        client_id=client_id,
        request_id=request_id,
        model_used=result.model_name,
        fallback_used=fallback_used,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        cost_usd=result.cost_usd,
        selection_reason=selection.reason,
    )

    return GenerateResponse(
        text=result.text,
        model_used=result.model_name,
        fallback_used=fallback_used,    # ← updated variable
        usage=UsageInfo(
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            total_tokens=result.input_tokens + result.output_tokens,
            estimated_cost_usd=result.cost_usd,
        ),
        request_id=request_id,
    )
