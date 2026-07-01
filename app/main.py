import uuid
import os
from contextlib import asynccontextmanager
from fastapi import Depends, FastAPI, HTTPException, status
from app.resilience.fallback import ServiceUnavailableError, execute_with_fallback
from fastapi.security import OAuth2PasswordRequestForm
from app.security.input_validator import (
    InputValidationError,
    validate_and_sanitise,
)
from app.security.output_validator import validate_output
from app.security.prompt_guard import PromptInjectionError, check_prompt_injection

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

    Request flow:
        1. JWT verified (dependency)
        2. Pydantic validates request body
        3. Prompt injection check
        4. Input sanitisation
        5. Adapter selected (cost-based routing)
        6. Model called (with retry + circuit breaker + fallback)
        7. Output validated (no_information / refusal detection)
        8. Pydantic validates response
        9. Structured audit log written
    """
    request_id = str(uuid.uuid4())

    log.info(
        "generate_request_received",
        client_id=client_id,
        request_id=request_id,
        prompt_len=len(request.prompt),
        has_system_context=request.system_context is not None,
    )

    # ── Security: prompt injection check ──────────────────────────────────────
    try:
        check_prompt_injection(request.prompt, client_id=client_id)
        if request.system_context:
            check_prompt_injection(request.system_context, client_id=client_id)
    except PromptInjectionError as e:
        log.warning("generate_injection_blocked", client_id=client_id, request_id=request_id)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )

    # ── Security: input sanitisation ──────────────────────────────────────────
    try:
        clean_prompt = validate_and_sanitise(request.prompt, client_id=client_id)
        clean_context = (
            validate_and_sanitise(request.system_context, client_id=client_id)
            if request.system_context
            else None
        )
    except InputValidationError as e:
        log.warning(
            "generate_input_invalid", client_id=client_id, request_id=request_id, error=str(e)
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )

    # ── Resolve optional fields to defaults ───────────────────────────────────
    temperature = request.temperature if request.temperature is not None else 0.7
    max_tokens = (
        request.max_tokens if request.max_tokens is not None else settings.max_output_tokens
    )

    # ── Adapter selection (cost-based routing) ────────────────────────────────
    selection = selector.select_adapter(
        caller_context=clean_context or "",
        user_prompt=clean_prompt,
        max_tokens=max_tokens,
    )

    is_primary_selected = selection.adapter.model_name == settings.primary_model
    primary_adapter = selection.adapter if is_primary_selected else selector._fallback
    fallback_adapter = selector._fallback if is_primary_selected else selector._primary

    # ── Model call (retry + circuit breaker + fallback) ───────────────────────
    try:
        result, failure_fallback_used = await execute_with_fallback(
            primary_adapter=primary_adapter,
            fallback_adapter=fallback_adapter,
            caller_context=clean_context or "",
            user_prompt=clean_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    except ServiceUnavailableError as e:
        log.error(
            "generate_both_adapters_failed",
            client_id=client_id,
            request_id=request_id,
            error=str(e),
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service temporarily unavailable. Please retry shortly.",
        )

    # ── Output validation ──────────────────────────────────────────────────────
    validated = validate_output(result.text, client_id=client_id)

    fallback_used = selection.adapter.model_name != settings.primary_model or failure_fallback_used

    log.info(
        "generate_request_completed",
        client_id=client_id,
        request_id=request_id,
        model_used=result.model_name,
        fallback_used=fallback_used,
        output_status=validated.status.value,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        cost_usd=result.cost_usd,
        selection_reason=selection.reason,
    )

    return GenerateResponse(
        text=validated.text,
        model_used=result.model_name,
        fallback_used=fallback_used,
        output_status=validated.status.value,
        usage=UsageInfo(
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            total_tokens=result.input_tokens + result.output_tokens,
            estimated_cost_usd=result.cost_usd,
        ),
        request_id=request_id,
    )
