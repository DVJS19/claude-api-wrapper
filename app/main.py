"""
FastAPI entry point.
Phase 1: health check, OAuth token endpoint, placeholder protected /generate route.
Phase 2: /generate becomes the full Pydantic-validated Claude call.
"""

from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI
from fastapi.security import OAuth2PasswordRequestForm

from app.auth.oauth import (
    TokenResponse,
    get_current_client_id,
    issue_token,
    seed_demo_client,
)
from app.config import settings
from app.observability.logger import get_logger, setup_logging

log = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    log.info("startup", env=settings.app_env)

    # Seed a demo OAuth client so the API is testable immediately.
    # In production, clients are registered out-of-band (admin tooling, not at boot).
    seed_demo_client()
    log.info("oauth_demo_client_ready")

    yield

    log.info("shutdown")


app = FastAPI(
    title="Claude API Wrapper",
    description="Production-hardened wrapper around the Anthropic API",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health():
    return {"status": "ok", "env": settings.app_env}


@app.post("/oauth/token", response_model=TokenResponse)
async def oauth_token(form_data: OAuth2PasswordRequestForm = Depends()):
    """
    OAuth2 Client Credentials grant.
    form_data.username = client_id, form_data.password = client_secret
    (FastAPI's OAuth2PasswordRequestForm uses these field names regardless of grant type.)
    """
    return issue_token(form_data.username, form_data.password)


@app.post("/generate")
async def generate(client_id: str = Depends(get_current_client_id)):
    """
    placeholder — todo: adds the full Pydantic request/response
    models and the adapter-based Claude call.
    """
    return {
        "message": "Authenticated successfully. Full generation endpoint coming in Phase 2.",
        "client_id": client_id,
    }
