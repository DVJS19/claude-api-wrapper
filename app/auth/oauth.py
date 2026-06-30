"""
OAuth2 Client Credentials flow — the correct pattern for machine-to-machine
API access (no human login UI, no user accounts, just service-to-service auth).

Flow:
    1. Client POSTs client_id + client_secret to /oauth/token
    2. We verify the secret hash, issue a short-lived JWT
    3. Client includes JWT in Authorization: Bearer <token> on every API call
    4. We verify the JWT signature + expiry on every request

Client registry is in-memory by design (project scope decision — no DB).
In production this would be a secrets manager or a dedicated auth service.
"""

from datetime import datetime, timedelta, timezone

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel

from app.config import settings
from app.observability.logger import get_logger

log = get_logger(__name__)

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# OAuth2PasswordBearer just tells FastAPI/OpenAPI docs where to get a token from.
# We implement client_credentials grant manually below — this is the standard
# pattern since FastAPI's built-in helper assumes the password grant.
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="oauth/token")


# ── In-memory client registry ───────────────────────────────────────────────────
# {client_id: hashed_secret}
# Seeded with one demo client at startup — see seed_demo_client() below.
_CLIENT_REGISTRY: dict[str, str] = {}


def register_client(client_id: str, client_secret: str) -> None:
    """Register a client. Secret is hashed immediately — never stored in plaintext."""
    _CLIENT_REGISTRY[client_id] = pwd_context.hash(client_secret)
    log.info("oauth_client_registered", client_id=client_id)


def seed_demo_client() -> None:
    """Seed one demo client at startup so the API is testable out of the box."""
    register_client("demo-client", "demo-secret-change-me")


def _verify_client(client_id: str, client_secret: str) -> bool:
    hashed = _CLIENT_REGISTRY.get(client_id)
    if not hashed:
        return False
    return pwd_context.verify(client_secret, hashed)


# ── Token models ─────────────────────────────────────────────────────────────────
class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int


class TokenPayload(BaseModel):
    sub: str  # client_id
    exp: int  # expiry as unix timestamp


# ── Token issuance ────────────────────────────────────────────────────────────────
def issue_token(client_id: str, client_secret: str) -> TokenResponse:
    """
    Validate client credentials and issue a short-lived JWT.
    Raises HTTPException(401) if credentials are invalid.
    """
    if not _verify_client(client_id, client_secret):
        log.warning("oauth_auth_failed", client_id=client_id)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid client credentials",
        )

    expire = datetime.now(timezone.utc) + timedelta(seconds=settings.oauth_token_expiry_seconds)
    payload = {"sub": client_id, "exp": int(expire.timestamp())}

    token = jwt.encode(
        payload,
        settings.oauth_jwt_secret,
        algorithm=settings.oauth_jwt_algorithm,
    )

    log.info(
        "oauth_token_issued", client_id=client_id, expires_in=settings.oauth_token_expiry_seconds
    )

    return TokenResponse(
        access_token=token,
        expires_in=settings.oauth_token_expiry_seconds,
    )


# ── Token verification (FastAPI dependency) ───────────────────────────────────────
async def get_current_client_id(token: str = Depends(oauth2_scheme)) -> str:
    """
    FastAPI dependency — verifies the JWT and returns the client_id.
    Use as: client_id: str = Depends(get_current_client_id)
    Raises HTTPException(401) on invalid/expired token.
    """
    try:
        payload = jwt.decode(
            token,
            settings.oauth_jwt_secret,
            algorithms=[settings.oauth_jwt_algorithm],
        )
        client_id = payload.get("sub")
        if client_id is None:
            raise JWTError("Missing subject claim")
        return client_id

    except JWTError as e:
        log.warning("oauth_token_invalid", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )
