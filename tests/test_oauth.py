"""
Phase 1 tests — OAuth2 client credentials flow.
Run: uv run pytest tests/test_oauth.py -v
"""

import pytest
from fastapi import HTTPException

from app.auth.oauth import (
    _CLIENT_REGISTRY,
    get_current_client_id,
    issue_token,
    register_client,
    seed_demo_client,
)


@pytest.fixture(autouse=True)
def clean_registry():
    """Reset the in-memory client registry before each test."""
    _CLIENT_REGISTRY.clear()
    yield
    _CLIENT_REGISTRY.clear()


def test_register_client_hashes_secret():
    """Secrets are never stored in plaintext."""
    register_client("test-client", "test-secret")
    stored = _CLIENT_REGISTRY["test-client"]
    assert stored != "test-secret"
    assert stored.startswith("$2b$")  # bcrypt hash prefix


def test_issue_token_with_valid_credentials():
    """Valid credentials produce a usable token."""
    register_client("test-client", "test-secret")
    response = issue_token("test-client", "test-secret")

    assert response.access_token
    assert response.token_type == "bearer"
    assert response.expires_in > 0


def test_issue_token_with_wrong_secret_raises_401():
    """Wrong secret is rejected."""
    register_client("test-client", "correct-secret")

    with pytest.raises(HTTPException) as exc:
        issue_token("test-client", "wrong-secret")
    assert exc.value.status_code == 401


def test_issue_token_with_unknown_client_raises_401():
    """Unknown client_id is rejected."""
    with pytest.raises(HTTPException) as exc:
        issue_token("nonexistent-client", "any-secret")
    assert exc.value.status_code == 401


@pytest.mark.asyncio
async def test_get_current_client_id_with_valid_token():
    """A valid token resolves back to the correct client_id."""
    register_client("test-client", "test-secret")
    token_response = issue_token("test-client", "test-secret")

    client_id = await get_current_client_id(token_response.access_token)
    assert client_id == "test-client"


@pytest.mark.asyncio
async def test_get_current_client_id_with_garbage_token_raises_401():
    """A malformed token is rejected."""
    with pytest.raises(HTTPException) as exc:
        await get_current_client_id("not-a-real-jwt-token")
    assert exc.value.status_code == 401


def test_seed_demo_client_creates_demo_client():
    """seed_demo_client() registers the expected demo client."""
    seed_demo_client()
    assert "demo-client" in _CLIENT_REGISTRY
