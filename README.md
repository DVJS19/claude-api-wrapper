# Claude API Wrapper

A production-hardened FastAPI wrapper around the Anthropic API. No agentic logic —
this project focuses entirely on the operational concerns required to run a single
LLM endpoint safely and reliably in production: auth, resilience, security, cost
control, and observability.

## Why this project exists

Most LLM demos skip the boring parts: who is allowed to call this API, what happens
when Anthropic returns a 529, how do you stop one client from burning your entire
monthly budget, how do you know if someone is trying to jailbreak your system prompt.
This project answers all of those questions with working code.

## What it covers

- **OAuth2 Client Credentials flow** — machine-to-machine auth, JWT-based
- **Pydantic-validated request/response schemas** — strict typing in and out
- **Adapter pattern** — swap between Sonnet/Haiku based on cost and failures
- **Resilience** — retry with exponential backoff, circuit breaker, timeouts
- **Security** — prompt injection detection, input/output validation
- **Rate limiting** — per-API-key token bucket
- **Cost control** — per-request cost estimation + per-key daily budget enforcement
- **Observability** — structured JSON audit logs for every request

## Tech stack

Python · FastAPI · Anthropic SDK · Pydantic · python-jose (JWT) · UV

## Auth

OAuth2 Client Credentials flow, JWT issuance/verification, bcrypt secret hashing

## Validation

Pydantic request/response schemas, field validators, custom constraints

## Model Routing

Adapter pattern — Sonnet/Haiku with cost-based and budget-based selection 

## Resilience

Retry with exponential backoff, circuit breaker (3 states), failure fallback

## Security

Prompt injection detection, input sanitisation, output validation

## Rate Limiting

Token bucket per API key with lazy refill and precise Retry-After

## Cost Control

Per-request cost estimation + per-key daily soft/hard budget enforcement

## Observability

Structured JSON audit logs, per-request latency, X-Request-ID header


## Quick start

```bash
uv sync --all-extras
cp .env.example .env
# Add your ANTHROPIC_API_KEY to .env

uv run uvicorn app.main:app --reload --port 8001

# Get a token (demo client seeded at startup)
curl -X POST http://localhost:8001/oauth/token \
  -d "username=demo-client&password=demo-secret-change-me"

# Use the token
curl -X POST http://localhost:8001/generate \
  -H "Authorization: Bearer <token>"

# Check usage
curl http://localhost:8001/usage \
  -H "Authorization: Bearer "

