# tests/test_rate_limit_and_cost.py

import pytest

from app.cost.tracker import BudgetHardLimitError, CostTracker
from app.rate_limit.rate_limiter import RateLimitExceededError, TokenBucket, RateLimiter

# ── TokenBucket tests ──────────────────────────────────────────────────────────


class TestTokenBucket:
    def test_starts_full(self):
        """Bucket starts at full capacity."""
        bucket = TokenBucket(
            client_id="test",
            capacity=5.0,
            refill_rate_per_sec=1.0,
        )
        assert bucket.tokens == 5.0

    def test_consume_reduces_tokens(self):
        bucket = TokenBucket(client_id="test", capacity=5.0, refill_rate_per_sec=1.0)
        bucket.consume()
        assert bucket.tokens < 5.0

    def test_empty_bucket_raises(self):
        """Consuming from an empty bucket raises RateLimitExceededError."""
        bucket = TokenBucket(client_id="test", capacity=1.0, refill_rate_per_sec=0.01)
        bucket.consume()  # empties it
        with pytest.raises(RateLimitExceededError) as exc:
            bucket.consume()
        assert exc.value.client_id == "test"
        assert exc.value.retry_after_seconds > 0

    def test_retry_after_is_positive(self):
        """retry_after_seconds gives caller a useful wait time."""
        bucket = TokenBucket(client_id="test", capacity=1.0, refill_rate_per_sec=1.0)
        bucket.consume()
        with pytest.raises(RateLimitExceededError) as exc:
            bucket.consume()
        assert exc.value.retry_after_seconds > 0


class TestRateLimiter:
    def test_different_clients_have_independent_buckets(self):
        """Each client gets its own bucket — one client's usage doesn't affect another."""
        limiter = RateLimiter()
        # Exhaust client A
        for _ in range(5):
            try:
                limiter.check("client-a")
            except RateLimitExceededError:
                break
        # Client B should still have tokens
        limiter.check("client-b")  # should not raise

    def test_burst_allows_multiple_quick_requests(self, monkeypatch):
        """Burst size allows requests up to the burst limit without waiting."""
        from app.config import settings

        monkeypatch.setattr(settings, "rate_limit_burst_size", 3)
        monkeypatch.setattr(settings, "rate_limit_requests_per_minute", 10)

        limiter = RateLimiter()
        # Should allow burst_size requests without error
        limiter.check("burst-client")
        limiter.check("burst-client")
        limiter.check("burst-client")

        # Next one should fail
        with pytest.raises(RateLimitExceededError):
            limiter.check("burst-client")


# ── CostTracker tests ──────────────────────────────────────────────────────────


class TestCostTracker:
    def test_no_constraint_below_soft_limit(self, monkeypatch):
        """Below soft limit, get_forced_model returns None."""
        from app.config import settings

        monkeypatch.setattr(settings, "daily_soft_budget_usd", 5.0)
        monkeypatch.setattr(settings, "daily_hard_budget_usd", 10.0)

        tracker = CostTracker()
        tracker.record_spend("client-a", 1.0)
        assert tracker.get_forced_model("client-a") is None

    def test_soft_limit_forces_fallback_model(self, monkeypatch):
        """Exceeding soft limit forces fallback model."""
        from app.config import settings

        monkeypatch.setattr(settings, "daily_soft_budget_usd", 1.0)
        monkeypatch.setattr(settings, "daily_hard_budget_usd", 10.0)

        tracker = CostTracker()
        tracker.record_spend("client-b", 1.50)  # over soft limit

        forced = tracker.get_forced_model("client-b")
        assert forced == settings.fallback_model

    def test_hard_limit_raises(self, monkeypatch):
        """Exceeding hard limit raises BudgetHardLimitError."""
        from app.config import settings

        monkeypatch.setattr(settings, "daily_soft_budget_usd", 1.0)
        monkeypatch.setattr(settings, "daily_hard_budget_usd", 2.0)

        tracker = CostTracker()
        tracker.record_spend("client-c", 2.50)  # over hard limit

        with pytest.raises(BudgetHardLimitError) as exc:
            tracker.get_forced_model("client-c")
        assert exc.value.client_id == "client-c"

    def test_spend_accumulates_across_requests(self, monkeypatch):
        """Multiple small requests accumulate toward the budget."""
        from app.config import settings

        monkeypatch.setattr(settings, "daily_soft_budget_usd", 5.0)
        monkeypatch.setattr(settings, "daily_hard_budget_usd", 10.0)

        tracker = CostTracker()
        tracker.record_spend("client-d", 0.01)
        tracker.record_spend("client-d", 0.02)
        tracker.record_spend("client-d", 0.03)

        summary = tracker.get_summary("client-d")
        assert abs(summary["spent_usd"] - 0.06) < 0.0001

    def test_summary_shows_remaining_budget(self, monkeypatch):
        """get_summary returns remaining budget correctly."""
        from app.config import settings

        monkeypatch.setattr(settings, "daily_hard_budget_usd", 10.0)
        monkeypatch.setattr(settings, "daily_soft_budget_usd", 5.0)

        tracker = CostTracker()
        tracker.record_spend("client-e", 3.0)

        summary = tracker.get_summary("client-e")
        assert summary["remaining_usd"] == 7.0
        assert summary["soft_limit_hit"] is False

    def test_daily_reset_clears_spend(self, monkeypatch):
        """Spend resets when the UTC date changes."""
        from datetime import date
        from app.config import settings

        monkeypatch.setattr(settings, "daily_soft_budget_usd", 1.0)
        monkeypatch.setattr(settings, "daily_hard_budget_usd", 2.0)

        tracker = CostTracker()
        tracker.record_spend("client-f", 1.50)

        # Simulate a new day by backdating the spend_date
        state = tracker._get_state("client-f")
        state.spend_date = date(2000, 1, 1)  # far in the past

        # Should reset and no longer be over limit
        assert tracker.get_forced_model("client-f") is None
        summary = tracker.get_summary("client-f")
        assert summary["spent_usd"] == 0.0
