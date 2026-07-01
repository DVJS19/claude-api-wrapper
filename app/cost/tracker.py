from dataclasses import dataclass, field
from datetime import date, timezone, datetime

from app.config import settings
from app.observability.logger import get_logger

log = get_logger(__name__)


class BudgetHardLimitError(Exception):
    """Raised when a client's daily hard budget is exhausted."""

    def __init__(self, client_id: str, spent: float, limit: float):
        self.client_id = client_id
        self.spent = spent
        self.limit = limit
        super().__init__(
            f"Daily budget exhausted for client '{client_id}'. "
            f"Spent ${spent:.4f} of ${limit:.2f} daily limit."
        )


@dataclass
class BudgetState:
    """Per-client daily spend state."""

    client_id: str
    spend_date: date = field(default_factory=lambda: datetime.now(timezone.utc).date())
    accumulated_usd: float = 0.0
    soft_limit_hit: bool = False

    def reset_if_new_day(self) -> None:
        """Reset spend if the UTC date has changed since last update."""
        today = datetime.now(timezone.utc).date()
        if today != self.spend_date:
            log.info(
                "budget_daily_reset",
                client_id=self.client_id,
                previous_spend=self.accumulated_usd,
                previous_date=self.spend_date.isoformat(),
            )
            self.spend_date = today
            self.accumulated_usd = 0.0
            self.soft_limit_hit = False


class CostTracker:
    """
    Per-API-key daily cost tracker with soft and hard budget enforcement.

    Soft limit: downgrade all requests to the fallback (cheaper) model.
    Hard limit: reject all requests until the next UTC day.

    In-memory — resets on server restart.
    Production version would use Redis with a TTL key expiring at midnight.
    """

    def __init__(self) -> None:
        self._states: dict[str, BudgetState] = {}

    def _get_state(self, client_id: str) -> BudgetState:
        if client_id not in self._states:
            self._states[client_id] = BudgetState(client_id=client_id)
        state = self._states[client_id]
        state.reset_if_new_day()
        return state

    def get_forced_model(self, client_id: str) -> str | None:
        """
        Check budget state before a request.

        Returns:
            None               — no budget constraint, use cost routing as normal
            fallback model name — soft limit hit, force downgrade to cheaper model
        Raises:
            BudgetHardLimitError — hard limit hit, reject the request
        """
        state = self._get_state(client_id)

        # Hard limit — reject entirely
        if state.accumulated_usd >= settings.daily_hard_budget_usd:
            log.warning(
                "budget_hard_limit_hit",
                client_id=client_id,
                spent=state.accumulated_usd,
                limit=settings.daily_hard_budget_usd,
            )
            raise BudgetHardLimitError(
                client_id=client_id,
                spent=state.accumulated_usd,
                limit=settings.daily_hard_budget_usd,
            )

        # Soft limit — downgrade to fallback model
        if state.accumulated_usd >= settings.daily_soft_budget_usd:
            if not state.soft_limit_hit:
                log.warning(
                    "budget_soft_limit_hit",
                    client_id=client_id,
                    spent=state.accumulated_usd,
                    limit=settings.daily_soft_budget_usd,
                )
                state.soft_limit_hit = True
            return settings.fallback_model

        return None

    def record_spend(self, client_id: str, cost_usd: float) -> None:
        """
        Record spend after a completed request.
        Called after the model returns — uses actual token counts, not estimates.
        """
        state = self._get_state(client_id)
        state.accumulated_usd += cost_usd

        log.info(
            "budget_spend_recorded",
            client_id=client_id,
            request_cost=round(cost_usd, 6),
            daily_total=round(state.accumulated_usd, 6),
            soft_limit=settings.daily_soft_budget_usd,
            hard_limit=settings.daily_hard_budget_usd,
        )

    def get_summary(self, client_id: str) -> dict:
        """Return current budget state — used by the /usage endpoint."""
        state = self._get_state(client_id)
        return {
            "client_id": client_id,
            "date": state.spend_date.isoformat(),
            "spent_usd": round(state.accumulated_usd, 6),
            "soft_limit_usd": settings.daily_soft_budget_usd,
            "hard_limit_usd": settings.daily_hard_budget_usd,
            "soft_limit_hit": state.soft_limit_hit,
            "remaining_usd": round(
                max(settings.daily_hard_budget_usd - state.accumulated_usd, 0.0), 6
            ),
        }


# Single shared instance
cost_tracker = CostTracker()
