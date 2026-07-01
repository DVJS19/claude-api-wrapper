"""
AdapterSelector — implements the Adapter pattern's selection logic.

cost-based routing only (estimate request cost, route to fallback
if it exceeds threshold).

will extend this with per-API-key budget enforcement (soft/hard
daily limits) — see select_adapter()'s docstring for where that plugs in.
"""

from dataclasses import dataclass

from app.adapters.base import ModelAdapter
from app.adapters.haiku_adapter import HaikuAdapter
from app.adapters.sonnet_adapter import SonnetAdapter
from app.config import settings
from app.observability.logger import get_logger

log = get_logger(__name__)


@dataclass
class SelectionResult:
    adapter: ModelAdapter
    reason: str  # human-readable reason, used in audit logs


class AdapterSelector:
    """
    Chooses between the primary and fallback model adapter.

    Singleton adapter instances — created once at selector construction,
    reused across every request. Avoids re-creating Anthropic clients
    per-request.
    """

    def __init__(self) -> None:
        self._primary = SonnetAdapter()
        self._fallback = HaikuAdapter()

    def select_adapter(
        self,
        caller_context: str,
        user_prompt: str,
        max_tokens: int,
    ) -> SelectionResult:
        """
        Cost-based routing:
            Estimate cost using the primary adapter's pricing.
            If estimated cost > settings.cost_route_threshold_usd,
            route to the fallback model instead.

        will add a budget check BEFORE this cost check:
            if per-key daily spend > hard budget: reject the request entirely
            if per-key daily spend > soft budget: force fallback regardless of cost estimate
        will add a third path: failure-triggered fallback, where
        the primary adapter is tried first regardless of this selection,
        and the resilience layer falls back to whatever this method did NOT pick.
        """
        estimated_cost = self._primary.estimate_cost(
            caller_context=caller_context,
            user_prompt=user_prompt,
            max_tokens=max_tokens,
        )

        if estimated_cost > settings.cost_route_threshold_usd:
            log.info(
                "adapter_selected_fallback_cost_routing",
                estimated_cost_usd=estimated_cost,
                threshold_usd=settings.cost_route_threshold_usd,
            )
            return SelectionResult(
                adapter=self._fallback,
                reason=(
                    f"estimated cost ${estimated_cost:.6f} exceeds "
                    f"threshold ${settings.cost_route_threshold_usd:.2f} — "
                    f"routed to fallback model"
                ),
            )

        log.info(
            "adapter_selected_primary",
            estimated_cost_usd=estimated_cost,
        )
        return SelectionResult(
            adapter=self._primary,
            reason=f"estimated cost ${estimated_cost:.6f} within threshold — primary model",
        )


# Single shared instance — imported by the route handler
selector = AdapterSelector()
