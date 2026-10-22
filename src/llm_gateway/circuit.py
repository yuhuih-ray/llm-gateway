import logging
import math
import time
from dataclasses import dataclass

from llm_gateway.errors import GatewayError

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Permit:
    generation: int
    probe: bool


class CircuitBreaker:
    def __init__(
        self, provider: str
    ) -> None:  # Shared provider name, never tenant data.
        self.provider = provider
        self.state = "closed"
        self.failures = 0
        self.open_until = 0.0
        self.probing = False
        self.generation = 0

    def transition(self, state: str) -> None:  # Log changes only.
        logger.warning(
            "Circuit transition provider=%s from=%s to=%s",
            self.provider,
            self.state,
            state,
        )
        self.state = state
        self.generation += 1

    def acquire(self) -> Permit:
        # No await between checking and claiming the single half-open probe.
        now = time.monotonic()
        if self.state == "open" and now >= self.open_until:
            self.transition("half-open")
        if self.state == "open" or self.probing:
            raise GatewayError(
                503,
                "Upstream temporarily unavailable",
                "upstream_unavailable",
                headers={"Retry-After": str(max(1, math.ceil(self.open_until - now)))},
                fallback_reason="open_circuit",
            )
        probe = self.state == "half-open"
        if probe:
            self.probing = True
        return Permit(self.generation, probe)

    def finish(self, permit: Permit, outcome: bool | None) -> None:
        # Ignore late completions from calls admitted before a state transition.
        if permit.generation != self.generation:
            return
        self.probing = False
        if outcome is None:
            # Cancellation, validation errors and slot waits are neutral. Busy is
            # not broken: a slot-wait 503 must never increment circuit failures.
            return
        if outcome:
            self.failures = 0
            if permit.probe:
                self.transition("closed")
        else:
            self.failures += 1
            if permit.probe or self.failures >= 5:
                self.open_until = time.monotonic() + 30
                self.transition("open")
