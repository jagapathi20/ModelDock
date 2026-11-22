"""Retry delays for loops that must survive a Redis outage (worker loop, API reply listener)."""

import random
from dataclasses import dataclass, field


@dataclass
class Backoff:
    """Capped exponential backoff with full jitter (C1.19).

    Full jitter (a uniform random wait between 0 and the cap for this attempt) spreads retries
    out, so many processes that lost Redis at the same moment don't all reconnect at the same
    moment (a thundering herd).
    """

    base_s: float = 0.05
    cap_s: float = 1.0
    attempt: int = 0
    rng: random.Random = field(default_factory=random.Random)

    def next_delay(self) -> float:
        delay = self.rng.uniform(0, min(self.cap_s, self.base_s * 2**self.attempt))
        self.attempt += 1
        return delay

    def reset(self) -> None:
        self.attempt = 0
