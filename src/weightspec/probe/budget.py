"""The forward-pass budget: the scarce resource of weight-space programming.

One *request* (a generation batch or a teacher-forced batch) costs one unit, as in
Takehi et al. (2026) and Hu et al. (2026). Only successful requests are charged.
Every charge is recorded, so a run's results can state exactly what access it had.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = ["Budget", "BudgetExceeded"]


class BudgetExceeded(RuntimeError):
    pass


@dataclass
class Budget:
    forward: int | None = 50  # None = unlimited
    used: int = 0
    ledger: list[dict[str, Any]] = field(default_factory=list)

    @property
    def remaining(self) -> int | None:
        return None if self.forward is None else self.forward - self.used

    def check(self, cost: int = 1) -> None:
        if self.forward is not None and self.used + cost > self.forward:
            raise BudgetExceeded(f"forward budget exhausted ({self.used}/{self.forward} used)")

    def charge(self, kind: str, cost: int = 1, **info: Any) -> None:
        self.check(cost)
        self.used += cost
        self.ledger.append({"n": self.used, "kind": kind, "cost": cost, **info})

    def status(self) -> dict[str, Any]:
        return {"used": self.used, "limit": self.forward, "remaining": self.remaining}

    def to_dict(self) -> dict[str, Any]:
        return {"forward": self.forward, "used": self.used, "ledger": self.ledger}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Budget:
        return cls(forward=d.get("forward"), used=d.get("used", 0), ledger=list(d.get("ledger", [])))
