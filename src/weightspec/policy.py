"""Edit policy: rules enforced by the library, not by a prompt (RFC-0001 §7).

A patch that violates the active policy is refused at apply time with a
:class:`PolicyViolation` explaining why. ``Policy.paper()`` reproduces the rules
of the published weight-space-programming experiments.
"""

from __future__ import annotations

import contextlib
import fnmatch

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from weightspec.selectors import Target
from weightspec.store import WeightStore

__all__ = ["Policy", "PolicyViolation"]


class PolicyViolation(ValueError):
    pass


class Policy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    frozen: list[str] = Field(default_factory=list, description="glob patterns of tensors that may not change")
    frozen_roles: list[str] = Field(
        default_factory=list, description="logical tensors that may not change: embed, unembed, final_norm"
    )
    allow_only: list[str] | None = Field(default=None, description="if set, only matching tensors may change")
    max_changed: int | None = Field(default=None, description="max stored values changed by the whole patch")
    max_changed_fraction: float | None = Field(default=None, description="max fraction of any one tensor changed")
    forbid_whole_tensor_scale: bool = Field(default=False, description="refuse scale ops on an entire tensor")
    require_finite: bool = True

    @classmethod
    def permissive(cls) -> Policy:
        return cls()

    @classmethod
    def paper(cls) -> Policy:
        """Rules of Takehi et al. / Hu et al.: embeddings, LM head and final norm are frozen,
        and an edit that only rescales a whole tensor is refused."""
        return cls(frozen_roles=["embed", "unembed", "final_norm"], forbid_whole_tensor_scale=True)

    def _frozen_names(self, store: WeightStore) -> set[str]:
        out = set()
        for role in self.frozen_roles:
            with contextlib.suppress(KeyError):
                out.add(store.arch.global_tensor(role))
        return out

    def check_op(self, store: WeightStore, op_name: str, target: Target) -> None:
        name = target.tensor
        if name in self._frozen_names(store) or any(fnmatch.fnmatch(name, p) for p in self.frozen):
            raise PolicyViolation(f"{target.selector!r} edits frozen tensor {name!r}")
        if self.allow_only is not None and not any(fnmatch.fnmatch(name, p) for p in self.allow_only):
            raise PolicyViolation(f"{name!r} is outside allow_only {self.allow_only}")
        if self.forbid_whole_tensor_scale and op_name == "scale" and target.is_whole:
            raise PolicyViolation(f"{target.selector!r}: scaling a whole tensor by a constant is not allowed")

    def check_result(self, store: WeightStore, working: dict[str, np.ndarray], changed: dict[str, int]) -> None:
        if self.require_finite:
            for name, w in working.items():
                if not np.isfinite(w).all():
                    raise PolicyViolation(f"edit makes {name!r} non-finite")
        total = sum(changed.values())
        if self.max_changed is not None and total > self.max_changed:
            raise PolicyViolation(f"patch changes {total:,} values; policy allows {self.max_changed:,}")
        if self.max_changed_fraction is not None:
            for name, n in changed.items():
                frac = n / max(1, store.numel(name))
                if frac > self.max_changed_fraction:
                    raise PolicyViolation(f"{name!r}: {frac:.2%} changed; policy allows {self.max_changed_fraction:.2%}")
