"""WeightSpec: the open standard and runtime for weight-space programming.

Describe a weight edit once as a small, typed, verifiable patch; apply it to any
compatible checkpoint; check it bit for bit; measure it with paired statistics.

    import weightspec as ws

    store = ws.load("Qwen/Qwen3.5-0.8B")                     # lazy: nothing big is downloaded
    u = (ws.readout(store, ["Wait", "However"]) - ws.readout(store, ["Therefore", "Thus"])).normalize()
    patch = ws.Patch.new(store, "doubt-column").add("layer[23].mlp.down.col[507]", u, alpha=-1.0)
    patch.build(); patch.save("doubt-column.wspec")
"""

from __future__ import annotations

from typing import Any

from weightspec._version import WSPEC_VERSION, __version__
from weightspec.agent import Session, Toolkit
from weightspec.arch import ArchSpec, register_arch
from weightspec.controls import controls
from weightspec.directions import Direction, embedding, readout, vector, weights
from weightspec.evals import EvalReport, evaluate, load_items
from weightspec.patch import Applied, Patch, VerifyReport, diff
from weightspec.policy import Policy, PolicyViolation
from weightspec.probe import Budget, BudgetExceeded, Probe
from weightspec.selectors import SelectorError, Target, resolve
from weightspec.stats import PairedResult, load_scores, paired
from weightspec.store import WeightStore, load

__all__ = [
    "WSPEC_VERSION",
    "Applied",
    "ArchSpec",
    "Budget",
    "BudgetExceeded",
    "Direction",
    "EvalReport",
    "PairedResult",
    "Patch",
    "Policy",
    "PolicyViolation",
    "Probe",
    "SelectorError",
    "Session",
    "Target",
    "Toolkit",
    "VerifyReport",
    "WeightStore",
    "__version__",
    "apply",
    "controls",
    "diff",
    "embedding",
    "evaluate",
    "load",
    "load_items",
    "load_scores",
    "paired",
    "readout",
    "register_arch",
    "resolve",
    "vector",
    "weights",
]


def apply(model: Any, patch: Patch | str, policy: Policy | None = None) -> Applied:
    """Apply a patch (object or path) to a ``torch.nn.Module`` in place, or to a store (returns the overlay).

    >>> ws.apply(model, "doubt-column.wspec")
    """
    p = Patch.load(patch) if isinstance(patch, str) else patch
    if hasattr(model, "state_dict") and not isinstance(model, WeightStore):
        from weightspec.integrations.torch import apply_to_module

        return apply_to_module(model, p, policy)
    return p.apply(model, policy)
