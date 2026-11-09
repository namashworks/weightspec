"""Forward-pass probes: run the base and edited model on the same inputs, under a budget.

    probe = ws.Probe.from_pretrained("Qwen/Qwen3-0.6B", budget=50, device="cuda")
    probe.set_patch(patch)
    res = probe.generate(["What is 17*23?"], weights="both", n=4, max_new_tokens=1024)
    tf = probe.teacher_forced(["The answer is 391."], weights="edited", mlp_layers=[23])

Two request kinds, each costing one unit of budget (Takehi et al.; Hu et al.):

* **generation**: sample answers (optionally from both arms with shared seeds);
* **teacher-forced**: read given texts and return per-token log-probs, top-k
  predictions, hidden states and MLP activations.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from weightspec.patch import Applied, Patch
from weightspec.policy import Policy
from weightspec.probe.budget import Budget, BudgetExceeded
from weightspec.selectors import SelectorError
from weightspec.store import WeightStore

__all__ = ["Budget", "BudgetExceeded", "GenerateResult", "Probe"]

Which = Literal["original", "edited", "both"]


@dataclass
class GenerateResult:
    """``arms[arm][item][sample] -> {"text", "tokens", "truncated"}``."""

    arms: dict[str, list[list[dict[str, Any]]]]
    items: list[Any] = field(default_factory=list)

    def texts(self, arm: str = "original") -> list[list[str]]:
        return [[s["text"] for s in item] for item in self.arms[arm]]

    def summary(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for arm, items in self.arms.items():
            samples = [s for item in items for s in item]
            out[arm] = {
                "samples": len(samples),
                "mean_tokens": float(np.mean([s["tokens"] for s in samples])) if samples else 0.0,
                "truncated_share": float(np.mean([s["truncated"] for s in samples])) if samples else 0.0,
            }
        return out


class Probe:
    """A model loaded once, the current patch, and a forward-pass budget."""

    def __init__(self, backend: Any, budget: Budget | int | None = 50, store: WeightStore | None = None):
        self.backend = backend
        self.budget = budget if isinstance(budget, Budget) else Budget(forward=budget)
        self.store = store or WeightStore.from_torch(backend.model, copy=False)
        self.patch: Patch | None = None
        self.applied: Applied | None = None

    @classmethod
    def from_pretrained(
        cls, model_id: str, budget: Budget | int | None = 50, device: str = "cpu", dtype: str = "auto",
        revision: str | None = None, **kwargs: Any,
    ) -> Probe:
        from weightspec.probe.transformers_backend import TransformersBackend

        return cls(TransformersBackend.from_pretrained(model_id, revision, device, dtype, **kwargs), budget)

    @classmethod
    def from_model(cls, model: Any, tokenizer: Any, budget: Budget | int | None = 50) -> Probe:
        from weightspec.probe.transformers_backend import TransformersBackend

        return cls(TransformersBackend(model, tokenizer), budget)

    # ------------------------------------------------------------------ patch
    def set_patch(self, patch: Patch | None, policy: Policy | None = None) -> Applied | None:
        """Use ``patch`` for 'edited' requests (``None`` clears it). Costs no budget."""
        if patch is None:
            self.patch = self.applied = None
            self.backend.clear_edits()
            return None
        try:
            applied = patch.apply(self.store, policy)
        except (SelectorError, KeyError):
            # Raw tensor names from another checkpoint layout: apply on the patch's own base,
            # then let the backend map names onto this module.
            applied = patch.apply(None, policy)
        self.backend.set_edits({n: (r, applied.base.dtype(n)) for n, r in applied.tensors.items()})
        self.patch, self.applied = patch, applied
        return applied

    # ------------------------------------------------------------------ requests
    def generate(
        self,
        items: Sequence[str | list[dict[str, str]]],
        weights: Which = "original",
        n: int = 1,
        max_new_tokens: int = 512,
        temperature: float = 0.6,
        top_p: float = 0.95,
        top_k: int = 20,
        seed: int = 0,
        chat: bool = True,
        enable_thinking: bool | None = None,
    ) -> GenerateResult:
        """Sample answers. ``weights='both'`` runs both arms on the same items and seeds (one budget unit)."""
        self.budget.check()
        arms = ["original", "edited"] if weights == "both" else [weights]
        kw = dict(n=n, max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p, top_k=top_k,
                  seed=seed, chat=chat, enable_thinking=enable_thinking)
        res = GenerateResult({arm: self.backend.generate(items, arm, **kw) for arm in arms}, list(items))
        self.budget.charge("generate", weights=weights, items=len(items), n=n)
        return res

    def teacher_forced(
        self,
        texts: Sequence[str | list[dict[str, str]]],
        weights: Literal["original", "edited"] = "original",
        hidden_layers: Sequence[int] = (),
        mlp_layers: Sequence[int] = (),
        topk: int = 0,
        chat: bool = False,
    ) -> list[dict[str, Any]]:
        """Read texts and return per-token log-probs (+ optional top-k, hidden states, MLP activations)."""
        self.budget.check()
        names = {layer: self.store.arch.tensor(layer, "mlp.down") for layer in mlp_layers}
        out = self.backend.teacher_forced(texts, weights, hidden_layers, mlp_layers, topk, chat, names)
        self.budget.charge("teacher_forced", weights=weights, texts=len(texts))
        return out

    def decode(self, ids: Sequence[int]) -> str:
        return str(self.backend.tokenizer.decode(list(ids)))
