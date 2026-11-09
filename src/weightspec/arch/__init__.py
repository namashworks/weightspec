"""Architecture maps: how logical selectors map onto a family's tensor names.

A selector such as ``layer[23].mlp.down.col[507]`` means the same thing for every
decoder-only transformer. The :class:`ArchSpec` of a checkpoint says which tensor
that is and how attention heads are laid out (head width, gated queries, ...).

Most Hugging Face decoders are handled by one generic map whose layer prefix is
discovered from the tensor names. Families with unusual layouts register a
factory with :func:`register_arch`.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from weightspec.store import WeightStore

__all__ = ["ArchSpec", "register_arch", "resolve_arch"]

# Logical component -> tensor-name suffix after the layer prefix (HF conventions).
HF_COMPONENTS: dict[str, str] = {
    "mlp.down": "mlp.down_proj.weight",
    "mlp.up": "mlp.up_proj.weight",
    "mlp.gate": "mlp.gate_proj.weight",
    "attn.q": "self_attn.q_proj.weight",
    "attn.k": "self_attn.k_proj.weight",
    "attn.v": "self_attn.v_proj.weight",
    "attn.out": "self_attn.o_proj.weight",
    "norm.input": "input_layernorm.weight",
    "norm.post_attn": "post_attention_layernorm.weight",
}

# Model types whose RMSNorm multiplies by (1 + weight) instead of weight.
ONE_CENTERED_NORM = re.compile(r"^(gemma.*|qwen3_next.*|qwen3_5.*)$")
# Model types whose q_proj interleaves a per-head output gate: rows [q_h (d), gate_h (d)] per head.
GATED_QUERY = re.compile(r"^(qwen3_next.*|qwen3_5.*)$")


@dataclass(frozen=True)
class ArchSpec:
    name: str
    layer_prefix: str | None = None  # e.g. "model.language_model.layers.{i}."
    num_layers: int | None = None
    components: dict[str, str] = field(default_factory=lambda: dict(HF_COMPONENTS))
    embed: str | None = None
    unembed: str | None = None  # None -> tied to embed
    final_norm: str | None = None
    norm_offset: float = 0.0  # effective gain = norm_offset + stored weight
    head_dim: int | None = None
    num_heads: int | None = None
    num_kv_heads: int | None = None
    gated_query: bool = False
    layer_types: tuple[str, ...] | None = None

    def tensor(self, layer: int, component: str) -> str:
        if self.layer_prefix is None:
            raise ValueError(f"architecture {self.name!r} has no layer map; use tensor['...'] selectors")
        if self.num_layers is not None and not 0 <= layer < self.num_layers:
            raise IndexError(f"layer {layer} out of range (model has {self.num_layers})")
        if component not in self.components:
            raise KeyError(f"unknown component {component!r}; known: {sorted(self.components)}")
        return self.layer_prefix.format(i=layer) + self.components[component]

    def global_tensor(self, which: str) -> str:
        name = {"embed": self.embed, "unembed": self.unembed or self.embed, "final_norm": self.final_norm}.get(which)
        if name is None:
            raise KeyError(f"architecture {self.name!r} defines no {which!r} tensor")
        return name


_FACTORIES: dict[str, Callable[[WeightStore], ArchSpec]] = {}


def register_arch(model_type: str) -> Callable[[Callable[[WeightStore], ArchSpec]], Callable[[WeightStore], ArchSpec]]:
    """Register a factory building an :class:`ArchSpec` for ``config.model_type``."""

    def deco(fn: Callable[[WeightStore], ArchSpec]) -> Callable[[WeightStore], ArchSpec]:
        _FACTORIES[model_type] = fn
        return fn

    return deco


def _find(names: list[str], pattern: str) -> str | None:
    rx = re.compile(pattern)
    hits = sorted((n for n in names if rx.search(n)), key=len)
    return hits[0] if hits else None


def generic_hf(store: WeightStore) -> ArchSpec:
    names = store.names
    cfg: dict[str, Any] = store.text_config
    model_type = str(cfg.get("model_type") or store.config.get("model_type") or "unknown")
    # Discover the decoder layer prefix from the shortest name ending in layers.0.<something>,
    # ignoring vision towers and multi-token-prediction heads.
    cands = [
        n for n in names
        if re.search(r"(^|\.)layers\.0\.", n) and not re.match(r"^(mtp|model\.visual|visual|vision)", n)
    ]
    prefix = None
    if cands:
        best = min(cands, key=lambda n: n.index("layers.0."))
        prefix = best[: best.index("layers.0.")] + "layers.{i}."
    layer_ids = set()
    if prefix:
        rx = re.compile("^" + re.escape(prefix).replace(r"\{i\}", r"(\d+)"))
        for n in names:
            m = rx.match(n)
            if m:
                layer_ids.add(int(m.group(1)))
    root = prefix[: -len("layers.{i}.")] if prefix else ""
    embed = _find(names, r"(^|\.)embed_tokens\.weight$") or _find(names, r"(^|\.)wte\.weight$")
    if embed and root and not embed.startswith(root):
        embed = _find([n for n in names if n.startswith(root)], r"embed_tokens\.weight$") or embed
    unembed = _find(names, r"^lm_head\.weight$")
    final_norm = (root + "norm.weight") if (root + "norm.weight") in store else _find(names, r"(^|\.)norm\.weight$")
    hidden = cfg.get("hidden_size")
    heads = cfg.get("num_attention_heads")
    head_dim = cfg.get("head_dim") or (hidden // heads if hidden and heads else None)
    lt = cfg.get("layer_types")
    return ArchSpec(
        name=model_type,
        layer_prefix=prefix,
        num_layers=(max(layer_ids) + 1) if layer_ids else cfg.get("num_hidden_layers"),
        embed=embed,
        unembed=unembed if unembed and unembed != embed else None,
        final_norm=final_norm,
        norm_offset=1.0 if ONE_CENTERED_NORM.match(model_type) else 0.0,
        head_dim=head_dim,
        num_heads=heads,
        num_kv_heads=cfg.get("num_key_value_heads") or heads,
        gated_query=bool(cfg.get("attn_output_gate")) or bool(GATED_QUERY.match(model_type)),
        layer_types=tuple(lt) if isinstance(lt, list) else None,
    )


PLAIN = ArchSpec(name="plain")


def resolve_arch(arg: str | ArchSpec | None, store: WeightStore) -> ArchSpec:
    if isinstance(arg, ArchSpec):
        return arg
    if arg in (None, "plain"):
        return PLAIN
    model_type = arg if arg != "auto" else str(store.text_config.get("model_type") or store.config.get("model_type") or "")
    if model_type in _FACTORIES:
        return _FACTORIES[model_type](store)
    spec = generic_hf(store)
    if spec.layer_prefix is None and arg == "auto":
        return PLAIN
    return spec
