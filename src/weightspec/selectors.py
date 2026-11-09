"""Portable selectors: name a block of weights the same way across architectures.

Grammar (RFC-0001 §4), indices are zero-based and ranges half-open like Python::

    selector  := base ("." accessor)*
    base      := "layer[" INT "]." component
               | "embed" | "unembed" | "final_norm"
               | "tensor[" QUOTED_NAME "]"
    component := "mlp.down" | "mlp.up" | "mlp.gate"
               | "attn.q" | "attn.k" | "attn.v" | "attn.out"
               | "attn.head[" INT "]." ("out" | "q" | "gate" | "k" | "v")
               | "norm.input" | "norm.post_attn"
    accessor  := ("col" | "cols" | "row" | "rows") "[" index "]"
    index     := INT | INT ":" INT | INT ("," INT)*

Examples::

    layer[23].mlp.down.col[507]        # MLP unit 507's write direction into the residual stream
    layer[15].attn.head[7].out         # the 256 o_proj columns that belong to head 7
    layer[19].attn.head[11].gate       # gate rows of a gated-query head (Qwen3.5 / Qwen3-Next)
    tensor['lm_head.weight'].row[42]   # escape hatch: any tensor by its exact name

A selector resolves to a :class:`Target`: one tensor plus a row index and a column
index. Every block is treated as 2-D (1-D tensors have a single column).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from weightspec.store import WeightStore

__all__ = ["SelectorError", "Target", "parse_index", "resolve"]


class SelectorError(ValueError):
    pass


_BASE = re.compile(
    r"""^(?:
        layer\[(?P<layer>\d+)\]\.(?P<comp>
            mlp\.(?:down|up|gate)
          | attn\.(?:q|k|v|out)
          | attn\.head\[(?P<head>\d+)\]\.(?P<hpart>out|q|gate|k|v)
          | norm\.(?:input|post_attn)
        )
      | (?P<glob>embed|unembed|final_norm)
      | tensor\[(?P<q>['"])(?P<tname>.+?)(?P=q)\]
    )""",
    re.X,
)
_ACC = re.compile(r"^\.(?P<kind>cols?|rows?)\[(?P<idx>[^\]]*)\]")


def parse_index(text: str) -> slice | list[int]:
    t = text.replace(" ", "")
    if re.fullmatch(r"\d+:\d+", t):
        a, b = (int(x) for x in t.split(":"))
        if b <= a:
            raise SelectorError(f"empty range [{text}]")
        return slice(a, b)
    if re.fullmatch(r"\d+(,\d+)*", t):
        return [int(x) for x in t.split(",")]
    raise SelectorError(f"bad index [{text}]; use 5, 1,2,3 or 0:64")


@dataclass(frozen=True)
class Target:
    """A rectangular block of one tensor: ``W[rows][:, cols]`` (``None`` means all)."""

    selector: str
    tensor: str
    shape: tuple[int, ...]
    rows: np.ndarray | None = None
    cols: np.ndarray | None = None

    @property
    def is_whole(self) -> bool:
        return self.rows is None and self.cols is None

    @property
    def block_shape(self) -> tuple[int, int]:
        r = self.shape[0] if self.rows is None else len(self.rows)
        c = (self.shape[1] if len(self.shape) > 1 else 1) if self.cols is None else len(self.cols)
        return (r, c)

    def _ix(self) -> tuple[np.ndarray | slice, ...]:
        r = slice(None) if self.rows is None else self.rows
        if len(self.shape) == 1:
            return (r,)
        c = slice(None) if self.cols is None else self.cols
        if isinstance(r, np.ndarray) and isinstance(c, np.ndarray):
            return np.ix_(r, c)
        return (r, c)

    def get(self, w: np.ndarray) -> np.ndarray:
        block = w[self._ix()]
        return np.array(block.reshape(-1, 1) if w.ndim == 1 else block, dtype=np.float32)

    def set(self, w: np.ndarray, block: np.ndarray) -> None:
        if w.ndim == 1:
            w[self._ix()] = block.reshape(-1)
        else:
            w[self._ix()] = block

    def describe(self) -> str:
        def fmt(a: np.ndarray | None) -> str:
            if a is None:
                return ":"
            if len(a) > 1 and np.all(np.diff(a) == 1):
                return f"{a[0]}:{a[-1] + 1}"
            return ",".join(map(str, a[:8])) + (",..." if len(a) > 8 else "")

        return f"{self.tensor}[{fmt(self.rows)}, {fmt(self.cols)}] {self.block_shape}"


def _compose(existing: np.ndarray | None, size: int, idx: slice | list[int]) -> np.ndarray:
    base = np.arange(size) if existing is None else existing
    try:
        out = base[idx]
    except IndexError as e:
        raise SelectorError(f"index {idx} out of range for axis of length {len(base)}") from e
    if isinstance(idx, slice) and idx.stop is not None and idx.stop > len(base):
        raise SelectorError(f"range {idx.start}:{idx.stop} out of range for axis of length {len(base)}")
    return np.asarray(out, dtype=np.int64)


def resolve(store: WeightStore, selector: str) -> Target:
    """Resolve ``selector`` against ``store`` into a concrete :class:`Target`."""
    sel = selector.strip()
    m = _BASE.match(sel)
    if not m:
        raise SelectorError(f"cannot parse selector {selector!r}")
    arch = store.arch
    rows: np.ndarray | None = None
    cols: np.ndarray | None = None
    if m.group("tname"):
        tensor = m.group("tname")
    elif m.group("glob"):
        tensor = arch.global_tensor(m.group("glob"))
    else:
        layer = int(m.group("layer"))
        comp = m.group("comp")
        if m.group("head") is not None:
            h, part = int(m.group("head")), m.group("hpart")
            d = arch.head_dim
            if d is None:
                raise SelectorError(f"architecture {arch.name!r} has no head_dim; cannot select heads")
            if arch.layer_types and arch.layer_types[layer] != "full_attention":
                raise SelectorError(f"layer {layer} is {arch.layer_types[layer]!r}, not full attention")
            n_heads = arch.num_heads or 0
            n_kv = arch.num_kv_heads or n_heads
            limit = n_kv if part in ("k", "v") else n_heads
            if not 0 <= h < limit:
                raise SelectorError(f"head {h} out of range ({limit} heads for {part!r})")
            if part == "out":
                tensor = arch.tensor(layer, "attn.out")
                cols = np.arange(h * d, (h + 1) * d)
            elif part in ("q", "gate"):
                tensor = arch.tensor(layer, "attn.q")
                if arch.gated_query:
                    start = 2 * h * d + (d if part == "gate" else 0)
                    rows = np.arange(start, start + d)
                elif part == "gate":
                    raise SelectorError(f"architecture {arch.name!r} has no gated queries")
                else:
                    rows = np.arange(h * d, (h + 1) * d)
            else:
                tensor = arch.tensor(layer, f"attn.{part}")
                rows = np.arange(h * d, (h + 1) * d)
        else:
            tensor = arch.tensor(layer, comp)
    if tensor not in store:
        raise SelectorError(f"{selector!r} -> tensor {tensor!r}, which this checkpoint does not have")
    shape = store.shape(tensor)
    if len(shape) not in (1, 2):
        raise SelectorError(f"{tensor!r} has shape {shape}; only 1-D and 2-D tensors are addressable")
    rest = sel[m.end():]
    while rest:
        a = _ACC.match(rest)
        if not a:
            raise SelectorError(f"cannot parse {rest!r} in selector {selector!r}")
        idx = parse_index(a.group("idx"))
        if a.group("kind").startswith("col"):
            if len(shape) == 1:
                raise SelectorError(f"{tensor!r} is 1-D; use row[...]")
            cols = _compose(cols, shape[1], idx)
        else:
            rows = _compose(rows, shape[0], idx)
        rest = rest[a.end():]
    return Target(selector=sel, tensor=tensor, shape=shape, rows=rows, cols=cols)
