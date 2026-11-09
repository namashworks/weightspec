"""Direction algebra: the vocabulary weight-space programs are written in.

Published edits build their vectors from a small set of recipes: the readout
direction of a few tokens, a contrast of two means, a projection that protects
other directions, a unit normalization, a random control of equal norm. Here each
recipe is a typed, serializable expression. A patch can therefore store *how* a
vector was made (a few hundred bytes, readable, auditable) instead of 1,024 opaque
floats, and anyone can rebuild it from the base weights and check the hash.

Use the functions at the bottom (:func:`readout`, :func:`embedding`,
:func:`weights`, :func:`vector`) and ordinary arithmetic::

    doubt = ws.readout(store, ["Wait", "However", "But", "Maybe", "Perhaps"])
    progress = ws.readout(store, ["Therefore", "Thus", "So", "Now"])
    u = (doubt - progress).normalize().project_off([ws.readout(store, list("0123456789"), reduce="none")])
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Annotated, Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from weightspec.store import WeightStore

__all__ = [
    "Direction",
    "DirectionExpr",
    "EvalContext",
    "embedding",
    "readout",
    "vector",
    "weights",
]


class _Node(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Tokens(_Node):
    """Token rows of the embedding (``fn='embedding'``) or of the effective readout (``fn='readout'``).

    The readout of token t is its unembedding row times the effective final-norm gain,
    ``r(t) = U_t ⊙ (offset + γ)``, where ``offset`` is 1 for one-centered RMSNorm
    families (Gemma, Qwen3.5) and 0 otherwise. ``ids`` is authoritative; ``tokens``
    records the human-readable words they came from.
    """

    fn: Literal["readout", "embedding"]
    ids: list[int]
    tokens: list[str] | None = None
    reduce: Literal["mean", "sum", "none"] = "mean"


class Weights(_Node):
    """A vector read from the weights: a single row/column, or the mean of a block's rows/columns."""

    fn: Literal["weights"] = "weights"
    target: str
    reduce: Literal["none", "mean_cols", "mean_rows"] = "none"
    original: bool = True


class Payload(_Node):
    """An explicit vector stored in the patch's ``tensors.safetensors``."""

    fn: Literal["payload"] = "payload"
    key: str


class Sum(_Node):
    """Weighted sum ``Σ cᵢ·xᵢ`` (subtraction is a coefficient of -1)."""

    fn: Literal["sum"] = "sum"
    terms: list[DirectionExpr]
    coefs: list[float]


class Scaled(_Node):
    fn: Literal["scale"] = "scale"
    of: DirectionExpr
    by: float


class Normalize(_Node):
    fn: Literal["normalize"] = "normalize"
    of: DirectionExpr


class ProjectOff(_Node):
    """Remove the components along span(``span``): ``x - B Bᵀ x`` with B an orthonormal basis (QR)."""

    fn: Literal["project_off"] = "project_off"
    of: DirectionExpr
    span: list[DirectionExpr]


class RandomOrthogonal(_Node):
    """A seeded Gaussian direction, projected off ``like`` (and ``also_off``), with ``like``'s norm.

    This is the 'equal-norm random direction' control used to show an edit is specific.
    """

    fn: Literal["random_orthogonal"] = "random_orthogonal"
    like: DirectionExpr
    seed: int
    also_off: list[DirectionExpr] = Field(default_factory=list)


DirectionExpr = Annotated[
    Tokens | Weights | Payload | Sum | Scaled | Normalize | ProjectOff | RandomOrthogonal,
    Field(discriminator="fn"),
]
for _m in (Sum, Scaled, Normalize, ProjectOff, RandomOrthogonal):
    _m.model_rebuild()


class EvalContext:
    """Evaluates expressions against a store; memoizes results."""

    def __init__(self, store: WeightStore, payloads: dict[str, np.ndarray] | None = None):
        self.store = store
        self.payloads = payloads or {}
        self._memo: dict[str, np.ndarray] = {}

    def eval(self, expr: Any) -> np.ndarray:
        key = expr.model_dump_json()
        if key not in self._memo:
            v = self._eval(expr)
            v.setflags(write=False)
            self._memo[key] = v
        return self._memo[key]

    def _eval(self, e: Any) -> np.ndarray:
        s = self.store
        if isinstance(e, Tokens):
            arch = s.arch
            name = arch.global_tensor("embed" if e.fn == "embedding" else "unembed")
            rows = s.rows(name, e.ids).astype(np.float64)
            if e.fn == "readout":
                if arch.final_norm is None:
                    raise ValueError(f"{arch.name!r} has no final norm; use fn='embedding'")
                gain = s.load_original(arch.final_norm).astype(np.float64) + arch.norm_offset
                rows = rows * gain
            if e.reduce == "mean":
                return rows.mean(axis=0)
            if e.reduce == "sum":
                return rows.sum(axis=0)
            return rows  # 2-D: a set of directions (only valid inside project_off spans)
        if isinstance(e, Weights):
            from weightspec.selectors import resolve

            t = resolve(s, e.target)
            w = s.load_original(t.tensor) if e.original else s.load(t.tensor)
            block = t.get(w).astype(np.float64)
            if e.reduce == "mean_cols":
                return block.mean(axis=1)
            if e.reduce == "mean_rows":
                return block.mean(axis=0)
            if 1 not in block.shape:
                raise ValueError(f"{e.target!r} is a {block.shape} block; pick reduce='mean_cols' or 'mean_rows'")
            return block.reshape(-1)
        if isinstance(e, Payload):
            if e.key not in self.payloads:
                raise KeyError(f"payload {e.key!r} missing from patch")
            return np.asarray(self.payloads[e.key], dtype=np.float64).reshape(-1)
        if isinstance(e, Sum):
            if len(e.terms) != len(e.coefs):
                raise ValueError("sum: terms and coefs differ in length")
            vs = [self._vec(t) for t in e.terms]
            return sum(c * v for c, v in zip(e.coefs, vs, strict=True))  # type: ignore[return-value]
        if isinstance(e, Scaled):
            return e.by * self._vec(e.of)
        if isinstance(e, Normalize):
            v = self._vec(e.of)
            n = float(np.linalg.norm(v))
            if n == 0.0:
                raise ValueError("cannot normalize a zero vector")
            return v / n
        if isinstance(e, ProjectOff):
            return _project_off(self._vec(e.of), [self.eval(x) for x in e.span])
        if isinstance(e, RandomOrthogonal):
            like = self._vec(e.like)
            rng = np.random.default_rng(e.seed)
            noise = np.asarray(rng.standard_normal(like.shape[0]), dtype=np.float64)
            noise = _project_off(noise, [like, *(self.eval(x) for x in e.also_off)])
            ratio = float(np.linalg.norm(like) / np.linalg.norm(noise))
            return noise * ratio
        raise TypeError(f"unknown direction node {type(e).__name__}")

    def _vec(self, e: Any) -> np.ndarray:
        v = self.eval(e)
        if v.ndim != 1:
            raise ValueError(f"{e.fn!r} with reduce='none' gives a set of vectors; use it as a project_off span")
        return v


def _project_off(x: np.ndarray, span: Sequence[np.ndarray]) -> np.ndarray:
    cols = [s.reshape(-1, x.shape[0]) for s in span]
    if not cols:
        return x
    a = np.concatenate(cols, axis=0).T  # (dim, k)
    q, r = np.linalg.qr(a)
    keep = np.abs(np.diag(r)) > 1e-10 * max(1.0, float(np.abs(r).max()))
    q = q[:, keep]
    return x - q @ (q.T @ x)


class Direction:
    """A vector plus the recipe that made it. Supports ``+ - *`` and chaining.

    Evaluated lazily (and cached) against the store it was created from; call
    :meth:`numpy` to see the values.
    """

    __array_priority__ = 1000

    def __init__(self, expr: Any, store: WeightStore | None = None, payloads: dict[str, np.ndarray] | None = None):
        self.expr = expr
        self.store = store
        self.payloads = dict(payloads or {})

    # arithmetic --------------------------------------------------------------
    def _join(self, other: Direction) -> tuple[WeightStore | None, dict[str, np.ndarray]]:
        if self.store is not None and other.store is not None and self.store is not other.store:
            raise ValueError("cannot combine directions built from different stores")
        return self.store or other.store, {**self.payloads, **other.payloads}

    def __add__(self, other: Direction) -> Direction:
        st, pl = self._join(other)
        return Direction(Sum(terms=[self.expr, other.expr], coefs=[1.0, 1.0]), st, pl)

    def __sub__(self, other: Direction) -> Direction:
        st, pl = self._join(other)
        return Direction(Sum(terms=[self.expr, other.expr], coefs=[1.0, -1.0]), st, pl)

    def __mul__(self, k: float) -> Direction:
        return Direction(Scaled(of=self.expr, by=float(k)), self.store, self.payloads)

    __rmul__ = __mul__

    def __neg__(self) -> Direction:
        return self * -1.0

    def __truediv__(self, k: float) -> Direction:
        return self * (1.0 / float(k))

    def normalize(self) -> Direction:
        return Direction(Normalize(of=self.expr), self.store, self.payloads)

    def project_off(self, span: Sequence[Direction]) -> Direction:
        pl = dict(self.payloads)
        for d in span:
            pl.update(d.payloads)
        return Direction(ProjectOff(of=self.expr, span=[d.expr for d in span]), self.store, pl)

    def random_orthogonal(self, seed: int, also_off: Sequence[Direction] = ()) -> Direction:
        """Equal-norm random control direction orthogonal to this one."""
        return Direction(
            RandomOrthogonal(like=self.expr, seed=seed, also_off=[d.expr for d in also_off]),
            self.store, self.payloads,
        )

    # evaluation --------------------------------------------------------------
    def numpy(self, store: WeightStore | None = None) -> np.ndarray:
        st = store or self.store
        if st is None:
            raise ValueError("this direction is not bound to a store; pass one")
        return EvalContext(st, self.payloads).eval(self.expr).astype(np.float32)

    @property
    def norm(self) -> float:
        return float(np.linalg.norm(self.numpy()))

    def to_json(self) -> str:
        return self.expr.model_dump_json(exclude_none=True)

    def __repr__(self) -> str:
        return f"Direction({self.to_json()[:120]})"


# ---------------------------------------------------------------- constructors
def readout(
    store: WeightStore, tokens: Sequence[str | int], reduce: Literal["mean", "sum", "none"] = "mean"
) -> Direction:
    """Mean effective readout direction of ``tokens`` (first sub-token of each word, no leading space)."""
    ids = store.token_ids(tokens)
    words = [t if isinstance(t, str) else str(t) for t in tokens]
    return Direction(Tokens(fn="readout", ids=ids, tokens=words, reduce=reduce), store)


def embedding(
    store: WeightStore, tokens: Sequence[str | int], reduce: Literal["mean", "sum", "none"] = "mean"
) -> Direction:
    ids = store.token_ids(tokens)
    words = [t if isinstance(t, str) else str(t) for t in tokens]
    return Direction(Tokens(fn="embedding", ids=ids, tokens=words, reduce=reduce), store)


def weights(
    store: WeightStore, selector: str, reduce: Literal["none", "mean_cols", "mean_rows"] = "none"
) -> Direction:
    """A vector taken from the (original) weights, e.g. ``weights(s, 'layer[23].mlp.down.col[507]')``."""
    return Direction(Weights(target=selector, reduce=reduce), store)


def vector(values: Any, key: str | None = None, store: WeightStore | None = None) -> Direction:
    """An explicit vector (e.g. from activations you measured). Saved as a payload in the patch."""
    import hashlib

    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    k = key or "v_" + hashlib.sha256(arr.tobytes()).hexdigest()[:12]
    return Direction(Payload(key=k), store, {k: arr})
