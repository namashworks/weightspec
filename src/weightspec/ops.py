"""Typed edit operations (RFC-0001 §6).

Every op acts on the block ``B`` selected by its ``target``, in float32, in place
on a working copy of the tensor. The set is deliberately small: it covers every
edit published in Takehi et al. (2026) and Hu et al. (2026).

=================  ==========================================  ===============================
op                 effect on block B (r × c)                   paper example
=================  ==========================================  ===============================
``add``            B += α·u (u added to every selected column  column 507 write: W[:,j] -= u
                   if len(u)==r, to every row if len(u)==c)
``scale``          B *= s                                      zero/halve a head, negate a column
``set``            B = value (scalar) or payload               restore / transplant values
``rank1``          B += α·u vᵀ                                 reader/writer rank-one edits
``project_scale``  B ← (I − k·û ûᵀ) B   (side='left')          IFEval head-15.7 edit, k=7
                   B ← B (I − k·û ûᵀ)   (side='right')
``delta``          B += α·D (payload matrix)                   any dense edit, e.g. from a diff
=================  ==========================================  ===============================
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from weightspec.directions import DirectionExpr, EvalContext
from weightspec.selectors import Target

if TYPE_CHECKING:
    pass

__all__ = ["Add", "Delta", "Op", "ProjectScale", "Rank1", "Scale", "Set"]


class _Op(BaseModel):
    model_config = ConfigDict(extra="forbid")
    target: str
    note: str | None = None

    def run(self, block: np.ndarray, ctx: EvalContext, target: Target) -> np.ndarray:
        raise NotImplementedError


def _f32(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=np.float32)


class Add(_Op):
    op: Literal["add"] = "add"
    u: DirectionExpr
    alpha: float = 1.0
    along: Literal["auto", "cols", "rows"] = "auto"

    def run(self, block: np.ndarray, ctx: EvalContext, target: Target) -> np.ndarray:
        u = _f32(ctx.eval(self.u))
        r, c = block.shape
        along: str = self.along
        if along == "auto":
            if r == c:
                raise ValueError(f"{self.target}: square block, set along='cols' or 'rows'")
            along = "cols" if len(u) == r else "rows" if len(u) == c else "?"
        if along == "cols" and len(u) == r:
            return block + np.float32(self.alpha) * u[:, None]
        if along == "rows" and len(u) == c:
            return block + np.float32(self.alpha) * u[None, :]
        raise ValueError(f"{self.target}: vector of length {len(u)} does not fit block {block.shape} ({along})")


class Scale(_Op):
    op: Literal["scale"] = "scale"
    factor: float

    def run(self, block: np.ndarray, ctx: EvalContext, target: Target) -> np.ndarray:
        return block * np.float32(self.factor)


class Set(_Op):
    op: Literal["set"] = "set"
    value: float | None = None
    payload: str | None = None

    def run(self, block: np.ndarray, ctx: EvalContext, target: Target) -> np.ndarray:
        if (self.value is None) == (self.payload is None):
            raise ValueError(f"{self.target}: set needs exactly one of value / payload")
        if self.value is not None:
            return np.full_like(block, np.float32(self.value))
        v = _f32(ctx.payloads[self.payload])  # type: ignore[index]
        return v.reshape(block.shape)


class Rank1(_Op):
    op: Literal["rank1"] = "rank1"
    u: DirectionExpr
    v: DirectionExpr
    alpha: float = 1.0

    def run(self, block: np.ndarray, ctx: EvalContext, target: Target) -> np.ndarray:
        u, v = _f32(ctx.eval(self.u)), _f32(ctx.eval(self.v))
        if (len(u), len(v)) != block.shape:
            raise ValueError(f"{self.target}: u⊗v is {(len(u), len(v))}, block is {block.shape}")
        return block + np.float32(self.alpha) * np.outer(u, v)


class ProjectScale(_Op):
    """Multiply the component of the block along unit direction û by (1 − k); keep the rest.

    k=1 removes the component, k=2 reflects it, k=7 multiplies it by −6.
    """

    op: Literal["project_scale"] = "project_scale"
    u: DirectionExpr
    k: float
    side: Literal["left", "right"] = "left"

    def run(self, block: np.ndarray, ctx: EvalContext, target: Target) -> np.ndarray:
        u = np.asarray(ctx.eval(self.u), dtype=np.float64)
        u = u / np.linalg.norm(u)
        b = block.astype(np.float64)
        if self.side == "left":
            if len(u) != b.shape[0]:
                raise ValueError(f"{self.target}: û has length {len(u)}, block has {b.shape[0]} rows")
            out = b - self.k * np.outer(u, u @ b)
        else:
            if len(u) != b.shape[1]:
                raise ValueError(f"{self.target}: û has length {len(u)}, block has {b.shape[1]} cols")
            out = b - self.k * np.outer(b @ u, u)
        return out.astype(np.float32)


class Delta(_Op):
    op: Literal["delta"] = "delta"
    payload: str
    alpha: float = 1.0

    def run(self, block: np.ndarray, ctx: EvalContext, target: Target) -> np.ndarray:
        d = _f32(ctx.payloads[self.payload]).reshape(block.shape)
        return block + np.float32(self.alpha) * d


Op = Annotated[Add | Scale | Set | Rank1 | ProjectScale | Delta, Field(discriminator="op")]
