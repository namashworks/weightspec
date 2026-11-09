"""Control patches: the evidence that an edit is specific, generated automatically.

Hu et al. (2026) accepted the 1,023-value MATH edit only after it beat a set of
matched controls on the same questions: the unit ablated, the direction reversed,
an equal-norm random direction, and a dose sweep. :func:`controls` builds those
variants for any patch whose ops write a direction (``add``, ``rank1``,
``project_scale``), so the comparison is one loop away::

    for name, ctrl in ws.controls(patch).items():
        ctrl.build(); ctrl.save(f"controls/{name}.json")
"""

from __future__ import annotations

from collections.abc import Sequence

from weightspec.directions import RandomOrthogonal
from weightspec.ops import Add, ProjectScale, Rank1, Set
from weightspec.patch import Patch, Provenance

__all__ = ["controls"]


def _variant(patch: Patch, suffix: str, note: str) -> Patch:
    p = Patch(
        name=f"{patch.name}--{suffix}",
        description=f"Control for {patch.name!r}: {note}",
        base=patch.base.model_copy(deep=True),
        numerics=patch.numerics.model_copy(),
        ops=[op.model_copy(deep=True) for op in patch.ops],
        provenance=Provenance(editor="program:weightspec.controls", method=f"control/{suffix}",
                              cites=list(patch.provenance.cites)),
    )
    p.base.tensors = {}
    p._payloads = dict(patch.payloads)
    p._store = patch._store
    return p


def controls(
    patch: Patch,
    doses: Sequence[float] = (0.5, 0.75, 1.5, 2.0),
    reverse: float = -0.5,
    seeds: Sequence[int] = (0, 1, 2),
    ablate: bool = True,
) -> dict[str, Patch]:
    """Return ``{name: control_patch}`` for a direction-writing patch.

    * ``dose-<k>``: every op's strength multiplied by k (alpha, or k for project_scale).
    * ``reverse``: strength multiplied by ``reverse`` (Hu et al. used -0.5).
    * ``random-<seed>``: each direction replaced by an equal-norm random direction orthogonal to it.
    * ``ablate``: each edited block set to zero instead (the 'native unit ablation' control).
    """
    kinds = (Add, Rank1, ProjectScale)
    if not patch.ops or not all(isinstance(op, kinds) for op in patch.ops):
        raise ValueError("controls() needs a patch whose ops are all add / rank1 / project_scale")
    out: dict[str, Patch] = {}

    def scaled(factor: float, suffix: str, note: str) -> Patch:
        p = _variant(patch, suffix, note)
        for op in p.ops:
            if isinstance(op, ProjectScale):
                op.k *= factor
            else:
                op.alpha *= factor  # type: ignore[union-attr]
        return p

    for d in doses:
        out[f"dose-{d:g}"] = scaled(d, f"dose-{d:g}", f"strength x{d:g}")
    if reverse:
        out["reverse"] = scaled(reverse, "reverse", f"strength x{reverse:g} (reversed direction)")
    for seed in seeds:
        p = _variant(patch, f"random-{seed}", f"equal-norm random direction orthogonal to the original (seed {seed})")
        for op in p.ops:
            op.u = RandomOrthogonal(like=op.u, seed=seed)  # type: ignore[union-attr]
        out[f"random-{seed}"] = p
    if ablate:
        p = _variant(patch, "ablate", "edited blocks set to zero")
        p.ops = [Set(target=op.target, value=0.0) for op in patch.ops]
        out["ablate"] = p
    return out
