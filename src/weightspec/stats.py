"""Paired evaluation statistics for weight edits (RFC-0001 §8).

Compares the base and edited model on the *same* questions, clustering samples by
question as in Hu et al. (2026), Eq. 2::

    d_q  = mean_s(y'_{q,s}) - mean_s(y_{q,s})
    Δ    = mean_q d_q
    SE   = sqrt( Σ_q (d_q - Δ)² / (Q (Q - 1)) )
    95% CI = Δ ± 1.96·SE

A verdict is ``confirmed`` only when the whole interval is above zero.
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

__all__ = ["PairedResult", "load_scores", "paired"]

Verdict = Literal["confirmed", "regressed", "not_significant", "insufficient_data"]

_ID_KEYS = ("id", "question_id", "problem_id", "qid", "idx", "doc_id", "task_id", "unique_id")
_SCORE_KEYS = ("correct", "score", "is_correct", "acc", "accuracy", "exact_match", "pass", "passed", "reward")


@dataclass(frozen=True)
class PairedResult:
    n_questions: int
    n_base_samples: int
    n_edited_samples: int
    base_acc: float
    edited_acc: float
    delta: float
    se: float
    ci_low: float
    ci_high: float
    rescues: int
    regressions: int
    verdict: Verdict
    dropped_base_only: int = 0
    dropped_edited_only: int = 0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def __str__(self) -> str:
        def pp(x: float) -> str:
            return f"{100 * x:+.2f}"

        lines = [
            f"Paired comparison on {self.n_questions} shared questions "
            f"({self.n_base_samples} base / {self.n_edited_samples} edited samples)",
            f"  base    {100 * self.base_acc:6.2f}%",
            f"  edited  {100 * self.edited_acc:6.2f}%",
            f"  delta   {pp(self.delta)} pp   SE {100 * self.se:.2f} pp   95% CI [{pp(self.ci_low)}, {pp(self.ci_high)}] pp",
            f"  sample pairs: {self.rescues} wrong->right, {self.regressions} right->wrong",
            f"  verdict {self.verdict.upper()}",
        ]
        if self.dropped_base_only or self.dropped_edited_only:
            lines.append(
                f"  note    {self.dropped_base_only} base-only and {self.dropped_edited_only} edited-only "
                "questions were excluded (unpaired)"
            )
        return "\n".join(lines)


def _mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs)


def paired(
    base: Mapping[str, Sequence[float]],
    edited: Mapping[str, Sequence[float]],
    z: float = 1.96,
) -> PairedResult:
    """Question-clustered paired comparison of per-sample scores in [0, 1]."""
    shared = [q for q in base if q in edited and len(base[q]) and len(edited[q])]
    q_n = len(shared)
    nb = sum(len(base[q]) for q in shared)
    ne = sum(len(edited[q]) for q in shared)
    if q_n < 2:
        nan = float("nan")
        return PairedResult(q_n, nb, ne, nan, nan, nan, nan, nan, nan, 0, 0, "insufficient_data",
                            len(set(base) - set(edited)), len(set(edited) - set(base)))
    d = [_mean(edited[q]) - _mean(base[q]) for q in shared]
    delta = _mean(d)
    se = math.sqrt(sum((x - delta) ** 2 for x in d) / (q_n * (q_n - 1)))
    lo, hi = delta - z * se, delta + z * se
    rescues = regressions = 0
    for q in shared:
        for yb, ye in zip(base[q], edited[q], strict=False):
            rescues += int(yb < 0.5 <= ye)
            regressions += int(ye < 0.5 <= yb)
    verdict: Verdict = "confirmed" if lo > 0 else "regressed" if hi < 0 else "not_significant"
    return PairedResult(
        n_questions=q_n, n_base_samples=nb, n_edited_samples=ne,
        base_acc=_mean([_mean(base[q]) for q in shared]),
        edited_acc=_mean([_mean(edited[q]) for q in shared]),
        delta=delta, se=se, ci_low=lo, ci_high=hi, rescues=rescues, regressions=regressions, verdict=verdict,
        dropped_base_only=len(set(base) - set(edited)), dropped_edited_only=len(set(edited) - set(base)),
    )


def _score(v: Any) -> list[float]:
    if isinstance(v, bool):
        return [float(v)]
    if isinstance(v, (int, float)):
        return [float(v)]
    if isinstance(v, list):
        return [x for item in v for x in _score(item)]
    raise ValueError(f"cannot read a score from {v!r}")


def _pick(rec: Mapping[str, Any], keys: Sequence[str], explicit: str | None, what: str) -> Any:
    if explicit:
        return rec[explicit]
    for k in keys:
        if k in rec:
            return rec[k]
    raise KeyError(f"no {what} field in record with keys {sorted(rec)}; pass --{what}-key")


def load_scores(
    path: str | os.PathLike[str], id_key: str | None = None, score_key: str | None = None
) -> dict[str, list[float]]:
    """Read per-question scores from JSON or JSONL.

    Accepts JSONL records (one sample or one question per line), a JSON list of
    records, or a JSON object with a ``per_problem`` / ``results`` / ``samples`` list.
    Records for the same id are pooled as samples.
    """
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    recs: list[Any]
    doc: Any = None
    if p.suffix != ".jsonl":
        try:
            doc = json.loads(text)
        except json.JSONDecodeError:
            doc = None
    if doc is None:
        recs = [json.loads(line) for line in text.splitlines() if line.strip()]
    elif isinstance(doc, dict):
        lists = [k for k in ("per_problem", "per_question", "results", "samples", "records", "data")
                 if isinstance(doc.get(k), list)]
        if lists:
            recs = doc[lists[0]]
        elif all(isinstance(v, (int, float, bool, list)) for v in doc.values()):
            recs = [{"id": k, "score": v} for k, v in doc.items()]  # {question_id: score(s)}
        else:
            recs = [doc]
    else:
        recs = list(doc)
    out: dict[str, list[float]] = {}
    for r in recs:
        qid = str(_pick(r, _ID_KEYS, id_key, "id"))
        out.setdefault(qid, []).extend(_score(_pick(r, _SCORE_KEYS, score_key, "score")))
    return out
