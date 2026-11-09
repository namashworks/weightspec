"""Paired evaluation of a weight edit on held-out items, with optional controls.

    items = ws.load_items("math_test.jsonl")
    report = ws.evaluate(probe, patch, items, grader="math", n=4, max_new_tokens=4096,
                         controls=ws.controls(patch))
    print(report)

Both arms answer the same items with the same seeds in one request (one budget
unit). Each control reuses the base answers and costs one more request. Scores
are compared with question-clustered paired statistics (:mod:`weightspec.stats`).
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import TYPE_CHECKING, Any

from weightspec.stats import PairedResult, paired

if TYPE_CHECKING:
    from weightspec.patch import Patch
    from weightspec.probe import Probe

__all__ = ["GRADERS", "EvalReport", "evaluate", "extract_answer", "grade_math", "load_items"]

Item = dict[str, Any]
Grader = Callable[[str, Item], float]


# ------------------------------------------------------------------ answer extraction
def _last_boxed(text: str) -> str | None:
    i = max(text.rfind("\\boxed{"), text.rfind("\\fbox{"))
    if i < 0:
        return None
    j = text.index("{", i)
    depth = 0
    for k in range(j, len(text)):
        if text[k] == "{":
            depth += 1
        elif text[k] == "}":
            depth -= 1
            if depth == 0:
                return text[j + 1 : k]
    return None


_NUM = re.compile(r"-?\d[\d,]*(?:\.\d+)?(?:/\d+)?")


def extract_answer(text: str) -> str | None:
    """The final answer in a response: last ``\\boxed{}``, else 'answer is X', else the last number."""
    if (b := _last_boxed(text)) is not None:
        return b
    m = re.findall(r"answer is[:\s]*\$?([^\n$]+)", text, flags=re.I)
    if m:
        return m[-1].strip().rstrip(".")
    nums = _NUM.findall(text)
    return nums[-1] if nums else None


def _normalize(s: str) -> str:
    s = s.strip().strip("$").strip()
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\(left|right|!|,|;|quad|displaystyle)", "", s)
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac").replace("^\\circ", "").replace("^{\\circ}", "")
    s = s.replace(" ", "").rstrip(".")
    s = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"(\1)/(\2)", s)
    s = re.sub(r"\\frac(\d)(\d)", r"\1/\2", s)
    return s


def _as_number(s: str) -> Fraction | None:
    t = s.replace(",", "").replace("(", "").replace(")", "")
    t = re.sub(r"^\\?\$|%$", "", t)
    try:
        if "/" in t:
            a, b = t.split("/", 1)
            return Fraction(a) / Fraction(b)
        return Fraction(t)
    except (ValueError, ZeroDivisionError):
        return None


def math_equal(pred: str | None, gold: str) -> bool:
    if pred is None:
        return False
    p, g = _normalize(pred), _normalize(str(gold))
    if p == g:
        return True
    a, b = _as_number(p), _as_number(g)
    if a is not None and b is not None:
        return abs(float(a) - float(b)) <= 1e-6 * max(1.0, abs(float(b)))
    return False


# ------------------------------------------------------------------ graders
def grade_math(text: str, item: Item) -> float:
    gold = item["answer"]
    gold = _last_boxed(str(gold)) or str(gold)
    return float(math_equal(extract_answer(text), gold))


def grade_exact(text: str, item: Item) -> float:
    return float(text.strip() == str(item["answer"]).strip())


def grade_contains(text: str, item: Item) -> float:
    return float(str(item["answer"]).strip().lower() in text.lower())


def grade_choice(text: str, item: Item) -> float:
    """Multiple choice: the last standalone letter A-J (or the boxed one) must equal the answer letter."""
    boxed = _last_boxed(text)
    found = re.findall(r"\b([A-J])\b", boxed if boxed else text)
    return float(bool(found) and found[-1].upper() == str(item["answer"]).strip().upper())


def numeric(rel_tol: float = 0.05) -> Grader:
    """SciBench-style: the last number must be within ``rel_tol`` of the answer."""

    def grade(text: str, item: Item) -> float:
        pred = extract_answer(text)
        a = _as_number(_normalize(pred)) if pred else None
        b = _as_number(_normalize(str(item["answer"])))
        if a is None or b is None:
            return 0.0
        return float(abs(float(a) - float(b)) <= rel_tol * max(abs(float(b)), 1e-12))

    return grade


GRADERS: dict[str, Grader] = {
    "math": grade_math,
    "exact": grade_exact,
    "contains": grade_contains,
    "choice": grade_choice,
    "numeric": numeric(0.05),
}


def get_grader(g: str | Grader) -> Grader:
    if callable(g):
        return g
    if g not in GRADERS:
        raise KeyError(f"unknown grader {g!r}; known: {sorted(GRADERS)}")
    return GRADERS[g]


# ------------------------------------------------------------------ data
_PROMPT_KEYS = ("prompt", "question", "problem", "input", "messages")
_ANSWER_KEYS = ("answer", "target", "solution", "output", "label")


def load_items(path: str | os.PathLike[str], prompt_key: str | None = None, answer_key: str | None = None,
               id_key: str | None = None, limit: int | None = None) -> list[Item]:
    """Read a JSONL (or JSON list) of items into ``{"id", "prompt", "answer"}`` dicts."""
    text = Path(path).read_text(encoding="utf-8")
    rows = json.loads(text) if text.lstrip().startswith("[") else [json.loads(x) for x in text.splitlines() if x.strip()]
    items = []
    for i, r in enumerate(rows[:limit] if limit else rows):
        pk = prompt_key or next((k for k in _PROMPT_KEYS if k in r), None)
        ak = answer_key or next((k for k in _ANSWER_KEYS if k in r), None)
        if pk is None or ak is None:
            raise KeyError(f"row {i}: cannot find prompt/answer fields in {sorted(r)}; pass prompt_key/answer_key")
        qid = r.get(id_key) if id_key else r.get("id", r.get("unique_id", r.get("idx", i)))
        items.append({**r, "id": str(qid), "prompt": r[pk], "answer": r[ak]})
    return items


# ------------------------------------------------------------------ evaluation
@dataclass
class EvalReport:
    paired: PairedResult
    arms: dict[str, dict[str, Any]]
    scores: dict[str, dict[str, list[float]]]
    controls: dict[str, PairedResult] = field(default_factory=dict)

    @property
    def verdict(self) -> str:
        return self.paired.verdict

    def beats_controls(self) -> dict[str, bool]:
        """For each control, whether the edit scored higher than the control on the same items."""
        return {name: self.paired.edited_acc > c.edited_acc for name, c in self.controls.items()}

    def as_dict(self) -> dict[str, Any]:
        return {
            "paired": self.paired.as_dict(), "arms": self.arms,
            "controls": {k: v.as_dict() for k, v in self.controls.items()},
            "beats_controls": self.beats_controls(),
        }

    def __str__(self) -> str:
        lines = [str(self.paired)]
        for arm, s in self.arms.items():
            lines.append(f"  {arm:<8} mean tokens {s['mean_tokens']:.0f}, truncated {100 * s['truncated_share']:.1f}%")
        if self.controls:
            lines.append("  controls (edited-arm accuracy; the edit should beat every one):")
            for name, c in self.controls.items():
                mark = "beats" if self.paired.edited_acc > c.edited_acc else "DOES NOT BEAT"
                lines.append(f"    {name:<12} {100 * c.edited_acc:6.2f}%   edit {mark} it")
        return "\n".join(lines)


def _scores(items: Sequence[Item], answers: list[list[dict[str, Any]]], grader: Grader) -> dict[str, list[float]]:
    return {str(it["id"]): [grader(s["text"], it) for s in samples] for it, samples in zip(items, answers, strict=True)}


def evaluate(
    probe: Probe,
    patch: Patch | None,
    items: Sequence[Item],
    grader: str | Grader = "math",
    n: int = 1,
    controls: Mapping[str, Patch] | None = None,
    **generate_kwargs: Any,
) -> EvalReport:
    """Paired comparison of base vs ``patch`` on ``items`` (and optionally each control)."""
    g = get_grader(grader)
    prompts = [it["prompt"] for it in items]
    previous = probe.patch
    try:
        if patch is not None:
            probe.set_patch(patch)
        res = probe.generate(prompts, weights="both", n=n, **generate_kwargs)
        summary = res.summary()
        base = _scores(items, res.arms["original"], g)
        edited = _scores(items, res.arms["edited"], g)
        report = EvalReport(paired(base, edited), summary, {"original": base, "edited": edited})
        for name, ctrl in (controls or {}).items():
            probe.set_patch(ctrl)
            c = probe.generate(prompts, weights="edited", n=n, **generate_kwargs)
            report.controls[name] = paired(base, _scores(items, c.arms["edited"], g))
        return report
    finally:
        probe.set_patch(previous)
