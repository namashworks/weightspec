import json

import pytest

import weightspec as ws
from weightspec.evals import extract_answer, grade_choice, grade_math, math_equal, numeric


@pytest.mark.parametrize(
    ("text", "gold", "ok"),
    [
        ("so the result is \\boxed{\\frac{1}{2}}", "0.5", True),
        ("\\boxed{\\dfrac{3}{4}}", "\\frac34", True),
        ("The answer is 1,234.", "1234", True),
        ("first 3 then 7", "7", True),
        ("\\boxed{x^2+1}", "x^2 + 1", True),
        ("\\boxed{\\text{(B)}}", "(B)", True),
        ("\\boxed{12}", "13", False),
        ("no number here", "4", False),
        ("nested \\boxed{\\frac{\\sqrt{2}}{2}}", "\\frac{\\sqrt{2}}{2}", True),
    ],
)
def test_math_grading(text, gold, ok):
    assert bool(grade_math(text, {"answer": gold})) is ok


def test_gold_solution_with_boxed():
    assert grade_math("\\boxed{5}", {"answer": "We compute ... so \\boxed{5}."}) == 1.0


def test_extract_and_equal_edge_cases():
    assert extract_answer("a \\boxed{1} then \\boxed{2}") == "2"
    assert math_equal("10%", "10")
    assert not math_equal(None, "1")


def test_choice_and_numeric():
    assert grade_choice("I think the answer is (C). Final: C", {"answer": "C"}) == 1.0
    assert grade_choice("A or B... \\boxed{D}", {"answer": "D"}) == 1.0
    g = numeric(0.05)
    assert g("about 101.0", {"answer": "100"}) == 1.0 and g("about 110", {"answer": "100"}) == 0.0


def test_load_items(tmp_path):
    (tmp_path / "a.jsonl").write_text("\n".join(json.dumps(r) for r in [
        {"unique_id": "q1", "problem": "1+1?", "answer": "2"},
        {"unique_id": "q2", "problem": "2+2?", "answer": "4"},
    ]))
    items = ws.load_items(tmp_path / "a.jsonl")
    assert items[0]["id"] == "q1" and items[1]["prompt"] == "2+2?" and items[1]["answer"] == "4"
    (tmp_path / "b.jsonl").write_text(json.dumps({"x": 1}))
    with pytest.raises(KeyError):
        ws.load_items(tmp_path / "b.jsonl")


class FakeProbe:
    """Answers come from a function of (arm patch name, prompt, sample index)."""

    def __init__(self, answer):
        self.answer, self.patch, self.calls = answer, None, []

    def set_patch(self, patch, policy=None):
        self.patch = patch

    def generate(self, prompts, weights="original", n=1, **kw):
        from weightspec.probe import GenerateResult

        self.calls.append(weights)
        arms = ["original", "edited"] if weights == "both" else [weights]
        out = {}
        for arm in arms:
            name = None if arm == "original" else (self.patch.name if self.patch else None)
            out[arm] = [[{"text": self.answer(name, p, s), "tokens": 10, "truncated": False} for s in range(n)]
                        for p in prompts]
        return GenerateResult(out, list(prompts))


def test_evaluate_with_controls(store):
    items = [{"id": str(i), "prompt": f"{i}+{i}?", "answer": str(2 * i)} for i in range(20)]

    def answer(patch_name, prompt, s):
        i = int(prompt.split("+")[0])
        if patch_name == "fix":
            return f"\\boxed{{{2 * i}}}"  # always right
        if patch_name == "fix--random-0":
            return f"\\boxed{{{2 * i if i % 4 == 0 else -1}}}"
        return f"\\boxed{{{2 * i if i % 5 == 0 else -1}}}"  # base: right 20% of the time

    patch = ws.Patch.new(store, "fix").add("layer[0].mlp.down.col[0]", [1.0] * 8)
    patch.build()
    probe = FakeProbe(answer)
    controls = {"random-0": ws.controls(patch, doses=(), reverse=0, seeds=(0,), ablate=False)["random-0"]}
    rep = ws.evaluate(probe, patch, items, grader="math", n=2, controls=controls)
    assert rep.paired.base_acc == pytest.approx(0.2) and rep.paired.edited_acc == 1.0
    assert rep.verdict == "confirmed"
    assert rep.controls["random-0"].edited_acc == pytest.approx(0.25)
    assert rep.beats_controls() == {"random-0": True}
    assert probe.calls == ["both", "edited"] and probe.patch is None  # previous patch restored
    assert "controls" in str(rep)
