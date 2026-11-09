import json
import math

import pytest

import weightspec as ws


def test_matches_hand_computation():
    base = {"a": [0, 0], "b": [1, 0], "c": [0, 0], "d": [1, 1]}
    edited = {"a": [1, 1], "b": [1, 1], "c": [0, 1], "d": [1, 1]}
    r = ws.paired(base, edited)
    d = [1.0, 0.5, 0.5, 0.0]
    mean = sum(d) / 4
    se = math.sqrt(sum((x - mean) ** 2 for x in d) / (4 * 3))
    assert r.delta == pytest.approx(mean) and r.se == pytest.approx(se)
    assert r.base_acc == pytest.approx(0.375) and r.edited_acc == pytest.approx(0.875)
    assert (r.rescues, r.regressions) == (4, 0)
    assert r.verdict == "confirmed"


def test_verdicts():
    same = {str(i): [i % 2] for i in range(50)}
    assert ws.paired(same, same).verdict == "not_significant"
    worse = {k: [0] for k in same}
    best = {k: [1] for k in same}
    assert ws.paired(best, worse).verdict == "regressed"
    assert ws.paired({"a": [1]}, {"a": [0]}).verdict == "insufficient_data"


def test_unpaired_questions_are_dropped_and_reported():
    r = ws.paired({"a": [0], "b": [0], "x": [1]}, {"a": [1], "b": [1], "y": [0]})
    assert r.n_questions == 2 and r.dropped_base_only == 1 and r.dropped_edited_only == 1
    assert "excluded" in str(r)


def test_load_scores_formats(tmp_path):
    (tmp_path / "a.json").write_text(json.dumps({"per_problem": [{"id": "q1", "correct": [0, 1]}, {"id": "q2", "correct": [1]}]}))
    (tmp_path / "b.jsonl").write_text('{"question_id": "q1", "score": true}\n{"question_id": "q1", "score": 0}\n')
    (tmp_path / "c.json").write_text(json.dumps({"q1": 1, "q2": 0.5}))
    (tmp_path / "d.jsonl").write_text('{"name": "q1", "ok": 1}\n')
    assert ws.load_scores(tmp_path / "a.json") == {"q1": [0.0, 1.0], "q2": [1.0]}
    assert ws.load_scores(tmp_path / "b.jsonl") == {"q1": [1.0, 0.0]}
    assert ws.load_scores(tmp_path / "c.json") == {"q1": [1.0], "q2": [0.5]}
    assert ws.load_scores(tmp_path / "d.jsonl", id_key="name", score_key="ok") == {"q1": [1.0]}
    with pytest.raises(KeyError):
        ws.load_scores(tmp_path / "d.jsonl")
