"""Agent sessions and editors (offline: tiny model, scripted / fake LLM clients)."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

import weightspec as ws  # noqa: E402
from weightspec.agent.editors import AnthropicEditor, ScriptedEditor, ToolResult  # noqa: E402
from weightspec.agent.editors_openai import OpenAICompatibleEditor  # noqa: E402
from weightspec.probe import Probe  # noqa: E402

from .test_probe import char_tokenizer  # noqa: E402


def tiny_probe(budget):
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=100, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=256,
                      bos_token_id=1, eos_token_id=2, pad_token_id=0, tie_word_embeddings=False)
    return Probe.from_model(LlamaForCausalLM(cfg).eval(), char_tokenizer(), budget=budget)


DEV = [{"id": f"d{i}", "prompt": f"say {c}", "answer": c} for i, c in enumerate("abcdef")]
HOLDOUT = [{"id": f"h{i}", "prompt": f"say {c}", "answer": c} for i, c in enumerate("xyz")]
GEN = {"max_new_tokens": 6, "chat": False}


def full_script():
    op = {"op": "add", "target": "layer[1].mlp.down.col[3]", "alpha": -1.0, "u": {"fn": "payload", "key": "doubt"}}
    return [
        [("list_items", {"limit": 3}), ("budget_status", {})],
        [("generate", {"item_ids": ["d0", "d1"], "weights": "original"})],
        [("contrast_activations", {"texts_a": ["wait, maybe", "but perhaps"], "texts_b": ["so then", "thus"],
                                   "layer": 1, "site": "residual", "name": "doubt"})],
        [("readout_tokens", {"vector": "doubt", "k": 3}),
         ("create_patch", {"name": "fix"}), ("add_op", {"patch": "fix", "op_json": json.dumps(op)})],
        [("build_patch", {"patch": "fix"}), ("make_controls", {"patch": "fix"}),
         ("scale_everything_please", {})],  # unknown tool -> error result, session continues
        [("evaluate_patch", {"patch": "fix", "item_ids": ["d2", "d3", "d4", "d5"], "controls": ["random-0"]})],
        [("save_patch", {"patch": "fix"}), ("finish", {"patch": "fix", "explanation": "subtracted the contrast"})],
    ]


def test_scripted_session_end_to_end(tmp_path):
    probe = tiny_probe(budget=10)
    editor = ScriptedEditor(full_script())
    s = ws.Session(probe=probe, editor=editor, goal="echo the letter", dev=DEV, holdout=HOLDOUT,
                   grader="contains", workspace=tmp_path / "run", generate_defaults=GEN)
    events = []
    result = s.run(on_event=lambda kind, d: events.append(kind))
    assert result.status == "finished" and result.steps == 7
    assert probe.budget.used == 4  # generate + contrast + evaluate(both) + 1 control
    assert result.patch is not None and result.patch.result.changed > 0
    assert result.patch.provenance.access_level == "weights+4fwd"
    assert (tmp_path / "run" / "final.wspec").exists()
    assert ws.Patch.load(tmp_path / "run" / "final.wspec").verify(probe.store).ok
    assert result.holdout is not None and result.holdout.paired.n_questions == 3  # sealed, separate budget
    assert probe.budget.used == 4
    errs = [r for r in editor.log if isinstance(r, list) and any(x["is_error"] for x in r)]
    assert errs and "unknown tool" in json.dumps(errs)
    traj = [json.loads(x) for x in (tmp_path / "run" / "trajectory.jsonl").read_text().splitlines()]
    assert traj[0]["type"] == "start" and traj[-1]["type"] == "end"
    assert "tool" in events
    assert "subtracted the contrast" in str(result)
    # the system prompt and task message reached the editor
    assert "weight-space programmer" in editor.log[0]["system"] and "BUDGET. 10" in editor.log[0]["task"]


def test_holdout_must_be_disjoint(tmp_path):
    with pytest.raises(ValueError, match="holdout"):
        ws.Session(probe=tiny_probe(5), editor=ScriptedEditor([]), goal="g", dev=DEV, holdout=DEV[:1],
                   workspace=tmp_path)


def test_budget_exhaustion_notice_and_refusal_of_extra_passes(tmp_path):
    probe = tiny_probe(budget=1)
    turns = [[("generate", {"item_ids": ["d0"]})], [("generate", {"item_ids": ["d1"]})],
             [("finish", {"explanation": "no edit beat the original", "patch": None})]]
    editor = ScriptedEditor(turns)
    s = ws.Session(probe=probe, editor=editor, goal="g", dev=DEV, workspace=tmp_path, grader="contains",
                   generate_defaults=GEN)
    r = s.run()
    assert r.status == "finished" and r.patch is None and probe.budget.used == 1
    assert any("budget is used up" in json.dumps(x) for x in editor.log)
    second = editor.log[3]  # results of the second generate call
    assert second[0]["is_error"] and "budget exhausted" in second[0]["content"]


def test_session_stops_when_editor_goes_quiet(tmp_path):
    s = ws.Session(probe=tiny_probe(5), editor=ScriptedEditor([]), goal="g", dev=DEV, workspace=tmp_path)
    r = s.run()
    assert r.status == "stopped" and r.steps == 3 and r.patch is None


def test_crash_and_resume(tmp_path):
    probe = tiny_probe(budget=10)
    script = full_script()

    def crash(_):
        raise ConnectionError("network down")

    first = ScriptedEditor([*script[:3], crash])
    s = ws.Session(probe=probe, editor=first, goal="g", dev=DEV, workspace=tmp_path, grader="contains",
                   generate_defaults=GEN)
    with pytest.raises(ConnectionError):
        s.run()
    state = json.loads((tmp_path / "state.json").read_text())
    assert state["steps"] == 3 and state["budget"]["used"] == 2
    second = ScriptedEditor(script)
    resumed = ws.Session.resume(tmp_path, probe=probe, editor=second, goal="g", dev=DEV, grader="contains",
                                generate_defaults=GEN)
    assert resumed.probe.budget.used == 2 and second.i == 3 and "doubt" in resumed.toolkit.vectors
    r = resumed.run()
    assert r.status == "finished" and r.patch is not None and probe.budget.used == 4


# ---------------------------------------------------------------- Anthropic editor (fake client)
class _Block:
    def __init__(self, **d):
        self.d = d

    def to_dict(self):
        return dict(self.d)


class FakeAnthropic:
    def __init__(self, replies):
        self.replies, self.requests = list(replies), []
        self.beta = SimpleNamespace(messages=SimpleNamespace(stream=self._stream))

    def _stream(self, **params):
        self.requests.append(json.loads(json.dumps(params, default=str)))
        reply = self.replies.pop(0)
        outer = self

        class Ctx:
            def __enter__(self):
                return SimpleNamespace(get_final_message=lambda: reply)

            def __exit__(self, *a):
                return False

        del outer
        return Ctx()


def _msg(blocks, stop, **extra):
    usage = SimpleNamespace(input_tokens=10, output_tokens=5, cache_read_input_tokens=3, cache_creation_input_tokens=0)
    return SimpleNamespace(content=[_Block(**b) for b in blocks], stop_reason=stop, usage=usage, **extra)


def test_anthropic_editor_request_shape_and_history():
    replies = [
        _msg([{"type": "thinking", "thinking": "", "signature": "sig"},
              {"type": "tool_use", "id": "tu_1", "name": "budget_status", "input": {}}], "tool_use"),
        _msg([{"type": "text", "text": "done"}], "end_turn"),
    ]
    client = FakeAnthropic(replies)
    ed = AnthropicEditor(client=client)
    tk = ws.Toolkit(".", probe=None)
    tools = tk.specs()
    ed.start("SYS", "TASK")
    t1 = ed.step(tools)
    assert [c.name for c in t1.calls] == ["budget_status"] and t1.stop == "tool_use"
    req = client.requests[0]
    assert req["model"] == "claude-opus-5-5" and req["thinking"] == {"type": "adaptive"}
    assert req["output_config"] == {"effort": "high"} and req["cache_control"] == {"type": "ephemeral"}
    assert req["fallbacks"] == "default" and req["betas"] == ["server-side-fallback-2026-07-01"]
    assert "tool_choice" not in req and all(t["eager_input_streaming"] for t in req["tools"])
    assert req["system"] == "SYS" and req["messages"][0] == {"role": "user", "content": "TASK"}
    ed.add_results([ToolResult("tu_1", '{"ok": 1}')])
    t2 = ed.step(tools)
    assert t2.text == "done" and t2.usage["cache_read_input_tokens"] == 3
    hist = client.requests[1]["messages"]
    assert hist[1]["content"][0] == {"type": "thinking", "thinking": "", "signature": "sig"}  # passed back unchanged
    assert hist[2]["content"] == [{"type": "tool_result", "tool_use_id": "tu_1", "content": '{"ok": 1}', "is_error": False}]
    assert ed.state()["messages"][-1]["role"] == "assistant"


def test_anthropic_refusal_ends_session(tmp_path):
    refusal = _msg([], "refusal", stop_details=SimpleNamespace(category="cyber", explanation="x"))
    ed = AnthropicEditor(client=FakeAnthropic([refusal]))
    s = ws.Session(probe=tiny_probe(5), editor=ed, goal="g", dev=DEV, workspace=tmp_path)
    r = s.run()
    assert r.status == "refused"
    assert "cyber" in (tmp_path / "trajectory.jsonl").read_text()


def test_max_tokens_truncated_calls_are_not_run(tmp_path):
    cut = _msg([{"type": "tool_use", "id": "tu_9", "name": "create_patch", "input": {"name": "x"}}], "max_tokens")
    done = _msg([{"type": "tool_use", "id": "tu_10", "name": "finish", "input": {"explanation": "none"}}], "tool_use")
    client = FakeAnthropic([cut, done])
    s = ws.Session(probe=tiny_probe(5), editor=AnthropicEditor(client=client), goal="g", dev=DEV, workspace=tmp_path)
    r = s.run()
    assert r.status == "finished" and "x" not in s.toolkit.patches
    last_user = client.requests[1]["messages"][-1]["content"][0]
    assert last_user["is_error"] and "max_tokens" in last_user["content"]


# ---------------------------------------------------------------- OpenAI-compatible editor (fake client)
def test_openai_compatible_editor():
    calls = []

    def create(**kw):
        calls.append(kw)
        tc = SimpleNamespace(id="c1", function=SimpleNamespace(name="budget_status", arguments="{}"))
        msg = SimpleNamespace(content=None, tool_calls=[tc])
        return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason="tool_calls")],
                               usage=SimpleNamespace(prompt_tokens=7, completion_tokens=2))

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    ed = OpenAICompatibleEditor("local-model", base_url="http://localhost:8000/v1", client=client)
    ed.start("SYS", "TASK")
    turn = ed.step(ws.Toolkit(".").specs())
    assert turn.stop == "tool_use" and turn.calls[0].name == "budget_status"
    assert calls[0]["tools"][0]["type"] == "function" and calls[0]["messages"][0]["role"] == "system"
    ed.add_results([ToolResult("c1", "boom", True)])
    assert ed.messages[-1] == {"role": "tool", "tool_call_id": "c1", "content": "ERROR: boom"}


def test_make_editor_specs(tmp_path):
    assert ws.agent.make_editor("anthropic:claude-opus-5-5").model == "claude-opus-5-5"
    assert ws.agent.make_editor("openai:m@http://h/v1").name == "openai:m@http://h/v1"
    (tmp_path / "s.json").write_text(json.dumps([[{"name": "budget_status"}]]))
    assert ws.agent.make_editor(f"scripted:{tmp_path / 's.json'}").turns[0][0][0] == "budget_status"
    with pytest.raises(ValueError):
        ws.agent.make_editor("nope:x")


def test_toolkit_schemas_are_strict_objects():
    tk = ws.Toolkit(".", probe=tiny_probe(1), items=DEV)
    names = {s.name for s in tk.specs()}
    assert {"generate", "teacher_forced", "contrast_activations", "evaluate_patch", "finish"} <= names
    sch = {s.name: s.input_schema() for s in tk.specs()}
    assert sch["generate"]["additionalProperties"] is False
    assert set(sch["contrast_activations"]["required"]) == {"texts_a", "texts_b", "layer"}
    no_probe = {s.name for s in ws.Toolkit(".").specs(include_session_only=False)}
    assert "generate" not in no_probe and "finish" not in no_probe
    with pytest.raises(ws.agent.ToolFailure, match="extra"):
        tk.call("budget_status", {"bogus": 1})
    np.testing.assert_equal(tk.call("budget_status"), {"used": 0, "limit": 1, "remaining": 1})


def test_cli_run_with_scripted_editor(tmp_path, capsys):
    from weightspec.cli import main

    probe = tiny_probe(budget=None)
    model_dir = tmp_path / "tiny"
    probe.backend.model.save_pretrained(model_dir)
    probe.backend.tokenizer.save_pretrained(model_dir)
    (tmp_path / "dev.jsonl").write_text("\n".join(json.dumps(x) for x in DEV))
    (tmp_path / "test.jsonl").write_text("\n".join(json.dumps(x) for x in HOLDOUT))
    op = {"op": "scale", "target": "layer[0].attn.head[1].out", "factor": 0.5}
    script = [[{"name": "generate", "arguments": {"item_ids": ["d0"], "weights": "original"}}],
              [{"name": "create_patch", "arguments": {"name": "half-head"}},
               {"name": "add_op", "arguments": {"patch": "half-head", "op_json": json.dumps(op)}},
               {"name": "finish", "arguments": {"patch": "half-head", "explanation": "halved head 1"}}]]
    (tmp_path / "script.json").write_text(json.dumps(script))
    ws_dir = tmp_path / "run"
    rc = main(["run", str(model_dir), "--goal", "echo", "--dev", str(tmp_path / "dev.jsonl"),
               "--holdout", str(tmp_path / "test.jsonl"), "--grader", "contains",
               "--editor", f"scripted:{tmp_path / 'script.json'}", "--budget", "3", "--no-chat",
               "--max-new-tokens", "4", "--workspace", str(ws_dir)])
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "Session finished" in out and "sealed holdout" in out and "halved head 1" in out
    assert (ws_dir / "final.json").exists() and (ws_dir / "holdout.json").exists()
    assert ws.Patch.load(ws_dir / "final.json").result.changed > 0
    assert main(["run", str(model_dir), "--goal", "x"]) == 2  # --dev missing: clean error
