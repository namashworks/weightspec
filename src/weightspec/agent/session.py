"""Agent sessions: an LLM edits a model's weights under a budget, with evidence.

    probe = ws.Probe.from_pretrained("Qwen/Qwen3-0.6B", budget=50, device="cuda")
    session = ws.Session(
        probe=probe, editor="anthropic:claude-opus-5-5",
        goal="Improve accuracy on grade-school math word problems.",
        dev=ws.load_items("gsm8k_train.jsonl"), holdout=ws.load_items("gsm8k_test.jsonl", limit=200),
        workspace="runs/gsm8k-1",
    )
    result = session.run()
    print(result)

The session is checkpointed after every step (``state.json`` plus drafts and
vectors), so it can be resumed after a crash with :meth:`Session.resume`. Every
tool call and editor turn is appended to ``trajectory.jsonl``. The holdout items
are sealed: the editor never sees them, and they are evaluated once, after the
editor has made its final choice, with a separate budget.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from weightspec.agent.editors import EditorTurn, ToolResult, make_editor
from weightspec.agent.toolkit import ToolFailure, Toolkit
from weightspec.evals import EvalReport, Item, evaluate
from weightspec.patch import Patch
from weightspec.policy import Policy
from weightspec.probe import Budget, Probe

__all__ = ["SYSTEM_PROMPT", "Session", "SessionResult"]

SYSTEM_PROMPT = """\
You are a weight-space programmer. You improve a neural network by editing its weights directly, \
with tools, based on an understanding of what its components compute. You never train it: no \
gradients, no optimizers, no fitting weights to a loss.

How to work:
- Diagnose first. Look at what the model gets wrong on development items: wrong answers, missing \
answers, answers cut off at the token limit. Separate symptoms from causes and consider competing \
explanations before spending budget.
- Localize causally. Being more active on correct answers is not enough. Before relying on a \
component, show that changing it changes the behavior you predicted, and try more than one strength.
- Prefer small, structured edits you can explain: one MLP unit's write direction, one attention \
head, one direction projected out. The number of changed values does not predict success; the \
choice of target does.
- Measure with enough power. Compare original and edited weights on the same items in one paired \
request. Keep an edit only if its gain is clearly larger than its standard error on items that \
were not used to choose it, and only if it beats its controls (make_controls, then evaluate_patch).
- Keeping the original weights is a valid result when no edit is clearly better.
- Forward passes are scarce. Every generate, teacher_forced, contrast_activations and \
evaluate_patch call costs budget. Plan them.

When you are done, call finish with the patch to submit (or null) and an explanation: what you \
changed, the evidence, and what you did not verify."""


@dataclass
class SessionResult:
    status: str  # finished | stopped | budget_exhausted | max_steps | refused
    patch: Patch | None
    explanation: str
    steps: int
    budget: dict[str, Any]
    usage: dict[str, int]
    holdout: EvalReport | None = None
    workspace: Path | None = None
    files: dict[str, str] = field(default_factory=dict)

    def __str__(self) -> str:
        lines = [f"Session {self.status} after {self.steps} steps; budget {self.budget['used']}/{self.budget['limit']}"]
        if self.patch is not None and self.patch.result is not None:
            lines.append(f"  submitted {self.patch.name!r}: {self.patch.result.changed:,} values changed")
        else:
            lines.append("  submitted: original weights (no edit)")
        if self.explanation:
            lines.append("  explanation: " + self.explanation.strip().replace("\n", "\n    ")[:1500])
        if self.holdout is not None:
            lines.append("  sealed holdout:\n    " + str(self.holdout).replace("\n", "\n    "))
        for k, v in self.files.items():
            lines.append(f"  {k}: {v}")
        return "\n".join(lines)


class Session:
    def __init__(
        self,
        *,
        probe: Probe,
        editor: Any,
        goal: str,
        dev: Sequence[Item],
        holdout: Sequence[Item] | None = None,
        grader: Any = "math",
        workspace: str | os.PathLike[str] = "wsp-run",
        policy: Policy | None = None,
        max_steps: int = 300,
        holdout_n: int = 1,
        generate_defaults: dict[str, Any] | None = None,
        system_prompt: str = SYSTEM_PROMPT,
        max_result_chars: int = 12000,
    ):
        self.workspace = Path(workspace)
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.probe = probe
        self.editor = make_editor(editor) if isinstance(editor, str) else editor
        self.goal = goal
        self.dev = list(dev)
        self.holdout = list(holdout or [])
        dev_ids = {str(i["id"]) for i in self.dev}
        leaked = [str(i["id"]) for i in self.holdout if str(i["id"]) in dev_ids]
        if leaked:
            raise ValueError(f"{len(leaked)} holdout items also appear in dev (e.g. {leaked[:3]})")
        self.grader = grader
        self.holdout_n = holdout_n
        self.max_steps = max_steps
        self.system_prompt = system_prompt
        self.max_result_chars = max_result_chars
        self.toolkit = Toolkit(self.workspace, probe=probe, items=self.dev, grader=grader,
                               policy=policy or Policy.paper(), generate_defaults=generate_defaults)
        self.steps = 0
        self.usage: dict[str, int] = {}
        self.nudges = 0
        self.budget_notice = False
        self.status = "running"
        self._resumed = False

    # ------------------------------------------------------------------ prompts
    def task_message(self) -> str:
        info = self.toolkit.inspect_model()
        b = self.probe.budget
        return (
            f"GOAL. {self.goal}\n\n"
            f"MODEL. {info['source']} ({info['arch']}, {info['layers']} layers, {info['heads']} heads of width "
            f"{info['head_dim']}, gated queries: {info['gated_query']}; {info['params']:,} parameters).\n"
            f"Full-attention layers: {info['full_attention_layers'] or 'all'}.\n\n"
            f"DATA. {len(self.dev)} development items (list_items). "
            + (f"A sealed held-out set of {len(self.holdout)} items will be evaluated after you finish; "
               "you never see it, so only edits that generalize will count.\n" if self.holdout else "\n")
            + f"\nBUDGET. {b.forward if b.forward is not None else 'unlimited'} forward passes "
            f"(generation default: {self.toolkit.generate_defaults}).\n"
            "RULES. Embeddings, the LM head and the final norm are frozen; whole-tensor rescaling is refused. "
            "Every build reports exactly how many stored values change.\n\n"
            "Start by diagnosing failures on development items."
        )

    # ------------------------------------------------------------------ persistence
    def _log(self, event: dict[str, Any]) -> None:
        event = {"time": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"), "step": self.steps, **event}
        with open(self.workspace / "trajectory.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(event, default=str) + "\n")

    def checkpoint(self) -> None:
        drafts = self.workspace / "drafts"
        drafts.mkdir(exist_ok=True)
        for name, p in self.toolkit.patches.items():
            (drafts / f"{name}.wspec").write_bytes(p.to_bytes(container="wspec"))
        if self.toolkit.vectors:
            np.savez(self.workspace / "vectors.npz", **self.toolkit.vectors)  # type: ignore[arg-type]
        state = {
            "goal": self.goal, "steps": self.steps, "status": self.status, "usage": self.usage,
            "nudges": self.nudges, "budget_notice": self.budget_notice, "budget": self.probe.budget.to_dict(),
            "finished": self.toolkit.finished, "editor": self.editor.state(),
        }
        tmp = self.workspace / "state.json.tmp"
        tmp.write_text(json.dumps(state, default=str), encoding="utf-8")
        os.replace(tmp, self.workspace / "state.json")

    @classmethod
    def resume(cls, workspace: str | os.PathLike[str], **kwargs: Any) -> Session:
        """Rebuild a session from its checkpoint. Pass the same probe, editor, dev and holdout as before."""
        s = cls(workspace=workspace, **kwargs)
        state = json.loads((s.workspace / "state.json").read_text(encoding="utf-8"))
        s.steps, s.usage, s.nudges = state["steps"], state["usage"], state["nudges"]
        s.budget_notice = state["budget_notice"]
        s.probe.budget = Budget.from_dict(state["budget"])
        s.toolkit.finished = state["finished"]
        s.editor.load_state(state["editor"])
        for f in sorted((s.workspace / "drafts").glob("*.wspec")):
            p = Patch.load(f)
            s.toolkit.patches[p.name] = p
        if (s.workspace / "vectors.npz").exists():
            with np.load(s.workspace / "vectors.npz") as z:
                s.toolkit.vectors = {k: z[k] for k in z.files}
        s._resumed = True
        s._log({"type": "resume"})
        return s

    # ------------------------------------------------------------------ loop
    def _execute(self, turn: EditorTurn) -> list[ToolResult]:
        results = []
        for call in turn.calls:
            if turn.stop == "max_tokens":
                results.append(ToolResult(call.id, "Your reply was cut off by max_tokens, so this call's input may be "
                                                   "incomplete and was not run. Send it again, shorter.", True))
                continue
            try:
                out = self.toolkit.call(call.name, call.arguments)
                text = json.dumps(out, default=str)
                if len(text) > self.max_result_chars:
                    text = text[: self.max_result_chars] + ' ... [truncated; full outputs are saved in the workspace]'
                results.append(ToolResult(call.id, text))
            except ToolFailure as e:
                results.append(ToolResult(call.id, str(e), True))
            self._log({"type": "tool", "name": call.name, "arguments": call.arguments,
                       "is_error": results[-1].is_error, "result": results[-1].content[:2000]})
        return results

    def run(self, on_event: Callable[[str, dict[str, Any]], None] | None = None) -> SessionResult:
        tools = self.toolkit.specs()
        emit = on_event or (lambda kind, data: None)
        if not getattr(self, "_resumed", False):
            self.editor.start(self.system_prompt, self.task_message())
            self._log({"type": "start", "editor": self.editor.name, "goal": self.goal,
                       "budget": self.probe.budget.status(), "dev": len(self.dev), "holdout": len(self.holdout)})
        while self.toolkit.finished is None:
            if self.steps >= self.max_steps:
                self.status = "max_steps"
                break
            turn = self.editor.step(tools)
            self.steps += 1
            for k, v in turn.usage.items():
                self.usage[k] = self.usage.get(k, 0) + v
            self._log({"type": "editor", "stop": turn.stop, "text": turn.text,
                       "calls": [c.name for c in turn.calls], "usage": turn.usage})
            emit("editor", {"step": self.steps, "text": turn.text, "calls": [c.name for c in turn.calls]})
            if turn.stop == "refusal":
                self.status = "refused"
                self._log({"type": "refusal", "detail": turn.detail})
                break
            if turn.calls:
                results = self._execute(turn)
                self.editor.add_results(results)
                for c, r in zip(turn.calls, results, strict=True):
                    emit("tool", {"name": c.name, "is_error": r.is_error, "result": r.content[:300]})
                self.nudges = 0
            elif turn.stop == "pause_turn":
                pass  # the API paused a long turn; asking again continues it
            elif turn.stop == "max_tokens":
                self.editor.add_user("Your reply was cut off by max_tokens. Continue, more briefly.")
            else:
                self.nudges += 1
                if self.nudges > 2:
                    self.status = "stopped"
                    break
                self.editor.add_user("Continue with the tools, or call finish(patch, explanation) if you are done.")
            remaining = self.probe.budget.remaining
            if remaining == 0 and not self.budget_notice and self.toolkit.finished is None:
                self.budget_notice = True
                self.editor.add_user("The forward-pass budget is used up. Make your final choice now: call finish "
                                     "with the patch to submit (or null to keep the original weights) and your evidence.")
            self.checkpoint()
        if self.toolkit.finished is not None:
            self.status = "finished"
        self.checkpoint()
        return self._finalize()

    def _finalize(self) -> SessionResult:
        fin = self.toolkit.finished or {}
        name = fin.get("patch")
        patch = self.toolkit.patches.get(name) if name else None
        files: dict[str, str] = {"trajectory": str(self.workspace / "trajectory.jsonl")}
        if patch is not None:
            patch.provenance.editor = f"agent:{self.editor.name}"
            patch.provenance.access_level = f"weights+{self.probe.budget.used}fwd"
            path = self.workspace / f"final{'.wspec' if patch.payloads else '.json'}"
            patch.save(path)
            files["patch"] = str(path)
        holdout = None
        if self.holdout:
            saved = self.probe.budget
            self.probe.budget = Budget(forward=None)  # the sealed evaluation does not spend the editor's budget
            try:
                holdout = evaluate(self.probe, patch, self.holdout, self.grader, n=self.holdout_n,
                                   **self.toolkit.generate_defaults) if patch is not None else None
            finally:
                self.probe.budget = saved
            if holdout is not None:
                (self.workspace / "holdout.json").write_text(json.dumps(holdout.as_dict(), indent=2, default=str))
                files["holdout"] = str(self.workspace / "holdout.json")
        result = SessionResult(self.status, patch, fin.get("explanation", ""), self.steps,
                               self.probe.budget.status(), self.usage, holdout, self.workspace, files)
        (self.workspace / "result.txt").write_text(str(result), encoding="utf-8")
        self._log({"type": "end", "status": self.status, "patch": name,
                   "holdout": holdout.paired.as_dict() if holdout else None})
        return result
