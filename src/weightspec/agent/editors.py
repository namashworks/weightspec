"""Editors: the LLM (or program) that decides which weight edits to make.

An editor turns the conversation so far into the next tool calls. WeightSpec's
session loop owns everything else (tools, budget, checkpoints, evaluation), so
any model can drive it:

* :class:`AnthropicEditor`: Claude via the Anthropic SDK (default ``claude-opus-5-5``).
* :class:`~weightspec.agent.editors_openai.OpenAICompatibleEditor`: any server that
  speaks the OpenAI chat-completions protocol (local vLLM / Ollama / OpenRouter / OpenAI).
* :class:`ScriptedEditor`: a fixed list of tool calls, for tests, demos and replays.

Editor specs for the CLI: ``anthropic:claude-opus-5-5``, ``openai:<model>``,
``openai:<model>@http://localhost:8000/v1``, ``scripted:calls.json``.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from weightspec.agent.toolkit import ToolSpec

__all__ = [
    "AnthropicEditor", "Editor", "EditorRefusal", "EditorTurn", "ScriptedEditor", "ToolCall", "ToolResult",
    "make_editor",
]


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ToolResult:
    id: str
    content: str
    is_error: bool = False


@dataclass
class EditorTurn:
    """One editor response. ``stop``: tool_use | end_turn | max_tokens | pause_turn | refusal."""

    text: str
    calls: list[ToolCall]
    stop: str
    usage: dict[str, int] = field(default_factory=dict)
    detail: str | None = None


class EditorRefusal(RuntimeError):
    pass


class Editor(Protocol):
    name: str

    def start(self, system: str, task: str) -> None: ...

    def step(self, tools: Sequence[ToolSpec]) -> EditorTurn: ...

    def add_results(self, results: Sequence[ToolResult]) -> None: ...

    def add_user(self, text: str) -> None: ...

    def state(self) -> dict[str, Any]: ...

    def load_state(self, state: dict[str, Any]) -> None: ...


# ---------------------------------------------------------------------- Anthropic
class AnthropicEditor:
    """Claude as the editor, through the Anthropic Python SDK.

    Defaults follow current guidance for long agentic work: ``claude-opus-5-5`` with
    adaptive thinking at ``effort='high'`` (Opus 5.5 defaults to medium), streaming
    with a large ``max_tokens``, automatic prompt caching of the growing prefix, and
    server-side refusal fallbacks. The history is append-only and every content
    block, thinking blocks included, is sent back exactly as returned.
    """

    def __init__(
        self,
        model: str = "claude-opus-5-5",
        effort: str = "high",
        max_tokens: int = 64000,
        fallbacks: bool = True,
        client: Any = None,
    ):
        self.name = f"anthropic:{model}"
        self.model = model
        self.effort = effort
        self.max_tokens = max_tokens
        self.fallbacks = fallbacks
        self._client = client
        self.system = ""
        self.messages: list[dict[str, Any]] = []

    @property
    def client(self) -> Any:
        if self._client is None:
            try:
                import anthropic
            except ImportError as e:  # pragma: no cover
                raise ImportError("pip install 'weightspec[anthropic]'") from e
            self._client = anthropic.Anthropic()
        return self._client

    def start(self, system: str, task: str) -> None:
        self.system = system
        self.messages = [{"role": "user", "content": task}]

    def _tools(self, tools: Sequence[ToolSpec]) -> list[dict[str, Any]]:
        # eager_input_streaming streams tool inputs as they are generated; inputs are then
        # validated by the toolkit (pydantic) before anything runs.
        return [{"name": t.name, "description": t.description, "input_schema": t.input_schema(),
                 "eager_input_streaming": True} for t in tools]

    def step(self, tools: Sequence[ToolSpec]) -> EditorTurn:
        params: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": self.system,
            "messages": self.messages,
            "tools": self._tools(tools),
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": self.effort},
            "cache_control": {"type": "ephemeral"},
        }
        if self.fallbacks:
            params["betas"] = ["server-side-fallback-2026-07-01"]
            params["fallbacks"] = "default"
        with self.client.beta.messages.stream(**params) as stream:
            msg = stream.get_final_message()
        content = [b.to_dict() for b in msg.content]
        self.messages.append({"role": "assistant", "content": content})
        u = msg.usage
        usage = {k: int(getattr(u, k, 0) or 0) for k in
                 ("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens")}
        text = "\n".join(b.get("text", "") for b in content if b.get("type") == "text")
        calls = [ToolCall(b["id"], b["name"], dict(b.get("input") or {})) for b in content if b.get("type") == "tool_use"]
        detail = None
        if msg.stop_reason == "refusal":
            sd = getattr(msg, "stop_details", None)
            detail = f"{getattr(sd, 'category', None)}: {getattr(sd, 'explanation', '')}" if sd else None
        return EditorTurn(text=text, calls=calls, stop=str(msg.stop_reason), usage=usage, detail=detail)

    def add_results(self, results: Sequence[ToolResult]) -> None:
        # All results of one assistant turn go back in a single user message.
        self.messages.append({"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": r.id, "content": r.content, "is_error": r.is_error}
            for r in results]})

    def add_user(self, text: str) -> None:
        self.messages.append({"role": "user", "content": text})

    def state(self) -> dict[str, Any]:
        return {"editor": self.name, "system": self.system, "messages": self.messages}

    def load_state(self, state: dict[str, Any]) -> None:
        self.system = state["system"]
        self.messages = state["messages"]


# ---------------------------------------------------------------------- scripted
class ScriptedEditor:
    """Replays a fixed sequence of turns. Each turn is a list of ``(tool_name, arguments)``.

    A callable turn receives the previous turn's results and returns the calls. After
    the script ends the editor stops (end_turn), which ends the session unless the
    script called ``finish``.
    """

    def __init__(self, turns: Sequence[Sequence[tuple[str, dict[str, Any]]] | Callable[[list[ToolResult]], Any]],
                 name: str = "scripted"):
        self.name = name
        self.turns = list(turns)
        self.i = 0
        self.last: list[ToolResult] = []
        self.log: list[Any] = []

    @classmethod
    def from_file(cls, path: str | Path) -> ScriptedEditor:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls([[(c["name"], c.get("arguments", {})) for c in turn] for turn in data], name=f"scripted:{path}")

    def start(self, system: str, task: str) -> None:
        self.log.append({"system": system, "task": task})

    def step(self, tools: Sequence[ToolSpec]) -> EditorTurn:
        if self.i >= len(self.turns):
            return EditorTurn(text="(script finished)", calls=[], stop="end_turn")
        turn = self.turns[self.i]
        calls = turn(self.last) if callable(turn) else turn
        out = [ToolCall(f"call_{self.i}_{k}", name, dict(args)) for k, (name, args) in enumerate(calls)]
        self.i += 1
        return EditorTurn(text="", calls=out, stop="tool_use")

    def add_results(self, results: Sequence[ToolResult]) -> None:
        self.last = list(results)
        self.log.append([r.__dict__ for r in results])

    def add_user(self, text: str) -> None:
        self.log.append({"user": text})

    def state(self) -> dict[str, Any]:
        return {"editor": self.name, "i": self.i}

    def load_state(self, state: dict[str, Any]) -> None:
        self.i = state.get("i", 0)


def make_editor(spec: str, **kwargs: Any) -> Any:
    """``anthropic:<model>`` | ``openai:<model>[@base_url]`` | ``scripted:<file.json>``."""
    kind, _, rest = spec.partition(":")
    if kind == "anthropic":
        return AnthropicEditor(model=rest or "claude-opus-5-5", **kwargs)
    if kind in ("openai", "openai-compat"):
        from weightspec.agent.editors_openai import OpenAICompatibleEditor

        model, _, base_url = rest.partition("@")
        return OpenAICompatibleEditor(model=model, base_url=base_url or None)
    if kind == "scripted":
        return ScriptedEditor.from_file(rest)
    raise ValueError(f"unknown editor {spec!r}; use anthropic:<model>, openai:<model>[@url] or scripted:<file>")
