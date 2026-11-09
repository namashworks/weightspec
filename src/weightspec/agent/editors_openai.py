"""Editor for any server that speaks the OpenAI chat-completions protocol.

Use it to drive weight-space programming with local or open models (vLLM, Ollama,
LM Studio, llama.cpp server), OpenRouter, or OpenAI:

    OpenAICompatibleEditor("Qwen/Qwen3-32B", base_url="http://localhost:8000/v1")

Kept in its own module, separate from the Anthropic editor.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from weightspec.agent.editors import EditorTurn, ToolCall, ToolResult
from weightspec.agent.toolkit import ToolSpec

__all__ = ["OpenAICompatibleEditor"]


class OpenAICompatibleEditor:
    def __init__(self, model: str, base_url: str | None = None, api_key: str | None = None,
                 max_tokens: int = 16000, client: Any = None):
        self.name = f"openai:{model}" + (f"@{base_url}" if base_url else "")
        self.model = model
        self.max_tokens = max_tokens
        self._client = client
        self.base_url = base_url
        self.api_key = api_key
        self.messages: list[dict[str, Any]] = []

    @property
    def client(self) -> Any:
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as e:  # pragma: no cover
                raise ImportError("pip install openai") from e
            # Local servers usually ignore the key, but the client requires one.
            key = self.api_key or ("not-needed" if self.base_url else None)
            self._client = OpenAI(base_url=self.base_url, api_key=key)
        return self._client

    def start(self, system: str, task: str) -> None:
        self.messages = [{"role": "system", "content": system}, {"role": "user", "content": task}]

    def step(self, tools: Sequence[ToolSpec]) -> EditorTurn:
        resp = self.client.chat.completions.create(
            model=self.model, messages=self.messages, max_tokens=self.max_tokens,
            tools=[{"type": "function", "function": {"name": t.name, "description": t.description,
                                                     "parameters": t.input_schema()}} for t in tools],
        )
        choice = resp.choices[0]
        m = choice.message
        calls = []
        for tc in m.tool_calls or []:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {"__invalid_json__": tc.function.arguments}
            calls.append(ToolCall(tc.id, tc.function.name, args))
        self.messages.append({
            "role": "assistant", "content": m.content or "",
            **({"tool_calls": [{"id": tc.id, "type": "function",
                                "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                               for tc in m.tool_calls]} if m.tool_calls else {}),
        })
        stop = {"tool_calls": "tool_use", "stop": "end_turn", "length": "max_tokens"}.get(
            choice.finish_reason or "", choice.finish_reason or "end_turn")
        if calls and stop != "max_tokens":
            stop = "tool_use"
        usage = {}
        if getattr(resp, "usage", None):
            usage = {"input_tokens": resp.usage.prompt_tokens or 0, "output_tokens": resp.usage.completion_tokens or 0}
        return EditorTurn(text=m.content or "", calls=calls, stop=stop, usage=usage)

    def add_results(self, results: Sequence[ToolResult]) -> None:
        for r in results:
            content = ("ERROR: " + r.content) if r.is_error else r.content
            self.messages.append({"role": "tool", "tool_call_id": r.id, "content": content})

    def add_user(self, text: str) -> None:
        self.messages.append({"role": "user", "content": text})

    def state(self) -> dict[str, Any]:
        return {"editor": self.name, "messages": self.messages}

    def load_state(self, state: dict[str, Any]) -> None:
        self.messages = state["messages"]
