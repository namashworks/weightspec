"""Agents for weight-space programming: a shared toolkit, LLM editors and checkpointed sessions."""

from weightspec.agent.editors import AnthropicEditor, ScriptedEditor, make_editor
from weightspec.agent.session import Session, SessionResult
from weightspec.agent.toolkit import ToolFailure, Toolkit

__all__ = ["AnthropicEditor", "ScriptedEditor", "Session", "SessionResult", "ToolFailure", "Toolkit", "make_editor"]
