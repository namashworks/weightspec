"""MCP server: let any coding agent do weight-space programming through WeightSpec.

    weightspec mcp --workspace ./wsp-work                         # weight tools only
    weightspec mcp --workspace ./wsp-work --probe Qwen/Qwen3-0.6B --dev dev.jsonl --budget 50

Add to Claude Code with::

    claude mcp add weightspec -- weightspec mcp --workspace ./wsp-work

The tools are those of :class:`~weightspec.agent.toolkit.Toolkit` (shared with
WeightSpec's own agent sessions). With ``--probe``, the budgeted forward-pass tools
(generate, teacher_forced, contrast_activations, evaluate_patch) are added.
"""

from __future__ import annotations

import functools
import os
from typing import Any

from weightspec._version import __version__
from weightspec.agent.toolkit import ToolFailure, Toolkit

__all__ = ["INSTRUCTIONS", "build_server", "main"]

INSTRUCTIONS = """\
WeightSpec edits neural-network weights directly (weight-space programming), with no gradient descent.

Workflow:
1. inspect_model to learn the architecture, then resolve_selector / tensor_stats / readout_tokens
   to understand components (what does MLP unit j or head h write into the residual stream?).
2. With a probe attached: diagnose with generate (graded dev items) and teacher_forced; localize with
   contrast_activations (it saves named vectors you can use as edit directions). Every forward pass
   costs budget (budget_status).
3. create_patch, then add_op once per edit. Ops and directions are JSON (call op_schema first).
   Selectors: layer[L].mlp.down.col[j], layer[L].attn.head[h].out, layer[L].attn.head[h].gate,
   embed, unembed, final_norm, tensor['exact.name'].row[i]. Indices are zero-based; ranges half-open.
4. build_patch (default policy 'paper': embeddings, LM head and final norm are frozen, and
   whole-tensor rescaling is refused). It reports exactly how many stored values change.
5. make_controls, then evaluate_patch with controls on fresh items. An edit is only credible if it
   beats the original and its controls on items that were not used to choose it.
6. save_patch to write a .json/.wspec file anyone can verify bit for bit with verify_patch.
Never claim an edit improves a model without a paired comparison on held-out data.
"""


def build_server(workspace: str | os.PathLike[str] = ".", toolkit: Toolkit | None = None) -> Any:
    """Create the MCP server (requires ``pip install 'weightspec[mcp]'``)."""
    try:
        from mcp.server.mcpserver import MCPServer
        from mcp.server.mcpserver.exceptions import ToolError
    except ImportError as e:  # pragma: no cover
        raise ImportError("pip install 'weightspec[mcp]' (needs mcp>=2.3)") from e

    kit = toolkit or Toolkit(workspace)
    server = MCPServer(
        name="weightspec", title="WeightSpec", version=__version__, instructions=INSTRUCTIONS,
        website_url="https://github.com/namashworks/weightspec",
    )

    for spec in kit.specs(include_session_only=False):

        def make(spec_name: str, fn: Any) -> Any:
            @functools.wraps(fn)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                try:
                    bound = dict(zip(kit._specs[spec_name].args_model.model_fields, args, strict=False))
                    return kit.call(spec_name, {**bound, **kwargs})
                except ToolFailure as e:
                    raise ToolError(str(e)) from e

            return wrapper

        server.tool(name=spec.name, description=spec.description)(make(spec.name, spec.fn))

    @server.prompt(description="Guide an agent through a careful, evidence-backed weight edit.")
    def weight_space_edit(model: str, goal: str) -> str:
        return (
            f"You will edit the weights of {model} directly to: {goal}.\n"
            "Rules: no gradients or training; do not edit embeddings, the LM head or the final norm; "
            "no whole-tensor rescaling.\n"
            "1. Diagnose: state the failure and competing explanations.\n"
            "2. Localize: use readout_tokens, tensor_stats and (with a probe) contrast_activations to find "
            "components whose write directions match the failure. Prefer the smallest structured edit.\n"
            "3. Edit: create_patch, add_op, build_patch. Report the exact changed-value count.\n"
            "4. Control: make_controls, then evaluate_patch with controls on fresh items. The edit is only "
            "credible if it beats the original and its controls.\n"
            "5. Ship: save_patch; say what you changed, the evidence, and what you did NOT verify."
        )

    return server


def main(workspace: str = ".", transport: str = "stdio", toolkit: Toolkit | None = None, **kwargs: Any) -> None:
    build_server(workspace, toolkit).run(transport=transport, **kwargs)
