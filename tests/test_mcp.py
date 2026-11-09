"""End-to-end test of the MCP server through the in-process MCP client."""

import json

import pytest

pytest.importorskip("mcp")
from mcp import Client

import weightspec as ws
from weightspec.io.safetensors import write_safetensors
from weightspec.mcp_server import build_server
from weightspec.numerics import encode

from .conftest import tiny_arrays, tiny_config

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


def _data(result):
    assert not result.is_error, result.content
    if result.structured_content is not None:
        return result.structured_content
    return json.loads(result.content[0].text)


async def test_agent_workflow(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    write_safetensors(model / "model.safetensors", {k: (encode(v, "BF16"), "BF16") for k, v in tiny_arrays().items()})
    (model / "config.json").write_text(json.dumps(tiny_config()))
    server = build_server(tmp_path)

    async with Client(server) as c:
        names = {t.name for t in (await c.list_tools()).tools}
        assert {"inspect_model", "add_op", "build_patch", "make_controls", "verify_patch", "readout_tokens"} <= names

        info = _data(await c.call_tool("inspect_model", {"source": "model", "tensor_filter": "down_proj"}))
        assert info["layers"] == 2 and len(info["matching_tensors"]) == 2

        r = _data(await c.call_tool("resolve_selector", {"source": "model", "selector": "layer[1].attn.head[1].out"}))
        assert r["block_shape"] == [8, 4]

        lens = _data(await c.call_tool("readout_tokens", {"source": "model", "selector": "layer[0].mlp.down.col[3]", "k": 3}))
        assert len(lens["promotes"]) == 3 and lens["promotes"][0]["score"] >= lens["suppresses"][0]["score"]

        await c.call_tool("create_patch", {"name": "fix", "source": "model"})
        op = {"op": "add", "target": "layer[0].mlp.down.col[3]", "alpha": -1.0,
              "u": {"fn": "normalize", "of": {"fn": "sum", "coefs": [1, -1],
                    "terms": [{"fn": "readout", "ids": [1, 2]}, {"fn": "readout", "ids": [5]}]}}}
        _data(await c.call_tool("add_op", {"patch": "fix", "op_json": json.dumps(op)}))
        built = _data(await c.call_tool("build_patch", {"patch": "fix"}))
        assert built["changed"] > 0

        ctrl = _data(await c.call_tool("make_controls", {"patch": "fix"}))
        assert "random-0" in ctrl["controls"] and "ablate" in ctrl["controls"]

        saved = _data(await c.call_tool("save_patch", {"patch": "fix"}))
        assert saved["saved"].endswith("fix.json")
        assert ws.Patch.load(saved["saved"]).verify(ws.load(model)).ok

        loaded = _data(await c.call_tool("load_patch", {"uri": "fix.json"}))
        assert _data(await c.call_tool("verify_patch", {"patch": loaded["loaded"]}))["ok"]

        # Policy is enforced: the final norm is frozen under the default 'paper' policy.
        await c.call_tool("create_patch", {"name": "bad", "source": "model"})
        await c.call_tool("add_op", {"patch": "bad", "op_json": json.dumps({"op": "scale", "target": "final_norm", "factor": 2})})
        res = await c.call_tool("build_patch", {"patch": "bad"})
        assert res.is_error and "frozen" in res.content[0].text

        # Writes and file reads are confined to the workspace.
        res = await c.call_tool("save_patch", {"patch": "fix", "filename": "../escape.json"})
        assert res.is_error and "outside the workspace" in res.content[0].text
        res = await c.call_tool("load_patch", {"uri": str(tmp_path.parent / "elsewhere.json")})
        assert res.is_error

        (tmp_path / "b.jsonl").write_text("\n".join(json.dumps({"id": i, "correct": 0}) for i in range(10)))
        (tmp_path / "e.jsonl").write_text("\n".join(json.dumps({"id": i, "correct": 1}) for i in range(10)))
        st = _data(await c.call_tool("paired_stats", {"base_file": "b.jsonl", "edited_file": "e.jsonl"}))
        assert st["verdict"] == "confirmed"

        prompts = (await c.list_prompts()).prompts
        assert prompts[0].name == "weight_space_edit"
