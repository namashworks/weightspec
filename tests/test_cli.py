import json

import numpy as np

import weightspec as ws
from weightspec.cli import main
from weightspec.io.safetensors import write_safetensors
from weightspec.numerics import encode

from .conftest import tiny_arrays, tiny_config


def _model_dir(tmp_path):
    d = tmp_path / "model"
    d.mkdir()
    write_safetensors(d / "model.safetensors", {k: (encode(v, "BF16"), "BF16") for k, v in tiny_arrays().items()})
    (d / "config.json").write_text(json.dumps(tiny_config()))
    return d


def test_cli_end_to_end(tmp_path, capsys):
    d = _model_dir(tmp_path)
    assert main(["inspect", str(d), "--filter", "down_proj"]) == 0
    assert "layers.0.mlp.down_proj" in capsys.readouterr().out
    assert main(["resolve", str(d), "layer[1].attn.head[0].out"]) == 0
    assert "o_proj.weight[:, 0:4]" in capsys.readouterr().out

    # A hand-written draft patch: no result yet.
    draft = {
        "name": "hand-written",
        "base": {"source": f"file://{d.as_posix()}"},
        "ops": [
            {"op": "add", "target": "layer[0].mlp.down.col[3]", "alpha": -1.0,
             "u": {"fn": "normalize", "of": {"fn": "sum", "coefs": [1, -1],
                   "terms": [{"fn": "readout", "ids": [1, 2]}, {"fn": "readout", "ids": [3]}]}}},
            {"op": "set", "target": "layer[1].attn.head[1].out", "value": 0.0},
        ],
    }
    (tmp_path / "p.json").write_text(json.dumps(draft))
    assert main(["build", str(tmp_path / "p.json"), "--policy", "paper"]) == 0
    built = ws.Patch.load(tmp_path / "p.json")
    assert built.result is not None and built.result.changed > 0
    assert main(["verify", str(tmp_path / "p.json")]) == 0
    assert "OK" in capsys.readouterr().out
    assert main(["show", str(tmp_path / "p.json")]) == 0
    assert main(["apply", str(tmp_path / "p.json"), "--out", str(tmp_path / "edited")]) == 0
    edited = ws.load(tmp_path / "edited")
    assert np.all(edited.load("model.layers.1.self_attn.o_proj.weight")[:, 4:] == 0)
    assert main(["diff", str(d), str(tmp_path / "edited"), "--out", str(tmp_path / "d.wspec")]) == 0
    assert main(["verify", str(tmp_path / "d.wspec"), "--base", str(d)]) == 0


def test_cli_verify_fails_on_tampering(tmp_path, capsys):
    d = _model_dir(tmp_path)
    p = ws.Patch.new(ws.load(d), "t").scale("layer[0].mlp.down.col[0]", 2.0)
    p.build()
    p.ops[0].factor = 2.5
    p.save(tmp_path / "t.json")
    assert main(["verify", str(tmp_path / "t.json")]) == 1


def test_cli_stats_and_schema(tmp_path, capsys):
    (tmp_path / "b.jsonl").write_text("\n".join(json.dumps({"id": str(i), "correct": 0}) for i in range(20)))
    (tmp_path / "e.jsonl").write_text("\n".join(json.dumps({"id": str(i), "correct": int(i < 15)}) for i in range(20)))
    assert main(["stats", str(tmp_path / "b.jsonl"), str(tmp_path / "e.jsonl"), "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["verdict"] == "confirmed" and abs(out["delta"] - 0.75) < 1e-9
    assert main(["schema", "--out", str(tmp_path / "s.json")]) == 0
    assert "Patch" in json.loads((tmp_path / "s.json").read_text())["title"]


def test_bad_input_is_a_clean_error(tmp_path, capsys):
    assert main(["show", str(tmp_path / "missing.wspec")]) == 2
    assert "error:" in capsys.readouterr().err
