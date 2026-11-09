import json

import numpy as np
import pytest

import weightspec as ws
from weightspec.numerics import decode

DOWN0 = "model.layers.0.mlp.down_proj.weight"
O1 = "model.layers.1.self_attn.o_proj.weight"


# ----------------------------------------------------------------- directions
def test_readout_uses_norm_gain_and_offset(store, gated_store):
    for s, offset in ((store, 0.0), (gated_store, 1.0)):
        r = ws.readout(s, [3, 5]).numpy()
        e = s.load("model.embed_tokens.weight")[[3, 5]].astype(np.float64)
        g = s.load("model.norm.weight").astype(np.float64) + offset
        np.testing.assert_allclose(r, (e * g).mean(0), rtol=1e-6)


def test_direction_algebra(store):
    a, b = ws.readout(store, [1, 2]), ws.embedding(store, [3])
    u = (a - b).normalize()
    assert abs(u.norm - 1.0) < 1e-6
    np.testing.assert_allclose((2 * a - a).numpy(), a.numpy(), rtol=1e-6)
    off = u.project_off([b])
    assert abs(float(off.numpy() @ b.numpy())) < 1e-4
    ctrl = u.random_orthogonal(seed=7)
    assert abs(float(ctrl.numpy() @ u.numpy())) < 1e-6 and abs(ctrl.norm - 1.0) < 1e-5
    assert np.array_equal(ctrl.numpy(), u.random_orthogonal(seed=7).numpy())


def test_weights_direction_and_vector(store):
    w = ws.weights(store, "layer[0].mlp.down.col[3]").numpy()
    np.testing.assert_array_equal(w, store.load(DOWN0)[:, 3])
    v = ws.vector(np.arange(8.0))
    assert v.numpy(store).tolist() == list(range(8))
    with pytest.raises(ValueError, match="reduce"):
        ws.weights(store, "layer[0].attn.head[0].out").numpy()


def test_directions_serialize(store):
    u = (ws.readout(store, [1]) - ws.readout(store, [2])).normalize().project_off([ws.embedding(store, [4, 5], reduce="none")])
    d = json.loads(u.to_json())
    assert d["fn"] == "project_off" and d["of"]["fn"] == "normalize"


# ----------------------------------------------------------------- ops
def test_add_column_write(store):
    u = np.arange(8, dtype=np.float32)
    p = ws.Patch.new(store, "t").add("layer[0].mlp.down.col[2]", u, alpha=-0.5)
    applied = p.build()
    w0, w1 = store.load(DOWN0), applied.numpy(DOWN0)
    expected = decode(ws.numerics.encode(w0[:, 2] - 0.5 * u, "BF16"), "BF16")
    np.testing.assert_array_equal(w1[:, 2], expected)
    assert np.array_equal(np.delete(w1, 2, axis=1), np.delete(w0, 2, axis=1))


def test_scale_zero_set_rank1_delta(store):
    p = (
        ws.Patch.new(store, "t")
        .scale("layer[1].attn.head[0].out", 0.5)
        .zero("layer[1].attn.head[1].out")
        .rank1("layer[0].mlp.down.cols[0:2]", np.ones(8), np.array([1.0, 2.0]), alpha=0.25)
        .delta("layer[0].mlp.down.col[5]", np.full((8, 1), 2.0))
        .set("tensor['model.norm.weight'].row[0]", 3.0)
    )
    a = p.build()
    o0, o1 = store.load(O1), a.numpy(O1)
    np.testing.assert_array_equal(o1[:, 0:4], decode(ws.numerics.encode(o0[:, 0:4] * 0.5, "BF16"), "BF16"))
    assert np.all(o1[:, 4:8] == 0)
    d0, d1 = store.load(DOWN0), a.numpy(DOWN0)
    np.testing.assert_allclose(d1[:, 1] - d0[:, 1], 0.5, atol=0.05)
    np.testing.assert_allclose(d1[:, 5] - d0[:, 5], 2.0, atol=0.05)
    assert a.numpy("model.norm.weight")[0] == 3.0


def test_project_scale_semantics(store):
    u = np.random.default_rng(1).standard_normal(8)
    uh = u / np.linalg.norm(u)
    store32 = ws.WeightStore.from_arrays({DOWN0: store.load(DOWN0)}, dtype="F32")
    a = ws.Patch.new(store32, "t").project_scale(f"tensor['{DOWN0}'].cols[0:4]", u, k=7).apply()
    b0, b1 = store32.load(DOWN0)[:, :4], a.numpy(DOWN0)[:, :4]
    np.testing.assert_allclose(uh @ b1, -6 * (uh @ b0), rtol=1e-4, atol=1e-5)  # component times (1 - k)
    perp = lambda m: m - np.outer(uh, uh @ m)  # noqa: E731
    np.testing.assert_allclose(perp(b1), perp(b0), atol=1e-5)  # orthogonal part preserved


def test_bad_shapes_raise(store):
    with pytest.raises(ValueError, match="does not fit"):
        ws.Patch.new(store, "t").add("layer[0].mlp.down.col[0]", np.ones(5)).apply()
    with pytest.raises(ValueError):
        ws.Patch.new(store, "t").rank1("layer[0].mlp.down", np.ones(3), np.ones(3)).apply()


# ----------------------------------------------------------------- policy
def test_paper_policy(store):
    with pytest.raises(ws.PolicyViolation, match="frozen"):
        ws.Patch.new(store, "t").scale("final_norm", 2.0).apply(policy=ws.Policy.paper())
    with pytest.raises(ws.PolicyViolation, match="whole tensor"):
        ws.Patch.new(store, "t").scale("layer[0].mlp.down", 2.0).apply(policy=ws.Policy.paper())
    ws.Patch.new(store, "t").scale("layer[0].mlp.down.col[0]", 2.0).apply(policy=ws.Policy.paper())


def test_policy_limits(store):
    p = ws.Patch.new(store, "t").scale("layer[0].mlp.down.cols[0:3]", 2.0)
    with pytest.raises(ws.PolicyViolation, match="changes"):
        p.apply(policy=ws.Policy(max_changed=10))
    with pytest.raises(ws.PolicyViolation, match="non-finite"):
        ws.Patch.new(store, "t").scale("layer[0].mlp.down.col[0]", float("inf")).apply()


# ----------------------------------------------------------------- build / io / verify
@pytest.mark.parametrize("suffix", [".json", ".wspec"])
def test_save_load_verify_roundtrip(store, tmp_path, suffix):
    u = (ws.readout(store, [1, 2]) - ws.readout(store, [3])).normalize()
    p = ws.Patch.new(store, "roundtrip", editor="human").add("layer[0].mlp.down.col[4]", u, alpha=-1)
    if suffix == ".wspec":
        p.rank1("layer[1].attn.head[0].out", np.ones(8), np.arange(4.0), alpha=0.1)
    p.build()
    path = p.save(tmp_path / f"p{suffix}")
    q = ws.Patch.load(path)
    assert q.result == p.result and q.ops == p.ops
    report = q.verify(store)
    assert report.ok, str(report)


def test_json_refuses_payloads(store, tmp_path):
    p = ws.Patch.new(store, "t").add("layer[0].mlp.down.col[0]", np.ones(8))
    p.build()
    with pytest.raises(ValueError, match="wspec"):
        p.save(tmp_path / "p.json")


def test_wspec_files_are_deterministic(store, tmp_path):
    def make(path):
        p = ws.Patch.new(store, "t").add("layer[0].mlp.down.col[0]", np.ones(8))
        p.build()
        p.provenance.created = "2026-01-01T00:00:00+00:00"
        return p.save(path).read_bytes()

    assert make(tmp_path / "a.wspec") == make(tmp_path / "b.wspec")


def test_verify_detects_other_base_and_tampering(store, tmp_path):
    p = ws.Patch.new(store, "t").scale("layer[0].mlp.down.col[1]", 3.0)
    p.build()
    other = ws.WeightStore.from_arrays(
        {**{n: store.load(n) for n in store.names}, DOWN0: store.load(DOWN0) + 1.0}, dtype="BF16",
        config=store.config,
    )
    bad = p.verify(other)
    assert not bad.ok and any("differs" in x for x in bad.problems)
    p.ops[0].factor = 3.5
    tampered = p.verify(store)
    assert not tampered.ok and any("hash" in x for x in tampered.problems)


def test_verify_requires_build(store):
    with pytest.raises(ValueError, match="build"):
        ws.Patch.new(store, "t").scale("layer[0].mlp.down.col[1]", 3.0).verify(store)


def test_bf16_rounding_can_hide_tiny_edits(store):
    p = ws.Patch.new(store, "t").add("layer[0].mlp.down.col[0]", np.full(8, 1e-7))
    assert p.apply().total_changed == 0


# ----------------------------------------------------------------- stacking, diff, export
def test_overlay_stacking(store):
    a = ws.Patch.new(store, "a").scale("layer[0].mlp.down.col[0]", 2.0).apply()
    b = ws.Patch.new(a.store, "b").scale("layer[0].mlp.down.col[0]", 0.5).apply()
    np.testing.assert_array_equal(b.numpy(DOWN0)[:, 0], store.load(DOWN0)[:, 0])
    assert np.array_equal(a.store.load_original(DOWN0), store.load(DOWN0))


def test_diff_reproduces_target_exactly(store):
    edited = ws.Patch.new(store, "e").scale("layer[0].mlp.down.col[3,7]", -1.0).zero("layer[1].attn.head[1].out").apply()
    target = edited.store
    d = ws.diff(store, target, name="d")
    applied = d.build(store)
    for name, raw in edited.tensors.items():
        assert np.array_equal(applied.tensors[name], raw)
    assert set(applied.tensors) == set(edited.tensors)
    assert d.verify(store).ok


def test_export_and_reload(store, tmp_path):
    a = ws.Patch.new(store, "exp").scale("layer[0].mlp.down.col[0]", 2.0).apply()
    out = a.export(tmp_path / "edited")
    reloaded = ws.load(out)
    assert np.array_equal(reloaded.raw(DOWN0), a.tensors[DOWN0])
    assert np.array_equal(reloaded.raw(O1), store.raw(O1))
    assert (out / "exp.wspec").exists()
    only = a.save_safetensors(tmp_path / "changed.safetensors")
    assert ws.load(only).names == [DOWN0]


def test_summary_mentions_result(store):
    p = ws.Patch.new(store, "s").scale("layer[0].mlp.down.col[0]", 2.0)
    p.build()
    assert "stored values changed" in p.summary()


def test_torch_module_in_place(tmp_path):
    torch = pytest.importorskip("torch")
    net = torch.nn.Sequential(torch.nn.Linear(4, 3), torch.nn.ReLU(), torch.nn.Linear(3, 2)).to(torch.bfloat16)
    before = net[0].weight.detach().clone()
    store = ws.load(net)
    p = ws.Patch.new(store, "t").scale("tensor['0.weight'].row[1]", -1.0)
    p.build()
    applied = ws.apply(net, p)
    assert applied.total_changed > 0
    assert torch.equal(net[0].weight[1], -before[1]) and torch.equal(net[0].weight[0], before[0])
    p.save(tmp_path / "t.json")
    net2 = torch.nn.Sequential(torch.nn.Linear(4, 3), torch.nn.ReLU(), torch.nn.Linear(3, 2)).to(torch.bfloat16)
    net2.load_state_dict({k: v for k, v in zip(net2.state_dict(), [before, *list(net.state_dict().values())[1:]], strict=True)})
    ws.apply(net2, str(tmp_path / "t.json"))
    assert torch.equal(net2[0].weight, net[0].weight)


# ----------------------------------------------------------------- controls
def test_controls_family(store):
    u = (ws.readout(store, [1, 2]) - ws.readout(store, [3])).normalize()
    p = ws.Patch.new(store, "edit", cites=["x"]).add("layer[0].mlp.down.col[4]", u, alpha=-1.0)
    p.build()
    ctrl = ws.controls(p, doses=(0.5, 2.0), seeds=(0, 1))
    assert set(ctrl) == {"dose-0.5", "dose-2", "reverse", "random-0", "random-1", "ablate"}
    assert ctrl["dose-2"].ops[0].alpha == -2.0 and ctrl["reverse"].ops[0].alpha == 0.5
    assert p.ops[0].alpha == -1.0  # original untouched
    orig_delta = p.apply().numpy(DOWN0)[:, 4] - store.load(DOWN0)[:, 4]
    rand = ctrl["random-0"].apply()
    rand_delta = rand.numpy(DOWN0)[:, 4] - store.load(DOWN0)[:, 4]
    assert abs(np.linalg.norm(rand_delta) - np.linalg.norm(orig_delta)) < 0.05
    assert abs(float(rand_delta @ orig_delta)) < 0.05
    assert np.all(ctrl["ablate"].apply().numpy(DOWN0)[:, 4] == 0)
    for c in ctrl.values():
        c.build()
        assert c.verify(store).ok
    with pytest.raises(ValueError):
        ws.controls(ws.Patch.new(store, "s").scale("layer[0].mlp.down.col[0]", 2.0))


# ----------------------------------------------------------------- sharing
def test_bytes_roundtrip_and_version_check(store):
    p = ws.Patch.new(store, "b").add("layer[0].mlp.down.col[0]", np.ones(8))
    p.build()
    data = p.to_bytes()
    assert data[:2] == b"PK"
    q = ws.Patch.from_bytes(data)
    assert q.verify(store).ok
    bad = json.loads(ws.Patch.new(store, "v").scale("layer[0].mlp.down.col[0]", 2.0).to_json())
    bad["wspec"] = "9.0"
    with pytest.raises(ValueError, match="unsupported wspec"):
        ws.Patch.from_bytes(json.dumps(bad).encode())


def test_load_from_hub_uri(store, monkeypatch):
    import weightspec.io.hub as hub

    p = ws.Patch.new(store, "h").scale("layer[0].mlp.down.col[0]", 2.0)
    p.build()
    seen = {}

    class Resp:
        def __init__(self, data):
            self.data = data

        def read(self, n=-1):
            return self.data

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake_urlopen(req, timeout=None):
        seen["url"] = req.full_url
        return Resp(p.to_bytes())

    monkeypatch.setattr(hub.urllib.request, "urlopen", fake_urlopen)
    q = ws.Patch.load("hf://someone/patches/fixes/h.json@abc123")
    assert seen["url"].endswith("/someone/patches/resolve/abc123/fixes/h.json")
    assert q.result == p.result
    ws.Patch.load("hf://datasets/someone/patches/h.json")
    assert "/datasets/someone/patches/resolve/main/h.json" in seen["url"]
    with pytest.raises(ValueError):
        hub.parse_hf_uri("hf://only-owner/file")


def test_push_to_hub_uses_hfapi(store, monkeypatch):
    hfh = pytest.importorskip("huggingface_hub")
    calls = []

    class FakeApi:
        def __init__(self, token=None):
            pass

        def create_repo(self, repo_id, **kw):
            calls.append(("create", repo_id, kw))

        def upload_file(self, **kw):
            calls.append(("upload", kw["repo_id"], kw["path_in_repo"], kw["path_or_fileobj"][:1]))

    monkeypatch.setattr(hfh, "HfApi", FakeApi)
    p = ws.Patch.new(store, "pushed").scale("layer[0].mlp.down.col[0]", 2.0)
    with pytest.raises(ValueError, match="build"):
        p.push_to_hub("me/patches")
    p.build()
    assert p.push_to_hub("me/patches") == "hf://me/patches/pushed.json"
    assert calls[1] == ("upload", "me/patches", "pushed.json", b"{")
