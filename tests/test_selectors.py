import numpy as np
import pytest

import weightspec as ws
from weightspec.selectors import SelectorError, resolve


def test_arch_detection(store, gated_store):
    a = store.arch
    assert a.layer_prefix == "model.layers.{i}."
    assert a.num_layers == 2 and a.head_dim == 4 and a.norm_offset == 0.0 and not a.gated_query
    assert a.embed == "model.embed_tokens.weight" and a.unembed is None
    g = gated_store.arch
    assert g.gated_query and g.norm_offset == 1.0


def test_mlp_column(store):
    t = resolve(store, "layer[1].mlp.down.col[5]")
    assert t.tensor == "model.layers.1.mlp.down_proj.weight"
    assert t.block_shape == (8, 1) and t.cols.tolist() == [5] and t.rows is None


def test_head_out_block(store):
    t = resolve(store, "layer[0].attn.head[1].out")
    assert t.tensor.endswith("o_proj.weight") and t.cols.tolist() == [4, 5, 6, 7]


def test_head_out_composes_with_accessor(store):
    t = resolve(store, "layer[0].attn.head[1].out.col[2]")
    assert t.cols.tolist() == [6]


def test_gated_query_rows(gated_store):
    q = resolve(gated_store, "layer[0].attn.head[1].q")
    g = resolve(gated_store, "layer[0].attn.head[1].gate")
    assert q.rows.tolist() == [8, 9, 10, 11] and g.rows.tolist() == [12, 13, 14, 15]


def test_ungated_has_no_gate(store):
    with pytest.raises(SelectorError, match="gated"):
        resolve(store, "layer[0].attn.head[0].gate")


def test_ranges_lists_and_globals(store):
    assert resolve(store, "layer[0].mlp.down.cols[2:5]").cols.tolist() == [2, 3, 4]
    assert resolve(store, "layer[0].mlp.down.col[1,3,7]").cols.tolist() == [1, 3, 7]
    assert resolve(store, "final_norm").tensor == "model.norm.weight"
    assert resolve(store, "unembed").tensor == "model.embed_tokens.weight"  # tied
    assert resolve(store, "tensor['model.norm.weight'].row[0:2]").rows.tolist() == [0, 1]


@pytest.mark.parametrize(
    "bad",
    ["layer[9].mlp.down", "layer[0].mlp.down.col[99]", "layer[0].attn.head[5].out", "nonsense",
     "layer[0].mlp.down.col[3:1]", "tensor['missing.weight']", "final_norm.col[0]"],
)
def test_errors(store, bad):
    with pytest.raises((SelectorError, IndexError, KeyError)):
        resolve(store, bad)


def test_get_set_roundtrip(store):
    t = resolve(store, "layer[0].attn.head[0].out")
    w = store.load(t.tensor)
    block = t.get(w)
    t.set(w, block * 0)
    assert np.all(w[:, 0:4] == 0) and np.any(w[:, 4:] != 0)


def test_plain_arch_only_takes_raw_tensors():
    s = ws.WeightStore.from_arrays({"fc1.weight": np.ones((3, 4))})
    assert s.arch.name == "plain"
    assert resolve(s, "tensor['fc1.weight'].col[0]").block_shape == (3, 1)
    with pytest.raises(ValueError):
        resolve(s, "layer[0].mlp.down")
