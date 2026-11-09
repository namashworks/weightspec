"""Probe layer on a tiny random Llama (offline, CPU)."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

import weightspec as ws  # noqa: E402
from weightspec.probe import Budget, BudgetExceeded, Probe  # noqa: E402


def char_tokenizer():
    from tokenizers import Regex, Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    chars = [chr(c) for c in range(32, 127)]
    vocab = {"<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3, **{c: i + 4 for i, c in enumerate(chars)}}
    tok = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Split(Regex("."), behavior="isolated")
    from tokenizers import decoders

    tok.decoder = decoders.Fuse()
    return PreTrainedTokenizerFast(tokenizer_object=tok, bos_token="<s>", eos_token="</s>",
                                   pad_token="<pad>", unk_token="<unk>")


@pytest.fixture(scope="module")
def tiny():
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=100, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=256,
                      bos_token_id=1, eos_token_id=2, pad_token_id=0, tie_word_embeddings=False)
    model = LlamaForCausalLM(cfg).to(torch.float32).eval()
    return model, char_tokenizer()


def test_store_reads_live_module_lazily(tiny):
    model, tok = tiny
    probe = Probe.from_model(model, tok, budget=None)
    w = probe.store.load("model.layers.0.mlp.down_proj.weight")
    np.testing.assert_array_equal(w, model.model.layers[0].mlp.down_proj.weight.detach().numpy())
    assert probe.store.arch.num_layers == 2 and probe.store.arch.head_dim == 8


def test_hot_swap_applies_and_restores_exactly(tiny):
    model, tok = tiny
    probe = Probe.from_model(model, tok, budget=None)
    before = model.model.layers[1].mlp.down_proj.weight.detach().clone()
    patch = ws.Patch.new(probe.store, "t").scale("layer[1].mlp.down.cols[0:16]", -3.0)
    probe.set_patch(patch)
    text = ["hello world, this is a test"]
    base = probe.teacher_forced(text, "original")[0]["token_logprobs"]
    edited = probe.teacher_forced(text, "edited")[0]["token_logprobs"]
    assert not np.allclose(base, edited)
    assert torch.equal(model.model.layers[1].mlp.down_proj.weight, before)  # restored
    again = probe.teacher_forced(text, "original")[0]["token_logprobs"]
    np.testing.assert_array_equal(base, again)


def test_restore_happens_even_on_error(tiny):
    model, tok = tiny
    probe = Probe.from_model(model, tok, budget=None)
    before = model.model.layers[0].mlp.down_proj.weight.detach().clone()
    probe.set_patch(ws.Patch.new(probe.store, "t").zero("layer[0].mlp.down"))
    with pytest.raises(ZeroDivisionError), probe.backend.weights("edited"):
        assert torch.all(model.model.layers[0].mlp.down_proj.weight == 0)  # swapped in
        _ = 1 / 0  # a crash mid-request
    assert torch.equal(model.model.layers[0].mlp.down_proj.weight, before)
    with pytest.raises(IndexError):
        probe.teacher_forced(["abc"], "edited", mlp_layers=[7])  # bad layer: rejected, not charged


def test_generate_both_arms_share_seeds_and_charge_once(tiny):
    model, tok = tiny
    probe = Probe.from_model(model, tok, budget=3)
    probe.set_patch(ws.Patch.new(probe.store, "noop").scale("layer[0].mlp.down.col[0]", 1.0))
    res = probe.generate(["abc", "xyz"], weights="both", n=2, max_new_tokens=8, chat=False)
    assert probe.budget.used == 1
    assert res.texts("original") == res.texts("edited")  # identical weights + shared seeds
    s = res.summary()
    assert s["original"]["samples"] == 4 and 0 <= s["edited"]["truncated_share"] <= 1
    again = probe.generate(["abc", "xyz"], weights="original", n=2, max_new_tokens=8, chat=False)
    assert again.texts("original") == res.texts("original")  # deterministic per item


def test_budget_is_enforced_and_failures_are_free(tiny):
    model, tok = tiny
    probe = Probe.from_model(model, tok, budget=Budget(forward=1))
    with pytest.raises(RuntimeError):
        probe.generate(["a"], weights="edited", max_new_tokens=2, chat=False)  # no patch: fails, not charged
    assert probe.budget.used == 0
    probe.teacher_forced(["abc"])
    with pytest.raises(BudgetExceeded):
        probe.teacher_forced(["abc"])
    assert probe.budget.status() == {"used": 1, "limit": 1, "remaining": 0}


def test_teacher_forced_captures(tiny):
    model, tok = tiny
    probe = Probe.from_model(model, tok, budget=None)
    r = probe.teacher_forced(["abcdef"], hidden_layers=[0, 1], mlp_layers=[1], topk=3)[0]
    n = len(r["token_ids"])
    assert r["token_logprobs"].shape == (n - 1,) and np.all(r["token_logprobs"] <= 0)
    assert r["topk_ids"].shape == (n - 1, 3)
    assert r["hidden"][1].shape == (n, 32) and r["mlp"][1].shape == (n, 64)
    # mlp capture equals the down_proj input: down_proj(mlp_act) is the MLP output
    out = model.model.layers[1].mlp.down_proj(torch.from_numpy(r["mlp"][1])).detach().numpy()
    assert out.shape == (n, 32)


def test_patch_with_foreign_tensor_names_maps_by_suffix(tiny):
    model, tok = tiny
    probe = Probe.from_model(model, tok, budget=None)
    arrays = {"model.language_model.layers.1.mlp.down_proj.weight": probe.store.load("model.layers.1.mlp.down_proj.weight")}
    foreign = ws.WeightStore.from_arrays(arrays, dtype="F32")
    p = ws.Patch.new(foreign, "f").scale("tensor['model.language_model.layers.1.mlp.down_proj.weight'].col[3]", 0.0)
    p._store = foreign
    probe.set_patch(p)
    assert set(probe.backend._edits) == {"model.layers.1.mlp.down_proj.weight"}
