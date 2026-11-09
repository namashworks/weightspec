from __future__ import annotations

import numpy as np
import pytest

import weightspec as ws

H, FF, HEADS, D, LAYERS, VOCAB = 8, 12, 2, 4, 2, 16


def tiny_arrays(seed: int = 0, gated: bool = False) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    arrays: dict[str, np.ndarray] = {
        "model.embed_tokens.weight": rng.standard_normal((VOCAB, H)),
        "model.norm.weight": rng.standard_normal(H) * 0.1,
    }
    q_rows = 2 * HEADS * D if gated else HEADS * D
    for i in range(LAYERS):
        p = f"model.layers.{i}."
        arrays[p + "mlp.down_proj.weight"] = rng.standard_normal((H, FF))
        arrays[p + "mlp.up_proj.weight"] = rng.standard_normal((FF, H))
        arrays[p + "mlp.gate_proj.weight"] = rng.standard_normal((FF, H))
        arrays[p + "self_attn.q_proj.weight"] = rng.standard_normal((q_rows, H))
        arrays[p + "self_attn.k_proj.weight"] = rng.standard_normal((HEADS * D, H))
        arrays[p + "self_attn.v_proj.weight"] = rng.standard_normal((HEADS * D, H))
        arrays[p + "self_attn.o_proj.weight"] = rng.standard_normal((H, HEADS * D))
        arrays[p + "input_layernorm.weight"] = np.ones(H)
    return {k: v.astype(np.float32) for k, v in arrays.items()}


def tiny_config(model_type: str = "llama") -> dict[str, object]:
    return {
        "model_type": model_type,
        "hidden_size": H,
        "num_attention_heads": HEADS,
        "num_key_value_heads": HEADS,
        "head_dim": D,
        "num_hidden_layers": LAYERS,
        "intermediate_size": FF,
        "vocab_size": VOCAB,
    }


@pytest.fixture
def store() -> ws.WeightStore:
    """A tiny Llama-style model stored in bfloat16."""
    return ws.WeightStore.from_arrays(tiny_arrays(), dtype="BF16", config=tiny_config())


@pytest.fixture
def gated_store() -> ws.WeightStore:
    """A tiny Qwen3.5-style model: gated queries, one-centered final norm."""
    return ws.WeightStore.from_arrays(tiny_arrays(gated=True), dtype="BF16", config=tiny_config("qwen3_5_text"))
