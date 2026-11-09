"""WeightSpec Space: verify weight patches and look inside model weights, in the browser.

Runs on a free CPU Space. Tensors are fetched from the Hugging Face Hub with HTTP
range reads, so nothing large is downloaded except when the readout lens needs
the embedding matrix (cached after the first call).
"""

from __future__ import annotations

import json
import tempfile
import traceback
from functools import lru_cache
from pathlib import Path

import gradio as gr
import numpy as np

import weightspec as ws

EXAMPLE_PATCHES = [
    "https://raw.githubusercontent.com/namashworks/weightspec/main/conformance/hu2026-math-primary.json",
    "https://raw.githubusercontent.com/namashworks/weightspec/main/conformance/takehi2026-exp8-math-p50.wspec",
]
LENS_MODELS = ["Qwen/Qwen3.5-0.8B", "RikiyaT/Qwen3.5-0.8B-weight-space-programming"]


@lru_cache(maxsize=4)
def _store(source: str) -> ws.WeightStore:
    return ws.load(source)


@lru_cache(maxsize=2)
def _readout_matrix(source: str) -> np.ndarray:
    st = _store(source)
    a = st.arch
    u = st.load(a.global_tensor("unembed")).astype(np.float32)
    if a.final_norm:
        u *= (st.load(a.final_norm) + a.norm_offset).astype(np.float32)
    return u


def verify_patch(uri: str, upload: str | None) -> tuple[str, str, dict]:
    try:
        patch = ws.Patch.load(upload) if upload else ws.Patch.load(uri.strip())
    except Exception as e:
        return f"### Could not load the patch\n`{e}`", "", {}
    try:
        report = patch.verify()
    except Exception as e:
        return f"### Verification error\n`{e}`", patch.summary(), {}
    if report.ok:
        verdict = (f"### ✅ Verified bit for bit\nRebuilt from `{patch.base.source}` at revision "
                   f"`{(patch.base.revision or '?')[:12]}`; **{report.changed:,}** stored values changed, "
                   "and every recorded SHA-256 matches.")
    else:
        verdict = "### ❌ Verification failed\n" + "\n".join(f"- {p}" for p in report.problems)
    manifest = json.loads(patch.to_json())
    return verdict, patch.summary(), manifest


def readout_lens(source: str, selector: str, k: int) -> tuple[str, list[list], list[list]]:
    try:
        st = _store(source)
        t = ws.resolve(st, selector)
        vec = t.get(st.load(t.tensor)).astype(np.float32)
        if 1 not in vec.shape:
            return f"`{selector}` is a {vec.shape} block: pick one column, e.g. `.col[507]`.", [], []
        scores = _readout_matrix(source) @ vec.reshape(-1)
        tok = st.tokenizer()

        def rows(ids: np.ndarray) -> list[list]:
            return [[int(i), tok.decode([int(i)]), round(float(scores[i]), 4)] for i in ids]

        head = (f"**{t.describe()}**: the tokens this direction pushes the model *towards* (left) "
                f"and *away from* (right), read through the unembedding and final norm.")
        return head, rows(np.argsort(-scores)[:k]), rows(np.argsort(scores)[:k])
    except Exception as e:
        return f"Error: `{e}`", [], []


def rebuild_paper_edit() -> tuple[str, str | None]:
    """Rebuild Hu et al.'s 1,023-value edit from its formula and offer the file."""
    try:
        qwen = _store("Qwen/Qwen3.5-0.8B")
        doubt = ws.readout(qwen, ["Wait", "However", "But", "Maybe", "Perhaps"])
        progress = ws.readout(qwen, ["Therefore", "Thus", "So", "Now", "The", "First", "Next", "Let"])
        patch = ws.Patch.new(qwen, "math-doubt-column", cites=["arXiv:2611.09382"]).add(
            "layer[23].mlp.down.col[507]", (doubt - progress).normalize(), alpha=-1.0)
        applied = patch.build(policy=ws.Policy.paper())
        out = Path(tempfile.mkdtemp()) / "math-doubt-column.json"
        patch.save(out)
        return (f"```\n{patch.summary()}\n```\n**{applied.total_changed:,}** stored values changed. "
                "Hu et al. report 1,023. Download the 2.7 KB patch below and run "
                "`weightspec verify math-doubt-column.json` anywhere."), str(out)
    except Exception:
        return f"```\n{traceback.format_exc()[-1500:]}\n```", None


INTRO = """
# WeightSpec: verifiable patches for neural-network weights

Frontier LLMs can now **edit another model's weights directly**, with no training. One edit to **1,023 numbers** took
Qwen3.5-0.8B from 8% to 38% on MATH-500 ([Hu et al.](https://arxiv.org/abs/2611.09382),
[Takehi et al.](https://rikiyat.github.io/blog/weight-space-programming/)).
[**WeightSpec**](https://github.com/namashworks/weightspec) is an open format for such edits: small, human-readable,
and checkable by anyone, bit for bit. `pip install weightspec`
"""

with gr.Blocks(title="WeightSpec") as demo:
    gr.Markdown(INTRO)
    with gr.Tab("Verify a patch"):
        gr.Markdown("Paste an `https://` or `hf://owner/repo/file` link to a `.json` / `.wspec` patch, or upload one. "
                    "The Space rebuilds it from the public base weights and checks every hash.")
        with gr.Row():
            uri = gr.Textbox(label="Patch URL", value=EXAMPLE_PATCHES[0], scale=4)
            upload = gr.File(label="…or upload", file_types=[".json", ".wspec"], type="filepath", scale=1)
        gr.Examples(EXAMPLE_PATCHES, inputs=uri)
        go = gr.Button("Verify", variant="primary")
        verdict = gr.Markdown()
        summary = gr.Code(label="Summary", language=None)
        manifest = gr.JSON(label="Manifest")
        go.click(verify_patch, [uri, upload], [verdict, summary, manifest])
    with gr.Tab("What does this weight write?"):
        gr.Markdown("Pick a single column, e.g. one MLP unit's write direction, and see which tokens it promotes.")
        with gr.Row():
            src = gr.Dropdown(LENS_MODELS, value=LENS_MODELS[0], label="Model", allow_custom_value=True)
            sel = gr.Textbox(value="layer[23].mlp.down.col[507]", label="Selector")
            k = gr.Slider(5, 40, value=15, step=1, label="Top k")
        gr.Examples([["layer[23].mlp.down.col[507]"], ["layer[19].mlp.down.col[1]"], ["layer[15].mlp.down.col[0]"]],
                    inputs=sel)
        lens_btn = gr.Button("Read out", variant="primary")
        lens_head = gr.Markdown()
        with gr.Row():
            pos = gr.Dataframe(headers=["id", "token", "score"], label="Promotes")
            neg = gr.Dataframe(headers=["id", "token", "score"], label="Suppresses")
        lens_btn.click(readout_lens, [src, sel, k], [lens_head, pos, neg])
    with gr.Tab("Rebuild the paper's edit"):
        gr.Markdown("Rebuild the 1,023-value MATH edit from Hu et al.'s Eq. 3, live, from the public Qwen weights.")
        rb = gr.Button("Rebuild", variant="primary")
        rb_out = gr.Markdown()
        rb_file = gr.File(label="Patch file")
        rb.click(rebuild_paper_edit, None, [rb_out, rb_file])

if __name__ == "__main__":
    demo.launch()
