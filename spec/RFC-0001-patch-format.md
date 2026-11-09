# RFC-0001: The WeightSpec Patch Format, version 0.1

| | |
|---|---|
| Status | **Draft, open for comment** |
| Format version | `0.1` |
| Schema | [`schema/patch-0.1.schema.json`](schema/patch-0.1.schema.json) |
| Reference implementation | `weightspec` (Python), this repository |
| Feedback | GitHub issues labelled `rfc` |

## 1. Motivation

Weight-space programming (WSP) edits a trained network's weights directly, without gradient descent, guided by an understanding of what its components compute (Takehi et al., 2026; Hu et al., 2026). Within a week of the first report, dozens of independent harnesses appeared. Each one stores edits differently (`.npz` diffs, `.npy` tensors, sparse index files, ledgers, full checkpoints), names weights with raw framework-specific tensor names, and reports "changed parameters" with different counting rules. Results cannot be exchanged, compared or verified across implementations.

This RFC defines a small, implementation-independent format for a **weight patch**. A patch should be:

1. **Readable:** a human can see *what* was changed and *how the vectors were made*.
2. **Verifiable:** anyone can rebuild it from the public base weights and check it bit for bit.
3. **Portable:** it names weights by role (`layer[23].mlp.down.col[507]`), not by one framework's tensor names.
4. **Small:** recipes are stored instead of tensors wherever possible.
5. **Neutral:** it does not depend on which agent, human or program produced the edit.

Non-goals: describing training (LoRA/PEFT adapters already do that well), describing activation-time interventions (hooks), and prescribing an editing method.

## 2. Terminology

The key words MUST, SHOULD and MAY are used as in RFC 2119. *Base* is the unmodified checkpoint. *Block* is a rectangular sub-matrix of one tensor. *Stored value* is one element in the checkpoint's storage dtype.

## 3. Container

A patch is one of:

* `*.json`: a UTF-8 JSON manifest. Allowed only when the patch has no payload tensors.
* `*.wspec`: a ZIP archive containing `patch.json` (the manifest) and, if any payloads exist, `tensors.safetensors`. Writers SHOULD use a fixed timestamp (1980-01-01) so identical patches produce identical files.
* A directory with the same two files (for development).

Payload tensors are stored as `F32` in safetensors under the keys referenced by ops or directions.

## 4. Selectors

A selector names a block. The grammar, with zero-based indices and half-open ranges, is:

```
selector  := base ("." accessor)*
base      := "layer[" INT "]." component | "embed" | "unembed" | "final_norm" | "tensor[" QUOTED "]"
component := "mlp.down" | "mlp.up" | "mlp.gate" | "attn.q" | "attn.k" | "attn.v" | "attn.out"
           | "attn.head[" INT "]." ("out" | "q" | "gate" | "k" | "v") | "norm.input" | "norm.post_attn"
accessor  := ("col" | "cols" | "row" | "rows") "[" (INT | INT ":" INT | INT ("," INT)*) "]"
```

Semantics (weights are `[out_features, in_features]`, as in PyTorch `Linear`):

* `mlp.down.col[j]` is the residual-stream write direction of MLP unit *j*.
* `attn.head[h].out` is columns `[h·d, (h+1)·d)` of the attention output projection, where *d* is the head width.
* `attn.head[h].q` / `.gate`: for architectures with gated queries (each head's rows are `[query (d), gate (d)]`), the query rows are `[2h·d, 2h·d+d)` and the gate rows `[2h·d+d, 2(h+1)·d)`. Otherwise `.q` is rows `[h·d, (h+1)·d)`.
* `unembed` is the output embedding, or the input embedding when the two are tied.
* Accessors compose: each one indexes into the block selected so far.

An implementation MUST record each selector's resolved tensor and indices in `result.targets`. `tensor['…']` is an escape hatch for weights that have no role name; patches SHOULD prefer role selectors.

## 5. Numerics

1. Directions are evaluated in float64 and rounded to float32 when used.
2. Ops run in float32 on a float32 copy of the stored tensor, in the order listed.
3. After all ops, each touched tensor is converted back to its storage dtype **once**, using IEEE round-to-nearest-even (for bfloat16, the rounding of `torch.Tensor.to(torch.bfloat16)`).
4. A *changed value* is a stored value whose bits differ from the base. `result.changed` counts these.
5. Hashes are SHA-256 of a tensor's raw little-endian stored bytes in C order. This matches the convention of the published Takehi et al. logs.

## 6. Operations

Each op has a `target` selector and acts on the selected block *B* (shape *r × c*; 1-D tensors are treated as *r × 1*).

| `op` | Fields | Effect |
|---|---|---|
| `add` | `u` (direction), `alpha`, `along` | `B += α·u` broadcast over columns (`len(u)=r`) or rows (`len(u)=c`) |
| `scale` | `factor` | `B *= factor` |
| `set` | `value` *or* `payload` | `B = value` (scalar or payload matrix) |
| `rank1` | `u`, `v`, `alpha` | `B += α·u vᵀ` |
| `project_scale` | `u`, `k`, `side` | left: `B ← (I − k·ûûᵀ)B`; right: `B ← B(I − k·ûûᵀ)`; û = u/‖u‖ |
| `delta` | `payload`, `alpha` | `B += α·D` |

These six cover every edit published in Takehi et al. (2026) and Hu et al. (2026) Tables 5–6. For example, a single-column write is `add`; head ablation is `set` with value 0; "multiply the component along û by −6" is `project_scale` with `k = 7`.

## 7. Directions

A direction is a JSON expression tree with a `fn` discriminator:

| `fn` | Meaning |
|---|---|
| `readout` | Rows `U_t ⊙ (o + γ)` of the unembedding for token `ids`, where γ is the final norm weight and *o* is 1 for one-centered RMSNorm families (Gemma, Qwen3.5) and 0 otherwise. `reduce`: mean, sum or none. |
| `embedding` | Rows of the input embedding for `ids`. |
| `weights` | A vector read from the *base* weights (`target`; `reduce` none, mean_cols or mean_rows). |
| `payload` | An explicit vector stored in the patch. |
| `sum` | `Σ cᵢ·termᵢ` |
| `scale` | `by · of` |
| `normalize` | `of / ‖of‖` |
| `project_off` | `of − QQᵀ·of`, with Q an orthonormal basis (QR) of the `span` vectors |
| `random_orthogonal` | Seeded (`numpy.random.default_rng(seed).standard_normal`) Gaussian vector projected off `like` and `also_off`, rescaled to ‖like‖ (the equal-norm random control) |

Token `ids` are authoritative; `tokens` records the words they came from.

## 8. Manifest

```jsonc
{
  "wspec": "0.1",
  "name": "qwen3.5-0.8b-math-doubt-column",
  "description": "…",
  "base": {
    "source": "hf://Qwen/Qwen3.5-0.8B",          // or file://…, memory, torch
    "revision": "2fc06364715b967f1860aea9cf38778875588b17",
    "arch": "qwen3_5_text",
    "tensors": { "<edited tensor>": "<sha256 before edit>" }
  },
  "numerics": { "directions": "F64", "compute": "F32", "store": "native", "rounding": "nearest-even" },
  "ops": [ /* §6 */ ],
  "result": {
    "changed": 1023,
    "tensors": { "<edited tensor>": { "dtype": "BF16", "shape": [1024, 3584], "changed": 1023, "sha256": "<after>" } },
    "targets": { "<selector>": "<resolved block>" }
  },
  "provenance": { "authors": [], "editor": "agent:<model> | human | program:<name> | diff",
                  "access_level": "weights+50fwd", "cites": [], "evidence": [], "license": "…" }
}
```

`result` is filled in by *building* the patch. A patch without `result` is a *draft*.

## 9. Verification

A conforming verifier MUST:

1. Check that each `base.tensors` hash matches the base checkpoint.
2. Re-apply the ops under §5 and §6.
3. Check that each `result.tensors[*].sha256` and changed count matches, and that no other tensor was touched.

Verification needs only the edited tensors and the rows that directions read, so it can run with HTTP range reads and no GPU.

## 10. Evaluation evidence (informative)

Edit claims SHOULD be reported as question-clustered paired differences against the base on held-out data (Hu et al., 2026, Eq. 2), with a 95% interval and the counts of wrong→right and right→wrong samples. They SHOULD include an equal-norm random-direction control. `provenance.evidence` MAY link such reports.

## 11. Conformance suite

The repository's `conformance/` folder holds reference patches with recorded hashes against public checkpoints. As of this draft:

| Patch | Source | Check |
|---|---|---|
| `hu2026-math-primary.json` | Hu et al. 2026, Eq. 3 (recipe, 2.7 KB) | 1,023 changed values, as reported in the paper |
| `takehi2026-exp8-math-p50.wspec` | Takehi et al. 2026, Exp. 8 (diff, 250 KB) | Base and edited hashes equal the authors' published manifest |

## 12. Versioning

The format version is `MAJOR.MINOR`. Readers MUST reject an unknown major version and SHOULD accept unknown fields in `provenance`. New ops or direction functions increment the minor version.

## 13. Open questions

* Quantized storage (FP8, GPTQ/AWQ blocks, GGUF): edit-then-requantize rules.
* Cross-architecture transfer: a standard map from one family's roles to another's.
* A registry of selector maps for non-transformer networks (CNNs, RL policies, diffusion UNets).
* Signing patches (Sigstore) for provenance.

## References

* R. Takehi, J. Teoh, A. Zhang, O. Khattab. *Sparks of Weight-Space Programming: LLMs can natively read & write neural network weights.* MIT CSAIL blog, 2 Nov 2026.
* J. Hu et al. *Exploring Weight-Space Programming: Capabilities, Generalization, and Limitations.* arXiv:2611.09382, 2026.
