"""What did GPT-6 Astra actually do to Qwen? Explain MIT's Exp. 8 edit from the weights alone.

Takehi et al. (2026) released their edited Qwen3.5-0.8B. ``weightspec diff`` turned
it into a 250 KB patch that rebuilds the 1.6 GB checkpoint bit for bit
(conformance/takehi2026-exp8-math-p50.wspec). Here we look at the structure of the
change: in each of two MLP layers, 64 columns change, and the change is rank one.
It is exactly ``project_scale`` with k = 6: the component of those 64 write
directions along one residual direction û is multiplied by (1 - 6) = -5.

    python examples/explain_takehi_exp8.py
"""

import sys

import numpy as np

import weightspec as ws

sys.stdout.reconfigure(encoding="utf-8")  # token strings include non-Latin scripts
patch =ws.Patch.load("conformance/takehi2026-exp8-math-p50.wspec")
base = patch.base_store()
applied = patch.apply()

directions = {}
for name in applied.tensors:
    w0 = base.load(name).astype(np.float64)
    w1 = applied.numpy(name).astype(np.float64)
    cols = np.flatnonzero((w0 != w1).any(axis=0))
    d = w1[:, cols] - w0[:, cols]
    u_svd, s, _ = np.linalg.svd(d, full_matrices=False)
    u = u_svd[:, 0]
    k = -(u @ d) / (u @ w0[:, cols])
    layer = name.split("layers.")[1].split(".")[0]
    directions[layer] = u
    print(f"layer {layer}: {applied.changed[name]:,} values in {len(cols)} columns changed")
    print(f"  rank of the change: s2/s1 = {s[1] / s[0]:.4f}  (rank one, up to bf16 rounding)")
    print(f"  fitted k per column: median {np.median(k):.4f}, range [{k.min():.3f}, {k.max():.3f}]")

    # The same edit written as a 1-op recipe, with the fitted direction as a payload:
    recipe = ws.Patch.new(base, f"exp8-layer{layer}-recipe").project_scale(
        f"layer[{layer}].mlp.down.col[{','.join(map(str, cols))}]", u, k=6.0
    )
    r = recipe.apply()
    same = r.tensors[name] == applied.tensors[name]
    print(f"  recipe 'project_scale k=6' reproduces {same.mean():.2%} of stored values exactly "
          f"(the rest are one bf16 step away: the authors' exact direction was not published)")

u15, u19 = directions["15"], directions["19"]
print(f"\ncosine between the two layers' directions: {abs(u15 @ u19):.3f}")
readout = base.arch.global_tensor("unembed")
gain = base.load(base.arch.final_norm).astype(np.float64) + base.arch.norm_offset
print("tokens whose readout aligns most with the layer-19 direction (ids):")
emb = base.load(readout).astype(np.float64)  # 508 MB: downloads the embedding once
scores = (emb * gain) @ u19
for sign, idx in (("+", np.argsort(-scores)[:10]), ("-", np.argsort(scores)[:10])):
    words = [base.tokenizer().decode([int(i)]) for i in idx]
    print(f"  {sign}: {words}")
