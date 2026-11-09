"""Rebuild the 1,023-value MATH edit of Hu et al. (2026) from its published formula.

Hu et al., "Exploring Weight-Space Programming" (arXiv:2611.09382), Appendix C.1, Eq. 3:

    u = N( (1/5 Σ_{w∈A} E_t(w) - 1/8 Σ_{w∈B} E_t(w)) ⊙ (1 + γ) )
    W23 ← W23 - 1 · u e_507ᵀ

A = {Wait, However, But, Maybe, Perhaps}       ("reconsideration" words)
B = {Therefore, Thus, So, Now, The, First, Next, Let}   ("forward progress" words)

The paper reports 37.80% vs 8.35% on held-out MATH-500 (+29.45 pp) and that
1,023 of the column's 1,024 stored BF16 values change. This script needs no GPU
and downloads ~30 KB of weights (13 embedding rows, one norm, one 7 MB matrix).

    python examples/hu2026_math_primary.py
"""

import weightspec as ws

store = ws.load("Qwen/Qwen3.5-0.8B")

doubt = ws.readout(store, ["Wait", "However", "But", "Maybe", "Perhaps"])
progress = ws.readout(store, ["Therefore", "Thus", "So", "Now", "The", "First", "Next", "Let"])
u = (doubt - progress).normalize()

patch = ws.Patch.new(
    store,
    "qwen3.5-0.8b-math-doubt-column",
    description="Subtract the doubt-minus-progress readout direction from MLP unit 507 of layer 23.",
    authors=["Hu et al. (2026)"],
    editor="agent:GPT-6 Astra",
    method="single-column write edit",
    access_level="weights+50fwd",
    cites=["arXiv:2611.09382"],
    license="CC-BY-4.0",
).add("layer[23].mlp.down.col[507]", u, alpha=-1.0, note="Eq. 3, alpha = 1")

applied = patch.build(policy=ws.Policy.paper())
print(patch.summary())
assert applied.total_changed == 1023, applied.total_changed  # the paper's reported count
patch.save("conformance/hu2026-math-primary.json")
print("saved conformance/hu2026-math-primary.json")
print(ws.Patch.load("conformance/hu2026-math-primary.json").verify())
