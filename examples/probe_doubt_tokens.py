"""Does the 1,023-value edit do what its authors say? Check it with one teacher-forced probe.

Hu et al. (2026) built their MATH edit from the contrast between reconsideration words
(Wait, However, But, Maybe, Perhaps) and forward-progress words (Therefore, Thus, So,
Now, ...). If the mechanism is real, the edited model should put less probability on
doubt tokens, relative to progress tokens, at the start of a new reasoning paragraph.

Runs on CPU (~4 GB RAM). Downloads Qwen3.5-0.8B once.

    python examples/probe_doubt_tokens.py

Result on 2026-11-09 (CPU, float32, 3 prompts): P(doubt) fell from 0.205 / 0.026 / 0.088
to 0.0001 / 0.017 / 0.0002, and the doubt/progress odds fell by x0.005 (geometric mean).
Three prompts test the mechanism, not accuracy. The model is loaded in float32 here, so
all 1,024 values of the column change; in the bfloat16 checkpoint one update rounds away
(1,023, as the paper reports).
"""

import numpy as np

import weightspec as ws

probe = ws.Probe.from_pretrained("Qwen/Qwen3.5-0.8B", budget=2, device="cpu", dtype="float32")
patch = ws.Patch.load("conformance/hu2026-math-primary.json")
applied = probe.set_patch(patch)
print(f"patch applied: {applied.total_changed:,} values changed in {list(applied.tensors)}")

PREFIXES = [
    "<|im_start|>user\nWhat is 12 * 13?<|im_end|>\n<|im_start|>assistant\n<think>\n12 * 13 = 12 * 10 + 12 * 3 = 120 + 36 = 156.\n\n",
    "<|im_start|>user\nSolve 2x + 6 = 14.<|im_end|>\n<|im_start|>assistant\n<think>\nSubtract 6 from both sides: 2x = 8. Divide by 2: x = 4.\n\n",
    "<|im_start|>user\nHow many primes are below 20?<|im_end|>\n<|im_start|>assistant\n<think>\nThe primes below 20 are 2, 3, 5, 7, 11, 13, 17, 19. That is 8 primes.\n\n",
]
DOUBT = ["Wait", "But", "However", "Hmm"]
PROGRESS = ["So", "Therefore", "Thus", "The"]
texts = [p + w for p in PREFIXES for w in DOUBT + PROGRESS]

result = {}
for arm in ("original", "edited"):
    res = probe.teacher_forced(texts, weights=arm)
    result[arm] = np.array([r["token_logprobs"][-1] for r in res]).reshape(len(PREFIXES), -1)

k = len(DOUBT)
for arm, lp in result.items():
    p_doubt = np.exp(lp[:, :k]).sum(axis=1)
    p_prog = np.exp(lp[:, k:]).sum(axis=1)
    print(f"{arm:<9} P(doubt) = {np.round(p_doubt, 4)}   P(progress) = {np.round(p_prog, 4)}   "
          f"doubt/progress = {np.round(p_doubt / p_prog, 3)}")
ratio_o = np.exp(result["original"][:, :k]).sum(1) / np.exp(result["original"][:, k:]).sum(1)
ratio_e = np.exp(result["edited"][:, :k]).sum(1) / np.exp(result["edited"][:, k:]).sum(1)
print(f"\nThe edit changes the doubt/progress odds by x{np.round(ratio_e / ratio_o, 3)} per prefix "
      f"(geometric mean x{np.exp(np.mean(np.log(ratio_e / ratio_o))):.3f}).")
print(f"budget used: {probe.budget.status()}")
