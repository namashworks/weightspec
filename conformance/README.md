# Conformance suite

Reference patches that reproduce published weight edits from public checkpoints. Any implementation of the WeightSpec format must rebuild each one **bit for bit**. CI checks this with `pytest -m network`, which downloads about 22 MB with HTTP range reads and needs no GPU.

| Patch | Source | Base | Recorded check |
|---|---|---|---|
| `hu2026-math-primary.json` | Hu et al., *Exploring Weight-Space Programming*, arXiv:2611.09382, App. C.1, Eq. 3. Rebuilt from the formula. | `Qwen/Qwen3.5-0.8B@2fc06364` | 1,023 changed BF16 values (paper, Table 5) |
| `takehi2026-exp8-math-p50.wspec` | Takehi, Teoh, Zhang, Khattab, *Sparks of Weight-Space Programming*, Exp. 8 (50-pass edit), extracted with `weightspec diff` from `RikiyaT/Qwen3.5-0.8B-weight-space-programming@d1671a17` | `Qwen/Qwen3.5-0.8B@2fc06364` | Base and edited SHA-256 of both tensors equal `exp8_math/runs/p50c/edit/manifest.json` in [RikiyaT/lm_understand-logs](https://github.com/RikiyaT/lm_understand-logs) |

Notes:

* Hu et al. did not release their edited checkpoint, so the first patch is checked against the changed-value count the paper reports, not a hash. The formula fully determines the edit. A hash published by the authors would make the check exact.
* The second patch stores the edited columns explicitly. Its structure is a rank-one `project_scale` with k = 6 (see `examples/explain_takehi_exp8.py`). It is not stored as a recipe because the authors' exact direction vector was not published.

To verify locally:

```bash
weightspec verify conformance/hu2026-math-primary.json
weightspec verify conformance/takehi2026-exp8-math-p50.wspec
```
