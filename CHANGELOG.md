# Changelog

## 0.1.0a1 (unreleased)

First public alpha.

- Patch format `wspec 0.1` (RFC-0001 draft) with JSON Schema; `.json` and `.wspec` containers.
- Portable selectors with architecture maps for Hugging Face decoders, including gated-query Qwen3.5 / Qwen3-Next and one-centered RMSNorm families.
- Ops: `add`, `scale`, `set`, `rank1`, `project_scale`, `delta`.
- Direction algebra: `readout`, `embedding`, `weights`, `vector`, sums, scaling, `normalize`, `project_off`, `random_orthogonal`.
- Exact numerics: pure-numpy bfloat16 round-to-nearest-even, bit-identical to PyTorch.
- Stores: Hugging Face Hub with HTTP range reads, local safetensors, numpy dicts, live `torch.nn.Module`.
- `Policy` (including `Policy.paper()`), `diff`, `controls`, paired statistics.
- CLI: `inspect`, `resolve`, `show`, `build`, `verify`, `apply`, `diff`, `stats`, `schema`.
- MCP server (`weightspec mcp`): 15 tools and a guided prompt so coding agents can inspect weights and build, control, verify and save patches inside a workspace, under policy.
- Sharing: `Patch.push_to_hub`, loading from `hf://owner/repo/file[@rev]` and `https://`, `Patch.to_bytes` / `from_bytes`; readers reject unknown major format versions.
- Launch assets: Colab quickstart notebook, Hugging Face Space (`spaces/verify`), terminal demo rendered from real output (`scripts/render_demo.py`).
- Probes (`ws.Probe`): budgeted generation and teacher-forced requests (log-probs, top-k, hidden states, MLP activations) on a transformers model, with in-place weight hot-swap that always restores the original tensors; paired "both" requests share seeds; tensor names map across multimodal / text-only layouts.
- Evaluation (`ws.evaluate`, `ws.load_items`): paired base-vs-edit comparison with graders (`math`, `exact`, `contains`, `choice`, `numeric`) and controls, in 1 + len(controls) requests.
- Agents: a shared `Toolkit` (used by the MCP server and sessions) with forward-pass tools (`generate`, `teacher_forced`, `contrast_activations` with named vectors, `evaluate_patch`); `ws.Session` with checkpoint/resume, trajectory log, budget notices and a sealed holdout; editors for Claude (Anthropic SDK), OpenAI-compatible servers, and scripted replays; `weightspec run` and `weightspec mcp --probe`.
- Conformance suite: Hu et al. (2026) MATH primary edit; Takehi et al. (2026) Exp. 8 edit, with hashes equal to the authors' manifest.
