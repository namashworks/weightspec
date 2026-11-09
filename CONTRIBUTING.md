# Contributing to WeightSpec

Thanks for helping build a shared foundation for weight-space programming.

## Ways to help

| Contribution | Where | Size |
|---|---|---|
| Comment on the format | Issues labelled `rfc`, or a PR to [`spec/RFC-0001-patch-format.md`](spec/RFC-0001-patch-format.md) | Any |
| Add an architecture map | `src/weightspec/arch/` (see below) | Small, self-contained: good first issue |
| Add a reference patch | `conformance/` (see below) | Small |
| Report a reproduction that disagrees | Issue with the patch file and your environment | Small, high value |
| Importer for another WSP tool's edit format | `src/weightspec/importers/` (planned) | Medium |

## Development

```bash
uv venv && uv pip install -e ".[dev]"
pytest                 # offline, CPU-only
pytest -m network      # conformance against real Hugging Face checkpoints
ruff check src tests && mypy src
```

All offline tests must pass on Linux, macOS and Windows. New behaviour needs a test. Changes to numerics or hashing must keep every conformance patch verifying bit for bit.

## Architecture maps

Most Hugging Face decoders work automatically: the layer prefix is discovered from tensor names, and head layout comes from `config.json`. A family needs its own map only when it differs, for example fused QKV projections or experts. Register a factory:

```python
import dataclasses
from weightspec.arch import generic_hf, register_arch

@register_arch("my_model_type")
def my_arch(store):
    spec = generic_hf(store)
    return dataclasses.replace(spec, components={**spec.components, "attn.out": "attn.c_proj.weight"})
```

Add a test with a tiny synthetic model in `tests/` that checks `resolve()` for a column, a head, and a global tensor.

## Reference patches

A reference patch reproduces a *published* edit from a *public* checkpoint:

1. Build it with WeightSpec (from the paper's formula, or with `weightspec diff` from released weights).
2. Fill `provenance` with authors, `cites` (arXiv id or URL) and the license of the source.
3. Add a check in `tests/test_conformance.py` against a number or hash the authors published.
4. Save it to `conformance/` with a name `<firstauthor><year>-<short-name>`.

## Pull requests

Keep PRs focused. Explain *why* in the description. For changes to the format, open an issue first. The format has its own version (`wspec`), and it changes more slowly than the code.

By contributing you agree that your contributions are licensed under Apache-2.0.
