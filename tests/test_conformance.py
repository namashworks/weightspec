"""Conformance: rebuild the reference patches against the real public checkpoints.

Run with ``pytest -m network``. Needs internet; downloads about 22 MB on first run
(cached afterwards) and no GPU.
"""

from pathlib import Path

import pytest

import weightspec as ws

ROOT = Path(__file__).resolve().parents[1] / "conformance"
PATCHES = sorted(p for p in ROOT.iterdir() if p.suffix in (".json", ".wspec"))


@pytest.mark.network
@pytest.mark.parametrize("path", PATCHES, ids=[p.name for p in PATCHES])
def test_reference_patch_rebuilds_bit_for_bit(path):
    patch = ws.Patch.load(path)
    report = patch.verify()
    assert report.ok, str(report)


@pytest.mark.network
def test_hu2026_count_matches_paper():
    patch = ws.Patch.load(ROOT / "hu2026-math-primary.json")
    assert patch.result is not None and patch.result.changed == 1023  # Hu et al. 2026, Table 5


@pytest.mark.network
def test_takehi2026_hashes_match_authors_manifest():
    # From github.com/RikiyaT/lm_understand-logs, exp8_math/runs/p50c/edit/manifest.json
    expected = {
        "model.language_model.layers.15.mlp.down_proj.weight": (
            "ef60d2c687bab19464dca9c8bfd7777e46ad226df0c9e25643b0bab04b5a1cca",
            "82e2d733ea64feaeb4af93a1e065641d3df98b494a8b7576ef2f52e90fe2ee53",
            65260,
        ),
        "model.language_model.layers.19.mlp.down_proj.weight": (
            "638eedeb701b94aa861f4438e615c07ec6c60ee703db1bfee2de577bb6498c3e",
            "8ef3c61bc36af113ad58e89dbb0a4a14dcba5a45cb750c15ab515c82ddd12eb2",
            64928,
        ),
    }
    patch = ws.Patch.load(ROOT / "takehi2026-exp8-math-p50.wspec")
    assert patch.result is not None
    for name, (base_h, edited_h, changed) in expected.items():
        assert patch.base.tensors[name] == base_h
        assert patch.result.tensors[name].sha256 == edited_h
        assert patch.result.tensors[name].changed == changed
