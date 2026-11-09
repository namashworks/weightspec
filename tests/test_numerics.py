import numpy as np
import pytest

from weightspec.numerics import (
    bf16_bits_to_f32,
    changed_count,
    decode,
    encode,
    f32_to_bf16_bits,
    sha256_raw,
)


def test_bf16_roundtrip_is_exact_for_representable_values():
    bits = np.arange(0, 65536, 7, dtype=np.uint16)
    f = bf16_bits_to_f32(bits)
    finite = np.isfinite(f)
    assert np.array_equal(f32_to_bf16_bits(f[finite]), bits[finite])


def test_bf16_rounding_matches_torch():
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(0)
    x = np.concatenate([
        rng.standard_normal(100_000).astype(np.float32) * 10.0 ** rng.integers(-30, 30, 100_000),
        np.array([0.0, -0.0, np.inf, -np.inf, 1.00390625, 1.01171875, 3.4e38], dtype=np.float32),
    ]).astype(np.float32)
    ours = f32_to_bf16_bits(x)
    theirs = torch.from_numpy(x).to(torch.bfloat16).view(torch.int16).numpy().view(np.uint16)
    assert np.array_equal(ours, theirs)


def test_round_to_nearest_even_ties():
    # 1 + 2^-8 is exactly halfway between two bf16 values: ties go to the even mantissa (1.0).
    assert bf16_bits_to_f32(f32_to_bf16_bits(np.array([1.00390625], np.float32)))[0] == 1.0
    assert bf16_bits_to_f32(f32_to_bf16_bits(np.array([1.01171875], np.float32)))[0] == 1.015625


def test_nan_stays_nan():
    out = bf16_bits_to_f32(f32_to_bf16_bits(np.array([np.nan], np.float32)))
    assert np.isnan(out[0])


def test_changed_count_counts_stored_bits():
    a = encode(np.array([1.0, 2.0, 3.0], np.float32), "BF16")
    b = encode(np.array([1.0, 2.0 + 1e-4, 3.5], np.float32), "BF16")  # 1e-4 is below bf16 spacing at 2.0
    assert changed_count(a, b) == 1


def test_decode_encode_f32_and_f16():
    x = np.array([0.1, -2.5], np.float32)
    assert np.array_equal(decode(encode(x, "F32"), "F32"), x)
    assert decode(encode(x, "F16"), "F16").dtype == np.float32


def test_sha256_is_stable():
    a = np.array([1, 2, 3], np.uint16)
    assert sha256_raw(a) == sha256_raw(a.copy())
    assert sha256_raw(a) != sha256_raw(a[::-1])
