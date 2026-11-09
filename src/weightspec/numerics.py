"""Storage dtypes and the exact rounding rules of the WeightSpec format.

Edits are computed in float32 and written back in the tensor's storage dtype
(RFC-0001 §5). Every implementation must round the same way, otherwise patch
hashes would not be portable. We implement bfloat16 round-to-nearest-even in pure
numpy (bit-identical to PyTorch's ``tensor.to(torch.bfloat16)``) so the core needs
no deep-learning framework.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

__all__ = [
    "DTYPES",
    "DType",
    "changed_count",
    "decode",
    "encode",
    "sha256_raw",
]


@dataclass(frozen=True)
class DType:
    """A safetensors dtype: its name, the numpy array that holds its raw bits, and size."""

    name: str
    raw: np.dtype
    itemsize: int
    floating: bool


DTYPES: dict[str, DType] = {
    "BF16": DType("BF16", np.dtype("<u2"), 2, True),
    "F16": DType("F16", np.dtype("<f2"), 2, True),
    "F32": DType("F32", np.dtype("<f4"), 4, True),
    "F64": DType("F64", np.dtype("<f8"), 8, True),
    "I64": DType("I64", np.dtype("<i8"), 8, False),
    "I32": DType("I32", np.dtype("<i4"), 4, False),
    "I16": DType("I16", np.dtype("<i2"), 2, False),
    "I8": DType("I8", np.dtype("i1"), 1, False),
    "U8": DType("U8", np.dtype("u1"), 1, False),
    "BOOL": DType("BOOL", np.dtype("?"), 1, False),
}

EDITABLE = ("BF16", "F16", "F32", "F64")


def bf16_bits_to_f32(bits: np.ndarray) -> np.ndarray:
    """Widen raw bfloat16 bits (uint16) to float32. Exact."""
    return (np.asarray(bits, dtype=np.uint16).astype(np.uint32) << np.uint32(16)).view(np.float32)


def f32_to_bf16_bits(x: np.ndarray) -> np.ndarray:
    """Round float32 to bfloat16 bits with round-to-nearest-even (PyTorch semantics)."""
    x = np.ascontiguousarray(x, dtype=np.float32)
    b = x.view(np.uint32)
    nan = np.isnan(x)
    rounding = ((b >> np.uint32(16)) & np.uint32(1)) + np.uint32(0x7FFF)
    with np.errstate(over="ignore"):
        out = ((b + rounding) >> np.uint32(16)).astype(np.uint16)
    if nan.any():
        out[nan] = ((b[nan] >> np.uint32(16)) | np.uint32(0x0040)).astype(np.uint16)
    return out


def decode(raw: np.ndarray, dtype: str) -> np.ndarray:
    """Raw storage array -> float32 working array (exact for BF16/F16/F32)."""
    if dtype == "BF16":
        return bf16_bits_to_f32(raw)
    if dtype not in DTYPES:
        raise ValueError(f"unsupported dtype {dtype!r}")
    return np.asarray(raw).astype(np.float32)


def encode(x: np.ndarray, dtype: str) -> np.ndarray:
    """float32 working array -> raw storage array, using the format's rounding rule."""
    if dtype == "BF16":
        return f32_to_bf16_bits(x)
    if dtype not in EDITABLE:
        raise ValueError(f"tensors of dtype {dtype!r} cannot be edited")
    return np.asarray(x, dtype=np.float32).astype(DTYPES[dtype].raw)


def changed_count(before_raw: np.ndarray, after_raw: np.ndarray) -> int:
    """Number of stored values that differ, compared bit for bit (the paper's 'changed BF16 count')."""
    a = np.ascontiguousarray(before_raw)
    b = np.ascontiguousarray(after_raw)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch {a.shape} vs {b.shape}")
    width = a.dtype.itemsize
    view = {1: np.uint8, 2: np.uint16, 4: np.uint32, 8: np.uint64}[width]
    return int(np.count_nonzero(a.view(view) != b.view(view)))


def sha256_raw(raw: np.ndarray) -> str:
    """Hash of a tensor's raw little-endian bytes, C order."""
    return hashlib.sha256(np.ascontiguousarray(raw).tobytes()).hexdigest()


def sha256_f32(x: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(x, dtype="<f4").tobytes()).hexdigest()
