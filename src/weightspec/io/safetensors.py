"""Minimal, dependency-free safetensors reader/writer.

Two readers share one interface:

* :class:`LocalSafetensors` memory-maps a file on disk.
* :class:`RemoteSafetensors` reads a file over HTTP with ``Range`` requests, so you
  can inspect or edit one tensor (or a few rows of the embedding matrix) of a
  multi-GB checkpoint on the Hugging Face Hub without downloading it.

Raw arrays are returned in their storage representation (``uint16`` bits for BF16).
"""

from __future__ import annotations

import json
import os
import struct
import urllib.error
import urllib.request
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from weightspec.numerics import DTYPES

__all__ = ["LocalSafetensors", "RemoteSafetensors", "TensorInfo", "write_safetensors"]


@dataclass(frozen=True)
class TensorInfo:
    name: str
    dtype: str
    shape: tuple[int, ...]
    start: int  # absolute byte offset in the file
    end: int

    @property
    def numel(self) -> int:
        n = 1
        for s in self.shape:
            n *= s
        return n

    @property
    def row_bytes(self) -> int:
        if not self.shape:
            raise ValueError(f"{self.name} is a scalar")
        return (self.end - self.start) // self.shape[0]


def _parse_header(raw: bytes, data_start: int) -> tuple[dict[str, TensorInfo], dict[str, str]]:
    header = json.loads(raw)
    meta = header.pop("__metadata__", {}) or {}
    infos = {}
    for name, h in header.items():
        s, e = h["data_offsets"]
        infos[name] = TensorInfo(name, h["dtype"], tuple(h["shape"]), data_start + s, data_start + e)
    return infos, meta


class SafetensorsReader(Protocol):
    tensors: dict[str, TensorInfo]
    metadata: dict[str, str]

    def read(self, name: str) -> np.ndarray: ...

    def read_rows(self, name: str, rows: Iterable[int]) -> np.ndarray: ...


def _to_array(buf: bytes, info: TensorInfo, shape: tuple[int, ...] | None = None) -> np.ndarray:
    dt = DTYPES[info.dtype].raw
    return np.frombuffer(buf, dtype=dt).reshape(shape if shape is not None else info.shape)


class LocalSafetensors:
    """Memory-mapped local safetensors file."""

    def __init__(self, path: str | os.PathLike[str]):
        self.path = Path(path)
        with open(self.path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            self.tensors, self.metadata = _parse_header(f.read(n), 8 + n)
        self._mm = np.memmap(self.path, dtype=np.uint8, mode="r")

    def read(self, name: str) -> np.ndarray:
        info = self.tensors[name]
        return _to_array(self._mm[info.start : info.end].tobytes(), info)

    def read_rows(self, name: str, rows: Iterable[int]) -> np.ndarray:
        info = self.tensors[name]
        rb = info.row_bytes
        out = [self._mm[info.start + r * rb : info.start + (r + 1) * rb].tobytes() for r in rows]
        return _to_array(b"".join(out), info, (len(out), *info.shape[1:]))


def _cache_dir() -> Path:
    root = os.environ.get("WEIGHTSPEC_CACHE") or os.path.join(
        os.environ.get("XDG_CACHE_HOME") or os.path.join(Path.home(), ".cache"), "weightspec"
    )
    return Path(root)


class RemoteSafetensors:
    """A safetensors file read lazily over HTTP range requests, with an on-disk cache.

    ``cache_key`` must identify immutable content (e.g. ``repo@commit_sha/filename``);
    cached tensors are reused across runs.
    """

    def __init__(self, url: str, cache_key: str, headers: Mapping[str, str] | None = None):
        self.url = url
        self.headers = dict(headers or {})
        self.cache = _cache_dir() / "tensors" / cache_key.replace(":", "_")
        n = struct.unpack("<Q", self._get(0, 8))[0]
        self.tensors, self.metadata = _parse_header(self._get(8, 8 + n), 8 + n)

    def _get(self, start: int, end: int) -> bytes:
        """Fetch bytes [start, end)."""
        req = urllib.request.Request(
            self.url, headers={**self.headers, "Range": f"bytes={start}-{end - 1}"}
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                data: bytes = r.read()
        except urllib.error.HTTPError as e:  # pragma: no cover - network
            raise OSError(f"HTTP {e.code} reading {self.url}") from e
        if len(data) != end - start:
            raise OSError(f"short read from {self.url}: wanted {end - start} bytes, got {len(data)}")
        return data

    def read(self, name: str) -> np.ndarray:
        info = self.tensors[name]
        path = self.cache / (name + ".raw")
        if path.exists() and path.stat().st_size == info.end - info.start:
            buf = path.read_bytes()
        else:
            buf = b"".join(
                self._get(s, min(s + (64 << 20), info.end)) for s in range(info.start, info.end, 64 << 20)
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_bytes(buf)
            os.replace(tmp, path)
        return _to_array(buf, info)

    def read_rows(self, name: str, rows: Iterable[int]) -> np.ndarray:
        info = self.tensors[name]
        full = self.cache / (name + ".raw")
        rows = list(rows)
        if full.exists():
            arr = self.read(name)
            return np.ascontiguousarray(arr[rows])
        rb = info.row_bytes
        out = []
        for r in rows:
            p = self.cache / f"{name}.rows" / f"{r}.raw"
            if p.exists():
                out.append(p.read_bytes())
                continue
            b = self._get(info.start + r * rb, info.start + (r + 1) * rb)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b)
            out.append(b)
        return _to_array(b"".join(out), info, (len(rows), *info.shape[1:]))


def write_safetensors(
    path: str | os.PathLike[str],
    tensors: Mapping[str, tuple[np.ndarray, str]],
    metadata: Mapping[str, str] | None = None,
) -> None:
    """Write ``{name: (raw_array, dtype_name)}`` to a safetensors file (sorted, aligned)."""
    header: dict[str, Any] = {}
    if metadata:
        header["__metadata__"] = {str(k): str(v) for k, v in metadata.items()}
    offset = 0
    blobs = []
    for name in sorted(tensors):
        raw, dtype = tensors[name]
        b = np.ascontiguousarray(raw, dtype=DTYPES[dtype].raw).tobytes()
        header[name] = {"dtype": dtype, "shape": list(np.shape(raw)), "data_offsets": [offset, offset + len(b)]}
        offset += len(b)
        blobs.append(b)
    hb = json.dumps(header, separators=(",", ":")).encode()
    hb += b" " * ((8 - len(hb) % 8) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hb)))
        f.write(hb)
        for b in blobs:
            f.write(b)
