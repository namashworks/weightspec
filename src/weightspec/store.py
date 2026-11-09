"""WeightStore: uniform, lazy, read-only access to a model's weights.

A store can come from the Hugging Face Hub (tensors fetched on demand with HTTP
range reads), a local directory or file, a dict of numpy arrays, or a live
``torch.nn.Module``. Edits never mutate a store; they produce an overlay
(:meth:`WeightStore.overlay`), so the original weights are always available.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from weightspec.io.safetensors import LocalSafetensors, SafetensorsReader
from weightspec.numerics import DTYPES, decode

if TYPE_CHECKING:
    from weightspec.arch import ArchSpec

__all__ = ["WeightStore", "load"]


class _ArraySource:
    """In-memory tensors: {name: (raw_array, dtype)}."""

    def __init__(self, arrays: Mapping[str, tuple[np.ndarray, str]]):
        self.arrays = dict(arrays)

    def read(self, name: str) -> np.ndarray:
        return self.arrays[name][0]

    def read_rows(self, name: str, rows: Iterable[int]) -> np.ndarray:
        return np.ascontiguousarray(self.arrays[name][0][list(rows)])


class WeightStore:
    """Read-only, lazily loaded weights plus the metadata needed to address them."""

    def __init__(
        self,
        tensors: Mapping[str, tuple[Any, str, tuple[int, ...]]],
        *,
        source: str,
        revision: str | None = None,
        config: dict[str, Any] | None = None,
        tokenizer_file: Path | None = None,
        arch: str | ArchSpec | None = "auto",
    ):
        # name -> (reader, dtype, shape)
        self._tensors = dict(tensors)
        self._overrides: dict[str, np.ndarray] = {}
        self._parent: WeightStore | None = None
        self.source = source
        self.revision = revision
        self.config = config or {}
        self._tokenizer_file = tokenizer_file
        self._tokenizer: Any = None
        self._arch_arg = arch
        self._arch: ArchSpec | None = None

    # ------------------------------------------------------------------ constructors
    @classmethod
    def from_hub(cls, repo_id: str, revision: str = "main", arch: str | None = "auto") -> WeightStore:
        from weightspec.io.hub import HubRepo

        repo = HubRepo(repo_id, revision)
        tensors: dict[str, tuple[Any, str, tuple[int, ...]]] = {}
        for fn in repo.safetensors_files():
            reader = repo.open(fn)
            for name, info in reader.tensors.items():
                tensors[name] = (reader, info.dtype, info.shape)
        config = json.loads(repo.download("config.json").read_text()) if "config.json" in repo.files else {}
        tok = repo.download("tokenizer.json") if "tokenizer.json" in repo.files else None
        return cls(tensors, source=f"hf://{repo_id}", revision=repo.sha, config=config, tokenizer_file=tok, arch=arch)

    @classmethod
    def from_path(cls, path: str | os.PathLike[str], arch: str | None = "auto") -> WeightStore:
        p = Path(path)
        files = sorted(p.glob("*.safetensors")) if p.is_dir() else [p]
        if not files:
            raise FileNotFoundError(f"no .safetensors files in {p}")
        tensors: dict[str, tuple[Any, str, tuple[int, ...]]] = {}
        for f in files:
            reader = LocalSafetensors(f)
            for name, info in reader.tensors.items():
                tensors[name] = (reader, info.dtype, info.shape)
        root = p if p.is_dir() else p.parent
        cfg_path = root / "config.json"
        config = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
        tok = root / "tokenizer.json"
        return cls(
            tensors, source=f"file://{p.resolve().as_posix()}", config=config,
            tokenizer_file=tok if tok.exists() else None, arch=arch,
        )

    @classmethod
    def from_arrays(
        cls,
        arrays: Mapping[str, np.ndarray | tuple[np.ndarray, str]],
        *,
        dtype: str = "F32",
        config: dict[str, Any] | None = None,
        arch: str | ArchSpec | None = "auto",
        source: str = "memory",
    ) -> WeightStore:
        """Wrap numpy arrays. Plain float arrays are stored in ``dtype`` (rounded with the format's rules)."""
        from weightspec.numerics import encode

        norm: dict[str, tuple[np.ndarray, str]] = {}
        for k, v in arrays.items():
            if isinstance(v, tuple):
                norm[k] = (np.asarray(v[0]), v[1])
            else:
                norm[k] = (encode(np.asarray(v, dtype=np.float32), dtype), dtype)
        src = _ArraySource(norm)
        tensors = {k: (src, d, tuple(a.shape)) for k, (a, d) in norm.items()}
        return cls(tensors, source=source, config=config, arch=arch)

    @classmethod
    def from_torch(
        cls, module: Any, config: dict[str, Any] | None = None, arch: str | None = "auto", copy: bool = True
    ) -> WeightStore:
        """Weights of a ``torch.nn.Module``.

        ``copy=True`` (default) snapshots the weights now, so later in-place edits to the
        module do not change the store. ``copy=False`` reads the live module lazily (no
        extra memory); use it only when the module's weights stay fixed, as in a Probe.
        """
        from weightspec.integrations.torch import ModuleSource, state_dict_to_arrays

        cfg = config
        if cfg is None and getattr(module, "config", None) is not None:
            try:
                cfg = module.config.to_dict()
            except Exception:
                cfg = None
        if copy:
            return cls.from_arrays(state_dict_to_arrays(module), config=cfg, arch=arch, source="torch")
        return cls(ModuleSource(module).describe(), source="torch", config=cfg, arch=arch)

    # ------------------------------------------------------------------ access
    @property
    def names(self) -> list[str]:
        return list(self._tensors)

    def __contains__(self, name: object) -> bool:
        return name in self._tensors

    def dtype(self, name: str) -> str:
        return self._tensors[name][1]

    def shape(self, name: str) -> tuple[int, ...]:
        return self._tensors[name][2]

    def raw(self, name: str) -> np.ndarray:
        """Current raw storage array (with any overlay edits applied)."""
        if name in self._overrides:
            return self._overrides[name]
        if self._parent is not None:
            return self._parent.raw(name)
        if name not in self._tensors:
            raise KeyError(f"no tensor named {name!r}")
        reader: SafetensorsReader = self._tensors[name][0]
        return reader.read(name)

    def raw_original(self, name: str) -> np.ndarray:
        store = self
        while store._parent is not None:
            store = store._parent
        return store.raw(name)

    def load(self, name: str) -> np.ndarray:
        """Current weights as float32 (a fresh copy you may modify)."""
        return decode(self.raw(name), self.dtype(name)).copy()

    def load_original(self, name: str) -> np.ndarray:
        return decode(self.raw_original(name), self.dtype(name)).copy()

    def rows(self, name: str, rows: Iterable[int]) -> np.ndarray:
        """Selected rows of a tensor as float32, without loading the whole tensor if possible."""
        rows = [int(r) for r in rows]
        if name in self._overrides or self._parent is not None:
            return decode(self.raw(name)[rows], self.dtype(name))
        reader: SafetensorsReader = self._tensors[name][0]
        return decode(reader.read_rows(name, rows), self.dtype(name))

    def numel(self, name: str | None = None) -> int:
        names = [name] if name else self.names
        total = 0
        for n in names:
            k = 1
            for s in self.shape(n):
                k *= s
            total += k
        return total

    def overlay(self, edits: Mapping[str, np.ndarray]) -> WeightStore:
        """A new store that sees ``edits`` (raw arrays) on top of this one."""
        child = WeightStore.__new__(WeightStore)
        child.__dict__.update(self.__dict__)
        child._overrides = {k: np.asarray(v) for k, v in edits.items()}
        for k, v in child._overrides.items():
            if k not in self._tensors:
                raise KeyError(f"cannot overlay unknown tensor {k!r}")
            if tuple(v.shape) != self.shape(k) or v.dtype != DTYPES[self.dtype(k)].raw:
                raise ValueError(f"overlay for {k!r} has wrong shape/dtype")
        child._parent = self
        return child

    # ------------------------------------------------------------------ metadata
    @property
    def arch(self) -> ArchSpec:
        if self._arch is None:
            from weightspec.arch import resolve_arch

            self._arch = resolve_arch(self._arch_arg, self)
        return self._arch

    @property
    def text_config(self) -> dict[str, Any]:
        tc = self.config.get("text_config")
        return tc if isinstance(tc, dict) else self.config

    def tokenizer(self) -> Any:
        if self._tokenizer is None:
            if self._tokenizer_file is None:
                raise RuntimeError(f"{self.source} has no tokenizer.json; pass token ids instead of strings")
            try:
                from tokenizers import Tokenizer
            except ImportError as e:  # pragma: no cover
                raise ImportError("pip install 'weightspec[tokenizers]' to resolve tokens from strings") from e
            self._tokenizer = Tokenizer.from_file(str(self._tokenizer_file))
        return self._tokenizer

    def token_ids(self, words: Iterable[str | int], first_token: bool = True) -> list[int]:
        out = []
        for w in words:
            if isinstance(w, int):
                out.append(w)
                continue
            ids = self.tokenizer().encode(w, add_special_tokens=False).ids
            if not ids:
                raise ValueError(f"{w!r} tokenizes to nothing")
            if not first_token and len(ids) != 1:
                raise ValueError(f"{w!r} is {len(ids)} tokens; set first_token=True or pass an id")
            out.append(int(ids[0]))
        return out

    def describe(self) -> str:
        rev = f"@{self.revision[:12]}" if self.revision else ""
        return f"WeightStore({self.source}{rev}, {len(self._tensors)} tensors, {self.numel():,} params, arch={self.arch.name})"

    def __repr__(self) -> str:
        return self.describe()


def load(source: Any, revision: str = "main", arch: str | None = "auto") -> WeightStore:
    """Open weights from a Hub id, a local path, a dict of arrays, or a torch module.

    >>> store = ws.load("Qwen/Qwen3.5-0.8B")          # nothing large is downloaded yet
    """
    if isinstance(source, WeightStore):
        return source
    if isinstance(source, Mapping):
        return WeightStore.from_arrays(source, arch=arch)
    if isinstance(source, (str, os.PathLike)):
        s = os.fspath(source)
        if s.startswith("hf://"):
            s = s[5:]
        if os.path.exists(s):
            return WeightStore.from_path(s, arch=arch)
        if s.count("/") == 1 and not s.startswith((".", "/")):
            return WeightStore.from_hub(s, revision=revision, arch=arch)
        raise FileNotFoundError(f"{s!r} is neither a local path nor a Hub repo id")
    if hasattr(source, "state_dict"):
        return WeightStore.from_torch(source, arch=arch)
    raise TypeError(f"cannot load weights from {type(source).__name__}")
