"""The Patch: WeightSpec's unit of exchange (RFC-0001).

A patch is a small JSON manifest (``patch.json``) plus, only when needed, a
``tensors.safetensors`` file of explicit vectors/matrices. Saved as a single
``.wspec`` file (a deterministic zip), or as plain ``.json`` when it has no payloads.

Lifecycle::

    patch = ws.Patch.new(store, "my-fix").add("layer[23].mlp.down.col[507]", u, alpha=-1)
    applied = patch.build()          # apply + record hashes and changed counts in patch.result
    patch.save("my-fix.wspec")
    ws.Patch.load("my-fix.wspec").verify()   # rebuilds from the base weights, bit for bit
"""

from __future__ import annotations

import datetime as _dt
import io
import json
import os
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from weightspec._version import WSPEC_VERSION, __version__
from weightspec.directions import Direction, EvalContext, Payload
from weightspec.io.safetensors import LocalSafetensors, write_safetensors
from weightspec.numerics import changed_count, decode, encode, sha256_raw
from weightspec.ops import Add, Delta, Op, ProjectScale, Rank1, Scale, Set
from weightspec.policy import Policy
from weightspec.selectors import resolve
from weightspec.store import WeightStore, load

__all__ = ["Applied", "Patch", "VerifyReport", "diff"]


class BaseRef(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source: str = Field(description="hf://<repo>, file://<path>, memory or torch")
    revision: str | None = Field(default=None, description="immutable commit sha for Hub sources")
    arch: str | None = None
    tensors: dict[str, str] = Field(default_factory=dict, description="sha256 of each edited tensor before the edit")


class TensorResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    dtype: str
    shape: list[int]
    changed: int
    sha256: str


class Result(BaseModel):
    model_config = ConfigDict(extra="forbid")
    changed: int = Field(description="stored values that differ from the base, counted bit for bit")
    tensors: dict[str, TensorResult]
    targets: dict[str, str] = Field(default_factory=dict, description="selector -> resolved tensor block")


class Numerics(BaseModel):
    model_config = ConfigDict(extra="forbid")
    directions: Literal["F64"] = "F64"
    compute: Literal["F32"] = "F32"
    store: Literal["native"] = "native"
    rounding: Literal["nearest-even"] = "nearest-even"


class Provenance(BaseModel):
    model_config = ConfigDict(extra="allow")
    authors: list[str] = Field(default_factory=list)
    editor: str | None = Field(default=None, description="human | agent:<model> | program:<name> | diff")
    method: str | None = None
    access_level: str | None = Field(default=None, description="e.g. weights-only, weights+50fwd")
    cites: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
    license: str | None = None
    created: str | None = None
    tool: str | None = None
    notes: str | None = None


class Patch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    wspec: str = WSPEC_VERSION
    name: str
    description: str | None = None
    base: BaseRef
    numerics: Numerics = Field(default_factory=Numerics)
    ops: list[Op] = Field(default_factory=list)
    result: Result | None = None
    provenance: Provenance = Field(default_factory=Provenance)

    _payloads: dict[str, np.ndarray] = PrivateAttr(default_factory=dict)
    _store: WeightStore | None = PrivateAttr(default=None)

    # ------------------------------------------------------------------ building
    @classmethod
    def new(cls, store: WeightStore | Any, name: str, description: str | None = None, **provenance: Any) -> Patch:
        st = load(store)
        p = cls(
            name=name,
            description=description,
            base=BaseRef(source=st.source, revision=st.revision, arch=st.arch.name),
            provenance=Provenance(tool=f"weightspec {__version__}", **provenance),
        )
        p._store = st
        return p

    def _vec(self, v: Direction | Any, hint: str) -> Any:
        if isinstance(v, Direction):
            self._payloads.update(v.payloads)
            return v.expr
        arr = np.asarray(v, dtype=np.float32).reshape(-1)
        key = f"op{len(self.ops)}.{hint}"
        self._payloads[key] = arr
        return Payload(key=key)

    def _mat(self, m: Any, hint: str) -> str:
        key = f"op{len(self.ops)}.{hint}"
        self._payloads[key] = np.asarray(m, dtype=np.float32)
        return key

    def _push(self, op: Any) -> Patch:
        self.ops.append(op)
        self.result = None
        return self

    def add(self, target: str, u: Direction | Any, alpha: float = 1.0, along: Literal["auto", "cols", "rows"] = "auto", note: str | None = None) -> Patch:
        """``B += alpha·u``, u added to every selected column (or row)."""
        return self._push(Add(target=target, u=self._vec(u, "u"), alpha=alpha, along=along, note=note))

    def scale(self, target: str, factor: float, note: str | None = None) -> Patch:
        return self._push(Scale(target=target, factor=factor, note=note))

    def zero(self, target: str, note: str | None = None) -> Patch:
        return self._push(Set(target=target, value=0.0, note=note))

    def set(self, target: str, value: float | Any, note: str | None = None) -> Patch:
        if isinstance(value, (int, float, np.floating, np.integer)):
            return self._push(Set(target=target, value=float(value), note=note))
        return self._push(Set(target=target, payload=self._mat(value, "values"), note=note))

    def rank1(self, target: str, u: Direction | Any, v: Direction | Any, alpha: float = 1.0, note: str | None = None) -> Patch:
        return self._push(Rank1(target=target, u=self._vec(u, "u"), v=self._vec(v, "v"), alpha=alpha, note=note))

    def project_scale(self, target: str, u: Direction | Any, k: float, side: Literal["left", "right"] = "left", note: str | None = None) -> Patch:
        return self._push(ProjectScale(target=target, u=self._vec(u, "u"), k=k, side=side, note=note))

    def delta(self, target: str, d: Any, alpha: float = 1.0, note: str | None = None) -> Patch:
        return self._push(Delta(target=target, payload=self._mat(d, "delta"), alpha=alpha, note=note))

    # ------------------------------------------------------------------ applying
    @property
    def payloads(self) -> dict[str, np.ndarray]:
        return self._payloads

    def base_store(self) -> WeightStore:
        if self._store is None:
            src = self.base.source
            if src in ("memory", "torch"):
                raise ValueError("this patch was built on in-memory weights; pass the store explicitly")
            if src.startswith("file://"):
                src = src[len("file://"):]
            self._store = load(src, revision=self.base.revision or "main")
        return self._store

    def apply(self, store: WeightStore | Any | None = None, policy: Policy | None = None) -> Applied:
        """Compute the edited tensors. Does not modify ``store``; see :meth:`Applied.store`."""
        st = load(store) if store is not None else self.base_store()
        pol = policy or Policy.permissive()
        ctx = EvalContext(st, self._payloads)
        working: dict[str, np.ndarray] = {}
        targets: dict[str, str] = {}
        for i, op in enumerate(self.ops):
            t = resolve(st, op.target)
            pol.check_op(st, op.op, t)
            w = working.get(t.tensor)
            if w is None:
                w = working[t.tensor] = st.load(t.tensor)
            block = t.get(w)
            try:
                new = op.run(block, ctx, t)
            except Exception as e:
                raise type(e)(f"op {i} ({op.op} {op.target}): {e}") from e
            if new.shape != block.shape:
                raise ValueError(f"op {i} changed the block shape {block.shape} -> {new.shape}")
            t.set(w, new.astype(np.float32))
            targets[op.target] = t.describe()
        raw_new: dict[str, np.ndarray] = {}
        changed: dict[str, int] = {}
        for name, w in working.items():
            raw_new[name] = encode(w, st.dtype(name))
            changed[name] = changed_count(st.raw(name), raw_new[name])
        pol.check_result(st, working, changed)
        return Applied(patch=self, base=st, tensors=raw_new, changed=changed, targets=targets)

    def build(self, store: WeightStore | Any | None = None, policy: Policy | None = None) -> Applied:
        """Apply the patch and record base hashes, result hashes and changed counts in it."""
        applied = self.apply(store, policy)
        st = applied.base
        self.base.source, self.base.revision, self.base.arch = st.source, st.revision, st.arch.name
        self.base.tensors = {n: sha256_raw(st.raw(n)) for n in applied.tensors}
        self.result = applied.result()
        if self.provenance.created is None:
            self.provenance.created = _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat()
        return applied

    def verify(self, store: WeightStore | Any | None = None) -> VerifyReport:
        """Rebuild the patch from the base weights and compare every recorded hash."""
        if self.result is None:
            raise ValueError("patch has no recorded result; call build() first")
        st = load(store) if store is not None else self.base_store()
        problems: list[str] = []
        if self.base.revision and st.revision and st.revision != self.base.revision:
            problems.append(f"base revision {st.revision[:12]} != recorded {self.base.revision[:12]}")
        for name, h in self.base.tensors.items():
            if name not in st:
                problems.append(f"base is missing tensor {name}")
            elif sha256_raw(st.raw(name)) != h:
                problems.append(f"base tensor {name} differs from the one the patch was built on")
        applied = self.apply(st)
        got = applied.result()
        for name, exp in self.result.tensors.items():
            r = got.tensors.get(name)
            if r is None:
                problems.append(f"rebuild did not produce {name}")
            elif r.sha256 != exp.sha256:
                problems.append(
                    f"{name}: rebuilt hash {r.sha256[:12]} != recorded {exp.sha256[:12]} "
                    f"(changed {r.changed:,} vs {exp.changed:,})"
                )
        extra = set(got.tensors) - set(self.result.tensors)
        problems += [f"rebuild touched unrecorded tensor {n}" for n in sorted(extra)]
        return VerifyReport(ok=not problems, problems=problems, changed=got.changed, applied=applied)

    # ------------------------------------------------------------------ io
    def to_json(self, indent: int = 2) -> str:
        return self.model_dump_json(indent=indent, exclude_none=True)

    def save(self, path: str | os.PathLike[str]) -> Path:
        """Save as ``.wspec`` (zip with optional payloads) or ``.json`` (no payloads)."""
        p = Path(path)
        if p.suffix == ".json" and self._payloads:
            raise ValueError("patch has payload tensors; save as .wspec")
        p.write_bytes(self.to_bytes(container="json" if p.suffix == ".json" else "wspec"))
        return p

    def to_bytes(self, container: Literal["wspec", "json", "auto"] = "auto") -> bytes:
        """Serialize: plain JSON when there are no payloads (or when asked), else a ``.wspec`` zip."""
        manifest = self.to_json().encode()
        if container == "json" or (container == "auto" and not self._payloads):
            if self._payloads:
                raise ValueError("patch has payload tensors; use the wspec container")
            return manifest
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            def put(name: str, data: bytes) -> None:
                info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o644 << 16
                z.writestr(info, data)

            put("patch.json", manifest)
            if self._payloads:
                tmp = io.BytesIO()
                _write_payloads(tmp, self._payloads)
                put("tensors.safetensors", tmp.getvalue())
        return buf.getvalue()

    @classmethod
    def from_bytes(cls, data: bytes) -> Patch:
        payloads: dict[str, np.ndarray] = {}
        if data[:4] == b"PK\x03\x04":
            with zipfile.ZipFile(io.BytesIO(data)) as z:
                manifest = z.read("patch.json")
                if "tensors.safetensors" in z.namelist():
                    payloads = _read_payloads(z.read("tensors.safetensors"))
        else:
            manifest = data
        return cls._from_manifest(manifest, payloads)

    @classmethod
    def _from_manifest(cls, manifest: bytes, payloads: dict[str, np.ndarray]) -> Patch:
        version = json.loads(manifest).get("wspec", WSPEC_VERSION)  # drafts may omit it
        if str(version).split(".")[0] != WSPEC_VERSION.split(".")[0]:
            raise ValueError(f"unsupported wspec version {version!r}; this library reads {WSPEC_VERSION}")
        patch = cls.model_validate_json(manifest)
        patch._payloads = payloads
        return patch

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> Patch:
        """Load from a local ``.wspec`` / ``.json`` / directory, an ``https://`` URL,
        or the Hugging Face Hub: ``hf://<owner>/<repo>/<file>[@revision]``."""
        s = os.fspath(path)
        if s.startswith(("hf://", "https://", "http://")):
            from weightspec.io.hub import fetch_bytes

            return cls.from_bytes(fetch_bytes(s))
        p = Path(s)
        if p.is_dir():
            payloads: dict[str, np.ndarray] = {}
            if (p / "tensors.safetensors").exists():
                payloads = _read_payloads((p / "tensors.safetensors").read_bytes())
            return cls._from_manifest((p / "patch.json").read_bytes(), payloads)
        return cls.from_bytes(p.read_bytes())

    def push_to_hub(
        self,
        repo_id: str,
        path_in_repo: str | None = None,
        *,
        repo_type: Literal["model", "dataset"] = "model",
        private: bool = False,
        token: str | None = None,
        commit_message: str | None = None,
    ) -> str:
        """Upload this patch to a Hugging Face Hub repo and return its ``hf://`` address.

        Needs ``pip install huggingface_hub`` and a write token (``huggingface-cli login`` or ``HF_TOKEN``).
        """
        if self.result is None:
            raise ValueError("build() the patch before sharing it, so it carries verifiable hashes")
        try:
            from huggingface_hub import HfApi
        except ImportError as e:  # pragma: no cover
            raise ImportError("pip install 'weightspec[hub]' to push patches to the Hub") from e
        ext = ".wspec" if self._payloads else ".json"
        name = path_in_repo or f"{self.name}{ext}"
        api = HfApi(token=token)
        api.create_repo(repo_id, repo_type=repo_type, private=private, exist_ok=True)
        api.upload_file(
            path_or_fileobj=self.to_bytes(), path_in_repo=name, repo_id=repo_id, repo_type=repo_type,
            commit_message=commit_message or f"Add WeightSpec patch {self.name}",
        )
        prefix = "datasets/" if repo_type == "dataset" else ""
        return f"hf://{prefix}{repo_id}/{name}"

    # ------------------------------------------------------------------ display
    def summary(self) -> str:
        lines = [f"Patch {self.name!r}  (wspec {self.wspec})"]
        rev = f"@{self.base.revision[:12]}" if self.base.revision else ""
        lines.append(f"  base   {self.base.source}{rev}  arch={self.base.arch}")
        for i, op in enumerate(self.ops):
            extra = {k: v for k, v in op.model_dump(exclude={"target", "op", "u", "v", "note"}, exclude_none=True).items()}
            lines.append(f"  op {i}   {op.op:<14} {op.target}  {extra if extra else ''}")
        if self.result:
            total = sum(int(np.prod(t.shape)) for t in self.result.tensors.values())
            lines.append(f"  result {self.result.changed:,} stored values changed in {len(self.result.tensors)} tensor(s) "
                         f"({self.result.changed / max(total, 1):.4%} of those tensors)")
        if self._payloads:
            n = sum(a.size for a in self._payloads.values())
            lines.append(f"  payload {len(self._payloads)} array(s), {n:,} floats")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.summary()


def _write_payloads(f: io.BytesIO, payloads: Mapping[str, np.ndarray]) -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "t.safetensors"
        write_safetensors(path, {k: (np.asarray(v, dtype=np.float32), "F32") for k, v in payloads.items()})
        f.write(path.read_bytes())


def _read_payloads(data: bytes) -> dict[str, np.ndarray]:
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "t.safetensors"
        path.write_bytes(data)
        r = LocalSafetensors(path)
        out = {name: np.array(decode(r.read(name), info.dtype)) for name, info in r.tensors.items()}
        del r
        return out


@dataclass
class VerifyReport:
    ok: bool
    problems: list[str]
    changed: int
    applied: Applied | None = field(default=None, repr=False)

    def __bool__(self) -> bool:
        return self.ok

    def __str__(self) -> str:
        if self.ok:
            return f"OK: rebuilt bit for bit ({self.changed:,} values changed)"
        return "FAILED:\n  " + "\n  ".join(self.problems)


@dataclass
class Applied:
    """The result of applying a patch: edited tensors in storage format."""

    patch: Patch
    base: WeightStore
    tensors: dict[str, np.ndarray]
    changed: dict[str, int]
    targets: dict[str, str]

    @property
    def total_changed(self) -> int:
        return sum(self.changed.values())

    @property
    def store(self) -> WeightStore:
        """A store that sees the edited weights (for stacking patches or probing)."""
        return self.base.overlay(self.tensors)

    def numpy(self, name: str) -> np.ndarray:
        return decode(self.tensors[name], self.base.dtype(name))

    def result(self) -> Result:
        return Result(
            changed=self.total_changed,
            tensors={
                n: TensorResult(
                    dtype=self.base.dtype(n), shape=list(self.base.shape(n)), changed=self.changed[n], sha256=sha256_raw(raw)
                )
                for n, raw in self.tensors.items()
            },
            targets=self.targets,
        )

    def save_safetensors(self, path: str | os.PathLike[str]) -> Path:
        """Write only the edited tensors (load with ``model.load_state_dict(..., strict=False)``)."""
        write_safetensors(path, {n: (r, self.base.dtype(n)) for n, r in self.tensors.items()},
                          metadata={"weightspec_patch": self.patch.name})
        return Path(path)

    def export(self, out_dir: str | os.PathLike[str]) -> Path:
        """Write a complete edited checkpoint (all tensors, one file) plus config/tokenizer when available."""
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        st = self.base
        tensors = {n: (self.tensors.get(n, st.raw(n)), st.dtype(n)) for n in st.names}
        write_safetensors(out / "model.safetensors", tensors, metadata={"format": "pt", "weightspec_patch": self.patch.name})
        if st.config:
            (out / "config.json").write_text(json.dumps(st.config, indent=2))
        if st._tokenizer_file is not None:
            (out / "tokenizer.json").write_bytes(Path(st._tokenizer_file).read_bytes())
        self.patch.save(out / f"{self.patch.name}.wspec")
        return out

    def summary(self) -> str:
        parts = [f"{n}: {c:,} changed" for n, c in self.changed.items()]
        return f"Applied {self.patch.name!r}: {self.total_changed:,} values changed\n  " + "\n  ".join(parts)


def diff(
    a: WeightStore | Any,
    b: WeightStore | Any,
    tensors: Sequence[str] | None = None,
    name: str = "diff",
) -> Patch:
    """Express the difference between two checkpoints as a patch of exact ``set`` ops.

    Each changed tensor becomes one ``set`` on the smallest set of whole columns or
    rows that covers its changes. Rebuilding the patch on ``a`` reproduces ``b``
    bit for bit for those tensors.
    """
    sa, sb = load(a), load(b)
    names = list(tensors) if tensors is not None else [n for n in sa.names if n in sb]
    patch = Patch.new(sa, name, editor="diff")
    patch.provenance.notes = f"diff of {sb.source}" + (f"@{sb.revision}" if sb.revision else "")
    for n in names:
        if sa.shape(n) != sb.shape(n) or sa.dtype(n) != sb.dtype(n):
            raise ValueError(f"{n}: shape/dtype differ between checkpoints")
        ra, rb = sa.raw(n), sb.raw(n)
        width = ra.dtype.itemsize
        view = {1: np.uint8, 2: np.uint16, 4: np.uint32, 8: np.uint64}[width]
        mask = ra.view(view) != rb.view(view)
        if not mask.any():
            continue
        fb = decode(rb, sb.dtype(n))
        sel = f"tensor['{n}']"
        if mask.ndim == 1:
            idx = np.flatnonzero(mask)
            patch.set(f"{sel}.row[{','.join(map(str, idx))}]", fb[idx])
            continue
        cols = np.flatnonzero(mask.any(axis=0))
        rows = np.flatnonzero(mask.any(axis=1))
        if len(cols) * mask.shape[0] <= len(rows) * mask.shape[1]:
            patch.set(f"{sel}.col[{_fmt_idx(cols)}]", fb[:, cols])
        else:
            patch.set(f"{sel}.row[{_fmt_idx(rows)}]", fb[rows, :])
    return patch


def _fmt_idx(idx: np.ndarray) -> str:
    if len(idx) > 1 and np.all(np.diff(idx) == 1):
        return f"{idx[0]}:{idx[-1] + 1}"
    return ",".join(map(str, idx.tolist()))
