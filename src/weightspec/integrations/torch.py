"""PyTorch integration: read a live module's weights and apply patches in place."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from weightspec.patch import Applied, Patch
    from weightspec.policy import Policy

__all__ = ["apply_to_module", "state_dict_to_arrays"]

_TORCH_TO_ST = {
    "torch.bfloat16": "BF16", "torch.float16": "F16", "torch.float32": "F32", "torch.float64": "F64",
    "torch.int64": "I64", "torch.int32": "I32", "torch.int16": "I16", "torch.int8": "I8",
    "torch.uint8": "U8", "torch.bool": "BOOL",
}


def _to_raw(t: Any) -> tuple[np.ndarray, str]:
    import torch

    dt = _TORCH_TO_ST.get(str(t.dtype))
    if dt is None:
        raise TypeError(f"unsupported dtype {t.dtype}")
    t = t.detach().to("cpu").contiguous()
    if dt == "BF16":
        return t.view(torch.int16).numpy().view(np.uint16).copy(), dt
    return t.numpy().copy(), dt


def state_dict_to_arrays(module: Any) -> dict[str, tuple[np.ndarray, str]]:
    """Copy a module's state dict into raw numpy arrays (deduplicating tied weights by name)."""
    return {name: _to_raw(t) for name, t in module.state_dict().items()}


class ModuleSource:
    """Reads a live module's tensors on demand (no up-front copy of the whole model)."""

    def __init__(self, module: Any):
        self.tensors = dict(module.state_dict(keep_vars=True))

    def read(self, name: str) -> np.ndarray:
        return _to_raw(self.tensors[name])[0]

    def read_rows(self, name: str, rows: Any) -> np.ndarray:
        return _to_raw(self.tensors[name][list(rows)])[0]

    def describe(self) -> dict[str, tuple[Any, str, tuple[int, ...]]]:
        out = {}
        for name, t in self.tensors.items():
            dt = _TORCH_TO_ST.get(str(t.dtype))
            if dt is not None:
                out[name] = (self, dt, tuple(t.shape))
        return out


def apply_to_module(module: Any, patch: Patch, policy: Policy | None = None) -> Applied:
    """Apply ``patch`` to a ``torch.nn.Module`` in place and return what changed.

    >>> ws.apply(model, ws.Patch.load("fix.wspec"))
    """
    import torch

    from weightspec.store import WeightStore

    store = WeightStore.from_torch(module, arch=patch.base.arch or "auto")
    applied = patch.apply(store, policy)
    params = dict(module.named_parameters())
    params.update(dict(module.named_buffers()))
    with torch.no_grad():
        for name, raw in applied.tensors.items():
            p = params[name]
            if store.dtype(name) == "BF16":
                new = torch.from_numpy(raw.view(np.int16).copy()).view(torch.bfloat16)
            else:
                new = torch.from_numpy(raw.copy())
            p.data.copy_(new.reshape(p.shape).to(p.device))
    return applied
