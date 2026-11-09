"""Hugging Face ``transformers`` backend for forward-pass probes.

The model is loaded once and always rests at its *original* weights. For an
``edited`` request the edited tensors are copied in, the forward pass runs, and
the originals are restored (in a ``finally``), so the base weights can be read at
any time and only the edited tensors are ever held twice in memory.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator, Sequence
from typing import Any

import numpy as np

__all__ = ["TransformersBackend"]

Chat = list[dict[str, str]]


def _torch() -> Any:
    try:
        import torch
    except ImportError as e:  # pragma: no cover
        raise ImportError("pip install 'weightspec[probe]' (needs torch and transformers)") from e
    return torch


class TransformersBackend:
    def __init__(self, model: Any, tokenizer: Any, device: str | None = None):
        self.torch = _torch()
        self.model = model.eval()
        self.tokenizer = tokenizer
        if getattr(self.tokenizer, "pad_token", None) is None and getattr(self.tokenizer, "eos_token", None):
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.device = device or str(next(model.parameters()).device)
        self._params = dict(model.named_parameters())
        self._params.update(dict(model.named_buffers()))
        self._edits: dict[str, Any] = {}  # param name -> torch tensor (edited values)

    # ------------------------------------------------------------------ construction
    @classmethod
    def from_pretrained(
        cls, model_id: str, revision: str | None = None, device: str = "cpu", dtype: str = "auto", **kwargs: Any
    ) -> TransformersBackend:
        torch = _torch()
        from transformers import AutoModelForCausalLM, AutoTokenizer

        td = {"auto": "auto", "bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[dtype]
        model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision, dtype=td, **kwargs).to(device)  # type: ignore[arg-type]
        tok = AutoTokenizer.from_pretrained(model_id, revision=revision)
        return cls(model, tok, device)

    # ------------------------------------------------------------------ weights
    def param_name(self, name: str) -> str:
        """Map a checkpoint tensor name to this module's parameter name.

        Exact match first. Otherwise match the ``layers.N.…`` suffix uniquely, so
        patches built on a multimodal checkpoint (``model.language_model.layers.…``)
        apply to the text-only module (``model.layers.…``) and back.
        """
        if name in self._params:
            return name
        key = name[name.find("layers."):] if "layers." in name else name.split(".", 1)[-1]
        hits = [p for p in self._params if p.endswith(key) and not p.startswith(("mtp.", "model.visual", "visual"))]
        if len(hits) != 1:
            raise KeyError(f"cannot map tensor {name!r} onto the model ({len(hits)} candidates)")
        return hits[0]

    def set_edits(self, tensors: dict[str, tuple[np.ndarray, str]]) -> None:
        """Register edited tensors ``{checkpoint name: (raw array, dtype)}`` (not applied until a request)."""
        from weightspec.numerics import decode

        torch = self.torch
        edits = {}
        for name, (raw, dtype) in tensors.items():
            pname = self.param_name(name)
            p = self._params[pname]
            if tuple(p.shape) != tuple(raw.shape):
                raise ValueError(f"{name}: edit shape {raw.shape} != parameter shape {tuple(p.shape)}")
            if dtype == "BF16" and p.dtype == torch.bfloat16:
                t = torch.from_numpy(np.ascontiguousarray(raw).view(np.int16).copy()).view(torch.bfloat16)
            else:
                t = torch.from_numpy(decode(raw, dtype).copy()).to(p.dtype)
            edits[pname] = t.to(p.device)
        self._edits = edits

    def clear_edits(self) -> None:
        self._edits = {}

    @contextlib.contextmanager
    def weights(self, which: str) -> Iterator[None]:
        if which == "original" or not self._edits:
            if which == "edited" and not self._edits:
                raise RuntimeError("no edits registered; set a patch on the probe first")
            yield
            return
        torch = self.torch
        saved = {}
        try:
            with torch.no_grad():
                for name, t in self._edits.items():
                    p = self._params[name]
                    saved[name] = p.detach().clone()
                    p.data.copy_(t)
            yield
        finally:
            with torch.no_grad():
                for name, orig in saved.items():
                    self._params[name].data.copy_(orig)

    # ------------------------------------------------------------------ requests
    def _encode(self, item: str | Chat, chat: bool, enable_thinking: bool | None) -> list[int]:
        tok = self.tokenizer
        if chat and getattr(tok, "chat_template", None):
            msgs = item if isinstance(item, list) else [{"role": "user", "content": item}]
            kw = {} if enable_thinking is None else {"enable_thinking": enable_thinking}
            ids = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, **kw)
            if hasattr(ids, "input_ids"):
                ids = ids["input_ids"]
            if hasattr(ids, "tolist"):
                ids = ids.tolist()
            return list(ids[0] if ids and isinstance(ids[0], list) else ids)
        text = item if isinstance(item, str) else "\n".join(m["content"] for m in item)
        return list(tok(text, add_special_tokens=True)["input_ids"])

    def generate(
        self,
        items: Sequence[str | Chat],
        which: str,
        n: int = 1,
        max_new_tokens: int = 512,
        temperature: float = 0.6,
        top_p: float = 0.95,
        top_k: int = 20,
        seed: int = 0,
        chat: bool = True,
        enable_thinking: bool | None = None,
    ) -> list[list[dict[str, Any]]]:
        """Sample ``n`` answers per item. Item i always uses the same seed, in every arm."""
        torch = self.torch
        out: list[list[dict[str, Any]]] = []
        eos = self.tokenizer.eos_token_id
        with self.weights(which), torch.no_grad():
            for i, item in enumerate(items):
                ids = torch.tensor([self._encode(item, chat, enable_thinking)], device=self.device)
                torch.manual_seed(seed * 1_000_003 + i)
                gen = self.model.generate(
                    ids, attention_mask=torch.ones_like(ids), max_new_tokens=max_new_tokens,
                    do_sample=temperature > 0, temperature=temperature if temperature > 0 else None,
                    top_p=top_p if temperature > 0 else None, top_k=top_k if temperature > 0 else None,
                    num_return_sequences=n, pad_token_id=self.tokenizer.pad_token_id,
                )
                samples = []
                for s in range(n):
                    new = gen[s, ids.shape[1]:].tolist()
                    finished = eos is not None and eos in new
                    if finished:
                        new = new[: new.index(eos)]
                    samples.append({
                        "text": self.tokenizer.decode(new, skip_special_tokens=True),
                        "tokens": len(new),
                        "truncated": not finished and len(new) >= max_new_tokens,
                    })
                out.append(samples)
        return out

    def teacher_forced(
        self,
        texts: Sequence[str | Chat],
        which: str,
        hidden_layers: Sequence[int] = (),
        mlp_layers: Sequence[int] = (),
        topk: int = 0,
        chat: bool = False,
        down_proj_names: dict[int, str] | None = None,
    ) -> list[dict[str, Any]]:
        """Read texts; return per-token log-probs, optional top-k, hidden states and MLP activations.

        ``hidden[l]`` is the residual stream *after* layer ``l``. ``mlp[l]`` is the input
        to layer ``l``'s MLP down-projection (one value per intermediate unit).
        """
        torch = self.torch
        results: list[dict[str, Any]] = []
        captured: dict[int, Any] = {}
        hooks = []
        for layer in mlp_layers:
            if not down_proj_names or layer not in down_proj_names:
                raise ValueError(f"no down-projection name for layer {layer}")
            mod = self.model.get_submodule(self.param_name(down_proj_names[layer]).rsplit(".", 1)[0])

            def hook(_m: Any, args: tuple[Any, ...], layer: int = layer) -> None:
                captured[layer] = args[0].detach()

            hooks.append(mod.register_forward_pre_hook(hook))
        try:
            with self.weights(which), torch.no_grad():
                for item in texts:
                    ids_list = (self._encode(item, True, None) if chat and isinstance(item, list)
                                else self._encode(item, False, None))
                    ids = torch.tensor([ids_list], device=self.device)
                    out = self.model(ids, output_hidden_states=bool(hidden_layers))
                    logp = torch.log_softmax(out.logits[0, :-1].float(), dim=-1)
                    target = ids[0, 1:]
                    r: dict[str, Any] = {
                        "token_ids": ids_list,
                        "token_logprobs": logp.gather(1, target[:, None])[:, 0].cpu().numpy(),
                    }
                    if topk:
                        v, ix = logp.topk(topk, dim=-1)
                        r["topk_ids"] = ix.cpu().numpy()
                        r["topk_logprobs"] = v.cpu().numpy()
                    if hidden_layers:
                        r["hidden"] = {layer: out.hidden_states[layer + 1][0].float().cpu().numpy()
                                       for layer in hidden_layers}
                    if mlp_layers:
                        r["mlp"] = {layer: captured[layer][0].float().cpu().numpy() for layer in mlp_layers}
                    results.append(r)
        finally:
            for h in hooks:
                h.remove()
        return results
