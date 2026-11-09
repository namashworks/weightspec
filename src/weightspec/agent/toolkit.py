"""The weight-space programming toolkit: one set of tools for every agent surface.

The same :class:`Toolkit` backs the MCP server (any coding agent) and
:class:`~weightspec.agent.session.Session` (WeightSpec's own agent loop), so the
two cannot drift apart. Tools never execute agent-written code. Agents inspect
weights, write ops as JSON (RFC-0001), and ask for builds, probes and evaluations.
Every edit passes through the policy. Every file write stays in the workspace.
Every forward pass is charged to the budget.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, get_type_hints

import numpy as np
from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError, create_model

from weightspec.controls import controls as make_control_family
from weightspec.directions import DirectionExpr, EvalContext
from weightspec.evals import Item, evaluate, get_grader
from weightspec.ops import Op
from weightspec.patch import Patch, diff
from weightspec.policy import Policy
from weightspec.probe import BudgetExceeded, Probe
from weightspec.selectors import resolve
from weightspec.stats import load_scores, paired
from weightspec.store import WeightStore, load

__all__ = ["ToolFailure", "ToolSpec", "Toolkit"]

EXPECTED_ERRORS = (ValueError, KeyError, IndexError, PermissionError, FileNotFoundError, ValidationError,
                   BudgetExceeded, RuntimeError)


class ToolFailure(Exception):
    """An anticipated tool error whose message is shown to the agent."""


@dataclass
class ToolSpec:
    name: str
    description: str
    fn: Callable[..., Any]
    args_model: type[BaseModel]
    needs_probe: bool = False
    session_only: bool = False

    def input_schema(self) -> dict[str, Any]:
        schema = self.args_model.model_json_schema()
        schema.pop("title", None)
        return schema


def tool(description: str, needs_probe: bool = False, session_only: bool = False) -> Callable[[Any], Any]:
    def deco(fn: Any) -> Any:
        fn._ws_tool = (description, needs_probe, session_only)
        return fn

    return deco


def _excerpt(text: str, n: int = 300) -> str:
    return text if len(text) <= n else text[: n // 2] + " [...] " + text[-n // 2 :]


class Toolkit:
    """Tools over model weights, draft patches, a probe and development data.

    Args:
        workspace: folder for all files the tools write (and the only place they read patches/evals from).
        probe: optional :class:`~weightspec.probe.Probe`; enables the forward-pass tools.
        items: development items ``{"id", "prompt", "answer"}`` the agent may see and evaluate on.
        grader: grader name or callable for ``items``.
        policy: edit policy enforced on every build (default: the papers' rules).
        generate_defaults: default sampling settings for forward passes (e.g. max_new_tokens).
    """

    def __init__(
        self,
        workspace: str | os.PathLike[str] = ".",
        probe: Probe | None = None,
        items: Sequence[Item] | None = None,
        grader: Any = "math",
        policy: Policy | None = None,
        generate_defaults: dict[str, Any] | None = None,
    ):
        self.workspace = Path(workspace)
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.probe = probe
        self.items = {str(it["id"]): it for it in (items or [])}
        self.grader = get_grader(grader)
        self.policy = policy or Policy.paper()
        self.generate_defaults = {"max_new_tokens": 1024, **(generate_defaults or {})}
        self.stores: dict[tuple[str, str], WeightStore] = {}
        self.patches: dict[str, Patch] = {}
        self.vectors: dict[str, np.ndarray] = {}
        self.finished: dict[str, Any] | None = None
        self._counter = 0
        self._op = TypeAdapter(Op)
        self._dir = TypeAdapter(DirectionExpr)
        self._specs = self._collect()

    # ------------------------------------------------------------------ registry
    def _collect(self) -> dict[str, ToolSpec]:
        specs = {}
        for name in dir(type(self)):
            fn = getattr(type(self), name)
            meta = getattr(fn, "_ws_tool", None)
            if meta is None:
                continue
            hints = get_type_hints(fn)
            import inspect

            fields: dict[str, Any] = {}
            for p in list(inspect.signature(fn).parameters.values())[1:]:
                default = ... if p.default is inspect.Parameter.empty else p.default
                fields[p.name] = (hints[p.name], default)
            model = create_model(f"{name}_args", __config__=ConfigDict(extra="forbid"), **fields)
            specs[name] = ToolSpec(name, meta[0], getattr(self, name), model, meta[1], meta[2])
        return specs

    def specs(self, include_session_only: bool = True) -> list[ToolSpec]:
        return [s for s in self._specs.values()
                if (self.probe is not None or not s.needs_probe) and (include_session_only or not s.session_only)]

    def call(self, name: str, arguments: dict[str, Any] | None = None) -> Any:
        """Validate arguments and run a tool. Anticipated failures raise :class:`ToolFailure`."""
        spec = self._specs.get(name)
        if spec is None or spec not in self.specs():
            raise ToolFailure(f"unknown tool {name!r}")
        try:
            args = spec.args_model(**(arguments or {}))
            return spec.fn(**{k: getattr(args, k) for k in type(args).model_fields})
        except ToolFailure:
            raise
        except EXPECTED_ERRORS as e:
            msg = e.args[0] if isinstance(e, KeyError) and e.args else str(e)
            raise ToolFailure(f"{type(e).__name__}: {msg}") from e

    # ------------------------------------------------------------------ helpers
    def _store(self, source: str | None, revision: str = "main") -> WeightStore:
        alias = source == "model" and not (self.workspace / "model").exists() and not os.path.exists("model")
        if source in (None, "") or alias:
            if self.probe is not None:
                return self.probe.store
            if len({k for k in self.stores}) == 1:
                return next(iter(self.stores.values()))
            raise ValueError("give a source (Hub id or path) - no default model is attached")
        key = (source, revision)
        if key not in self.stores:
            src = source.removeprefix("file://")
            in_ws = self.workspace / src
            if not os.path.isabs(src) and in_ws.exists():
                src = str(in_ws)
            self.stores[key] = load(src, revision=revision)
        return self.stores[key]

    def _patch(self, name: str) -> Patch:
        if name not in self.patches:
            raise KeyError(f"no draft patch named {name!r}; existing: {sorted(self.patches)}")
        return self.patches[name]

    def safe_path(self, path: str, must_exist: bool = False) -> Path:
        p = Path(path)
        p = (self.workspace / p).resolve() if not p.is_absolute() else p.resolve()
        if self.workspace.resolve() not in (p, *p.parents):
            raise PermissionError(f"{path!r} is outside the workspace {self.workspace}")
        if must_exist and not p.exists():
            raise FileNotFoundError(path)
        return p

    def _next(self, prefix: str) -> Path:
        self._counter += 1
        return self.workspace / f"{prefix}_{self._counter:03d}"

    def _fill(self, node: Any, store: WeightStore, patch: Patch) -> None:
        """Resolve token words to ids and copy named vectors into the patch's payloads."""
        if isinstance(node, dict):
            if node.get("fn") in ("readout", "embedding") and "ids" not in node and "tokens" in node:
                node["ids"] = store.token_ids(node["tokens"])
            if node.get("fn") == "payload" and node.get("key") in self.vectors:
                patch.payloads[node["key"]] = self.vectors[node["key"]]
            for v in node.values():
                self._fill(v, store, patch)
        elif isinstance(node, list):
            for v in node:
                self._fill(v, store, patch)

    def _built(self, name: str | None) -> Patch | None:
        if name is None:
            return None
        p = self._patch(name)
        if p.result is None:
            p.build(policy=self.policy)
        return p

    def _gen_kwargs(self, max_new_tokens: int | None) -> dict[str, Any]:
        kw = dict(self.generate_defaults)
        if max_new_tokens is not None:
            kw["max_new_tokens"] = max_new_tokens
        return kw

    # ================================================================== weight tools
    @tool("Describe a checkpoint: architecture map (layers, heads, head width, gated queries, full-attention "
          "layers) and optionally tensors matching a regex (max 200). source defaults to the attached model.")
    def inspect_model(self, source: str | None = None, revision: str = "main",
                      tensor_filter: str | None = None) -> dict[str, Any]:
        st = self._store(source, revision)
        a = st.arch
        info: dict[str, Any] = {
            "source": st.source, "revision": st.revision, "tensors": len(st.names), "params": st.numel(),
            "arch": a.name, "layers": a.num_layers, "heads": a.num_heads, "kv_heads": a.num_kv_heads,
            "head_dim": a.head_dim, "gated_query": a.gated_query, "norm_offset": a.norm_offset,
            "layer_prefix": a.layer_prefix, "embed": a.embed, "unembed": a.unembed or "(tied to embed)",
            "final_norm": a.final_norm,
            "full_attention_layers": [i for i, t in enumerate(a.layer_types) if t == "full_attention"]
            if a.layer_types else None,
        }
        if tensor_filter:
            rx = re.compile(tensor_filter)
            names = [n for n in st.names if rx.search(n)][:200]
            info["matching_tensors"] = [{"name": n, "dtype": st.dtype(n), "shape": list(st.shape(n))} for n in names]
        return info

    @tool("Show which tensor block a selector names, e.g. layer[15].attn.head[7].out or layer[23].mlp.down.col[507].")
    def resolve_selector(self, selector: str, source: str | None = None, revision: str = "main") -> dict[str, Any]:
        t = resolve(self._store(source, revision), selector)
        return {"selector": selector, "tensor": t.tensor, "block": t.describe(), "block_shape": list(t.block_shape)}

    @tool("Summary statistics of a block of weights, plus the columns with the largest norms.")
    def tensor_stats(self, selector: str, source: str | None = None, revision: str = "main",
                     top_k: int = 10) -> dict[str, Any]:
        st = self._store(source, revision)
        t = resolve(st, selector)
        b = t.get(st.load(t.tensor)).astype(np.float64)
        col_norms = np.linalg.norm(b, axis=0)
        cols = t.cols if t.cols is not None else np.arange(b.shape[1])
        top = np.argsort(-col_norms)[:top_k]
        return {
            "block": t.describe(), "mean": float(np.mean(b)), "std": float(np.std(b)), "min": float(np.min(b)),
            "max": float(np.max(b)), "frobenius_norm": float(np.linalg.norm(b)),
            "top_columns_by_norm": [{"col": int(cols[i]), "norm": float(col_norms[i])} for i in top],
        }

    @tool("Readout lens: which tokens a residual-stream direction promotes (+) and suppresses (-). Give a selector "
          "naming one column/row of hidden size (e.g. layer[23].mlp.down.col[507]), a named vector, or a "
          "direction as JSON. Loads the unembedding once (can be hundreds of MB).")
    def readout_tokens(self, selector: str | None = None, vector: str | None = None,
                       direction_json: str | None = None, k: int = 15, source: str | None = None,
                       revision: str = "main") -> dict[str, Any]:
        st = self._store(source, revision)
        given = [x is not None for x in (selector, vector, direction_json)]
        if sum(given) != 1:
            raise ValueError("give exactly one of selector / vector / direction_json")
        if selector is not None:
            t = resolve(st, selector)
            vec = t.get(st.load(t.tensor)).astype(np.float64)
            if 1 not in vec.shape:
                raise ValueError(f"{selector} is a {vec.shape} block; select a single column or row")
            vec = vec.reshape(-1)
        elif vector is not None:
            vec = self.vectors[vector].astype(np.float64)
        else:
            vec = EvalContext(st).eval(self._dir.validate_json(str(direction_json))).reshape(-1)
        a = st.arch
        u = st.load(a.global_tensor("unembed")).astype(np.float32)
        if a.final_norm:
            u = u * (st.load(a.final_norm) + a.norm_offset).astype(np.float32)
        if u.shape[1] != vec.shape[0]:
            raise ValueError(f"direction has length {vec.shape[0]}, hidden size is {u.shape[1]}")
        scores = u @ vec.astype(np.float32)

        def words(ids: np.ndarray) -> list[dict[str, Any]]:
            try:
                tok = st.tokenizer()
                return [{"id": int(i), "token": tok.decode([int(i)]), "score": round(float(scores[i]), 5)} for i in ids]
            except Exception:
                return [{"id": int(i), "score": round(float(scores[i]), 5)} for i in ids]

        return {"promotes": words(np.argsort(-scores)[:k]), "suppresses": words(np.argsort(scores)[:k])}

    @tool("JSON Schema of ops and directions. Call this before add_op.")
    def op_schema(self) -> dict[str, Any]:
        return {"op": self._op.json_schema(), "direction": self._dir.json_schema(),
                "named_vectors": "use {'fn': 'payload', 'key': '<vector name>'} to reference a named vector"}

    # ================================================================== patch tools
    @tool("Start a new draft patch on a base checkpoint (default: the attached model).")
    def create_patch(self, name: str, source: str | None = None, description: str | None = None,
                     revision: str = "main", editor: str = "agent") -> dict[str, Any]:
        if not re.fullmatch(r"[A-Za-z0-9._-]{1,100}", name):
            raise ValueError("patch names may only contain letters, digits, '.', '_' and '-'")
        self.patches[name] = Patch.new(self._store(source, revision), name, description, editor=editor)
        return {"created": name, "base": self.patches[name].base.model_dump(exclude_none=True)}

    @tool("Append one op (a JSON object following op_schema) to a draft. Readout/embedding directions may give "
          "'tokens' without 'ids'. Named vectors (from contrast_activations) are referenced as payloads.")
    def add_op(self, patch: str, op_json: str) -> dict[str, Any]:
        p = self._patch(patch)
        raw = json.loads(op_json)
        store = p.base_store()
        self._fill(raw, store, p)
        op = self._op.validate_python(raw)
        resolve(store, op.target)
        p.ops.append(op)
        p.result = None
        return {"patch": patch, "ops": len(p.ops), "summary": p.summary()}

    @tool("Apply a draft to its base under the policy ('paper': embeddings, LM head and final norm frozen, no "
          "whole-tensor rescaling; or 'permissive'). Records hashes and the exact number of stored values changed.")
    def build_patch(self, patch: str, policy: Literal["paper", "permissive"] = "paper",
                    max_changed: int | None = None) -> dict[str, Any]:
        p = self._patch(patch)
        pol = self.policy if policy == "paper" else Policy.permissive()
        if max_changed is not None:
            pol = pol.model_copy(update={"max_changed": max_changed})
        applied = p.build(policy=pol)
        return {"patch": patch, "changed": applied.total_changed, "per_tensor": applied.changed,
                "targets": applied.targets, "summary": p.summary()}

    @tool("Create control variants of a draft (dose sweep, reversed direction, equal-norm random orthogonal "
          "directions, ablation) as new built drafts named '<patch>--<control>'.")
    def make_controls(self, patch: str) -> dict[str, Any]:
        p = self._patch(patch)
        out = {}
        for name, c in make_control_family(p).items():
            c.build(policy=self.policy)
            self.patches[c.name] = c
            out[name] = {"patch": c.name, "changed": c.result.changed if c.result else None}
        return {"controls": out}

    @tool("Save a draft to the workspace as .json (no payloads) or .wspec.")
    def save_patch(self, patch: str, filename: str | None = None) -> dict[str, Any]:
        p = self._patch(patch)
        if p.result is None:
            raise ValueError("build_patch first, so the file carries verifiable hashes")
        fn = filename or f"{p.name}{'.wspec' if p.payloads else '.json'}"
        path = self.safe_path(fn)
        path.parent.mkdir(parents=True, exist_ok=True)
        p.save(path)
        return {"saved": str(path), "bytes": path.stat().st_size}

    @tool("Load a patch from the workspace, an https:// URL or hf://owner/repo/file into the drafts.")
    def load_patch(self, uri: str) -> dict[str, Any]:
        src = uri if uri.startswith(("hf://", "https://", "http://")) else str(self.safe_path(uri, must_exist=True))
        p = Patch.load(src)
        self.patches[p.name] = p
        return {"loaded": p.name, "summary": p.summary()}

    @tool("Rebuild a built patch from its base weights and check every recorded hash.")
    def verify_patch(self, patch: str) -> dict[str, Any]:
        r = self._patch(patch).verify()
        return {"ok": r.ok, "changed": r.changed, "problems": r.problems}

    @tool("Turn the difference between two checkpoints into a patch of exact set ops. Pass tensor names to "
          "compare (comparing all tensors of a Hub model downloads it).")
    def diff_models(self, name: str, source_a: str, source_b: str, tensors: list[str] | None = None) -> dict[str, Any]:
        sa = self._store(source_a)
        p = diff(sa, self._store(source_b), tensors, name=name)
        p.build(sa)
        self.patches[name] = p
        return {"patch": name, "summary": p.summary()}

    @tool("Paired, question-clustered comparison of base vs edited eval results (workspace JSON/JSONL files "
          "with an id and a 0/1 score per sample).")
    def paired_stats(self, base_file: str, edited_file: str, id_key: str | None = None,
                     score_key: str | None = None) -> dict[str, Any]:
        r = paired(load_scores(self.safe_path(base_file, True), id_key, score_key),
                   load_scores(self.safe_path(edited_file, True), id_key, score_key))
        return {**r.as_dict(), "report": str(r)}

    @tool("List draft patches and named vectors in this session.")
    def list_patches(self) -> dict[str, Any]:
        return {
            "patches": {n: {"ops": len(p.ops), "built": p.result is not None,
                            "changed": p.result.changed if p.result else None} for n, p in self.patches.items()},
            "vectors": {n: {"length": int(v.shape[0]), "norm": float(np.linalg.norm(v))} for n, v in self.vectors.items()},
        }

    # ================================================================== probe tools
    @tool("Forward-pass budget: used, limit, remaining. Generation and teacher-forced requests cost 1 each.",
          needs_probe=True)
    def budget_status(self) -> dict[str, Any]:
        assert self.probe is not None
        return self.probe.budget.status()

    @tool("List development items (id, prompt, reference answer) you may use for diagnosis and confirmation.",
          needs_probe=True)
    def list_items(self, offset: int = 0, limit: int = 20) -> dict[str, Any]:
        ids = list(self.items)
        page = ids[offset: offset + limit]
        return {"total": len(ids), "items": [
            {"id": i, "prompt": _excerpt(str(self.items[i]["prompt"]), 400), "answer": str(self.items[i]["answer"])[:200]}
            for i in page]}

    @tool("COSTS 1 BUDGET. Sample answers from the model. weights: 'original', 'edited' (needs patch) or 'both' "
          "(same items and seeds; paired). Use item_ids to grade against reference answers, or free-form prompts. "
          "Returns accuracy, truncation and short excerpts; full texts are saved to the workspace.", needs_probe=True)
    def generate(self, item_ids: list[str] | None = None, prompts: list[str] | None = None,
                 weights: Literal["original", "edited", "both"] = "original", patch: str | None = None,
                 n: int = 1, max_new_tokens: int | None = None, show: int = 3) -> dict[str, Any]:
        assert self.probe is not None
        if (item_ids is None) == (prompts is None):
            raise ValueError("give exactly one of item_ids / prompts")
        items = [self.items[i] for i in item_ids] if item_ids is not None else [{"id": f"p{k}", "prompt": p} for k, p in enumerate(prompts or [])]
        if weights != "original":
            if patch is None:
                raise ValueError("weights 'edited'/'both' need a patch")
            self.probe.set_patch(self._built(patch), self.policy)
        res = self.probe.generate([it["prompt"] for it in items], weights=weights, n=n, **self._gen_kwargs(max_new_tokens))
        out: dict[str, Any] = {"budget": self.probe.budget.status(), "arms": res.summary()}
        path = self._next("gen").with_suffix(".jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for arm, per_item in res.arms.items():
                for it, samples in zip(items, per_item, strict=True):
                    for s_i, s in enumerate(samples):
                        score = self.grader(s["text"], it) if item_ids is not None else None
                        f.write(json.dumps({"arm": arm, "id": it["id"], "sample": s_i, "score": score, **s}) + "\n")
        out["saved"] = path.name
        if item_ids is not None:
            scores = {arm: {it["id"]: [self.grader(s["text"], it) for s in samples]
                            for it, samples in zip(items, per_item, strict=True)} for arm, per_item in res.arms.items()}
            for arm, sc in scores.items():
                out["arms"][arm]["accuracy"] = float(np.mean([x for v in sc.values() for x in v]))
            if weights == "both":
                out["paired"] = str(paired(scores["original"], scores["edited"]))
        out["excerpts"] = [{"arm": arm, "id": it["id"], "text": _excerpt(samples[0]["text"])}
                           for arm, per_item in res.arms.items() for it, samples in list(zip(items, per_item, strict=True))[:show]]
        return out

    @tool("COSTS 1 BUDGET. Read given texts (no generation): per-text mean token log-prob, and for each requested "
          "MLP layer the most active intermediate units (mean over tokens). Arrays are saved to the workspace.",
          needs_probe=True)
    def teacher_forced(self, texts: list[str], weights: Literal["original", "edited"] = "original",
                       patch: str | None = None, mlp_layers: list[int] | None = None,
                       hidden_layers: list[int] | None = None, top_units: int = 10) -> dict[str, Any]:
        assert self.probe is not None
        if weights == "edited":
            if patch is None:
                raise ValueError("weights 'edited' needs a patch")
            self.probe.set_patch(self._built(patch), self.policy)
        res = self.probe.teacher_forced(texts, weights, hidden_layers or (), mlp_layers or ())
        arrays: dict[str, np.ndarray] = {}
        summary = []
        for t_i, r in enumerate(res):
            lp = r["token_logprobs"]
            entry: dict[str, Any] = {"text": _excerpt(texts[t_i], 120), "tokens": len(r["token_ids"]),
                                     "mean_logprob": float(np.mean(lp)) if len(lp) else 0.0, "sum_logprob": float(np.sum(lp))}
            for layer, act in r.get("mlp", {}).items():
                mean = act.mean(axis=0)
                arrays[f"t{t_i}_mlp{layer}"] = act
                top = np.argsort(-np.abs(mean))[:top_units]
                entry[f"mlp{layer}_top_units"] = [{"unit": int(u), "mean_act": round(float(mean[u]), 4)} for u in top]
            for layer, h in r.get("hidden", {}).items():
                arrays[f"t{t_i}_hidden{layer}"] = h
            arrays[f"t{t_i}_logprobs"] = lp
            summary.append(entry)
        path = self._next("tf").with_suffix(".npz")
        np.savez_compressed(path, **arrays)  # type: ignore[arg-type]
        return {"budget": self.probe.budget.status(), "texts": summary, "saved": path.name}

    @tool("COSTS 1 BUDGET. Contrast two sets of texts at one layer: mean activation of set A minus set B, over "
          "the MLP intermediate units (site='mlp'; a reader direction for rank1 edits) or the residual stream "
          "after the layer (site='residual'; a write direction for add/project_scale). Saves the difference as "
          "a named vector you can use in ops.", needs_probe=True)
    def contrast_activations(self, texts_a: list[str], texts_b: list[str], layer: int,
                             site: Literal["mlp", "residual"] = "mlp",
                             weights: Literal["original", "edited"] = "original", patch: str | None = None,
                             name: str | None = None, top: int = 15) -> dict[str, Any]:
        assert self.probe is not None
        if not texts_a or not texts_b:
            raise ValueError("both text sets must be non-empty")
        if weights == "edited":
            if patch is None:
                raise ValueError("weights 'edited' needs a patch")
            self.probe.set_patch(self._built(patch), self.policy)
        res = self.probe.teacher_forced([*texts_a, *texts_b], weights,
                                        hidden_layers=[layer] if site == "residual" else (),
                                        mlp_layers=[layer] if site == "mlp" else ())
        key = "mlp" if site == "mlp" else "hidden"
        means = [r[key][layer].mean(axis=0) for r in res]
        d = np.mean(means[: len(texts_a)], axis=0) - np.mean(means[len(texts_a):], axis=0)
        vname = name or f"contrast_{site}{layer}_{self._counter + 1}"
        if not re.fullmatch(r"[A-Za-z0-9._-]{1,100}", vname):
            raise ValueError("vector names may only contain letters, digits, '.', '_' and '-'")
        self._counter += 1
        self.vectors[vname] = d.astype(np.float32)
        order = np.argsort(-np.abs(d))[:top]
        return {"budget": self.probe.budget.status(), "vector": vname, "length": int(d.shape[0]),
                "norm": float(np.linalg.norm(d)),
                "top_units": [{"unit": int(u), "diff": round(float(d[u]), 5)} for u in order]}

    @tool("COSTS 1 + len(controls) BUDGET. Paired evaluation of a patch vs the original weights on development "
          "items (same items and seeds), optionally against named controls from make_controls (e.g. "
          "['random-0', 'reverse']). Use fresh items not used to choose the edit.", needs_probe=True)
    def evaluate_patch(self, patch: str, item_ids: list[str] | None = None, n: int = 1,
                       controls: list[str] | None = None, max_new_tokens: int | None = None) -> dict[str, Any]:
        assert self.probe is not None
        p = self._built(patch)
        assert p is not None
        ids = item_ids or list(self.items)
        ctrl = {c: self._built(f"{patch}--{c}") for c in (controls or [])}
        rep = evaluate(self.probe, p, [self.items[i] for i in ids], self.grader, n=n,
                       controls={k: v for k, v in ctrl.items() if v is not None}, **self._gen_kwargs(max_new_tokens))
        return {"budget": self.probe.budget.status(), "report": str(rep), **rep.as_dict()}

    # ================================================================== session tools
    @tool("End the session. patch: the draft to submit, or null to keep the original weights (a valid outcome "
          "when no edit is clearly better). explanation: what you changed, the evidence, and what you did NOT verify.",
          session_only=True)
    def finish(self, explanation: str, patch: str | None = None) -> dict[str, Any]:
        if patch is not None:
            self._built(patch)
        self.finished = {"patch": patch, "explanation": explanation}
        return {"finished": True, "patch": patch}
