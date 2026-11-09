"""``weightspec`` command-line interface."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

from weightspec._version import __version__


def _store(src: str | None, revision: str = "main"):  # type: ignore[no-untyped-def]
    from weightspec.store import load

    return load(src, revision=revision) if src else None


def cmd_inspect(a: argparse.Namespace) -> int:
    from weightspec.store import load

    st = load(a.source, revision=a.revision)
    print(st.describe())
    arch = st.arch
    print(f"  arch      {arch.name}  layers={arch.num_layers}  heads={arch.num_heads}  kv_heads={arch.num_kv_heads}  "
          f"head_dim={arch.head_dim}  gated_query={arch.gated_query}  norm_offset={arch.norm_offset}")
    print(f"  layers    {arch.layer_prefix}")
    print(f"  embed     {arch.embed}   unembed {arch.unembed or '(tied)'}   final_norm {arch.final_norm}")
    if arch.layer_types:
        full = [i for i, t in enumerate(arch.layer_types) if t == "full_attention"]
        print(f"  full-attention layers {full}")
    if a.tensors or a.filter:
        rx = re.compile(a.filter or ".")
        for n in st.names:
            if rx.search(n):
                print(f"  {st.dtype(n):5} {list(st.shape(n))!s:22} {n}")
    return 0


def cmd_resolve(a: argparse.Namespace) -> int:
    from weightspec.selectors import resolve
    from weightspec.store import load

    st = load(a.source, revision=a.revision)
    for sel in a.selectors:
        print(f"{sel}\n  -> {resolve(st, sel).describe()}")
    return 0


def cmd_show(a: argparse.Namespace) -> int:
    from weightspec.patch import Patch

    p = Patch.load(a.patch)
    print(p.to_json() if a.json else p.summary())
    return 0


def cmd_build(a: argparse.Namespace) -> int:
    from weightspec.patch import Patch
    from weightspec.policy import Policy

    p = Patch.load(a.patch)
    applied = p.build(_store(a.base, a.revision), Policy.paper() if a.policy == "paper" else None)
    out = Path(a.out or a.patch)
    if out.suffix == ".json" and p.payloads:
        out = out.with_suffix(".wspec")
    p.save(out)
    print(p.summary())
    print(f"built: {applied.total_changed:,} values changed -> {out}")
    return 0


def cmd_verify(a: argparse.Namespace) -> int:
    from weightspec.patch import Patch

    p = Patch.load(a.patch)
    report = p.verify(_store(a.base, a.revision))
    print(report)
    return 0 if report.ok else 1


def cmd_apply(a: argparse.Namespace) -> int:
    from weightspec.patch import Patch
    from weightspec.policy import Policy

    p = Patch.load(a.patch)
    applied = p.apply(_store(a.base, a.revision), Policy.paper() if a.policy == "paper" else None)
    print(applied.summary())
    if a.only_changed:
        out = Path(a.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        applied.save_safetensors(out)
    else:
        out = applied.export(a.out)
    print(f"wrote {out}")
    return 0


def cmd_diff(a: argparse.Namespace) -> int:
    from weightspec.patch import diff
    from weightspec.store import load

    sa, sb = load(a.a, revision=a.revision_a), load(a.b, revision=a.revision_b)
    names = a.tensors.split(",") if a.tensors else None
    if names is None and (sa.source.startswith("hf://") or sb.source.startswith("hf://")):
        print("note: comparing every tensor of a Hub model downloads it in full; use --tensors to limit", file=sys.stderr)
    p = diff(sa, sb, names, name=a.name)
    p.build(sa)
    p.save(a.out)
    print(p.summary())
    print(f"wrote {a.out}")
    return 0


def cmd_stats(a: argparse.Namespace) -> int:
    from weightspec.stats import load_scores, paired

    r = paired(load_scores(a.base, a.id_key, a.score_key), load_scores(a.edited, a.id_key, a.score_key))
    print(json.dumps(r.as_dict(), indent=2) if a.json else r)
    return 0


def cmd_schema(a: argparse.Namespace) -> int:
    from weightspec.patch import Patch

    schema = Patch.model_json_schema()
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    schema["$id"] = "https://weightspec.org/schema/patch-0.1.json"
    text = json.dumps(schema, indent=2)
    if a.out:
        Path(a.out).write_text(text + "\n")
    else:
        print(text)
    return 0


def _probe_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--dev", help="development items (JSONL with id / prompt / answer)")
    p.add_argument("--dev-limit", type=int)
    p.add_argument("--grader", default="math", help="math | exact | contains | choice | numeric")
    p.add_argument("--budget", type=int, default=50, help="forward-pass budget (default 50)")
    p.add_argument("--device", default="cpu", help="cpu | cuda | cuda:0 | mps")
    p.add_argument("--dtype", default="auto", choices=["auto", "bfloat16", "float16", "float32"])
    p.add_argument("--max-new-tokens", type=int, default=1024)
    p.add_argument("--no-chat", action="store_true", help="send prompts as raw text, without the chat template")


def _make_probe(a: argparse.Namespace, model: str):  # type: ignore[no-untyped-def]
    from weightspec.probe import Probe

    print(f"loading {model} on {a.device} ...", file=sys.stderr)
    return Probe.from_pretrained(model, budget=a.budget, device=a.device, dtype=a.dtype)


def _gen_defaults(a: argparse.Namespace) -> dict[str, object]:
    return {"max_new_tokens": a.max_new_tokens, "chat": not a.no_chat}


def cmd_mcp(a: argparse.Namespace) -> int:
    from weightspec.agent.toolkit import Toolkit
    from weightspec.evals import load_items
    from weightspec.mcp_server import main as serve

    toolkit = None
    if a.probe:
        items = load_items(a.dev, limit=a.dev_limit) if a.dev else []
        toolkit = Toolkit(a.workspace, probe=_make_probe(a, a.probe), items=items, grader=a.grader,
                          generate_defaults=_gen_defaults(a))
    kwargs = {"host": a.host, "port": a.port} if a.transport != "stdio" else {}
    serve(workspace=a.workspace, transport=a.transport, toolkit=toolkit, **kwargs)
    return 0


def cmd_run(a: argparse.Namespace) -> int:
    from weightspec.agent.editors import make_editor
    from weightspec.agent.session import Session
    from weightspec.evals import load_items

    if not a.dev:
        raise ValueError("--dev is required: development items the editor diagnoses and confirms on")
    dev = load_items(a.dev, limit=a.dev_limit)
    holdout = load_items(a.holdout, limit=a.holdout_limit) if a.holdout else None
    editor = make_editor(a.editor, **({"effort": a.effort} if a.editor.startswith("anthropic") else {}))
    kwargs: dict[str, Any] = dict(probe=_make_probe(a, a.model), editor=editor, goal=a.goal, dev=dev, holdout=holdout,
                  grader=a.grader, max_steps=a.max_steps, holdout_n=a.holdout_n,
                  generate_defaults=_gen_defaults(a))

    def show(kind: str, d: dict[str, Any]) -> None:
        if kind == "editor":
            text = " ".join(str(d.get("text") or "").split())
            calls = ", ".join(str(c) for c in d.get("calls") or [])
            print(f"[{d['step']}] {text[:160]}" + (f"  -> {calls}" if calls else ""))
        elif kind == "tool" and d.get("is_error"):
            print(f"    ! {d['name']}: {str(d['result'])[:160]}")

    session = Session.resume(a.workspace, **kwargs) if a.resume else Session(workspace=a.workspace, **kwargs)
    result = session.run(on_event=show)
    print(result)
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="weightspec", description="Small, verifiable patches for neural-network weights.")
    ap.add_argument("--version", action="version", version=f"weightspec {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def src(p: argparse.ArgumentParser, name: str = "source", help: str = "Hub repo id or local path") -> None:
        p.add_argument(name, help=help)
        p.add_argument("--revision", default="main")

    p = sub.add_parser("inspect", help="describe a checkpoint and its architecture map")
    src(p)
    p.add_argument("--tensors", action="store_true", help="list all tensors")
    p.add_argument("--filter", help="regex filter for the tensor list")
    p.set_defaults(fn=cmd_inspect)

    p = sub.add_parser("resolve", help="show which tensor block a selector names")
    src(p)
    p.add_argument("selectors", nargs="+")
    p.set_defaults(fn=cmd_resolve)

    p = sub.add_parser("show", help="print a patch")
    p.add_argument("patch")
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=cmd_show)

    for name, fn, hlp in (
        ("build", cmd_build, "apply a patch to its base and record hashes and changed counts"),
        ("verify", cmd_verify, "rebuild a patch from its base and check every hash"),
        ("apply", cmd_apply, "write the edited checkpoint (or only the edited tensors)"),
    ):
        p = sub.add_parser(name, help=hlp)
        p.add_argument("patch")
        p.add_argument("--base", help="override the base recorded in the patch")
        p.add_argument("--revision", default="main")
        if name != "verify":
            p.add_argument("--policy", choices=["permissive", "paper"], default="permissive")
        if name == "build":
            p.add_argument("--out", help="where to save the built patch (default: in place)")
        if name == "apply":
            p.add_argument("--out", required=True, help="output directory (or .safetensors with --only-changed)")
            p.add_argument("--only-changed", action="store_true", help="write only the edited tensors")
        p.set_defaults(fn=fn)

    p = sub.add_parser("diff", help="turn the difference between two checkpoints into a patch")
    p.add_argument("a")
    p.add_argument("b")
    p.add_argument("--revision-a", default="main")
    p.add_argument("--revision-b", default="main")
    p.add_argument("--tensors", help="comma-separated tensor names to compare")
    p.add_argument("--name", default="diff")
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_diff)

    p = sub.add_parser("stats", help="paired, question-clustered comparison of two eval runs")
    p.add_argument("base")
    p.add_argument("edited")
    p.add_argument("--id-key")
    p.add_argument("--score-key")
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=cmd_stats)

    p = sub.add_parser("mcp", help="run the MCP server so coding agents can inspect weights and build patches")
    p.add_argument("--workspace", default=".", help="folder for saved patches and eval files (default: .)")
    p.add_argument("--transport", choices=["stdio", "streamable-http", "sse"], default="stdio")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--probe", help="load this model (Hub id / path) to enable budgeted forward-pass tools")
    _probe_args(p)
    p.set_defaults(fn=cmd_mcp)

    p = sub.add_parser("run", help="let an LLM edit a model's weights under a forward-pass budget")
    p.add_argument("model", help="Hub id or local path of the model to edit")
    p.add_argument("--goal", required=True, help="what to improve, in plain language")
    p.add_argument("--editor", default="anthropic:claude-opus-5-5",
                   help="anthropic:<model> | openai:<model>[@base_url] | scripted:<file.json>")
    p.add_argument("--effort", default="high", help="Claude effort: low | medium | high | xhigh | max")
    p.add_argument("--holdout", help="sealed held-out items, evaluated once after the editor finishes")
    p.add_argument("--holdout-limit", type=int)
    p.add_argument("--holdout-n", type=int, default=1, help="samples per held-out item")
    p.add_argument("--workspace", default="wsp-run")
    p.add_argument("--max-steps", type=int, default=300)
    p.add_argument("--resume", action="store_true", help="continue a checkpointed session in --workspace")
    _probe_args(p)
    p.set_defaults(fn=cmd_run)

    p = sub.add_parser("schema", help="print the JSON Schema of the patch format")
    p.add_argument("--out")
    p.set_defaults(fn=cmd_schema)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.fn(args))
    except Exception as e:  # pragma: no cover - user-facing error path
        if "--debug" in sys.argv or __import__("os").environ.get("WEIGHTSPEC_DEBUG"):
            raise
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
