"""Render the README's terminal demo (docs/assets/demo.svg) from real command output.

    pip install rich && python scripts/render_demo.py

Every line in the image is produced by actually running the command, so the demo
cannot drift from what the tool does.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

from rich.console import Console
from rich.text import Text

ROOT = Path(__file__).resolve().parents[1]
EVAL = "https://huggingface.co/RikiyaT/Qwen3.5-0.8B-weight-space-programming/resolve/main/eval/"


def main() -> None:
    exe = shutil.which("weightspec") or str(Path(sys.executable).with_name("weightspec"))
    tmp = Path(tempfile.mkdtemp())
    for f in ("scores_base_math500.json", "scores_edited_math500.json"):
        urllib.request.urlretrieve(EVAL + f, tmp / f)
    shutil.copy(ROOT / "conformance" / "hu2026-math-primary.json", tmp / "hu2026-math-primary.json")
    shutil.copy(ROOT / "conformance" / "takehi2026-exp8-math-p50.wspec", tmp / "takehi2026-exp8-math-p50.wspec")
    commands = [
        ("# A 2.7 KB patch: the 1,023-weight edit from Hu et al. (2026), rebuilt from the paper's formula", None),
        ("weightspec show hu2026-math-primary.json", ["show", "hu2026-math-primary.json"]),
        ("weightspec verify hu2026-math-primary.json", ["verify", "hu2026-math-primary.json"]),
        ("# MIT's 1.6 GB edited checkpoint as a 250 KB patch; hashes equal the authors' manifest", None),
        ("weightspec verify takehi2026-exp8-math-p50.wspec", ["verify", "takehi2026-exp8-math-p50.wspec"]),
        ("# Paired, question-clustered statistics on MIT's published MATH-500 scores", None),
        ("weightspec stats scores_base_math500.json scores_edited_math500.json",
         ["stats", "scores_base_math500.json", "scores_edited_math500.json"]),
    ]
    console = Console(record=True, width=104, force_terminal=True, color_system="truecolor")
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    for shown, args in commands:
        if args is None:
            console.print(Text(shown, style="italic #8b949e"))
            continue
        console.print(Text("$ ", style="bold #3fb950") + Text(shown, style="bold"))
        out = subprocess.run([exe, *args], cwd=tmp, capture_output=True, text=True, env=env, check=True).stdout
        for line in out.rstrip().splitlines():
            style = "bold #3fb950" if line.startswith("OK") or "CONFIRMED" in line else None
            console.print(Text(line, style=style) if style else Text(line))
        console.print()
    out_path = ROOT / "docs" / "assets" / "demo.svg"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    console.save_svg(str(out_path), title="weightspec")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
