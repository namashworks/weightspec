"""Hugging Face Hub access with no extra dependencies (urllib only).

Revisions are always resolved to an immutable commit sha, which is recorded in
patches so a patch names exactly the weights it was built against.
"""

from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from weightspec.io.safetensors import RemoteSafetensors, _cache_dir

__all__ = ["HubRepo"]

ENDPOINT = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")


def _headers() -> dict[str, str]:
    h = {"User-Agent": "weightspec"}
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        h["Authorization"] = f"Bearer {token}"
    return h


def _get_json(url: str) -> Any:
    with urllib.request.urlopen(urllib.request.Request(url, headers=_headers()), timeout=60) as r:
        return json.loads(r.read())


MAX_PATCH_BYTES = 512 << 20


def parse_hf_uri(uri: str) -> tuple[str, str, str, str]:
    """``hf://[datasets/]owner/repo/path/in/repo[@revision]`` -> (repo_type, repo_id, path, revision)."""
    rest = uri[len("hf://"):]
    repo_type = "model"
    if rest.startswith(("datasets/", "spaces/")):
        kind, rest = rest.split("/", 1)
        repo_type = kind[:-1]
    revision = "main"
    if "@" in rest:
        rest, revision = rest.rsplit("@", 1)
    parts = rest.split("/")
    if len(parts) < 3:
        raise ValueError(f"expected hf://owner/repo/file, got {uri!r}")
    return repo_type, "/".join(parts[:2]), "/".join(parts[2:]), revision


def fetch_bytes(uri: str, max_bytes: int = MAX_PATCH_BYTES) -> bytes:
    """Download a (small) file from ``hf://…`` or ``http(s)://…``."""
    if uri.startswith("hf://"):
        repo_type, repo_id, path, rev = parse_hf_uri(uri)
        prefix = {"model": "", "dataset": "datasets/", "space": "spaces/"}[repo_type]
        url = (f"{ENDPOINT}/{prefix}{repo_id}/resolve/{urllib.parse.quote(rev, safe='')}/"
               f"{urllib.parse.quote(path)}")
    else:
        url = uri
    with urllib.request.urlopen(urllib.request.Request(url, headers=_headers()), timeout=120) as r:
        data: bytes = r.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise ValueError(f"{uri} is larger than {max_bytes >> 20} MB; is it really a patch?")
    return data


class HubRepo:
    """A model repo pinned to one commit."""

    def __init__(self, repo_id: str, revision: str = "main"):
        self.repo_id = repo_id
        rev = urllib.parse.quote(revision, safe="")
        info = _get_json(f"{ENDPOINT}/api/models/{repo_id}/revision/{rev}")
        self.sha: str = info["sha"]
        self.files: list[str] = [s["rfilename"] for s in info.get("siblings", [])]

    def url(self, filename: str) -> str:
        return f"{ENDPOINT}/{self.repo_id}/resolve/{self.sha}/{urllib.parse.quote(filename)}"

    def download(self, filename: str) -> Path:
        """Download a (small) file into the cache and return its path."""
        path = _cache_dir() / "files" / self.repo_id.replace("/", "--") / self.sha / filename
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            req = urllib.request.Request(self.url(filename), headers=_headers())
            with urllib.request.urlopen(req, timeout=300) as r:
                data = r.read()
            tmp = path.with_suffix(path.suffix + ".tmp")
            tmp.write_bytes(data)
            os.replace(tmp, path)
        return path

    def safetensors_files(self) -> list[str]:
        index = [f for f in self.files if f.endswith(".safetensors.index.json")]
        if index:
            weight_map = json.loads(self.download(index[0]).read_text())["weight_map"]
            return sorted(set(weight_map.values()))
        files = sorted(f for f in self.files if f.endswith(".safetensors"))
        if not files:
            raise FileNotFoundError(f"{self.repo_id}@{self.sha} has no .safetensors files")
        return files

    def open(self, filename: str) -> RemoteSafetensors:
        key = f"{self.repo_id.replace('/', '--')}/{self.sha}/{filename}"
        return RemoteSafetensors(self.url(filename), key, _headers())
