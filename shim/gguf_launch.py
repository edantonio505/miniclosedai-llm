#!/usr/bin/env python3
"""Download a GGUF from HuggingFace in Python, then `exec` llama-server on the
local file(s).

Why: the project's `llama-server` is built with `-DLLAMA_CURL=ON` but without TLS
(no OpenSSL/BoringSSL), so its own `--hf-repo` downloader aborts with
"HTTPS is not supported". Python's `huggingface_hub` has real TLS and honors
`HF_TOKEN` (so gated repos work too), so we fetch the weights here and hand
llama-server a plain local path — no SSL needed in the binary, and the ternary
PrismML `llama-server` still does the serving.

This process REPLACES itself with llama-server via os.execv, so the manager's
one-child pid/log/state machinery (NativeEngine) is unchanged. Download progress
(huggingface_hub's tqdm) streams to the model log, so the dashboard shows the
"downloading" phase exactly as the old `--hf-repo` path did.

Env in:
  GGUF_REPO     HF repo id (required)
  GGUF_FILE     specific .gguf filename (optional; smallest is auto-picked)
  GGUF_SERVER   path to the llama-server binary (required)
  GGUF_ARGS     JSON list of llama-server args to append after `-m <file>`
  HF_TOKEN      optional; required for gated repos
"""
from __future__ import annotations

import json
import os
import re
import sys

from huggingface_hub import hf_hub_download, list_repo_files

# A sharded GGUF is named "<base>-00001-of-00005.gguf"; llama-server is pointed at
# the first shard and loads the siblings from the same dir, so we must fetch all.
_SHARD_RE = re.compile(r"^(?P<base>.+)-(?P<idx>\d+)-of-(?P<total>\d+)\.gguf$", re.I)


def _log(msg: str) -> None:
    print(f"[gguf] {msg}", flush=True)


def _pick(files: list[str]) -> str:
    """Pick a GGUF when none was specified: prefer a low-bit quant, else shortest."""
    ggufs = [f for f in files if f.lower().endswith(".gguf")]
    if not ggufs:
        _log("no .gguf files found in the repo")
        sys.exit(1)
    # Sharded set → the first shard (its siblings come along in _shards()).
    firsts = [f for f in ggufs if (m := _SHARD_RE.match(f.rsplit("/", 1)[-1]))
              and int(m.group("idx")) == 1]
    pool = firsts or ggufs
    return sorted(pool, key=lambda f: (len(f), f))[0]


def _shards(target: str, files: list[str]) -> list[str]:
    """All files to download for `target`: itself, plus every sibling shard."""
    base = target.rsplit("/", 1)[-1]
    m = _SHARD_RE.match(base)
    if not m:
        return [target]
    prefix = target[: len(target) - len(base)]  # dir part of target ("" or "sub/")
    stem = m.group("base")
    out = []
    for f in files:
        fm = _SHARD_RE.match(f.rsplit("/", 1)[-1])
        if fm and fm.group("base") == stem and f.startswith(prefix):
            out.append(f)
    return sorted(out) or [target]


def main() -> None:
    repo = os.environ.get("GGUF_REPO")
    server = os.environ.get("GGUF_SERVER")
    if not repo or not server:
        _log("GGUF_REPO and GGUF_SERVER are required"); sys.exit(2)
    token = os.environ.get("HF_TOKEN") or None
    args = json.loads(os.environ.get("GGUF_ARGS", "[]"))

    try:
        files = list_repo_files(repo, token=token)
    except Exception as ex:  # network / auth / not-found — surface it in the log
        _log(f"could not list {repo}: {type(ex).__name__}: {ex}")
        sys.exit(1)

    target = os.environ.get("GGUF_FILE") or _pick(files)
    to_fetch = _shards(target, files)
    _log(f"model {repo} → {target}"
         + (f"  ({len(to_fetch)} shards)" if len(to_fetch) > 1 else ""))

    local_first = None
    for f in to_fetch:
        _log(f"downloading {f} …")
        path = hf_hub_download(repo_id=repo, filename=f, token=token)
        if f == target:
            local_first = path
    if local_first is None:  # single-file repo returns its own path
        local_first = hf_hub_download(repo_id=repo, filename=target, token=token)

    _log(f"ready: {local_first}")
    _log("starting llama-server")
    os.execv(server, [server, "-m", local_first, *args])


if __name__ == "__main__":
    main()
