#!/usr/bin/env python3
"""Shared helper: turn a models.yaml entry into vLLM `vllm serve` arguments.

Single source of truth for how a model row becomes a command line, so that
docker-compose.yml (via gen_compose.py) and scripts/run_model.sh produce the
*identical* vLLM invocation.

CLI usage (consumed by run_model.sh):
    python3 _args.py shell <served_name|profile>
        -> prints shell assignments (HF_ID, SERVED_NAME, PORT, IMAGE, VLLM_ARGS).
    python3 _args.py args  <served_name|profile>
        -> prints the bare vllm serve args, space-joined (already shell-quoted).
    python3 _args.py list
        -> prints "profile  served_name  port  enabled" for each model.
"""
from __future__ import annotations

import json
import shlex
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
MODELS_YAML = ROOT / "models.yaml"


def load() -> tuple[dict, list[dict]]:
    data = yaml.safe_load(MODELS_YAML.read_text())
    return data.get("defaults", {}) or {}, data.get("models", []) or []


def find(key: str) -> tuple[dict, dict]:
    """Look up a model by served_name or profile (profile wins on ambiguity)."""
    defaults, models = load()
    for m in models:
        if m.get("profile") == key:
            return defaults, m
    matches = [m for m in models if m.get("served_name") == key]
    if len(matches) == 1:
        return defaults, matches[0]
    if len(matches) > 1:
        sys.exit(
            f"'{key}' matches {len(matches)} models by served_name; "
            f"use the unique profile instead: {[m['profile'] for m in matches]}"
        )
    sys.exit(f"No model with profile/served_name '{key}' in models.yaml")


def build_args(defaults: dict, m: dict, *, api_key: str | None = None) -> list[str]:
    """Return the argument list that follows `vllm serve` (incl. the hf_id)."""
    a: list[str] = [m["hf_id"]]
    a += ["--served-model-name", str(m["served_name"])]
    a += ["--host", "0.0.0.0", "--port", str(m["port"])]
    a += ["--max-model-len", str(m["max_model_len"])]
    a += ["--gpu-memory-utilization", str(m["gpu_memory_util"])]
    a += ["--tensor-parallel-size", str(m.get("tensor_parallel", 1))]

    # Multiple images per prompt (front+back / multi-page ID packets). Only for
    # multimodal models — text-only LLMs reject this flag, so it's gated on
    # max_images being set (presets set it; the GUI clears it for text models).
    if m.get("max_images"):
        a += ["--limit-mm-per-prompt", json.dumps({"image": int(m["max_images"])})]

    # Quantization. Pre-quantized FP8/AWQ repos (e.g. *-Instruct-FP8) carry a
    # quantization_config in their config.json and vLLM auto-detects them, so we
    # only pass --quantization when models.yaml sets it explicitly (e.g. "awq"
    # for repos that need it, or "fp8" to dynamically quantize a bf16 checkpoint).
    if m.get("quantization"):
        a += ["--quantization", str(m["quantization"])]
    if m.get("trust_remote_code"):
        a += ["--trust-remote-code"]
    if m.get("mm_processor_kwargs"):
        a += ["--mm-processor-kwargs", str(m["mm_processor_kwargs"])]
    if m.get("hf_overrides"):
        a += ["--hf-overrides", str(m["hf_overrides"])]

    gdb = defaults.get("guided_decoding_backend")
    if gdb:
        a += ["--guided-decoding-backend", str(gdb)]

    key = api_key if api_key is not None else defaults.get("api_key", "")
    if key:
        a += ["--api-key", str(key)]

    a += list(m.get("extra_args", []) or [])
    return a


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    cmd = sys.argv[1]

    if cmd == "list":
        _, models = load()
        for m in models:
            print(
                f"{m['profile']:<18} {m['served_name']:<16} "
                f"port={m['port']:<6} enabled={m.get('enabled', True)}"
            )
        return

    key = sys.argv[2]
    defaults, m = find(key)
    args = build_args(defaults, m)

    if cmd == "args":
        print(" ".join(shlex.quote(x) for x in args))
    elif cmd == "shell":
        image = m.get("image") or defaults.get("image", "vllm/vllm-openai:latest")
        print(f"HF_ID={shlex.quote(m['hf_id'])}")
        print(f"SERVED_NAME={shlex.quote(str(m['served_name']))}")
        print(f"PORT={shlex.quote(str(m['port']))}")
        print(f"IMAGE={shlex.quote(image)}")
        print(f"TRUST_REMOTE_CODE={'1' if m.get('trust_remote_code') else '0'}")
        print(f"VLLM_ARGS={shlex.quote(' '.join(shlex.quote(x) for x in args))}")
    else:
        sys.exit(f"Unknown command '{cmd}' (use: shell | args | list)")


if __name__ == "__main__":
    main()
