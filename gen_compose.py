#!/usr/bin/env python3
"""Generate docker-compose.yml from models.yaml.

Each model becomes one service guarded by its own compose `profile`, so nothing
starts unless you ask for it (`docker compose --profile qwen3-vl-8b up`). The
vLLM command for each service is built by _args.build_args, the same code the
run_*.sh scripts use, so Docker and bare-metal launches stay identical.

    python3 gen_compose.py            # writes ./docker-compose.yml
    python3 gen_compose.py --stdout   # print to stdout instead

Re-run this whenever you edit models.yaml.
"""
from __future__ import annotations

import sys
from pathlib import Path

import yaml

from _args import ROOT, build_args, load

HEADER = (
    "# ============================================================================\n"
    "# GENERATED FILE — do not edit by hand.\n"
    "# Edit models.yaml, then run:  python3 gen_compose.py\n"
    "# ----------------------------------------------------------------------------\n"
    "# One service per model, each behind a compose `profile` (so it only starts\n"
    "# when selected). Bring a model up with start.sh or directly:\n"
    "#     docker compose --profile qwen3-vl-8b up -d\n"
    "# vLLM binds 0.0.0.0:<port>; register it in miniclosedai as an `openai`\n"
    "# backend with base_url  http://host.docker.internal:<port>/v1\n"
    "# ============================================================================\n"
)

GPU_DEPLOY = {
    "resources": {
        "reservations": {
            "devices": [{"driver": "nvidia", "count": "all", "capabilities": ["gpu"]}]
        }
    }
}


def healthcheck(port: int) -> dict:
    # vLLM image ships python3 but not necessarily curl; use urllib.
    test = (
        "python3 -c \"import urllib.request,sys; "
        f"sys.exit(0 if urllib.request.urlopen('http://localhost:{port}/health')"
        ".status==200 else 1)\""
    )
    return {
        "test": ["CMD-SHELL", test],
        "interval": "30s",
        "timeout": "10s",
        "retries": 20,
        "start_period": "600s",  # first run downloads weights — give it time
    }


def service_for(defaults: dict, m: dict) -> dict:
    port = int(m["port"])
    return {
        "image": "${VLLM_IMAGE:-" + (m.get("image") or defaults.get("image")) + "}",
        "profiles": [m["profile"]],
        "entrypoint": ["vllm", "serve"],
        "command": build_args(defaults, m),
        "ports": [f"{port}:{port}"],
        "environment": {
            "HF_TOKEN": "${HF_TOKEN:-}",
            "HUGGING_FACE_HUB_TOKEN": "${HF_TOKEN:-}",
            "HF_HOME": "/root/.cache/huggingface",
            "VLLM_API_KEY": "${VLLM_API_KEY:-}",
        },
        "volumes": ["${HF_HOME:-~/.cache/huggingface}:/root/.cache/huggingface"],
        "ipc": "host",  # vLLM needs large /dev/shm for tensor parallel + workers
        "shm_size": "16gb",
        "deploy": GPU_DEPLOY,
        "restart": "unless-stopped",
        "healthcheck": healthcheck(port),
    }


def shim_service() -> dict:
    """transformers fallback shim — for models vLLM cannot serve yet."""
    return {
        "build": {"context": "./shim"},
        "image": "miniclosedai-vlm-shim:latest",
        "profiles": ["shim"],
        "ports": ["${SHIM_PORT:-8009}:${SHIM_PORT:-8009}"],
        "environment": {
            "HF_TOKEN": "${HF_TOKEN:-}",
            "HUGGING_FACE_HUB_TOKEN": "${HF_TOKEN:-}",
            "HF_HOME": "/root/.cache/huggingface",
            # Set these to serve a specific model via the shim:
            "SHIM_MODEL_ID": "${SHIM_MODEL_ID:-Qwen/Qwen2.5-VL-7B-Instruct}",
            "SHIM_SERVED_NAME": "${SHIM_SERVED_NAME:-qwen2.5-vl-7b}",
            "SHIM_PORT": "${SHIM_PORT:-8009}",
            "SHIM_MAX_IMAGES": "${SHIM_MAX_IMAGES:-5}",
            "SHIM_API_KEY": "${VLLM_API_KEY:-}",
        },
        "volumes": ["${HF_HOME:-~/.cache/huggingface}:/root/.cache/huggingface"],
        "ipc": "host",
        "shm_size": "16gb",
        "deploy": GPU_DEPLOY,
        "restart": "unless-stopped",
    }


def main() -> None:
    defaults, models = load()
    services = {m["profile"]: service_for(defaults, m) for m in models}
    services["shim"] = shim_service()
    doc = {"services": services}

    body = yaml.safe_dump(doc, sort_keys=False, default_flow_style=False, width=4096)
    out = HEADER + "\n" + body

    if "--stdout" in sys.argv:
        sys.stdout.write(out)
        return
    target = ROOT / "docker-compose.yml"
    target.write_text(out)
    print(f"Wrote {target}  ({len(services)} services: {', '.join(services)})")


if __name__ == "__main__":
    main()
