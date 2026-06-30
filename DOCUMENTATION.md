# miniclosedai-llm — Documentation

Complete reference for the **miniclosedai-llm** model server and its web control
plane. For a task-oriented guide, start with [README.md](README.md); this document
covers architecture, the full HTTP API, the data/status model, engine internals,
configuration, and operations.

> **What it is.** A self-hosted control plane that downloads and runs **any
> HuggingFace LLM that vLLM supports — text or vision** — behind an
> OpenAI-compatible `/v1` API, so the models can be registered in the
> [miniclosedai](../miniclosedai) gateway as `openai` backends. It is part of the
> miniclosedai family (gateway + voice + llm) and mirrors their conventions
> (FastAPI + a no-build static UI, `dev.sh`, CSS-variable theming, SSE).

---

## Table of contents

1. [Architecture](#1-architecture)
2. [Components & file layout](#2-components--file-layout)
3. [Request lifecycle](#3-request-lifecycle)
4. [Launch engines](#4-launch-engines)
5. [Model status model](#5-model-status-model)
6. [HTTP API reference](#6-http-api-reference)
7. [models.yaml schema](#7-modelsyaml-schema)
8. [Registry & state (`models.local.json`)](#8-registry--state-modelslocaljson)
9. [Memory & fit analysis](#9-memory--fit-analysis)
10. [Configuration (environment)](#10-configuration-environment)
11. [Registering in miniclosedai](#11-registering-in-miniclosedai)
12. [Config-file workflow (compose + scripts)](#12-config-file-workflow-compose--scripts)
13. [Transformers shim](#13-transformers-shim)
14. [Testing](#14-testing)
15. [Security](#15-security)
16. [Operations & deployment](#16-operations--deployment)
17. [Troubleshooting](#17-troubleshooting)
18. [Hardware notes (unified memory / GB10)](#18-hardware-notes-unified-memory--gb10)
19. [Known limitations & design decisions](#19-known-limitations--design-decisions)
20. [Command-line interface (`mc`)](#20-command-line-interface-mc)

---

## 1. Architecture

There are **two independent ways** to run models; they share `models.yaml` and
`_args.py` but are otherwise separate.

```
                         ┌────────────────────────────────────────────┐
   Browser ──HTTP──▶     │  app.py  (FastAPI control plane, :8099)     │
   (dashboard)           │   static/  index.html · style.css · app.js │
                         │                                            │
                         │  model_manager.py                          │
                         │   ├─ registry (models.local.json)          │
                         │   ├─ analyze_model() ── HuggingFace API     │
                         │   ├─ Engine ─┬─ DockerEngine                │
                         │   │           └─ NativeEngine               │
                         │   └─ derive_status() ── probe /v1/models     │
                         └───────────────┬────────────────────────────┘
                                         │ docker run / vllm serve
                                         ▼
              ┌───────────────────────────────────────────────┐
              │  per-model vLLM OpenAI server  (:8001, :8002…) │
              │  GET /v1/models · POST /v1/chat/completions     │
              └───────────────────────────────────────────────┘
                                         ▲
                                         │  base_url  http://host:port/v1
              ┌───────────────────────────────────────────────┐
              │  miniclosedai gateway (:8095)  Settings→Backends│
              └───────────────────────────────────────────────┘
```

- **The control plane holds no ML dependencies.** `app.py` / `model_manager.py`
  import only FastAPI, httpx, pyyaml, and the standard library. All GPU/torch/vLLM
  work happens inside the launched container (Docker engine) or subprocess (native
  engine). This keeps the dashboard fast to install on any box.
- **Single source of truth for vLLM flags.** Both engines, the compose generator,
  and the run scripts call `_args.build_args(defaults, model)` to turn a model
  spec into the exact `vllm serve` argument list — so a model behaves identically
  whether launched from the GUI, `docker compose`, or `run_model.sh`.
- **Config-file workflow** (`models.yaml` → `gen_compose.py` → `docker-compose.yml`
  / `run_*.sh`) is fully independent of the dashboard and is the reproducible path
  for curated fleets.

---

## 2. Components & file layout

| Path | Imports | Responsibility |
|---|---|---|
| `app.py` | fastapi, httpx, `model_manager` | HTTP API + SSE, static mount, auth, the control-plane surface |
| `model_manager.py` | stdlib, pyyaml, `_args` | Registry, HF analysis, port/name allocation, the `Engine` abstraction, status derivation, `base_url` |
| `_args.py` | pyyaml | `load()` (reads `models.yaml`) + `build_args()` (→ `vllm serve` flags) |
| `static/index.html` | — | SPA shell (theme boot, banner, add form, model cards, log/test panels) |
| `static/app.js` | — | Vanilla JS controller (fetch, render, SSE logs, analyze, test, copy) |
| `static/style.css` | — | CSS-variable theme (light/dark), shared with the sibling apps |
| `dev.sh` | — | venv + deps + preflight + `uvicorn app:app` |
| `manager-requirements.txt` | — | fastapi, uvicorn, pyyaml, httpx, python-multipart |
| `Dockerfile.manager` | — | optional containerized control plane (mounts docker socket) |
| `models.yaml` | — | preset fleet + per-model serving config (source of truth) |
| `gen_compose.py` | pyyaml, `_args` | generate `docker-compose.yml` from `models.yaml` |
| `docker-compose.yml` | — | generated; one profile-gated service per preset |
| `scripts/run_model.sh`, `run_<model>.sh` | — | per-model launchers (docker/native) |
| `start.sh` / `stop.sh` | — | compose up/down by profile |
| `e2e_test.py` | stdlib | end-to-end regression harness (drives the dashboard) |
| `smoke_test.py` | stdlib | direct vision smoke test vs a vLLM `/v1` |
| `tests/test_image.png` | — | labelled test image (blue card, "ID 12345", yellow circle) |
| `shim/server.py` | fastapi, transformers, torch | OpenAI-compatible transformers fallback |
| `models.local.json` | — | **generated** registry state (gitignored) |
| `.run/` | — | **generated** native logs + docker orchestration logs (gitignored) |

---

## 3. Request lifecycle

**Adding & running a model (`POST /api/models`)**

1. `normalize_hf_id` — accepts `owner/name` or a full `https://huggingface.co/…`
   URL; strips scheme/host/`/tree/…`/`.git`; validates the `owner/name` shape.
2. `analyze_model` — queries the HuggingFace API for existence, gating, params,
   dtype, and a `config.json` check for `vision_config`; computes weight size and
   compares estimated need to free memory. If the model doesn't exist → `400`. If
   it doesn't fit and `force` is not set → `409` with the analysis attached (the UI
   surfaces a "Run anyway" button).
3. Decide serving params: defaults merged with any overrides; `max_model_len`
   **capped to the model's native context** (`max_position_embeddings`, so vLLM
   isn't asked for more than the model supports); `gpu_memory_util` sized
   adaptively from free memory; image flags cleared for text-only models.
4. **Under a lock**: reject if the same `hf_id` is already running (one launch per
   model — prevents accidental duplicates from a double-click / two tabs), then
   allocate a unique `served_name` (slug, de-duplicated) and the next free port
   (≥ 8001, both registry-free and OS-bindable), and create the entry.
5. Persist the entry to `models.local.json`, then (if `run`) call `start()`.
6. `start()` → the active engine's `launch()`, which is **non-blocking**: it kicks
   off the image pull (Docker) or the `vllm serve` subprocess (native) and returns
   immediately. Status is derived afterward, not awaited.

**Serving a chat** — once the model's container/process is up and `GET
/v1/models` returns 200, miniclosedai (or the dashboard's test box) calls
`POST {base_url}/chat/completions` directly on the model server. The control plane
is **not** in the inference path except for the Quick-test proxy.

---

## 4. Launch engines

`model_manager.Engine` is an interface with two implementations. Selection
(`select_engine`): `LAUNCH_ENGINE=docker|native` forces one; `auto` (default) picks
Docker when `docker info` succeeds, else native when `vllm` is importable, else a
degraded "no engine" state surfaced in `/api/health`.

### DockerEngine

- **Launch** detaches a small `bash` orchestrator (so the API request never blocks
  on a multi-GB pull): `docker pull <image>` → `docker rm -f` any stale container →
  `docker run -d` with `--gpus all --ipc=host --shm-size 16g -p P:P`, `HF_TOKEN`
  env, the host `HF_HOME` bind-mounted to `/root/.cache/huggingface`, and
  `miniclosedai.*` labels. Pull progress + the run result stream to
  `.run/<name>.docker.log`.
- **Container name** `vlm-<served_name>`. **State** via `docker inspect`. **Logs**
  via `docker logs` once the container exists, else the orchestrator log file.
  **Stop** `docker rm -f`. **Discover** (reconcile) reads `miniclosedai.*` labels
  from `docker ps -a`.

### NativeEngine

- **Launch** runs `vllm serve <args>` as a subprocess in its own process group,
  redirecting output to `.run/<name>.log`, and records `{pid,port,…}` in
  `.run/<name>.json`.
- **State** via `pid` liveness. **Logs** by tailing the file. **Stop**
  `SIGTERM→SIGKILL` the process group. **Discover** scans `.run/*.json`.

Both build their vLLM flags from `_args.build_args`, so a model's behavior is
engine-independent.

### Startup reconcile

On boot, `Manager.reconcile()` loads the registry, seeds presets from
`models.yaml`, and re-attaches to any live instances the active engine discovers
(so a dashboard restart doesn't orphan running models). Models marked running but
with no live instance are reset to `stopped` (it never auto-relaunches GPU
workloads on boot, to avoid surprise contention).

---

## 5. Model status model

`derive_status(entry)` combines the engine's coarse `state()` (`running` /
`exited` / `absent`), a `GET /v1/models` probe, and a log scan:

| status | meaning | how it's derived |
|---|---|---|
| `stopped` | not running, user intent stopped | engine `absent`, `desired_state != running` |
| `pulling` | image/weights downloading, no container yet | engine `absent`, `desired_state == running`, no error in log |
| `downloading` | container up, fetching weights (only when NOT already cached) | `running`, not serving, download markers in logs **and** `is_cached()` is false |
| `loading` | container up, loading/compiling | `running`, not serving, no error markers |
| `ready` | serving | `running` **and** `GET /v1/models` returns 200 with the served name |
| `error` | crashed or failed to start | engine `exited`, or error markers in logs (OOM, `401`, `no matching manifest`, `MANAGER-ERROR`, traceback) |

`ready` is the authoritative signal — the dashboard reveals the Register box and
Quick-test panel only when a model is `ready`. vLLM provides no clean
download/compile percentage, so the UI shows an indeterminate state plus the live
log tail.

**Cache-aware labeling.** If a model's weights are already in the HF cache
(`is_cached(hf_id)`), a launch never shows `downloading` — it goes straight to
`loading`. (vLLM's *weight-loading* progress bar prints a `%` bar that would
otherwise be misread as a download.) Re-running a model you've used before loads
from disk; it does **not** re-download.

---

## 6. HTTP API reference

This is the **control plane** API — model management (run / stop / inspect / cache).
**Inference does not happen here**: once a model is `ready`, you chat with it on its
*own* OpenAI-compatible endpoint at `http://<host>:<port>/v1` (see §11). So an agent
uses `/api` (or `mc`) to manage models and `:<port>/v1` to talk to them.

Base: `http://<host>:8099`. All endpoints accept an optional
`Authorization: Bearer <MANAGER_API_KEY>` header (enforced only if that env var is
set). JSON unless noted.

### Meta

| Method · Path | Body | Returns |
|---|---|---|
| `GET /api/health` | — | `{ok, version, engine, docker_ok, native_ok, gpu_ok, image, hf_home, lan_ip, public_host, dashboard_url, runpod, no_engine}` |
| `GET /api/gpu` | — | `{gpus:[{index,name,mem_total_mb,mem_used_mb,util_pct}], error?}` (mem fields `null` on unified memory) |
| `GET /api/test-image` | — | the bundled `tests/test_image.png` (`image/png`) |

### Analyze

| Method · Path | Body | Returns |
|---|---|---|
| `POST /api/analyze` | `{hf_id}` | `{exists, hf_id, pipeline_tag, multimodal, is_llm, gated, hf_token_present, params, dtype, max_ctx, size_gb, need_gb, available_gb, total_gb, fits}` or `{exists:false, error}` |

### Cache (already-downloaded models)

| Method · Path | Body | Returns |
|---|---|---|
| `GET /api/cache` | — | `{models:[{hf_id, size_gb, multimodal, arch}], hf_home, total_gb}` — runnable LLMs already on disk in the HF cache (filtered: causal-LM / vision-LM only; ASR/TTS/embeddings/tokenizers excluded). The UI's **Downloaded models** list. |
| `POST /api/cache/delete` | `{hf_id}` | `{ok:true}` — delete a model's weights from the cache to free disk. `404` if not present. |

### Models

| Method · Path | Body / params | Returns / notes |
|---|---|---|
| `GET /api/models` | — | `{models:[<view>…]}` (see view shape below) |
| `POST /api/models` | `{hf_id, served_name?, port?, params?, run=true, force=false}` | `201 <view>`. `400` invalid id / **already running** (a model is launched once per hf_id); `409 {message, analysis}` if it won't fit and `force` is false; `503` engine unavailable. Allocation is lock-guarded so concurrent duplicate submits can't both create an entry. |
| `POST /api/models/{id}/start` | — | (re)launch a stopped entry → `<view>` |
| `POST /api/models/{id}/stop` | — | stop (container `rm -f` / process kill) → `<view>` |
| `DELETE /api/models/{id}` | — | stop + remove the entry (keeps weights) → `{ok:true}` |
| `GET /api/models/{id}/status` | — | `{status, ready, detail}` (cheap poll) |
| `GET /api/models/{id}/logs` | — | **SSE** (`text/event-stream`): `data:{line}` log lines, periodic `data:{status,ready}`, terminal `data:{eof}` |
| `POST /api/models/{id}/test` | multipart: `prompt`, `max_tokens?`, `image?` | `{answer, usage, latency_ms}` — proxies a chat to the model (text, or text+image if a file is attached) |

**Model view shape** (`GET /api/models`):

```jsonc
{
  "id": "qwen2-5-7b-instruct",       // == served_name
  "hf_id": "Qwen/Qwen2.5-7B-Instruct",
  "served_name": "qwen2-5-7b-instruct",
  "port": 8001,
  "source": "custom",                // "preset" | "custom"
  "params": { "max_model_len": 16384, "gpu_memory_util": 0.73, … },
  "desired_state": "running",
  "engine": "docker",
  "multimodal": false,
  "size_gb": 15.2,
  "status": "ready",                 // derived (see §5)
  "ready": true,
  "detail": "qwen2-5-7b-instruct",
  "error": "",
  "base_url": "http://192.168.0.110:8001/v1",      // for miniclosedai
  "alt_base_url": "http://host.docker.internal:8001/v1",
  "local_url": "http://127.0.0.1:8001/v1",         // used by the test proxy
  "container": "vlm-qwen2-5-7b-instruct"
}
```

---

## 7. models.yaml schema

`models.yaml` defines `defaults` and a list of preset `models`. The GUI seeds these
as one-click cards; the compose/scripts workflow uses them directly.

```yaml
defaults:
  image: "vllm/vllm-openai:latest"   # override per-model or via $VLLM_IMAGE
  guided_decoding_backend: null      # see note below
  api_key: ""                        # "" → server accepts any key

models:
  - hf_id: "Qwen/Qwen3-VL-8B-Instruct"
    served_name: "qwen3-vl-8b"       # the OpenAI model id miniclosedai sees
    port: 8001
    quantization: null               # null | "fp8" | "awq" | "gptq"
    max_model_len: 16384
    gpu_memory_util: 0.90            # fraction of GPU memory (see §9 for unified mem)
    tensor_parallel: 1               # # GPUs to shard across
    max_images: 5                    # multimodal only; omitted/null for text models
    trust_remote_code: false         # true for InternVL etc. (runs repo code)
    mm_processor_kwargs: '{"max_pixels": 1605632}'   # vision pixel budget (JSON)
    hf_overrides: '{"max_dynamic_patch": 24}'        # InternVL patch budget (JSON)
    extra_args: []                   # free-form extra `vllm serve` flags
    profile: "qwen3-vl-8b"           # compose profile name
    enabled: true                    # skipped by gen_compose/start.sh if false
```

`_args.build_args` emits, in order: the `hf_id`, `--served-model-name`,
`--host 0.0.0.0`, `--port`, `--max-model-len`, `--gpu-memory-utilization`,
`--tensor-parallel-size`, `--limit-mm-per-prompt '{"image":N}'` **(only if
`max_images` is set)**, `--quantization` (if set), `--trust-remote-code` (if set),
`--mm-processor-kwargs` (if set), `--hf-overrides` (if set),
`--guided-decoding-backend` (only if `defaults.guided_decoding_backend` is truthy),
`--api-key` (if set), then `extra_args`.

> **`guided_decoding_backend` is `null` by default.** Recent vLLM **removed** the
> `--guided-decoding-backend` flag (structured/JSON output is on by default via the
> `auto` backend). Passing it makes vLLM exit with `unrecognized arguments`. Leave
> it null unless you pin an older `VLLM_IMAGE` that still accepts it.

---

## 8. Registry & state (`models.local.json`)

The dashboard persists a per-machine registry (gitignored, regenerated on first
run). Engine runtime is the source of truth for liveness; this file holds user
intent + chosen settings.

```jsonc
{
  "version": 1,
  "models": [
    {
      "id": "qwen2-5-7b-instruct", "hf_id": "Qwen/Qwen2.5-7B-Instruct",
      "served_name": "qwen2-5-7b-instruct", "port": 8001,
      "source": "custom",
      "params": { "max_model_len": 16384, "gpu_memory_util": 0.73,
                  "tensor_parallel": 1, "max_images": null,
                  "quantization": null, "trust_remote_code": false,
                  "mm_processor_kwargs": null, "hf_overrides": null,
                  "extra_args": [] },
      "desired_state": "running", "engine": "docker",
      "multimodal": false, "size_gb": 15.2,
      "created_at": "2026-06-28T19:45:00Z"
    }
  ]
}
```

Delete this file to reset the dashboard's model list (running containers are
re-discovered via their `miniclosedai.*` labels on next start). Downloaded weights
in `HF_HOME` are never touched.

### Model weight cache & reuse

Weights live in `HF_HOME` (`~/.cache/huggingface`), bind-mounted into every
container. A model is downloaded **once**; re-running it (or running it on a
different port) loads from disk — it never re-downloads. The dashboard's
**Downloaded models** list (`GET /api/cache`) scans the cache and shows the
runnable LLMs already present, with a one-click **Run** (loads from cache) and a
**Free** action (`POST /api/cache/delete`, deletes weights to reclaim disk).
`is_cached(hf_id)` is also what keeps a cached re-run from being mislabeled
"downloading" (see §5).

---

## 9. Memory & fit analysis

`analyze_model` estimates a model's footprint and whether it fits:

- **Size.** Preferred: the HF API `safetensors.total` param count × bytes-per-param
  (BF16/F16 = 2, F32 = 4, FP8/INT8 = 1, AWQ/GPTQ/4-bit ≈ 0.5, inferred from dtype +
  tags). Fallback: sum of weight-file sizes from the repo tree.
- **Need.** `≈ size_gb × 1.15 + 1 GB` (weights + overhead/KV headroom).
- **Available.** From `/proc/meminfo` (`MemAvailable`). On **unified-memory** parts
  (GB10) the GPU shares system RAM, so this is the right signal; `nvidia-smi`
  reports VRAM as N/A there. On a discrete GPU this is system RAM (a coarse upper
  bound — the real limit is VRAM).
- **Fit.** `need ≤ available`. Advisory only: the UI offers "Run anyway" (`force`).

**Adaptive `gpu_memory_utilization`.** vLLM interprets this as a fraction of
**total** device memory. On unified memory the OS/desktop already holds a chunk, so
the default `0.9` can exceed what's free and vLLM aborts (`Free memory … less than
desired GPU memory utilization`). When you don't set it explicitly, the manager
sizes it from **free** memory: `min(0.90, available×0.85/total)`. On a 121 GB GB10
with ~102 GB free this yields ≈ `0.73`.

---

## 10. Configuration (environment)

Loaded from `.env` (copy `.env.example`). `dev.sh` sources it; the compose workflow
reads it via `docker compose`.

| Var | Default | Used by | Notes |
|---|---|---|---|
| `HF_TOKEN` | — | both | Required for gated repos; passed as `HF_TOKEN` + `HUGGING_FACE_HUB_TOKEN` |
| `HF_HOME` | `~/.cache/huggingface` | both | Weight cache; set to a persistent volume on RunPod (`/workspace/hf-cache`) |
| `MANAGER_PORT` | `8099` | dashboard | Bind port |
| `LAUNCH_ENGINE` | `auto` | dashboard | `auto` / `docker` / `native` |
| `PUBLIC_HOST` / `ADVERTISE_HOST` | auto LAN IP | dashboard | Host advertised in `base_url` |
| `MANAGER_API_KEY` | — | dashboard | If set, all API calls require a matching Bearer token |
| `RUNPOD_POD_ID` | (set by RunPod) | dashboard | Switches `base_url` to the pod proxy form |
| `VLLM_IMAGE` | `vllm/vllm-openai:latest` | both | Docker image tag |
| `VLLM_API_KEY` | — | both | If set, vLLM enforces it (`--api-key`); enter the same value in miniclosedai |
| `SHIM_*` | see `shim/server.py` | shim | Shim model id / served name / port / dtype / max images / api key |

---

## 11. Registering in miniclosedai

miniclosedai's `_base_url()` (in `miniclosedai/llm.py`) only strips a trailing
slash, then calls `{base_url}/models` and `{base_url}/chat/completions`. vLLM serves
those at `/v1/models` and `/v1/chat/completions`, so **the base URL you register
must end in `/v1`** — which is exactly what the dashboard's Register box gives you.

- **Kind:** `openai`
- **Base URL:** `http://<LAN-IP>:<port>/v1` (default), or
  `http://host.docker.internal:<port>/v1` if miniclosedai runs as a Docker container
  on the same host (its container needs
  `extra_hosts: ["host.docker.internal:host-gateway"]`), or the RunPod proxy URL.
- **API key:** any non-empty string, unless `VLLM_API_KEY` is set (then match it).

miniclosedai discovers the model name from `GET /v1/models` and sends it back as the
`model` field on each request. Multimodal `content` arrays (text + `image_url`
data-URLs) pass straight through.

---

## 12. Config-file workflow (compose + scripts)

Independent of the dashboard; the reproducible path for curated fleets.

- `models.yaml` is the source of truth. `python3 gen_compose.py` regenerates
  `docker-compose.yml` (one service per preset, each behind a compose **profile**).
- `start.sh [profile…]` / `stop.sh` bring services up/down. `start.sh --list` lists
  profiles. Default profile: `qwen3-vl-8b`.
- `scripts/run_model.sh <profile> [--docker|--native] [extra vllm flags]` launches a
  single model without compose; `scripts/run_<model>.sh` are thin wrappers.
- `smoke_test.py --base-url http://localhost:<port>/v1 --model <served>` verifies a
  vision model with one image and a two-image request.

All of these produce the **same** `vllm serve` invocation as the GUI, via
`_args.build_args`.

---

## 13. Transformers shim

`shim/server.py` is an OpenAI-compatible `/v1` server backed by HuggingFace
`transformers` (`AutoProcessor` + `AutoModelForImageTextToText`), for models vLLM
can't serve yet. It implements `GET /v1/models` and `POST /v1/chat/completions`
(streaming + non-streaming), decodes base64 `image_url` data-URLs into PIL images,
and enforces a max-image limit. Configured entirely via `SHIM_*` env vars. Run it
with `./start.sh shim` (Docker) or `python3 shim/server.py` (native). miniclosedai
can't tell it apart from vLLM.

---

## 14. Testing

| Harness | Scope | Run |
|---|---|---|
| `e2e_test.py` | End-to-end regression via the **dashboard API**: add+run → ready → text chat (two prompts must differ) → image chat (vision) → cleanup. Exits non-zero on failure. | `python3 e2e_test.py [--quick] [--models …] [--base …] [--timeout …] [--keep]` |
| `smoke_test.py` | Direct vision check vs a single vLLM `/v1` (1 image + 2 images) | `python3 smoke_test.py --base-url http://localhost:<port>/v1 --model <served>` |

`e2e_test.py`'s default set is a small text model + a small vision model; the
distinct-answers assertion specifically guards the "every answer is identical"
class of bug. Both have been validated on the GB10 (text model ready in ~190 s,
the 3B vision model in ~410 s on first compile; the vision check correctly reads
the test image's blue background).

---

## 15. Security

- The dashboard binds `0.0.0.0` for trusted LAN / pod use. **Do not expose it to the
  public internet.** Set `MANAGER_API_KEY` to require a Bearer token if others can
  reach it.
- `trust_remote_code` runs arbitrary repo code inside the model container/process.
  It's **off by default** and surfaced as an Advanced toggle with a warning.
- The Quick-test proxy only targets `127.0.0.1:<port>` of manager-owned models — no
  arbitrary-URL fetch (no SSRF).
- Secrets (`HF_TOKEN`, `VLLM_API_KEY`, `MANAGER_API_KEY`) live in `.env`
  (gitignored), never in `models.yaml` or the registry.

---

## 16. Operations & deployment

- **Run it:** `./dev.sh` (foreground; logs to stdout). To keep it running detached,
  `nohup ./dev.sh > manager.log 2>&1 &`, or install a systemd service that runs
  `.venv/bin/uvicorn app:app --host 0.0.0.0 --port $MANAGER_PORT` after
  `network-online.target` + `docker.service`.
- **Restart safely:** match the port when killing so you don't hit sibling apps that
  also run `uvicorn app:app` — `pkill -f "uvicorn app:app --host 0.0.0.0 --port 8099"`.
- **Models keep running** across a dashboard restart (Docker containers / detached
  subprocesses) and are re-attached on boot. Stopping the dashboard does **not** stop
  models; use the UI Stop or `docker rm -f vlm-*`.
- **Containerized control plane:** `Dockerfile.manager` runs the dashboard itself in
  a container (mount the docker socket + `HF_HOME`); not the default, and not usable
  on RunPod (no docker daemon).

---

## 17. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `nvidia-smi`: *Driver/library version mismatch* | Driver updated without reloading the kernel module. **Reboot** (look for `/var/run/reboot-required`). GPU passthrough fails until `nvidia-smi` works. |
| Launch hangs / *No such container* on first run | First-time multi-GB image pull. The Docker engine now pulls in a streamed background step — watch **Logs**; the manager surfaces the real result, not a timeout. |
| *Free memory … less than desired GPU memory utilization* | `gpu_memory_util` × total > free. The manager sizes adaptively from free memory; if you set it manually, lower it. |
| `unrecognized arguments: --guided-decoding-backend …` | Flag removed in this vLLM version. Keep `guided_decoding_backend: null` (default), or pin an older `VLLM_IMAGE`. |
| `max_model_len … VLLM_ALLOW_LONG_MAX_MODEL_LEN` | Requested `max_model_len` exceeds the model's context window. The GUI caps it to the model's `max_position_embeddings` automatically; in `models.yaml`, lower `max_model_len`. |
| `401/403` / `GatedRepoError` in logs | Gated model — set `HF_TOKEN` and accept the license on HuggingFace. Analyze flags gating up front. |
| Quick test returns the **same answer** to every prompt | Fixed: the test endpoint reads `prompt` as a multipart form field (`Form`). If you forked it, ensure `prompt`/`max_tokens` use `Form(...)`, not bare params. |
| *model architecture not supported* | vLLM version too old for that model — pin a newer `VLLM_IMAGE` or use the shim. |
| miniclosedai Test fails / `host.docker.internal` unresolved | Use the LAN-IP base URL, or add `extra_hosts: ["host.docker.internal:host-gateway"]` to miniclosedai. |
| *Reachable but 0 models* in miniclosedai | Base URL missing `/v1`, or the model is still loading. |
| Unreachable from another machine | Open firewall ports (`ufw allow 8099/tcp`, `8001:8010/tcp`); confirm `0.0.0.0` binding. |

---

## 18. Hardware notes (unified memory / GB10)

This project was validated on an **NVIDIA GB10 (Grace-Blackwell, aarch64)** with
unified LPDDR memory. Points specific to such hardware:

- **arm64 image.** `vllm/vllm-openai:latest` has an arm64 build; it pulls and runs
  on the GB10 (the image is ~21 GB — first pull takes a while, then it's cached).
- **Unified memory.** `nvidia-smi` reports VRAM total/used as `[N/A]`; the GPU
  shares system memory (e.g. 121 GB total, ~102 GB free). The manager reads
  `/proc/meminfo` for fit analysis and sizes `gpu_memory_util` from free memory (§9).
- **First-load time.** vLLM's `torch.compile` + CUDA-graph capture dominates the
  first start of each model (minutes for larger/vision models); the compiled graph
  is cached under `HF_HOME`-adjacent `~/.cache/vllm`, so re-runs are faster.
- **Driver updates require a reboot** before `nvidia-smi`/GPU passthrough work again.

---

## 19. Known limitations & design decisions

- **No download/compile percentage.** vLLM doesn't expose one; the UI shows an
  indeterminate state + live logs rather than a fake bar.
- **Fit analysis is advisory.** Estimates can be off (quantization variants, MoE,
  activation memory); "Run anyway" always available.
- **One running instance per `served_name`.** Re-adding the same model produces a
  de-duplicated name (`-2`, `-3`) on a new port.
- **No auto-relaunch on boot.** Models marked running but absent after a restart are
  reset to `stopped` to avoid surprise GPU contention — start them explicitly.
- **Dashboard is single-tenant / trusted-network.** No multi-user auth beyond an
  optional shared Bearer token.
- **Discrete-GPU fit estimates are coarse** (uses system RAM, not VRAM). The real
  constraint there is VRAM; rely on the VRAM table and OOM feedback.

---

## 20. Command-line interface (`mc`)

`cli.py` (run via the `mc` wrapper) is a terminal client for the dashboard — the
same operations as the GUI, scriptable. It is a **thin HTTP client over the `/api`
endpoints** (§6), so CLI and GUI share the one `Manager` and stay in live sync. It
imports **nothing** beyond the standard library (`argparse`, `urllib`, `json`), so
it runs under any `python3` with no venv — and never duplicates `model_manager`
logic.

**Config** (env, or read from `.env`): `MANAGER_URL` (default
`http://localhost:$MANAGER_PORT`, 8099), `MANAGER_API_KEY` (sent as
`Authorization: Bearer …` when set), `VLLM_API_KEY` (used by `mc chat` when the
served model enforces a key).

**Commands → endpoint:**

| Command | Endpoint | Notes |
|---|---|---|
| `mc info` | `GET /api/health` + `/api/gpu` | engine, GPU, dashboard URL |
| `mc gpu [--json]` | `GET /api/gpu` | |
| `mc ls` / `list [--json]` | `GET /api/models` | table; forgiving id matching |
| `mc analyze <hf_id> [--json]` | `POST /api/analyze` | size / gated / fits |
| `mc run <hf_id> [flags] [--wait]` | `POST /api/models` | flags → `served_name/port/params/force`; `--wait` polls status |
| `mc start <id> [--wait]` | `POST /api/models/{id}/start` | re-run an existing stopped model |
| `mc stop <id>` | `POST /api/models/{id}/stop` | |
| `mc rm <id>` | `DELETE /api/models/{id}` | keeps weights |
| `mc status <id> [--json]` | `GET /api/models/{id}/status` | |
| `mc logs <id> [-f]` | `GET /api/models/{id}/logs` (SSE) | snapshot (bounded) or follow |
| `mc test <id> [prompt] [--image P] [--max-tokens N]` | `POST /api/models/{id}/test` | one-shot |
| `mc chat <id> [--temperature --max-tokens]` | model's `…/v1/chat/completions` | REPL, streamed, multi-turn history; `/reset`, `/exit` |
| `mc url <id>` | `GET /api/models` view | base_url + alt |
| `mc cache [--json]` / `cache rm <hf_id>` | `GET /api/cache` / `POST /api/cache/delete` | |
| `mc free <hf_id>` | `POST /api/cache/delete` | |
| `mc serve` | execs `./dev.sh` | start the dashboard |

`mc chat` is the only command that bypasses the API — it streams from the model's
own OpenAI endpoint (`http://localhost:<port>/v1/chat/completions`, port read from
the `GET /api/models` view) to keep full conversation history client-side, matching
how miniclosedai itself calls models.

**Exit codes:** `0` success · `1` operation error (HTTP 4xx/5xx, surfaced verbatim —
e.g. doesn't-fit, "already running") · `2` dashboard unreachable / usage error.

**Install:** symlink onto `PATH` if desired — `ln -s "$PWD/mc" ~/.local/bin/mc`.

### Remote / LLM-agent access

Everything binds `0.0.0.0` and is plain HTTP, so a coding/agent LLM (e.g. Claude
Code) on another host can drive the project with no GUI, via the **two surfaces**:
the **control plane** (`http://<host>:8099/api`, wrapped by `mc` — run/manage
models) and **inference** (each running model's own `http://<host>:<port>/v1`,
OpenAI-compatible — chat). Point the CLI at a remote host with `MANAGER_URL`
(default `http://localhost:$MANAGER_PORT`); send `MANAGER_API_KEY` to the control
plane and `VLLM_API_KEY` to a model when those are set — all via environment, no
interactive prompts. A typical agent loop: `mc analyze` → `mc run … --wait` →
`mc ls`/`mc url <id>` to get the port → chat on `:<port>/v1` with any OpenAI client
(the served-name is the `model` id), or `mc chat`/`mc test` to stay in the terminal.
The README's [Command-line interface](README.md) section has copy-paste examples;
firewall/auth notes are in §15 (Security) and §16 (Operations).
