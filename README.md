# miniclosedai-llm — run any HuggingFace LLM locally, behind an OpenAI API

Paste a HuggingFace model id, click **Download & Run**, and this serves it on your
own GPU behind an **OpenAI-compatible `/v1` API** — then register it in a
self-hosted [**miniclosedai**](../miniclosedai) gateway (Settings → Backends, kind
`openai`) and use it for chats and bots, exactly like the built-in models.

It runs **any LLM that vLLM supports — text *or* vision (VLM)** — with a built-in
**Analyze** step (size / gated / does-it-fit before you download) and a **Quick
test** box (send a prompt, optionally an image, see the answer). It's the sibling
of [`miniclosedai`](../miniclosedai) (the gateway) and
[`miniclosedai-voice`](../miniclosedai-voice) (ASR/TTS), and shares their look and
conventions.

- **Web GUI control plane** — a FastAPI dashboard (`./dev.sh`, port **8099**) that
  downloads, launches, monitors, and tests models for you.
- **Terminal CLI (`mc`)** — everything the GUI does, from the shell
  (`./mc run …`, `mc ls`, `mc chat …`). Shares the dashboard's backend, so CLI and
  browser stay in live sync. → see **[Command-line interface](#command-line-interface-mc--local-remote--agent-access)**.
- **Format-aware serving, auto-detected** — safetensors models run on **vLLM**
  (**Docker** `vllm/vllm-openai`, or **native** `vllm serve` on RunPod pods with no
  Docker daemon); **GGUF** models — including **ternary Bonsai** — run on
  **llama.cpp** (`llama-server`). You just paste the repo. → see
  **[Serving engines](#serving-engines)**.
- **Config-file workflow** — for reproducible, version-controlled fleets, drive
  everything from [`models.yaml`](models.yaml) via `docker compose` / launch
  scripts.
- **Transformers shim** — an OpenAI-compatible fallback for models vLLM can't
  serve yet.

For deep internals (architecture, full API reference, status model, engine
internals, schemas) see **[DOCUMENTATION.md](DOCUMENTATION.md)**.

---

## Table of contents

- [Quick start (Web GUI)](#quick-start-web-gui)
- [Using the dashboard](#using-the-dashboard)
- [Command-line interface (`mc`)](#command-line-interface-mc--local-remote--agent-access)
- [Serving engines (vLLM & llama.cpp/GGUF)](#serving-engines)
- [Register a model in miniclosedai](#register-a-model-in-miniclosedai)
- [Network access (LAN / RunPod)](#network-access-lan--runpod)
- [Config-file workflow (compose + scripts)](#config-file-workflow-compose--scripts)
- [Transformers shim (fallback)](#transformers-shim-fallback)
- [Testing](#testing)
- [Environment variables](#environment-variables)
- [Requirements & VRAM](#requirements--vram)
- [File layout](#file-layout)
- [Troubleshooting](#troubleshooting)
- [License / scope](#license--scope)

---

## Quick start (Web GUI)

```bash
git clone … && cd miniclosedai-llm
cp .env.example .env          # set HF_TOKEN (only needed for gated models)
./dev.sh                      # builds a tiny venv, runs the dashboard on :8099
# open  http://<this-host>:8099   (or http://<LAN-IP>:8099 from another machine)
```

In the browser:

1. Paste a model — e.g. `Qwen/Qwen2.5-7B-Instruct`, `meta-llama/Llama-3.1-8B-Instruct`,
   or `Qwen/Qwen3-VL-8B-Instruct` (or a full `https://huggingface.co/…` URL).
2. *(Optional)* **Analyze** — confirms it exists, whether it's gated, its size, and
   whether it **fits in available memory** (warns with a "Run anyway" override).
3. **Download & Run** — a card appears and walks `pulling → downloading → loading →
   ready` (open **Logs** to watch). First run downloads weights (GB) and compiles
   CUDA graphs, so it can take a few minutes; later runs are faster (cached).
4. When **ready**, use the **Quick test** box (a prompt, plus an optional image for
   vision models) and copy the **base_url** into miniclosedai.

The dashboard itself needs **no GPU/ML libraries** — only FastAPI/httpx. All heavy
work runs inside the model it launches.

---

## Using the dashboard

| Element | What it does |
|---|---|
| **Banner** | Selected engine (Docker/native), GPU + memory readout, and the network URL the dashboard is reachable at. Red if no engine; amber if no GPU. |
| **Analyze** | Queries HuggingFace: existence, gated status, params/dtype, weight size, estimated need vs free memory, text-vs-vision. No download. |
| **Download & Run** | Allocates a port, writes the registry entry, and launches the model via the active engine. Returns immediately; status updates live. A model is launched **once** per id — re-running an already-running model is rejected, not duplicated. |
| **Downloaded models** | A library of LLMs **already in the HF cache** (with sizes). **Run** loads one straight from disk — no re-download; **Free** deletes its weights to reclaim space. |
| **Model card** | served-name, hf_id, port, a status pill (stopped / pulling / downloading / loading / ready / error), and Run / Stop / Logs / Remove. Click the header to **collapse/expand** the card. |
| **Logs** | Live stream (SSE) of image pull + vLLM startup, so you can see exactly where a slow or failing load is. |
| **Quick test** | Sends a chat to the running model and shows the reply + latency. For vision models, **+ Attach image** adds an image part (defaults to the bundled test image). |
| **Register box** | The exact `base_url` to paste into miniclosedai, plus a `host.docker.internal` alternative for same-host Docker. |

---

## Command-line interface (`mc`) — local, remote & agent access

`mc` is a terminal client for the dashboard — the same actions as the GUI, scriptable
from the shell. It's a thin HTTP client over the `/api` endpoints, so the CLI and
browser share one backend (run a model in the terminal → it shows in the GUI, and
vice-versa). **No dependencies** — pure standard library, runs under any `python3`.

The dashboard must be running (`./dev.sh`, or `./mc serve` to start it). Then:

```bash
./mc info                                   # engine + GPU + dashboard URL
./mc analyze Qwen/Qwen2.5-7B-Instruct       # type, size, gated, does-it-fit
./mc run Qwen/Qwen2.5-7B-Instruct --wait    # download + run, poll until ready
./mc ls                                     # list models (status · port · hf_id)
./mc test qwen2-5-7b "Capital of France?"   # one-shot prompt
./mc chat qwen2-5-7b                         # interactive REPL (multi-turn, streamed)
./mc logs qwen2-5-7b -f                      # follow live logs (Ctrl-C to stop)
./mc url qwen2-5-7b                          # base_url to register in miniclosedai
./mc cache                                   # already-downloaded models (run loads from disk)
./mc stop qwen2-5-7b   ./mc rm qwen2-5-7b    # stop / remove
```

| Command | Does |
|---|---|
| `mc info` · `gpu` | engine/GPU/dashboard status |
| `mc analyze <hf_id>` | inspect a model before downloading (size, gated, fits) |
| `mc run <hf_id> [--name --port --quant --max-len --gpu-mem --tp --trust-remote-code --gguf-file --force --wait]` | download + run (auto-picks vLLM or llama.cpp by format) |
| `mc ls` · `status <id>` | list / one model's status |
| `mc start <id>` · `stop <id>` · `rm <id>` | lifecycle (start re-runs an existing stopped model) |
| `mc logs <id> [-f]` | snapshot or follow logs |
| `mc test <id> [prompt] [--image P]` | one-shot chat/vision test |
| `mc chat <id>` | interactive REPL — `/reset`, `/exit` |
| `mc url <id>` | base_url for miniclosedai |
| `mc cache` · `free <hf_id>` | list cached models / delete weights |
| `mc serve` | start the dashboard (`./dev.sh`) |

Model ids are forgiving — `mc test llama …` matches `llama-3-1-8b-instruct`. Read
commands take `--json` for scripting. Configure the target with `MANAGER_URL`
(default `http://localhost:$MANAGER_PORT`) and `MANAGER_API_KEY` if the dashboard is
protected. Exit codes: `0` ok, `1` operation error, `2` dashboard unreachable.

> Tip: symlink it into your PATH — `ln -s "$PWD/mc" ~/.local/bin/mc` — then just
> `mc ls` from anywhere.

### From another machine, or from an LLM agent

Because everything binds `0.0.0.0` and is plain HTTP, a coding/agent LLM (e.g.
Claude Code) on another host can discover, run, and use models with no GUI. There
are **two surfaces** an agent drives:

- **Control plane** — `http://<host>:8099/api` (what `mc` wraps): list / analyze /
  run / stop / inspect models and the download cache.
- **Inference** — each *running* model's own `http://<host>:<port>/v1`
  (OpenAI-compatible): once a model is `ready`, chat with it from any OpenAI client.

Everything is configured by **environment variables** (no interactive prompts):
`MANAGER_URL` (default `http://localhost:$MANAGER_PORT`, point it at the remote
host), `MANAGER_API_KEY` (if the dashboard is protected), and `VLLM_API_KEY` (if a
model enforces a key). Typical agent flow:

```bash
# 1. point at the host and run a model (control plane, via mc)
export MANAGER_URL=http://192.168.0.110:8099
./mc analyze Qwen/Qwen2.5-7B-Instruct        # size / gated / fits?
./mc run Qwen/Qwen2.5-7B-Instruct --wait     # launches, polls to ready
./mc ls                                       # served-name · status · PORT
./mc url qwen2-5-7b-instruct                  # the /v1 base_url for this model
```

```python
# 2. chat with it (inference) — any OpenAI client, against the model's own /v1.
#    The served-name IS the OpenAI `model` id; the port comes from `mc ls`/`mc url`.
from openai import OpenAI
client = OpenAI(base_url="http://192.168.0.110:8001/v1", api_key="EMPTY")  # or VLLM_API_KEY
r = client.chat.completions.create(
    model="qwen2-5-7b-instruct",
    messages=[{"role": "user", "content": "Summarize this in one line: ..."}])
print(r.choices[0].message.content)
```

```bash
# …or raw HTTP — no SDK needed:
curl http://192.168.0.110:8001/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"qwen2-5-7b-instruct","messages":[{"role":"user","content":"hello"}]}'
```

An agent that prefers the terminal end-to-end can skip the SDK entirely:
`mc chat <id>` (interactive), `mc test <id> "prompt"` (one-shot), `mc cache` /
`mc analyze` to inspect. For exposing this beyond localhost (firewall ports, auth),
see [Network access](#network-access-lan--runpod).

---

## Serving engines

A model's **format** picks the engine automatically — you don't choose:

| Format | Engine | How it runs |
|---|---|---|
| **safetensors** (most HF models) | **vLLM**, or the **transformers shim** | `docker run vllm/vllm-openai` (server), `vllm serve` (native), or `python shim/server.py` (bare-metal — no Docker/vLLM) |
| **GGUF** (ternary Bonsai, any `*-GGUF` repo) | **llama.cpp** | `llama-server` as a subprocess |

For a safetensors model the manager auto-selects the first **available** engine in
this order (override with `LAUNCH_ENGINE=docker|native|shim`):

1. **Docker** (`vllm/vllm-openai` container) — default on a normal GPU server.
2. **native vLLM** (`vllm serve` subprocess) — when `vllm` is importable (e.g. a RunPod pod).
3. **transformers shim** (`shim/server.py`) — the universal **bare-metal** fallback: no
   Docker, no vLLM, works wherever `torch` runs (including **Jetson aarch64**, where vLLM
   can't be built). Run `./setup_shim.sh` once to provision it. → see
   [Transformers shim](#transformers-shim-fallback).

So on a box with **no Docker and no vLLM**, safetensors models still run — the manager
falls through to the shim instead of erroring. GGUF repos are routed to **llama.cpp**
regardless — see below.

### GGUF & ternary models (Bonsai) — `llama.cpp`

Paste any **GGUF** repo (e.g. PrismML's ternary Bonsai:
`prism-ml/Ternary-Bonsai-1.7B-gguf`, `-4B-gguf`, `-8B-gguf`) and it's auto-detected
and served by **`llama-server`** — which is OpenAI-compatible, so it registers in
miniclosedai and works with `mc`/the GUI exactly like a vLLM model.

```bash
./mc analyze prism-ml/Ternary-Bonsai-4B-gguf   # -> "GGUF → llama.cpp", picks the Q2_0 file
./mc run     prism-ml/Ternary-Bonsai-4B-gguf --wait
./mc chat    ternary-bonsai-4b-gguf
```

**The `llama-server` binary.** Ternary GGUFs (`Q2_0`, 1.58-bit) need the
**PrismML-Eng/llama.cpp** fork (upstream can't load them). The engine auto-detects a
binary in this order: `$LLAMACPP_SERVER_BIN` → the project's `./.llamacpp` build →
the `bonsai1bit_test` demo build → `PATH`.

**Auto-built on first run.** `./dev.sh` builds the fork for you the first time it
starts (if no `llama-server` is found and `git` + `cmake` are present) — it runs in
the **background** so the dashboard and the vLLM path come up immediately, and the
GGUF engine flips to available when the build finishes (watch
`.run/llamacpp-build.log`; the banner / `mc info` show its status). So a fresh clone
+ `./dev.sh` is ready for GGUF with no manual step. Set `LLAMACPP_AUTOBUILD=0` to
skip it, or build it yourself synchronously:

```bash
./setup_llamacpp.sh        # clones PrismML-Eng/llama.cpp (prism), builds llama-server (CUDA)
```

The first CUDA build takes ~10–30 min. On **Debian/Ubuntu** (standard GPU boxes and
RunPod pods) `setup_llamacpp.sh` **auto-installs the build deps** (`git`, `cmake`,
`ninja-build`, `build-essential`, `libcurl4-openssl-dev`) via `apt-get` — so a fresh
clone on a CUDA server builds with no manual setup (set `LLAMACPP_INSTALL_DEPS=0` to
opt out). It uses the CUDA toolkit (`nvcc`) if present and otherwise falls back to a
CPU build. On non-Debian distros, install those deps yourself first.

`mc info` / the dashboard banner shows whether `llama-server` is available. GGUFs
download via `llama-server --hf-repo` into `LLAMA_CACHE` (under `HF_HOME`). For a
multi-file GGUF repo, override the picked file with `mc run … --gguf-file NAME`.

---

## Register a model in miniclosedai

vLLM serves under `/v1`, and miniclosedai appends `/chat/completions` and `/models`
to your base URL — so **the base URL must end in `/v1`**.

In miniclosedai → **Settings → Backends → Add**:

| Field | Value |
|-------|-------|
| **Kind** | `openai` |
| **Name** | anything, e.g. `vLLM Qwen2.5-7B` |
| **Base URL** | the dashboard's **Register box** value, e.g. `http://192.168.0.110:8001/v1` (LAN IP), or `http://host.docker.internal:8001/v1` if miniclosedai is a Docker container on the same host |
| **API key** | any non-empty string (e.g. `EMPTY`) — vLLM ignores it **unless** you set `VLLM_API_KEY`, in which case enter that exact value |
| **Headers** | none |

Click **Test** → expect *"Reachable · 1 model"* → **Save**. The served-model-name
now appears in miniclosedai's model dropdown for conversations and bots.

> The served-model-name is stable and is what miniclosedai sends back as `model`.
> For preset models it's the short name (`qwen3-vl-8b`); for a pasted model it's a
> slug of the repo name (override it in **Advanced settings**).

---

## Network access (LAN / RunPod)

**LAN.** The dashboard and every model server bind `0.0.0.0`, so they're reachable
from other machines at `http://<LAN-IP>:8099` (dashboard) and `…:<port>` (models).
The Register box advertises the detected **LAN IP** by default; pin it with
`PUBLIC_HOST=192.168.0.110` if auto-detection picks the wrong interface. If a remote
machine can't connect, open the ports (trusted networks only):

```bash
sudo ufw allow 8099/tcp          # dashboard
sudo ufw allow 8001:8010/tcp     # model servers
```

**RunPod / pods without Docker.** Use a vLLM/PyTorch template (vLLM preinstalled),
or `pip install vllm`. The manager auto-selects the **native** engine. Persist
weights on the volume and launch:

```bash
echo "HF_HOME=/workspace/hf-cache" >> .env
echo "HF_TOKEN=hf_xxx" >> .env
./dev.sh
```

Expose port **8099** (and model ports **8001+**) via RunPod's HTTP proxy. The
Register box detects `RUNPOD_POD_ID` and shows the public form
`https://<podId>-<port>.proxy.runpod.net/v1`.

---

## Config-file workflow (compose + scripts)

For a reproducible, curated fleet (the original use case — vision models for
ID-document extraction), drive everything from [`models.yaml`](models.yaml) instead
of the GUI. The repo ships presets: `qwen3-vl-8b` (+FP8), `internvl3-8b`,
`qwen2.5-vl-7b`, `qwen3-vl-32b`, `internvl3-38b`.

```bash
cp .env.example .env                 # set HF_TOKEN
./start.sh                           # default preset (qwen3-vl-8b) on :8001
./start.sh --list                    # list profiles + ports
./start.sh internvl3-8b              # a specific model (compose profile)
./stop.sh                            # stop everything

# or launch one model directly, no compose:
./scripts/run_qwen3-vl-8b.sh         # docker run
./scripts/run_internvl3-8b.sh --native   # bare `vllm serve`
```

`models.yaml` is the single source of truth; `python3 gen_compose.py` regenerates
`docker-compose.yml` from it. Each model is a compose **profile** so nothing starts
unless asked. Per-model fields: `hf_id`, `served_name`, `port`, `quantization`,
`max_model_len`, `gpu_memory_util`, `tensor_parallel`, `max_images`,
`trust_remote_code`, `mm_processor_kwargs`, `hf_overrides`, `extra_args`. See
[DOCUMENTATION.md](DOCUMENTATION.md#7-modelsyaml-schema) for the full schema.

The bundled vision smoke test (one image + two images in one request):

```bash
python3 smoke_test.py --base-url http://localhost:8001/v1 --model qwen3-vl-8b
```

---

## Transformers shim (fallback)

The `transformers` shim (`shim/server.py`) is the universal **bare-metal** engine —
same OpenAI `/v1` surface, but **no Docker and no vLLM**. It's what makes the manager
serve safetensors models on hosts where vLLM can't run (e.g. **Jetson aarch64**), and
it's auto-selected when Docker + native vLLM are both unavailable.

**Set it up once** (creates `./.shim-venv/` with a CUDA-aware torch, which
`model_manager.py` auto-discovers):

```bash
./setup_shim.sh                 # auto-detect CUDA + build ./.shim-venv
./setup_shim.sh --reuse-venv ../miniclosedai-voice/env   # reuse an existing torch env (no re-download)
```

After that, just **Download & Run** any model from the dashboard (or `mc run …`) — the
manager launches the shim as a subprocess, and `mc info` / the banner show
`transformers (bare-metal): ready`. It serves both **text** models
(`AutoModelForCausalLM`, e.g. Llama/Qwen/Mistral) and **vision** models
(`AutoProcessor` + `AutoModelForImageTextToText`, decoding base64 `image_url`
data-URLs) — `SHIM_MODALITY=auto|text|vlm` picks (default `auto`: try VLM, fall back
to causal-LM). Force it globally with `LAUNCH_ENGINE=shim`.

You can still run it standalone (e.g. for a model vLLM can't serve on a Docker box):

```bash
SHIM_MODEL_ID=OpenGVLab/InternVL3-8B-HF SHIM_SERVED_NAME=internvl3-8b \
  SHIM_PORT=8009 ./.shim-venv/bin/python shim/server.py   # -> http://localhost:8009/v1
```

Config via env (top of `shim/server.py`); supports streaming + non-streaming.

---

## Testing

Two harnesses (standard-library only):

- **`e2e_test.py`** — regression test that drives the **running dashboard** like the
  GUI: for each small model, add+run → wait for `ready` → text chat (asserting two
  different prompts give two different answers) → image chat for vision models →
  stop + remove. Exits non-zero on any failure. `--quick` runs two tiny text models
  (`Qwen/Qwen2.5-0.5B-Instruct`, `HuggingFaceTB/SmolLM2-360M-Instruct`); the default
  adds a small vision model (`Qwen/Qwen2.5-VL-3B-Instruct`).

  ```bash
  ./dev.sh                                   # dashboard must be running
  python3 e2e_test.py --quick                # tiny text models only (fast)
  python3 e2e_test.py                        # + a small vision model
  python3 e2e_test.py --gguf                  # ternary Bonsai (needs ./setup_llamacpp.sh)
  python3 e2e_test.py --models Qwen/Qwen2.5-1.5B-Instruct
  python3 e2e_test.py --base http://192.168.0.110:8099 --timeout 1200
  ```

  All three default models have been verified loading + answering on an NVIDIA GB10.

- **`smoke_test.py`** — direct vision check against a single running vLLM `/v1`
  endpoint (one image + a two-image request). For the config-file workflow.

---

## Environment variables

All optional; copy `.env.example` → `.env`. (See [DOCUMENTATION.md](DOCUMENTATION.md#10-configuration-environment) for full descriptions.)

| Var | Default | Purpose |
|---|---|---|
| `HF_TOKEN` | — | HuggingFace token; required for **gated** models (Llama, etc.) |
| `HF_HOME` | `~/.cache/huggingface` | weight cache (set to a volume on RunPod) |
| `MANAGER_PORT` | `8099` | dashboard port |
| `LAUNCH_ENGINE` | `auto` | `auto` / `docker` / `native` / `shim` (auto order: docker → native vLLM → bare-metal shim) |
| `LLAMACPP_AUTOBUILD` | `auto` | `auto`/`1` = `dev.sh` builds the GGUF `llama-server` in the background if missing; `0` = skip |
| `SHIM_PYTHON` | `./.shim-venv` | interpreter for the bare-metal transformers shim (provision with `./setup_shim.sh`) |
| `SHIM_MODALITY` | `auto` | shim model type: `auto` / `text` / `vlm` |
| `PUBLIC_HOST` | auto LAN IP | host advertised in the Register box |
| `MANAGER_API_KEY` | — | optional Bearer token to protect the dashboard/API |
| `VLLM_IMAGE` | `vllm/vllm-openai:latest` | Docker image tag |
| `VLLM_API_KEY` | — | if set, vLLM enforces it; put the same value in miniclosedai |

---

## Requirements & VRAM

- **NVIDIA GPU + working driver** (`nvidia-smi` must print a table — see
  Troubleshooting if it errors after an update).
- **Docker + NVIDIA Container Toolkit** (Docker engine) *or* **vLLM installed**
  (native engine). The manager itself only needs Python 3.10+.
- **Disk** for weights (an 8B model is ~16–20 GB).

### First-time setup on a fresh Ubuntu GPU box

Everything the Docker engine needs (the default), end to end:

```bash
# 1. NVIDIA driver — verify it works (install via your distro if not):
nvidia-smi                              # must print a GPU table

# 2. Docker:
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker "$USER"         # then log out/in so `docker` works sans sudo

# 3. NVIDIA Container Toolkit (lets containers see the GPU):
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker

# 4. Confirm containers can see the GPU:
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi   # GPU table

# 5. Clone + run:
git clone <repo-url> && cd miniclosedai-llm
cp .env.example .env                    # set HF_TOKEN only for gated models
./dev.sh                                # -> http://<host>:8099
```

Then paste a HuggingFace model id in the browser and click **Download & Run**.
That's it — `dev.sh` builds the manager's venv, prints the selected engine + GPU,
and serves the dashboard. The first model launch pulls the `vllm/vllm-openai`
image (~21 GB, once) and the model weights; subsequent runs reuse both.

> On **RunPod** (no Docker daemon), skip steps 2–4 and use a vLLM template (or
> `pip install vllm`) — the manager auto-selects the native engine. See
> [Network access](#network-access-lan--runpod).

Approx. footprint of the bundled **preset** vision models:

| served-name | repo | quant | approx. weights |
|---|---|---|---|
| `qwen3-vl-8b` | `Qwen/Qwen3-VL-8B-Instruct` | bf16 | ~18–20 GB |
| `qwen3-vl-8b` (FP8) | `…-Instruct-FP8` | fp8 | ~12 GB |
| `internvl3-8b` | `OpenGVLab/InternVL3-8B-HF` | bf16 | ~18–20 GB |
| `qwen2.5-vl-7b` | `Qwen/Qwen2.5-VL-7B-Instruct` | bf16 | ~17–19 GB |
| `qwen3-vl-32b` | `Qwen/Qwen3-VL-32B-Instruct-FP8` | fp8 | ~34–38 GB |
| `internvl3-38b` | `OpenGVLab/InternVL3-38B-AWQ` | awq | ~48 GB card |

The **Analyze** step estimates this for any model you paste and compares it to free
memory. On **unified-memory** parts (e.g. NVIDIA GB10) the GPU shares system RAM, so
the manager sizes `gpu_memory_utilization` from **free** memory, not total.

---

## File layout

| Path | Purpose |
|---|---|
| `app.py` | FastAPI control plane (the dashboard backend) |
| `cli.py` + `mc` | Terminal client (`./mc …`) — stdlib HTTP client over the `/api` endpoints |
| `model_manager.py` | Engine abstraction (vLLM + llama.cpp), registry, HF analysis, status |
| `static/` | Dashboard UI (`index.html`, `style.css`, `app.js`) |
| `setup_llamacpp.sh` | Builds the PrismML llama.cpp fork (GGUF/ternary support) |
| `dev.sh` | One-command launcher (venv + preflight + uvicorn) |
| `manager-requirements.txt` | Dashboard deps (no torch/vLLM) |
| `Dockerfile.manager` | Optional: run the control plane itself in a container |
| `models.yaml` | Preset fleet + serving config (source of truth) |
| `_args.py` | `models.yaml` row → `vllm serve` flags (shared everywhere) |
| `gen_compose.py` | Generates `docker-compose.yml` from `models.yaml` |
| `docker-compose.yml` | Generated; one profile-gated service per preset |
| `scripts/run_*.sh` | Per-model launchers (docker or native) |
| `start.sh` / `stop.sh` | Compose up/down by profile |
| `e2e_test.py` | End-to-end regression harness (drives the dashboard) |
| `smoke_test.py` | Direct vision smoke test against a vLLM `/v1` |
| `tests/test_image.png` | Bundled labelled test image |
| `shim/` | transformers fallback (`server.py`, `Dockerfile`, `requirements.txt`) |
| `models.local.json` | (generated) per-machine registry state — gitignored |
| `.run/` | (generated) native logs + docker orchestration logs — gitignored |

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `nvidia-smi`: **"Driver/library version mismatch"** | The kernel module ≠ userspace driver after an update. **Reboot** (the package manager sets `/var/run/reboot-required`). GPU passthrough won't work until `nvidia-smi` prints a table. |
| Launch **times out / "No such container"** | First run downloads a multi-GB image. The manager now pulls in a streamed background step (watch **Logs**) — just wait. |
| **CUDA out of memory** / *"Free memory … less than desired GPU memory utilization"* | Lower `gpu_memory_util` (Advanced), reduce `max_model_len`, or pick a smaller/quantized model. On unified memory the manager already sizes from free memory. |
| Model errors immediately with **`unrecognized arguments`** | A vLLM flag changed/was removed in your image version (e.g. the old `--guided-decoding-backend`). Remove it from `models.yaml` / Advanced, or pin an older `VLLM_IMAGE`. |
| **`401 / 403` / GatedRepoError** in logs | Gated model — set `HF_TOKEN` in `.env` and accept the model's license on HuggingFace. Analyze flags this up front. |
| **"model architecture not supported"** | vLLM doesn't support it yet — pin a newer `VLLM_IMAGE`, or use the **shim**. |
| miniclosedai **Test** fails / `host.docker.internal` won't resolve | Use the LAN IP base URL, or ensure miniclosedai's container has `extra_hosts: ["host.docker.internal:host-gateway"]`. |
| **"Reachable but 0 models"** in miniclosedai | Base URL missing `/v1`, or the model is still loading. |
| Dashboard unreachable from another machine | Open the firewall ports (above); confirm `0.0.0.0` binding (default). |

More in [DOCUMENTATION.md → Troubleshooting](DOCUMENTATION.md#17-troubleshooting).

---

## License / scope

This project provides the **model server + dashboard + its miniclosedai
registration** only. The extraction/benchmarking prompts live elsewhere. Stable,
documented served-model-names make models selectable when creating bots in
miniclosedai. License: see [LICENSE](../miniclosedai/LICENSE) of the umbrella
project.
