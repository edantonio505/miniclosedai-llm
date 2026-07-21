#!/usr/bin/env bash
# One-command launcher for the miniclosedai-llm web control plane.
# Works on a fresh Ubuntu GPU box OR a RunPod pod:
#
#   git clone … && cd miniclosedai-llm
#   echo "HF_TOKEN=hf_xxx" >> .env      # (or: cp .env.example .env and edit)
#   ./dev.sh
#   -> open http://<this-host>:8099
#
# The manager itself needs NO GPU/ML deps — only FastAPI/httpx. It launches each
# model either as a Docker container (Ubuntu host) or a native `vllm serve`
# subprocess (RunPod), auto-detected (override with LAUNCH_ENGINE=docker|native).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

# --- config -------------------------------------------------------------------
[ -f .env ] || { echo "NOTE: no .env — copying .env.example -> .env (set HF_TOKEN in it)."; cp .env.example .env; }
set -a; source ./.env; set +a
PORT="${MANAGER_PORT:-8099}"
ENGINE="${LAUNCH_ENGINE:-auto}"
AUTOBUILD="${LLAMACPP_AUTOBUILD:-auto}"   # auto|1 = build GGUF engine if missing; 0 = skip

# --- GGUF/ternary engine (llama.cpp) helpers ----------------------------------
# True if a llama-server binary is already resolvable (mirrors model_manager's
# llamacpp_bin(): $LLAMACPP_SERVER_BIN → the ./setup_llamacpp.sh build → PATH).
llamacpp_bin_present() {
  { [ -n "${LLAMACPP_SERVER_BIN:-}" ] && [ -x "${LLAMACPP_SERVER_BIN:-}" ]; } && return 0
  [ -x .llamacpp/llama.cpp/build/bin/llama-server ] && return 0
  command -v llama-server >/dev/null 2>&1 && return 0
  return 1
}

# Kick off ./setup_llamacpp.sh in the BACKGROUND if the binary is missing, so the
# GGUF path becomes ready without a manual step — but the dashboard (and the vLLM
# path) come up immediately rather than waiting out a 10–30 min first CUDA build.
# Idempotent (skips if built or already building), and never fails startup.
maybe_build_llamacpp() {
  [ "$AUTOBUILD" = 0 ] && return 0
  if llamacpp_bin_present; then
    echo "  llama.cpp     : OK (GGUF/ternary engine present)"
    return 0
  fi
  mkdir -p .run
  local pidf=".run/llamacpp-build.pid" log=".run/llamacpp-build.log"
  if [ -f "$pidf" ] && kill -0 "$(cat "$pidf" 2>/dev/null)" 2>/dev/null; then
    echo "  llama.cpp     : build already running (tail -f $log)"
    return 0
  fi
  # Buildable if the toolchain is already here, OR apt-get can install it
  # (setup_llamacpp.sh best-effort installs the deps on Debian/Ubuntu).
  if { command -v git >/dev/null 2>&1 && command -v cmake >/dev/null 2>&1; } \
     || command -v apt-get >/dev/null 2>&1; then :; else
    echo "  llama.cpp     : GGUF engine absent; needs git + cmake (no apt-get to auto-install) — run ./setup_llamacpp.sh"
    return 0
  fi
  echo "  llama.cpp     : building GGUF/ternary engine in background -> $log"
  echo "                  (dashboard starts now; GGUF becomes available when the build finishes)"
  nohup ./setup_llamacpp.sh >"$log" 2>&1 &
  echo $! >"$pidf"
}

# --- python env ---------------------------------------------------------------
PY="${PYTHON:-python3}"
if [ ! -d .venv ]; then
  echo ">> creating venv (.venv)"
  "$PY" -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
pip install -q --upgrade pip >/dev/null 2>&1 || true
pip install -q -r manager-requirements.txt

# --- preflight (informational; the UI shows the same banner) ------------------
echo
echo "==================== preflight ===================="
if command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then
  echo "  docker        : OK (daemon reachable)"
  DOCKER_OK=1
else
  echo "  docker        : not usable (daemon unreachable or not installed)"
  DOCKER_OK=0
fi
if command -v vllm >/dev/null 2>&1 || "$PY" -c 'import vllm' >/dev/null 2>&1; then
  echo "  vllm (native) : OK"
  NATIVE_OK=1
else
  echo "  vllm (native) : not installed (pip install vllm) — needed only for native engine"
  NATIVE_OK=0
fi
if [ -x .shim-venv/bin/python ] && .shim-venv/bin/python -c 'import transformers' >/dev/null 2>&1; then
  echo "  shim (native) : OK (transformers, bare-metal — any model, no Docker/vLLM)"
  SHIM_OK=1
else
  echo "  shim (native) : not set up (./setup_shim.sh) — bare-metal fallback for safetensors"
  SHIM_OK=0
fi
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
  echo "  gpu           : $(nvidia-smi -L | wc -l) device(s)"
else
  echo "  gpu           : nvidia-smi not working (driver mismatch? reboot may be needed)"
fi
case "$ENGINE" in
  docker) SEL=docker ;;
  native) SEL=native ;;
  shim) SEL=shim ;;
  *) if [ "$DOCKER_OK" = 1 ]; then SEL=docker;
     elif [ "$NATIVE_OK" = 1 ]; then SEL=native;
     elif [ "$SHIM_OK" = 1 ]; then SEL=shim;
     else SEL="none"; fi ;;
esac
echo "  engine        : $SEL  (LAUNCH_ENGINE=$ENGINE)"
[ "$SEL" = "none" ] && echo "  WARNING: no usable launch engine — run ./setup_shim.sh (bare-metal) or ./setup_llamacpp.sh (GGUF), or install Docker / 'pip install vllm'."
maybe_build_llamacpp
echo "==================================================="
echo
echo ">> control plane on http://0.0.0.0:${PORT}  (open it in a browser)"
[ -n "${RUNPOD_POD_ID:-}" ] && echo ">> RunPod detected — expose port ${PORT} (and model ports 8001+) via the pod's HTTP proxy."
echo

exec uvicorn app:app --host 0.0.0.0 --port "$PORT"
