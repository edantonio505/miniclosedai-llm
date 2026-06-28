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
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
  echo "  gpu           : $(nvidia-smi -L | wc -l) device(s)"
else
  echo "  gpu           : nvidia-smi not working (driver mismatch? reboot may be needed)"
fi
case "$ENGINE" in
  docker) SEL=docker ;;
  native) SEL=native ;;
  *) if [ "$DOCKER_OK" = 1 ]; then SEL=docker; elif [ "$NATIVE_OK" = 1 ]; then SEL=native; else SEL="none"; fi ;;
esac
echo "  engine        : $SEL  (LAUNCH_ENGINE=$ENGINE)"
[ "$SEL" = "none" ] && echo "  WARNING: no usable launch engine — install Docker, or 'pip install vllm' for native."
echo "==================================================="
echo
echo ">> control plane on http://0.0.0.0:${PORT}  (open it in a browser)"
[ -n "${RUNPOD_POD_ID:-}" ] && echo ">> RunPod detected — expose port ${PORT} (and model ports 8001+) via the pod's HTTP proxy."
echo

exec uvicorn app:app --host 0.0.0.0 --port "$PORT"
