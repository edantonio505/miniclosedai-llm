#!/usr/bin/env bash
# Generic per-model launcher. Reads models.yaml (via _args.py) so there is no
# duplicated configuration — the same flags Docker uses are used here.
#
#   scripts/run_model.sh <profile|served_name> [--docker|--native] [extra vllm flags...]
#
#   --docker   (default) run the official vllm/vllm-openai image via `docker run`.
#   --native   run `vllm serve` directly (requires vLLM installed in your env).
#
# Examples:
#   scripts/run_model.sh qwen3-vl-8b
#   scripts/run_model.sh internvl3-8b --native
#   scripts/run_model.sh qwen3-vl-8b-fp8 --docker
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Load .env if present (HF_TOKEN, HF_HOME, VLLM_IMAGE, VLLM_API_KEY).
if [[ -f .env ]]; then
  set -a; source ./.env; set +a
fi

KEY="${1:?usage: run_model.sh <profile|served_name> [--docker|--native] [extra flags]}"
shift || true

MODE="--docker"
EXTRA=()
for a in "$@"; do
  case "$a" in
    --docker|--native) MODE="$a" ;;
    *) EXTRA+=("$a") ;;
  esac
done

# Pull resolved settings + the full vllm arg string from the single source of truth.
eval "$(python3 _args.py shell "$KEY")"
# VLLM_ARGS is a shell-quoted string; re-expand into a real array.
eval "ARGS=($VLLM_ARGS)"
if [[ ${#EXTRA[@]} -gt 0 ]]; then ARGS+=("${EXTRA[@]}"); fi

: "${VLLM_IMAGE:=${IMAGE}}"
: "${HF_HOME:=$HOME/.cache/huggingface}"
mkdir -p "$HF_HOME"

echo ">> model         : $HF_ID"
echo ">> served-name   : $SERVED_NAME   (this is what miniclosedai will list)"
echo ">> port          : 0.0.0.0:$PORT"
echo ">> base_url      : http://host.docker.internal:$PORT/v1  (register this in miniclosedai)"
echo ">> mode          : $MODE"
echo

if [[ "$MODE" == "--native" ]]; then
  command -v vllm >/dev/null 2>&1 || { echo "ERROR: 'vllm' not found in PATH. Install vLLM or use --docker."; exit 1; }
  exec vllm serve "${ARGS[@]}"
fi

# --docker
command -v docker >/dev/null 2>&1 || { echo "ERROR: docker not found."; exit 1; }
exec docker run --rm -it \
  --name "vlm-${SERVED_NAME}-${PORT}" \
  --gpus all \
  --ipc=host \
  --shm-size 16g \
  -p "${PORT}:${PORT}" \
  -e HF_TOKEN="${HF_TOKEN:-}" \
  -e HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}" \
  -e HF_HOME=/root/.cache/huggingface \
  -v "${HF_HOME}:/root/.cache/huggingface" \
  --entrypoint vllm \
  "${VLLM_IMAGE}" \
  serve "${ARGS[@]}"
