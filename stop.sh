#!/usr/bin/env bash
# Stop and remove the vLLM model server containers.
#
#   ./stop.sh           # stop everything started by this compose project
#   ./stop.sh -v        # also remove named volumes (NOT the HF cache, which is
#                       # a host bind mount and is left untouched)
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

# --profile '*' is needed so `down` also targets profile-gated services.
PROFILES=(qwen3-vl-8b qwen3-vl-8b-fp8 internvl3-8b qwen2.5-vl-7b qwen3-vl-32b internvl3-38b shim)
ARGS=()
for p in "${PROFILES[@]}"; do ARGS+=(--profile "$p"); done

echo ">> Stopping all vLLM model services..."
docker compose "${ARGS[@]}" down "$@"
echo ">> Stopped. (HuggingFace weight cache on the host is preserved.)"
