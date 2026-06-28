#!/usr/bin/env bash
# Bring vLLM model server(s) up via docker compose.
#
#   ./start.sh                      # start the default model (qwen3-vl-8b)
#   ./start.sh internvl3-8b         # start a specific model by profile
#   ./start.sh qwen3-vl-8b internvl3-8b   # start several (need enough VRAM!)
#   ./start.sh shim                 # start the transformers fallback shim
#   ./start.sh --list               # list available model profiles
#
# Profiles come from models.yaml. After it's up, verify with:
#   curl http://localhost:<port>/v1/models
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

if [[ "${1:-}" == "--list" ]]; then
  python3 _args.py list
  exit 0
fi

if [[ ! -f .env ]]; then
  echo "NOTE: no .env found — copying .env.example -> .env (edit it to set HF_TOKEN)."
  cp .env.example .env
fi

# Keep docker-compose.yml in sync with models.yaml.
python3 gen_compose.py >/dev/null

PROFILES=("${@:-qwen3-vl-8b}")
ARGS=()
for p in "${PROFILES[@]}"; do ARGS+=(--profile "$p"); done

echo ">> Starting profiles: ${PROFILES[*]}"
docker compose "${ARGS[@]}" up -d

echo
echo ">> Up. Tail logs with:   docker compose logs -f"
echo ">> First start downloads weights (can take several minutes)."
for p in "${PROFILES[@]}"; do
  port="$(python3 -c "import sys,_args; d,m=_args.find('$p'); print(m['port'])" 2>/dev/null || true)"
  [[ -n "$port" && "$p" != "shim" ]] && \
    echo "   $p  ->  http://localhost:$port/v1/models   (miniclosedai base_url: http://host.docker.internal:$port/v1)"
done
