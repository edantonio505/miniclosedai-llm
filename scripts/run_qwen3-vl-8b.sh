#!/usr/bin/env bash
# Convenience wrapper -> launches the 'qwen3-vl-8b' model defined in models.yaml.
# All real logic lives in run_model.sh. Pass --native or extra vllm flags through.
exec "$(dirname "${BASH_SOURCE[0]}")/run_model.sh" "qwen3-vl-8b" "$@"
