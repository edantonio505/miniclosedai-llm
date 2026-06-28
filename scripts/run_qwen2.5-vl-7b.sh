#!/usr/bin/env bash
# Convenience wrapper -> launches the 'qwen2.5-vl-7b' model defined in models.yaml.
# All real logic lives in run_model.sh. Pass --native or extra vllm flags through.
exec "$(dirname "${BASH_SOURCE[0]}")/run_model.sh" "qwen2.5-vl-7b" "$@"
