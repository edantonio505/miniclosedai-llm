#!/usr/bin/env bash
#
# setup_shim.sh — provision the bare-metal "transformers shim" launch engine.
#
# The shim (shim/server.py) serves any HuggingFace model behind an OpenAI-
# compatible /v1 API using plain `transformers` — no Docker, no vLLM. It is the
# universal fallback the manager auto-selects when Docker is unreachable and
# vLLM can't run (e.g. Jetson aarch64). torch is huge, so the shim gets its own
# venv at ./.shim-venv/ that model_manager.py's shim_python() auto-discovers.
#
# Usage:
#   ./setup_shim.sh                      # auto-detect CUDA + build ./.shim-venv
#   ./setup_shim.sh --cuda 12.6          # force a torch CUDA wheel channel
#   ./setup_shim.sh --cpu                # CPU-only (works, but slow)
#   ./setup_shim.sh --python python3.11  # use a specific interpreter
#   ./setup_shim.sh --reuse-venv PATH    # reuse an existing venv that ALREADY
#                                        # has torch + transformers (no re-download).
#                                        # e.g. ../miniclosedai-voice/env
#
# Re-running is safe: an existing ./.shim-venv is left alone (pass --force to
# rebuild). After it completes, (re)start the manager and the LLM Models page
# banner shows "Native (transformers shim) — Ready".
#
set -euo pipefail
cd "$(dirname "$0")"

VENV_DIR=".shim-venv"
PYTHON_BIN=""
CUDA_OVERRIDE=""
CPU_ONLY=0
REUSE_VENV=""
FORCE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --cuda)       CUDA_OVERRIDE="$2"; shift 2 ;;
    --cpu)        CPU_ONLY=1; shift ;;
    --python)     PYTHON_BIN="$2"; shift 2 ;;
    --reuse-venv) REUSE_VENV="$2"; shift 2 ;;
    --force)      FORCE=1; shift ;;
    -h|--help)    sed -n '2,/^set -e/p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown flag: $1" >&2; exit 1 ;;
  esac
done

c_blue=$'\e[1;34m'; c_green=$'\e[1;32m'; c_red=$'\e[1;31m'; c_yellow=$'\e[1;33m'; c_dim=$'\e[2m'; c_off=$'\e[0m'
step() { printf "\n%s▶ %s%s\n" "$c_blue"   "$1" "$c_off"; }
ok()   { printf   "%s✓ %s%s\n" "$c_green"  "$1" "$c_off"; }
warn() { printf   "%s! %s%s\n" "$c_yellow" "$1" "$c_off"; }
die()  { printf   "%s✗ %s%s\n" "$c_red"    "$1" "$c_off" >&2; exit 1; }

verify() {  # verify(python_bin)
  step "Verifying torch / transformers / GPU"
  "$1" - <<'PY'
import warnings; warnings.filterwarnings("ignore")
import torch, transformers
print(f"  torch:        {torch.__version__}")
print(f"  transformers: {transformers.__version__}")
gpu = torch.cuda.is_available()
print(f"  CUDA visible: {gpu}")
if gpu:
    print(f"  device:       {torch.cuda.get_device_name(0)}")
PY
}

# ─── Reuse an existing torch venv (fast path — no multi-GB re-download) ──────
if [[ -n "$REUSE_VENV" ]]; then
  py="$REUSE_VENV/bin/python"
  [[ -x "$py" ]] || die "no python at $py"
  "$py" -c "import torch, transformers" 2>/dev/null \
    || die "$REUSE_VENV is missing torch/transformers — pick a venv that has them, or run without --reuse-venv"
  target="$(cd "$REUSE_VENV" && pwd)"
  rm -rf "$VENV_DIR"
  ln -s "$target" "$VENV_DIR"
  ok "Linked $VENV_DIR → $target"
  verify "$VENV_DIR/bin/python"
  touch "$VENV_DIR/.ready"
  printf "\n%sShim ready (reusing %s).%s Restart the manager to pick it up.\n" "$c_green" "$target" "$c_off"
  exit 0
fi

# ─── Idempotency ────────────────────────────────────────────────────────────
if [[ -e "$VENV_DIR" && "$FORCE" != "1" ]]; then
  if [[ -x "$VENV_DIR/bin/python" ]] && "$VENV_DIR/bin/python" -c "import torch, transformers" 2>/dev/null; then
    ok "$VENV_DIR already set up (pass --force to rebuild)"
    verify "$VENV_DIR/bin/python"
    touch "$VENV_DIR/.ready"
    exit 0
  fi
  warn "$VENV_DIR exists but is incomplete — rebuilding"
  rm -rf "$VENV_DIR"
fi

# ─── Pick a Python interpreter ──────────────────────────────────────────────
if [[ -z "$PYTHON_BIN" ]]; then
  for v in python3.12 python3.11 python3.10 python3; do
    command -v "$v" >/dev/null 2>&1 && { PYTHON_BIN="$v"; break; }
  done
fi
[[ -z "$PYTHON_BIN" ]] && die "no python3 found; install python3.10+ or pass --python"
ok "Using $("$PYTHON_BIN" --version) at $(command -v "$PYTHON_BIN")"

# ─── Detect CUDA wheel channel (mirrors ../miniclosedai-voice/setup.sh) ──────
detect_cuda() {
  [[ "$CPU_ONLY" == "1" ]] && { echo "cpu"; return; }
  [[ -n "$CUDA_OVERRIDE" ]] && { echo "cu${CUDA_OVERRIDE//./}"; return; }
  command -v nvidia-smi >/dev/null 2>&1 || { echo "cpu"; return; }
  local v
  v=$(nvidia-smi 2>/dev/null | grep -oE 'CUDA Version: [0-9.]+' | head -1 | awk '{print $3}')
  [[ -z "$v" ]] && { echo "cpu"; return; }
  case "$v" in
    13.*|12.8*|12.9*) echo "cu130" ;;
    12.4|12.5|12.6|12.7) echo "cu124" ;;
    11.*) echo "cu118" ;;
    *) echo "cu124" ;;
  esac
}
CUDA_CHANNEL=$(detect_cuda)
if [[ "$CUDA_CHANNEL" == "cpu" ]]; then
  warn "No CUDA detected — installing CPU-only torch (shim works, generation is slow)"
else
  ok "CUDA detected → torch wheels from pytorch.org/whl/${CUDA_CHANNEL}"
fi

# ─── Create venv ────────────────────────────────────────────────────────────
step "Creating venv at ./${VENV_DIR}/"
"$PYTHON_BIN" -m venv "$VENV_DIR"
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"
python -m pip install --quiet --upgrade pip wheel setuptools
ok "pip / wheel / setuptools up to date"

# ─── torch (+ torchvision, best-effort for VLMs) ────────────────────────────
step "Installing torch (${CUDA_CHANNEL})"
if [[ "$CUDA_CHANNEL" == "cpu" ]]; then
  pip install --quiet torch || die "torch install failed"
  pip install --quiet torchvision || warn "torchvision skipped (VLMs may need it)"
else
  IDX="https://download.pytorch.org/whl/${CUDA_CHANNEL}"
  pip install --quiet torch --index-url "$IDX" || die "torch install failed for ${CUDA_CHANNEL}"
  pip install --quiet torchvision --index-url "$IDX" || warn "torchvision skipped (VLMs may need it)"
fi
ok "torch installed: $(python -c 'import torch; print(torch.__version__)')"

# ─── Shim runtime deps ──────────────────────────────────────────────────────
step "Installing transformers + serving deps"
pip install --quiet \
  transformers accelerate 'huggingface-hub[hf_transfer]' safetensors \
  sentencepiece protobuf einops pillow \
  fastapi 'uvicorn[standard]'
ok "shim deps installed"

verify "$VENV_DIR/bin/python"
touch "$VENV_DIR/.ready"

cat <<EOF

${c_green}Shim ready.${c_off}  The manager auto-discovers ./${VENV_DIR}.
Next: restart the manager (${c_dim}./dev.sh${c_off}) — the LLM Models banner will show
      "Native (transformers shim) — Ready", and Download & Run will serve
      safetensors models bare-metal (no Docker, no vLLM).
EOF
