#!/usr/bin/env bash
# setup_llamacpp.sh — build the llama.cpp `llama-server` used to serve GGUF models
# (Bonsai / Ternary-Bonsai and any other GGUF) in miniclosedai-llm.
#
# Ternary Bonsai GGUFs use PrismML's quant types (Q2_0 etc., ggml type ≥42), whose
# kernels live in the PrismML-Eng/llama.cpp fork (prism branch) — upstream llama.cpp
# and older builds can't load them. This clones that fork and builds `llama-server`
# with CUDA into ./.llamacpp, then prints the binary path for LLAMACPP_SERVER_BIN.
#
#   ./setup_llamacpp.sh                 # build with CUDA (default)
#   LLAMACPP_CUDA=0 ./setup_llamacpp.sh # CPU-only build (no NVCC needed)
#   LLAMACPP_CUDA_ARCH="120a" ./setup_llamacpp.sh   # override GPU arch list
#
# Requires: git, cmake, ninja (or make), a C++ compiler, libcurl dev headers, and
# for a CUDA build the CUDA toolkit (nvcc). On this GB10 box CUDA 13 + arch 120a
# (Blackwell) is known-good.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

REPO="${LLAMACPP_REPO:-https://github.com/PrismML-Eng/llama.cpp.git}"
BRANCH="${LLAMACPP_BRANCH:-prism}"
DIR=".llamacpp"
SRC="$DIR/llama.cpp"
BUILD="$SRC/build"
CUDA="${LLAMACPP_CUDA:-1}"

c() { printf '\033[1;34m>> %s\033[0m\n' "$1"; }

# CUDA arch(es) to compile for. We build ON the box we serve on, so by default we
# target exactly this machine's GPU(s) — auto-detected from nvidia-smi. This is what
# makes `git clone` + build "just work" on any CUDA server: no hardcoded arch list to
# go stale or exceed an older toolkit (e.g. sm_100/120a need CUDA ≥12.8). Override
# with LLAMACPP_CUDA_ARCH (e.g. "120a" for GB10, or a ";"-separated list for a binary
# you'll move between different GPUs). Falls back to a broad pre-Blackwell list if
# detection fails (works on any CUDA ≥11.8 toolkit).
_detect_cuda_arch() {
  command -v nvidia-smi >/dev/null 2>&1 || return 1
  nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null \
    | tr -d ' .' | grep -E '^[0-9]+$' | sort -u | paste -sd';' -
}
ARCH="${LLAMACPP_CUDA_ARCH:-}"
if [ "$CUDA" = 1 ] && [ -z "$ARCH" ]; then
  ARCH="$(_detect_cuda_arch || true)"
  if [ -n "$ARCH" ]; then
    c "auto-detected CUDA arch from GPU: $ARCH  (override with LLAMACPP_CUDA_ARCH)"
  else
    ARCH="80;86;89;90"
    c "could not detect GPU arch — falling back to $ARCH"
  fi
fi

# --- preflight -------------------------------------------------------------------
# Best-effort install of the build toolchain on Debian/Ubuntu (standard Ubuntu GPU
# boxes and RunPod pods), so a fresh `git clone` + build "just works". Skipped when
# the deps are already present, apt-get is absent (non-Debian), or LLAMACPP_INSTALL_DEPS=0.
# libssl-dev is required for HTTPS: llama-server downloads GGUFs via --hf-repo over
# https, and without OpenSSL dev headers the build has no TLS ("HTTPS is not supported").
_have_curl_dev() { [ -f /usr/include/curl/curl.h ] || pkg-config --exists libcurl 2>/dev/null; }
_have_ssl_dev()  { [ -f /usr/include/openssl/ssl.h ] || pkg-config --exists openssl 2>/dev/null; }
if [ "${LLAMACPP_INSTALL_DEPS:-1}" != 0 ] \
   && { ! command -v git >/dev/null || ! command -v cmake >/dev/null || ! _have_curl_dev || ! _have_ssl_dev; } \
   && command -v apt-get >/dev/null 2>&1; then
  SUDO=""; [ "$(id -u)" = 0 ] || SUDO="sudo"
  c "installing build deps (git cmake ninja-build build-essential libcurl4-openssl-dev libssl-dev)"
  DEBIAN_FRONTEND=noninteractive $SUDO apt-get update -qq || true
  DEBIAN_FRONTEND=noninteractive $SUDO apt-get install -y --no-install-recommends \
    git cmake ninja-build build-essential libcurl4-openssl-dev libssl-dev || \
    echo "WARN: apt-get install failed — install the build deps manually and re-run."
fi
for t in git cmake; do command -v "$t" >/dev/null || { echo "ERROR: '$t' not found (install it and re-run)."; exit 1; }; done
GEN=(); command -v ninja >/dev/null && GEN=(-G Ninja)

# Resolve a REAL nvcc. `/usr/bin/nvcc` is often a stale distro stub (e.g. CUDA 12);
# prefer the full toolkit under /usr/local/cuda. Pin it so CMake doesn't pick the
# stub and fail its compiler probe.
NVCC="${CUDACXX:-}"
if [ -z "$NVCC" ]; then
  for cand in /usr/local/cuda/bin/nvcc /usr/local/cuda-*/bin/nvcc "$(command -v nvcc 2>/dev/null || true)"; do
    [ -x "$cand" ] && { NVCC="$cand"; break; }
  done
fi
if [ "$CUDA" = 1 ] && [ -z "$NVCC" ]; then
  echo "WARN: no CUDA toolkit (nvcc) found — building CPU-only. Install CUDA + re-run for GPU."
  CUDA=0
fi
if [ "$CUDA" = 1 ]; then
  export CUDACXX="$NVCC"
  export PATH="$(dirname "$NVCC"):$PATH"     # so cudafe++/ptxas resolve from the same toolkit
  echo "   using nvcc: $NVCC ($($NVCC --version 2>/dev/null | sed -n 's/.*release //p' | head -1))"
fi

# --- clone / update --------------------------------------------------------------
mkdir -p "$DIR"
if [ -d "$SRC/.git" ]; then
  c "updating $SRC ($BRANCH)"
  git -C "$SRC" fetch --depth 1 origin "$BRANCH"
  git -C "$SRC" checkout -q "$BRANCH" 2>/dev/null || true
  git -C "$SRC" reset --hard -q "origin/$BRANCH"
else
  c "cloning $REPO ($BRANCH, shallow)"
  git clone --depth 1 -b "$BRANCH" "$REPO" "$SRC"
fi
echo "   at commit: $(git -C "$SRC" rev-parse --short HEAD)"

# --- configure + build (llama-server target only) --------------------------------
CMAKE_ARGS=(
  -S "$SRC" -B "$BUILD" "${GEN[@]}"
  -DCMAKE_BUILD_TYPE=Release
  -DLLAMA_CURL=ON          # enables --hf-repo/--hf-file model download
  -DLLAMA_BUILD_SERVER=ON
  -DLLAMA_BUILD_TESTS=OFF
  -DLLAMA_BUILD_EXAMPLES=OFF
)
if [ "$CUDA" = 1 ]; then
  c "configuring (CUDA, arch=$ARCH)"
  CMAKE_ARGS+=(-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="$ARCH"
               -DCMAKE_CUDA_COMPILER="$NVCC")
else
  c "configuring (CPU-only)"
fi
cmake "${CMAKE_ARGS[@]}"

c "building llama-server (this can take 10–30 min for a first CUDA build)"
cmake --build "$BUILD" --target llama-server -j "$(nproc)"

# --- report ----------------------------------------------------------------------
BIN="$(find "$BUILD" -name llama-server -type f -perm -u+x | head -1)"
if [ -z "$BIN" ]; then echo "ERROR: build finished but llama-server not found under $BUILD"; exit 1; fi
BIN="$(cd "$(dirname "$BIN")" && pwd)/$(basename "$BIN")"
c "done"
echo "   llama-server: $BIN"
echo
echo "miniclosedai-llm auto-detects this path. To pin it explicitly, add to .env:"
echo "   LLAMACPP_SERVER_BIN=$BIN"
