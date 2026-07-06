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
# Broad arch list incl. Blackwell (120a) — matches the known-good build on this box.
ARCH="${LLAMACPP_CUDA_ARCH:-80;86;89;90;100;120a}"

c() { printf '\033[1;34m>> %s\033[0m\n' "$1"; }

# --- preflight -------------------------------------------------------------------
for t in git cmake; do command -v "$t" >/dev/null || { echo "ERROR: '$t' not found."; exit 1; }; done
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
