#!/bin/bash
# The ggml/llama.cpp tree QwFNfer-xpu runs on: ggml-org/llama.cpp at a0e6dc7 (master of 2026-10-14; bbdd9f2 up to v0.4.0) plus the ggml-sycl patch series in
# patches/ggml-sycl (applied in file-name order), built for Intel GPUs with oneAPI (icx/icpx) and oneDNN.
#   scripts/b70/build-llama-sycl.sh [DIR]        (default ~/src/llama-b70)
# Needs: the oneAPI Base Toolkit (setvars.sh), a SYCL build of oneDNN (DNNL_DIR, default /opt/onednn-sycl),
# CMake and Ninja. The engine is then configured with
#   -DLLAMA_CPP_ROOT=DIR -DLLAMA_CPP_BUILD=DIR/build-sycl/bin
# and runs with QWFN_GGML_BACKENDS=DIR/build-sycl/bin.
set -euo pipefail
HERE=$(cd "$(dirname "$0")/../.." && pwd)
DIR=${1:-$HOME/src/llama-b70}
SHA=a0e6dc7384edb8f5d927155e1d26c10b4a310e6d
DNNL_DIR=${DNNL_DIR:-/opt/onednn-sycl/lib/cmake/dnnl}
ONEAPI=${ONEAPI:-/opt/intel/oneapi/setvars.sh}

if [ ! -d "$DIR/.git" ]; then
  mkdir -p "$DIR" && cd "$DIR"
  git init -q && git remote add origin https://github.com/ggml-org/llama.cpp.git
  git fetch -q --depth 1 origin $SHA && git checkout -q FETCH_HEAD
  for p in "$HERE"/patches/ggml-sycl/*.patch; do echo "applying $(basename "$p")"; git apply "$p"; done
else
  cd "$DIR"
  [ "$(git rev-parse HEAD)" = $SHA ] || { echo "$DIR is not at $SHA"; exit 1; }
fi
# setvars.sh exits (killing this script's set -eu shell) on some component probes, and
# 'source ... || true' cannot catch an exit from a sourced file: run it in a subshell
# and re-export what it set.
set +u; eval "$(ONEAPI="$ONEAPI" bash -c 'source "$ONEAPI" --force >/dev/null 2>&1; export')"; set -u
# ggml-sycl links oneMKL (MKL::MKL_SYCL::BLAS) and its CMake does find_package(MKL REQUIRED).
# The apt oneAPI does not register its cmake configs system-wide, so point MKL_DIR at the
# version directory setvars installed (mkl/<ver>/lib/cmake/mkl on the box and in CI).
MKL_DIR=${MKL_DIR:-$(find "$(dirname "$(dirname "$ONEAPI")")/mkl" -name MKLConfig.cmake -print -quit 2>/dev/null | xargs -r dirname)}
[ -n "$MKL_DIR" ] && echo "MKL_DIR: $MKL_DIR" || { echo "MKLConfig.cmake not found under $(dirname "$(dirname "$ONEAPI")")/mkl" >&2; exit 1; }
cmake -S . -B build-sycl -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx \
  -DGGML_SYCL=ON -DGGML_SYCL_TARGET=INTEL -DGGML_SYCL_F16=ON -DGGML_SYCL_DNN=ON -DGGML_SYCL_GRAPH=ON -DDNNL_DIR="$DNNL_DIR" -DMKL_DIR="$MKL_DIR" \
  -DGGML_NATIVE=OFF -DGGML_BACKEND_DL=ON -DBUILD_SHARED_LIBS=ON -DLLAMA_CURL=OFF
cmake --build build-sycl -j "$(nproc)" --target ggml llama test-backend-ops
echo "built: $DIR/build-sycl/bin"
