# Building and preparing the model

From a clean Linux box to the files [`RUNNING.md`](RUNNING.md) starts the server with. Count on ~110 GB of disk
for the model and a few hours of downloads.

## 1. Prerequisites

| | |
|---|---|
| GPU | Intel Arc Pro B70 (32 GB), a recent kernel with the `xe` driver (tested on Linux 7.1), Level Zero + the compute runtime (`intel-compute-runtime`, `level-zero-loader` or your distro's equivalent) |
| Host | 32 GB RAM, a fast NVMe for the model (experts that do not fit in VRAM/RAM are read from it every token) |
| Toolchain | Intel oneAPI Base Toolkit 2026.0 (`icx`/`icpx`, `/opt/intel/oneapi/setvars.sh`), CMake, Ninja, git, python3, `hf` (`pip install huggingface_hub`) |
| oneDNN | a SYCL build of oneDNN (tested: 3.11.3), below |

Check the GPU is visible: `source /opt/intel/oneapi/setvars.sh && sycl-ls` should list a `level_zero:gpu` device.

oneDNN for SYCL (installs to `/opt/onednn-sycl`, where the build script looks for it):

```sh
source /opt/intel/oneapi/setvars.sh
git clone --depth 1 -b v3.11.3 https://github.com/uxlfoundation/oneDNN && cd oneDNN
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx \
  -DDNNL_CPU_RUNTIME=SYCL -DDNNL_GPU_RUNTIME=SYCL -DCMAKE_INSTALL_PREFIX=/opt/onednn-sycl
cmake --build build -j && sudo cmake --install build
```

## 2. Build

The engine runs on a patched ggml tree (llama.cpp `bbdd9f2` + `patches/ggml-sycl/*.patch`), then builds against it:

```sh
git clone https://github.com/ccabrerah/QwFNfer-xpu && cd QwFNfer-xpu
scripts/b70/build-llama-sycl.sh ~/src/llama-b70          # DNNL_DIR=... if oneDNN is elsewhere
source /opt/intel/oneapi/setvars.sh
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DLLAMA_CPP_ROOT=$HOME/src/llama-b70 -DLLAMA_CPP_BUILD=$HOME/src/llama-b70/build-sycl/bin
cmake --build build -j
```

You get `build/qwfn-server` (the OpenAI/Anthropic-compatible server) and `build/gguf-requant` (used below).
After pulling new patches, delete `~/src/llama-b70` and run the build script again (it only applies patches to a
fresh tree).

## 3. Model

The preferred model is **overlay v4**: the GSQ-RCO Q2_0 base with the tensors where Q2 hurts most (expert down,
attention, shared experts, mixers) replaced by Unsloth's higher-bit versions, plus the MTP draft head and vision.
Nothing is copied: an overlay is a small metadata head plus symlinks.

```sh
M=~/models/qwen38-flash-next
# base model (66 GB) and the vision projector
hf download ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF --local-dir $M/gsq-rco \
  --include 'Q2_0/*' 'mmproj-Qwen3.8-Flash-Next-BF16.gguf'
# overlay v4: fetches ~31 GB of tensors from Unsloth by HTTP range, writes the head
scripts/b70/build-overlay.sh $M/gsq-rco/Q2_0 $M/overlay v4
# MTP draft head (2.8 GB), experts re-encoded to Q2_0 (~0.7 GB more)
hf download unsloth/Qwen3.8-Flash-Next-GGUF MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf --local-dir $M
scripts/b70/build-mtp-head.sh $M/MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf $M/mtp
# draft vocabulary (optional, makes each draft cheaper): Strata's English/code ids + Latin-script tokens
curl -L -o $M/mtp/draft_vocab_en.bin https://raw.githubusercontent.com/Niko1221/Strata/HEAD/data/draft_vocab_en.bin
python3 tools/overlay/mk_draft_vocab.py $M/gsq-rco/Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf \
  $M/mtp/draft_vocab_en.bin $M/mtp/draft_vocab_lat60.bin 60000          # 74,557 ids
```

The files `RUNNING.md` uses:

| Variable | File |
|---|---|
| `QWFN_B70_GSQ` | `$M/gsq-rco/Q2_0` |
| `QWFN_B70_HEAD` | `$M/overlay/v4/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00008.gguf` |
| `QWFN_B70_MTP` | `$M/mtp/mtp-q2-00001-of-00003.gguf` |
| `QWFN_B70_DRAFT_VOCAB` | `$M/mtp/draft_vocab_lat60.bin` |
| `--mmproj` | `$M/gsq-rco/mmproj-Qwen3.8-Flash-Next-BF16.gguf` |

Optional, saves 15 GB: drop the tensors the v4 head shadows from the stock first shard (after this only v4-style
heads work with it):

```sh
python3 tools/overlay/shard_prune.py $M/overlay/v4 $M/gsq-rco/Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf pruned.gguf \
  && mv pruned.gguf $M/gsq-rco/Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf
```

Other overlays (v2: faster, lower quality) and what each part is: [`dense-overlay.md`](dense-overlay.md) and the
README's [preferred config](../README.md#preferred-config). The MTP head's recipe: [`B70-config.md`](B70-config.md).
