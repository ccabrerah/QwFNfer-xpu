# Intel Arc Pro B70 (SYCL)

QwFNfer-xpu runs Qwen3.8-Flash-Next on an Intel Arc Pro B70 (32 GB) through llama.cpp's SYCL backend. The
engine uses only the ggml backend API; the GPU work specific to this card lives in a ggml-sycl patch series
on top of llama.cpp `bbdd9f2`.

Reference machine: Arc Pro B70 on PCIe 3.0 x16, Ryzen 7 5700 (8C/16T), 32 GB RAM, NVMe Gen3 x4, oneAPI
2026.0, Linux 7.1 (xe driver). Model: `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF` Q2_0.

## Build

1. llama.cpp with the patch series (fetches `bbdd9f2`, applies `patches/ggml-sycl/*.patch` in order,
   builds with icx/icpx and oneDNN):

   ```sh
   scripts/b70/build-llama-sycl.sh ~/src/llama-b70        # DNNL_DIR=... if oneDNN is not in /opt/onednn-sycl
   ```

2. The engine against that tree:

   ```sh
   cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
     -DLLAMA_CPP_ROOT=$HOME/src/llama-b70 -DLLAMA_CPP_BUILD=$HOME/src/llama-b70/build-sycl/bin
   cmake --build build -j
   ```

   At run time `QWFN_GGML_BACKENDS=$HOME/src/llama-b70/build-sycl/bin` points the engine at the SYCL backend
   and `QWFN_REQUIRE_GPU=1` makes a missing one an error instead of a slow CPU run.

## The ggml-sycl patch series

Every change is off by default unless noted, so the patched tree behaves as upstream until a switch is set.
`test-backend-ops` covers each (`-o MUL_MAT`, `-o MUL_MAT_ID`, `-o FLASH_ATTN_EXT`, `-o OUT_PROD`, and the
`HC_COMBINE_NORM` case the series adds; patch 09 is checked against q2_0 op by op, and patch 10 by replay NLL
and a long-context needle).

| Patch | What | Switch |
|---|---|---|
| `01-mmid-small-t` | `mul_mat_id` keeps expert ids on the device for 2-8 tokens (a speculative verify step) instead of a host round trip | always |
| `02-fa-quant-nc-strides` | flash-attention fast paths reconstruct the strides of a padded quantized K/V view correctly ([llama.cpp#28384](https://github.com/ggml-org/llama.cpp/issues/28384)) | always (bug fix) |
| `03-outprod-native` | native `OUT_PROD` kernel for the hyper-connection outer products | always |
| `04-hc-combine-norm-fusion` | fused `OUT_PROD -> ADD -> RMS_NORM -> MUL` (the hyper-connection combine + norm) for prefill | `GGML_SYCL_FUSE_HC=1` |
| `05-mmvw` | wide-load bf16/f16 matvec at one token (16-byte loads, f32 accumulate, 16-256 lanes per row), and a small-K f32 matmul (K <= 32) without oneMKL's per-call host cost | `GGML_SYCL_MMVW=1`, `GGML_SYCL_SMALLK=1` |
| `06-hc-fusion-decode` | the patch-04 fusion at decode sizes | `GGML_SYCL_FUSE_HC_DECODE=1` |
| `07-moe-q2_0-wide` | q2_0 MoE matvec with one whole 64-weight block per lane | `GGML_SYCL_MOE_Q2W=1` |
| `08-moe-q2_0-glu-fused` | fused q2_0 MoE gate + up + SwiGLU at one token (one quantize, one kernel) | with `GGML_SYCL_MOE_Q2W=1`; `GGML_SYCL_MOE_GLU_OFF=1` disables |
| `09-q2_0-soa` | `GGML_TYPE_Q2_0_SOA`: a q2_0 slice stored as [16-byte code groups][fp16 scales], reordered on upload and back on download; the one-token wide MoE kernel and the fused GLU read it with aligned vector loads (bit-identical results); large batches dequantize | used when the engine asks for it (`QWFN_Q2_SOA=1`) |
| `10-sparse-decode-attn` | the one-token sparse attention (gather the selected cells from the q8_0 cache, cast to f16, `FLASH_ATTN_EXT`) as one chunked online-softmax kernel reading the q8_0 rows directly, plus a merge | `GGML_SYCL_FUSE_SPARSE_DECODE=1` |
| `11-hc-gate-in-k1` | the combine gate `scale(sigmoid(inj))` evaluated inside the patch-04 kernel instead of two launches | `GGML_SYCL_FUSE_HC_GATE=1` (with `GGML_SYCL_FUSE_HC`) |
| `12-topk-moe-div-output` | the router fusion (softmax, top-k, gather, normalise) also matches when the caller keeps the normalised weights (the DIV) as an output | always (`GGML_SYCL_TOPK_DIV_OUT=0` for the old match) |
| `13-topk-workgroup` | work-group-wide top-k: the router fusion with one lane per expert, and an argsort read only through its first-k view (the next-layer predictor) writes only those k | `GGML_SYCL_TOPK_WG=1` |
| `14-hc-mix-fusions` | the hyper-connection mixer: SiLU as the one-token matvec's epilogue, and sigmoid·x → permute → stream mean as one kernel | `GGML_SYCL_FUSE_HC_MIX=1` |
| `15-decode-fusions` | chains of 2-4 ADDs as one kernel; the MoE weighted sum reading the expert rows in place (no CONT); the one-token DeltaNet conv (concat, state update, conv, SiLU) as one kernel | `GGML_SYCL_FUSE_ADDCHAIN=1`, `GGML_SYCL_FUSE_MOESUM=1`, `GGML_SYCL_FUSE_CONV=1` |
| `16-indexer-head-sum` | the sparse-attention indexer's per-head score sum at prefill (relu, permute, copy, sum_rows over 4 heads) as one kernel that reads the scores once, summing in sum_rows' own order (bit-identical) | `GGML_SYCL_FUSE_IDX=1` |

The measurements behind each are in the research repository's `docs/decode-fusion-plan.md`,
`docs/hc-fusion-plan.md`, `docs/flash-attention-sycl.md` and `docs/outprod-the-real-bottleneck.md`.

## What not to turn on, and why

| Setting | Measured effect |
|---|---|
| `--spec-block` | Slower: the duplicate block per layer costs more GPU time than its better expert prediction saves in I/O. |
| `--mtp` (draft head) | 15-25% slower than plain on the current stack: 79-92% of drafts are accepted, but a draft + verify step costs 2.1-2.4x a plain step (the verify step falls off the one-token kernels and the head runs its experts on the CPU). |
| `--batch 32768` | The prefill borrows too much of the expert tier: decode right after drops sharply. 16384 is the setting. |
| `--vram` above 24 | Leaves under 2 GB of VRAM after a long document; the host needs ~1.5 GB free to stay clear of the driver's VRAM-to-RAM eviction. |
| Power cap below 140 W | 140 W costs ~6% prefill and decode; 130 W ~13-15%. |
