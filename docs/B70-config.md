# The B70 configuration

What production runs on the Arc Pro B70 (`scripts/b70/qwfn-b70.sh`): the SYCL build from
[`B70-SYCL.md`](B70-SYCL.md), the v2 dense overlay from [`dense-overlay.md`](dense-overlay.md), and these
switches and flags. Every switch has been measured on this card; the research repository holds the
measurements and the rejected alternatives.

## Switches

| Switch | What it does | Measured |
|---|---|---|
| `GGML_SYCL_FA_ONEDNN=1` `GGML_SYCL_ENABLE_MKL_FA=0` | oneDNN flash attention for prefill | 89K prefill 90-192 -> ~300 tok/s |
| `QWFN_DEV_MASK=1` `QWFN_QSA_PACK=1` | causal mask and QSA bias built on the device, one packed upload per chunk | 89K prefill 361 -> 389 tok/s (the mask); the QSA pack is part of the -14% below |
| `QWFN_LOCK_HOST=1` | locked, driver-registered host memory for the RAM tier and prefill staging | 89K prefill 425 -> 482 tok/s |
| `GGML_SYCL_FUSE_HC=1` `GGML_SYCL_FUSE_HC_DECODE=1` | hyper-connection combine + norm fused (prefill; decode) | with the QSA pack, 89K prefill -14% time; decode graph -0.8% |
| `KMP_BLOCKTIME=0` | the CPU-expert OpenMP pool stops spinning next to the kernel-launch thread | decode +5% |
| `GGML_SYCL_MMVW=1` | wide-load bf16/f16 one-token matvec | decode +23% |
| `GGML_SYCL_SMALLK=1` | small-K f32 matmul without oneMKL's host cost | decode graph -4.9% |
| `GGML_SYCL_MOE_Q2W=1` | wide q2_0 MoE matvec + fused gate/up/SwiGLU | decode graph -1.3%, -3% with the fused GLU |
| `QWFN_PREDICT_CUR2=1` | cheaper next-layer expert prediction (prefetch only) | decode +4.7% |
| `QWFN_Q2_SOA=1` | VRAM-tier q2_0 experts in the [codes][scales] layout (patch 09) | decode graph -18%, 40K held decode +20% |
| `GGML_SYCL_FUSE_SPARSE_DECODE=1` | one-token sparse attention as one kernel on the q8_0 cache (patch 10) | decode graph -8.5% |
| `GGML_SYCL_FUSE_HC_GATE=1` | the combine gate inside the combine + norm kernel (patch 11) | decode graph -1.5% |
| `GGML_SYCL_TOPK_WG=1` `GGML_SYCL_FUSE_HC_MIX=1` `GGML_SYCL_FUSE_ADDCHAIN=1` `GGML_SYCL_FUSE_MOESUM=1` `GGML_SYCL_FUSE_CONV=1` | small-kernel fusions at decode (patches 12-15), bit-identical to the unfused ops | decode graph -8% (-2.0 ms/token), decode ~+5% |
| `GGML_SYCL_FUSE_IDX=1` | the prefill indexer's head sum as one kernel (patch 16), bit-identical | attention -10% at 89K (78 -> 70 s), -1.8 s at 40K |
| `QWFN_VOCAB_MODEL=<stock head>` | tokenizer from the stock model (needed with an overlay) | - |
| `QWFN_GGML_BACKENDS=<llama build>/bin` `QWFN_REQUIRE_GPU=1` | which ggml backends to load; fail rather than run on the CPU | - |

Engine defaults in this fork (no switch): the late fold sized to the promotion budget (`QWFN_LATE_FULL=1`
restores), deferred speculative prefetch (`QWFN_PREFETCH_EARLY=1` restores).

## Flags

`--ctx 131072 --kv q8_0 --vram 24 --ram 8 --batch 16384 --prefill-chunk 6144 --reserve 2048 --prefix-cache 3`

- `--ctx 131072` is the per-session cap; the engine runs one sequence at a time.
- `--vram 24` leaves the ~2 GB this host needs free after a long prompt; `--ram 8` plus the 3 GB prefix
  cache plus ~4 GB for the system is what a 31 GB machine can give.
- `--prefix-cache 3`: host-memory checkpoints of conversations switched away from (~2 at 64K, one at 130K).

## Result

On the reference machine, 160 W cap, v2 overlay: 25-27 tok/s warm decode at 20-100K of context (~28 on
short prompts; ~30 with the v1 overlay), 440-480 tok/s prefill at 20-100K. The first B70 enablement
(2026-09-15, UD-Q3_K_XL, no switches) ran 15-16 tok/s warm chat and 222-248 tok/s prefill at 21K.

## Benchmarks

`tools/perf`: `decab.py` (warm decode with the `/stats` decode split), `ctxdec.py` (prefill and decode after
a long prompt from a fixture: `{"messages": [...]}`), `mtpprobe.py` (tok/s, draft acceptance and step cost),
`pcswitch.py` (two alternating conversations: the prefix cache at work), and the `mv_bench` / `moe_bench`
kernel benches (CMake targets: `mv_bench BACKEND_DIR`).
