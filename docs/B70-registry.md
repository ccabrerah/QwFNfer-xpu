# B70 experiment registry

What has been tried for speed and quality on the Arc Pro B70, what it measured, and what is left to try. The point
is to avoid re-running dead ends: **before retrying a rejected item, check its "revisit when" column.** The patch
series itself is described in [`B70-SYCL.md`](B70-SYCL.md), the run configuration in [`B70-config.md`](B70-config.md).

Measurement notes that apply throughout:

- **Decode varies between server starts** by up to ~13% on the reference machine: how fast missed experts come off
  the NVMe differs from start to start. Compare configurations with interleaved arms (A-B-A-B), three or more
  starts each.
- **A change that alters rounding also alters routing.** Greedy decode drifts onto other tokens and other experts, so
  per-token counters stop being comparable. For kernel changes, time a teacher-forced replay
  (`qwfn-gen --replay-file`), which reports the decode split over identical tokens.
- **Quality** is natural-text NLL (lower is better) over 1,024 tokens after an 8K prompt, measured both through
  prefill and token by token through the decode path (the reference for the sparse attention).

Last updated: 2026-09-26.

## Adopted

| Change | Measured | Where |
|---|---|---|
| q2_0 experts stored as [codes][scales] per slice | decode layer graphs -18%, 40K decode +20% | patch 09, `QWFN_Q2_SOA` |
| Wide-load bf16/f16 one-token matvec | decode +23% | patch 05, `GGML_SYCL_MMVW` |
| One-token sparse attention as one kernel over the q8_0 cache | decode layer graphs -8.5% | patch 10 |
| Small-K f32 matmul without oneMKL's per-call host cost | decode layer graphs -4.9% | patch 05, `GGML_SYCL_SMALLK` |
| Wide q2_0 MoE matvec, fused gate/up/SwiGLU | decode layer graphs -4% | patches 07-08 |
| Hyper-connection combine + norm (+ its gate) as one kernel | decode -2.3%; 89K prefill -14% with the device-built inputs | patches 04, 06, 11 |
| Small-kernel fusions (router top-k, hc mixer, ADD chains, MoE weighted sum, DeltaNet conv) | decode ~+5% | patches 12-15 |
| Prefill indexer head sum as one kernel | prefill attention -10% at 40-89K; bit-identical | patch 16 |
| **IQ4_NL experts as [codes][scales], decoded through a local-memory table** | one-token IQ4_NL MoE matvec 2.75x; overlay v3 decode graph -12% | patch 17, `QWFN_IQ4_SOA` |
| OpenMP pool not spinning next to the launch thread | decode +5% | `KMP_BLOCKTIME=0` |
| Next-layer expert prediction from the FFN input | decode +4.7% | `QWFN_PREDICT_CUR2` |
| oneDNN flash attention for prefill | 89K prefill 90-192 -> ~300 tok/s | `GGML_SYCL_FA_ONEDNN` |
| Causal mask and sparse-attention bias built on the device | 89K prefill 361 -> 389 tok/s | `QWFN_DEV_MASK`, `QWFN_QSA_PACK` |
| Locked, driver-registered host memory for the RAM tier and staging | 89K prefill 425 -> 482 tok/s | `QWFN_LOCK_HOST` |
| Prefill sparse-attention selection fix (no slots on blocks after the query; the partial tail block's own cells) | usable cells/query 823 -> 1,998 at 40K; prefill NLL now agrees with decode | engine |
| Dense overlay v2, then **overlay v3** (the preferred config) | v3 NLL 1.51 vs v2 2.02 (decode path) | `scripts/b70/build-overlay.sh` |
| Shared System USM off (`NEOReadDebugKeys=1 EnableSharedSystemUsmSupport=0`) | no measured cost; removes the kernel SVM path for host copies | host environment (README) |

## Tried and rejected

| Experiment | Result | Revisit when |
|---|---|---|
| `--spec-block` | slower: the duplicate block costs more GPU time than its prediction saves in I/O | graph time drops far enough that prediction dominates |
| `--mtp` draft head | 15-25% slower: 79-92% acceptance, but a draft + verify step costs 2.1-2.4x a plain step | the verify step gets one-token kernels and GPU experts |
| `--batch 32768` | the prefill borrows too much of the expert tier: decode right after drops sharply | — |
| `--batch 8192` / `4096` | prefill -19% / -24%; each halving doubles the passes over the expert set | only if VRAM is needed elsewhere |
| `--batch 24576` | +3-5% prefill with v2, decode unaffected; not adopted yet | re-measure decode after a long prefill with v3 (51% VRAM coverage) |
| Power cap below 140 W | 140 W -6%, 130 W -13-15% | — |
| Rank-counting argsort (one barrier instead of bitonic stages) | +4% decode graph time | a top-k that reads each value once |
| Grouped oneMKL `gemm_batch` for the prefill MoE | no faster than per-expert GEMMs; prefill unchanged | — |
| Grouped XMX MUL_MAT_ID kernel (joint_matrix, one launch over all experts) | correct, +3% prefill at 40K; the kernel plateaus near 20 TFLOP/s | the kernel gets well past that |
| Device-side routing for the prefill MoE | same +3%: the host round trips were not the bottleneck | — |
| Tile-sparse prefill attention (skip K/V tiles the selection does not use) | not viable: tiles a kernel can run efficiently still touch 44-92% of the dense work | — |
| Async prefill expert uploads on a side queue | correct, no gain: the graph's small host-to-device copies wait behind queued upload pieces | a queue/engine setup where copies really run concurrently |
| Pinned vs pageable prefill staging; keeping the device staging between requests | no effect | — |
| Deeper expert prefetch pool | no effect: the demand wait comes from misses, not queue depth | — |
| IQ4_NL kernels: one block per lane (8-32 lanes/row), rows staged in local memory, table in registers, aligned layout alone | all ~110 GB/s: the stock table lookup is ALU-bound (`dpct::byte_level_permute` is written with 64-bit variable shifts) | superseded by patch 17's local-memory table |
| Q4_K expert gate/up | 2.3x faster kernel, but bigger blocks cut VRAM residency: net decode worse | the expert tier can hold the whole model |
| F16 / Q8_0 experts; MXFP4, Q5_0, Q3_K | slow or too big here | MXFP4: see ideas (its lookup is the same emulated permute) |
| Level Zero knobs (counter-based events, single-thread mode, in-order lists) | no change: layer-graph time is device-side | — |

## Ideas -- not yet tried

Ranked by expected gain. When one is tried, move it to a table above.

| Idea | Expected | First step |
|---|---|---|
| Find the start-to-start expert-read variance | up to ~13% decode on some starts | per start: io_uring depth reached, submit/complete CPUs vs NVMe interrupts, compressed extents on the model files |
| Patch 17's local-memory table for other table types (MXFP4, IQ4_XS) | MXFP4/IQ4_XS kernels several times faster; upstreamable | port the lookup, bench against the CPU with a 64-matmul graph |
| Expert residency for overlay v3 (51% VRAM coverage) | the largest remaining decode lever on v3 | frequency-weighted VRAM tier; misses/token at 20-100K |
| Q8_0 dense matvec (~67% of bandwidth) | ~3-4% decode on v3 | a wide-load kernel like `mmvw` |
| Prefill upload overlap on a dedicated copy engine | ~8-15% prefill | route uploads to a separate copy engine; keep the graph's small copies off that queue |
| GSQ-RCO IQ3_S (3.50 bpw) as a base model | quality/speed between v3 and a full Q4 | NLL through the decode path + decode speed vs v3 |
| XMX grouped MoE kernel past 20 TFLOP/s | a few seconds per 40K prefill | tile and SLM layout profile |
| Deterministic greedy decode on v3 | reproducibility | find the op whose result varies between runs |
| hc mixer weights in q8_0 (bf16 matvecs are at bandwidth) | ~2-3 ms/token | overlay + NLL |
| One graph per token (instead of 48 per-layer graphs) | launch overhead | large: needs the expert residency decided before the token |
| Router top-k over a work-group; radix top-k for the predictor | ~0.5-1 ms/token each | small kernels |
