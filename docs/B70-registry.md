# B70 experiment registry

What has been tried for speed and quality on the Arc Pro B70, what it measured, and what is left to try. The point
is to avoid re-running dead ends: **before retrying a rejected item, check its "revisit when" column.** The patch
series is described in [`B70-SYCL.md`](B70-SYCL.md), the run configuration in the README.

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

Everything adopted is in the README's **Major improvements** table (what, where, measured effect); each ggml
patch in detail in [`B70-SYCL.md`](B70-SYCL.md).

## Tried and rejected

| Experiment | Result | Revisit when |
|---|---|---|
| `--spec-block` | slower: the duplicate block costs more GPU time than its prediction saves in I/O | graph time drops far enough that prediction dominates |
| `--mtp` draft head | 15-25% slower: 79-92% acceptance, but a draft + verify step costs 2.1-2.4x a plain step | the verify step gets one-token kernels and GPU experts |
| `--batch 32768` | the prefill borrows too much of the expert tier: decode right after drops sharply | — |
| `--batch 8192` / `4096` | prefill -19% / -24%; each halving doubles the passes over the expert set | only if VRAM is needed elsewhere |
| `--batch 24576` | +3-5% prefill with v2, decode unaffected; not adopted yet | re-measure decode after a long prefill with v3 (51% VRAM coverage) |
| Power cap below 140 W | 140 W -6%, 130 W -13-15% | — |
| `--vram` above 24 | leaves under 2 GB of VRAM after a long document; the host needs ~1.5 GB free to stay clear of the driver's VRAM-to-RAM eviction | a card with more VRAM |
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
