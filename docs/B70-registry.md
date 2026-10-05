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

Last updated: 2026-10-04.

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
| Power cap 120 W (vs 140 / 160 W), overlay v4 with the draft head | 140 W +6-7%, 160 W +11-13% over 120 W, prefill and decode alike (short decode 30.5 / 32.6 / 33.8 tok/s; 100K prefill 443 / 471 / 499); card power in decode 120 / 139 / 156 W | 120 W kept for acoustics (coil whine at higher caps) |
| `--vram` above 25 (overlay v4) | 25 leaves ~0.7 GB free at a full context with an image and nothing evicted; 26 was below that already with v2 (1.8 GB free after init, under the engine's reserve check) | a card with more VRAM |
| Rank-counting argsort (one barrier instead of bitonic stages) | +4% decode graph time | a top-k that reads each value once |
| A larger locked RAM expert tier (`--ram 11` / `13` vs 8, preferred config) | 15% fewer expert misses but decode unchanged within the between-start I/O spread; 40K prefill 3.7% slower at 11, 7% at 13 | once expert-read speed is stable between starts |
| Q8_0 matvec with 2 or 4 rows per sub-group (activation loads shared across rows) | no faster than one row (21.6 / 21.6 / 22.6 us per call): the activation re-reads hit the cache | — |
| Hyper-connection mixers at Q8_0 without patch 21 | quality unchanged, GPU time -1.8 ms/token, but host launch time +1.8 ms/token (activation quantize, unfused SiLU, the scale not foldable exactly) | adopted with patch 21 (overlay v4) |
| Split-K Q8_0 matvec for matrices with few rows | no faster than one sub-group per row | — |
| One-token Q8_0 matvec on f32 activations (no quantize kernel; SiLU/scale epilogue) | launch time -33 us per layer graph, but GPU time +42 us (f32 converts and FMAs cost more than dp4a): +0.4 ms/token on the preferred config | produce q8_1 activations in the producing op instead |
| q8_1 reuse alone (patch 19 without 20) | 108 fewer quantize launches per token but only ~-0.1 ms/token: the quantize kernel is cheap, the generic matmul routine around it is not (hence patch 20) | — |
| Top-k with a row in one sub-group (values in registers, no work-group barriers; patch 13 pays one per selected value) | 2x slower at decode shapes (predictor 8.0 -> 16.9 us, router 6.9 -> 12.2 us); correct | never in this form: patch 13's top-k is 0.70 ms/token in total |
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

## Where decode time goes (preferred config, 2026-09-26)

Short-prompt warm decode on the reference machine, ~40 ms/token (~25 tok/s), from the server's `/stats` decode
split (`tools/perf/decab.py`):

| part | ms/token | what it is |
|---|---:|---|
| layer graphs on the GPU | ~26 | dense matvecs (Q8_0 in overlay v3: ~7.4; ~6.4 with patch 18; patches 19-20 then cut their host-side launch cost), bf16 hyper-connection matvecs (~5.5, at bandwidth), GPU experts (IQ4_NL down ~1.5, Q2_0 gate/up ~1.3), attention, DeltaNet, ~5,000 small kernels |
| expert I/O | ~7.6 (7-13 between starts) | waiting for missed experts read from the NVMe (~4 misses/token; 51% of expert blocks fit in VRAM) |
| host | ~6.5 | CPU-computed experts (~7/token), routing readback, promotions |

## MTP, the Strata recipe (2026-10)

Ideas from [Strata](https://github.com/Niko1221/Strata) (a CUDA/HIP engine for this model) ported to this engine,
measured on overlay v2 (stock GSQ-RCO Q2_0 + Unsloth tensors), vision loaded, one draft per step.

| Change | Measured | Status |
|---|---|---|
| Head with Q2_0 experts (per-block least-squares scale), all on the device, 0.81 GB | ~35% relative RMS on the experts (the 2-bit optimum); ~80% acceptance | adopted |
| Confidence gate (p >= 0.5), up to 3 drafts | 2.4-2.8 tokens per step, but T changes every step: graphs rebuilt per step, or a per-T graph cache that needs more device memory than `--vram 24` leaves; one device-lost fault under it | rejected for now (revisit with VRAM headroom) |
| Head logits read back only when sampling; draft vocabulary; no layer-0 prefetch pass | host time per step ~9.5 -> ~4.4 ms; the prefetch pass's speculative reads also crowded the drive | adopted |
| Draft vocabulary: English/code (40,525 ids) vs + Latin-script tokens (74,557) | Spanish acceptance ~46% -> ~66% (78% with the full vocabulary) for +0.3 ms of head | the Latin-script one |
| AVX2 Q2_0 dot product (patch 23) | 2.6x per core; end to end the CPU expert time -12% (per-call overhead dominates) | adopted |
| Verify-step attention: cache-free work, then indexer/scoring/top-k/masks batched over the positions | 167 -> 126 kernels per attention graph at T=2, layer graphs 28.4 -> 27.4 ms per step, identical answers | adopted |
| Promotion budget kept at the one-token value for verify steps | no change | rejected |
| Prefix cache with the head (checkpoints carry its state) | restores in ~110 ms, drafting continues | adopted |
| Exclusive expert tiers (`QWFN_RAM_EXCLUSIVE=1`), `--ram 8`, A/B in one window | expert I/O per step 14.4 / 12.3 / 11.6 -> 2.4 / 0.7 / 2.5 ms; essay / code / t=0.7 32.5 / 37.6 / 37.6 -> 44.9 / 46.8 / 44.1 tok/s; decode after a 20K prompt 19.1 -> 17.1 (64 tokens, one sample: the prefill-lent layers' VRAM experts go to disk); 40K prefill, prefix-cache restores, vision unchanged; every read-back byte-exact | adopted (opt-in switch) |
| Exclusive at `--ram 11`; and with the prefill-lent layers inclusive (`QWFN_RAM_LENT_INCLUSIVE=1`) | 47.7 / 47.9 / 46.2 tok/s, after 20K 20.8, but the VMs swap (+1.5 GiB, ~260K pages in); lent-inclusive: 40.8 / 45.9 / 38.8, after 20K 22.1 (less coverage for short prompts) | 8 for now; zone split not worth it |
| The exclusive tier's per-layer settle (stream sync after the promotion copies) | all of it is the copies' transfer at PCIe Gen3 speed (~14 GB/s: ~8 promotions/step, up to ~35 MB/step both ways), no host work; a read-back off the critical path would save <= ~1 ms/step and needs a second device queue | not pursued |
| `QWFN_SWAP_MIN_EF` (cold victims not read back) | fewer read-backs, more disk reads: slower at every threshold (2 / 4 / 8) | rejected (off) |
| Two or three head drafts every step | a step's layer graphs grow ~12 ms per extra position (1 / 1.37 / 1.73x) against +0.55 / +0.47 committed tokens: -6 / -10% on ordinary text, +8 / +12% on copying replies; three drafts run out of device memory at `--vram 24` | rejected: one draft |
| Prompt-lookup drafts, 2 per lookup step (`--lookup-drafts 2`) | exact; +1..+8% on copying replies (mean ~+4%), ordinary text unchanged, ~230 MiB of device memory. The policy needs a warm-up skip after a window-size change (the first steps run ~2x) and costs trusted only after several steady samples | built, off by default |
| Per-kernel cost of 1-4 positions (unitrace, plain / 1 / 2 / 3 drafts) | per extra position: q2_0 MoE +3.5-4.5 ms (each position's own experts), dense Q5_K/Q6_K +2.3 to the first extra (the one-column ESIMD kernel is left for the generic n-column MMVQ) then +1.5, attention +0.55, an element-wise broadcast +1.4 from 3 positions | targets below |
| 2-4-column Q5_K/Q6_K matvecs through the reordered ESIMD kernel (patch 24, `GGML_SYCL_DMMV_NCOLS=1`) | verify-step layer graphs 32.0 -> 29.9 ms; decode +5-7% (A/B/B/A); identical greedy answers; verify positions now use f32 activations as one-token decode does | adopted |
| q2_0 MoE GLU expert dedup across the verify step's positions (a shared expert's rows read once) | bit-identical by construction; no gain (layer graphs ~30.5 ms both ways) | rejected |
| Gathered sparse prefill attention, first versions (per-query gather; then DPAS with the GQA heads as matrix rows) | correct (max diff ~1e-3 relative per layer); slower than oneDNN's dense SDPA on XMX at 8K and 79K | superseded by the row below |
| Gathered sparse prefill attention (patch 25, `GGML_SYCL_FUSE_SPARSE_PREFILL=1`, `GGML_SYCL_SPF_MIN_KV=45056`): streaming cell compaction, Q held in registers, halfword q8_0 loads, dense below 45K cells or 256 queries | per-call ~0.22 -> ~0.14 s; prefill 100K 226.6 -> 213.3 s, 124.5K 295.1 -> 253.5 s, 79K equal; through the server +12% at ~100K and at 126K + image; at the VRAM limit nothing evicted. Without the 256-query floor, decode after a long prompt fell 12.8 -> 8.8 tok/s (one work-group per query and kv head) | adopted |
| Expert RAM tier size `--ram 11 / 12 / 13 / 14` (exclusive tiers) | short decode flat (~43-45 tok/s); at 14 the worst case (130K + image) swaps 7.0 vs 4.5 GiB, major faults 3.0 vs 1.5 M, its decode 8.4 vs 14.5 tok/s | 11 kept |
| **The configuration** (`--vram 24` + head vs `--vram 25` without) | ~+13% short-prompt decode; at the limit (image + 126K document) ~0.45 GB free, nothing evicted | adopted |

## Ideas -- not yet tried

Ranked by expected gain for the preferred config (overlay v4, patches 01-21, `--vram 25`) per effort. Decode is now
close to GPU-bound -- per layer graph ~310 us of host launch overlaps ~450 us of GPU work -- so GPU work, expert
misses and expert I/O are what is left; launch-side savings no longer pay. When one is tried, move it to a table
above. Effort: S = a kernel, a switch or one measurement session, M = a few days, L = a week or more.

| Idea | Expected | Effort | Helps the preferred config | First step |
|---|---|---|---|---|
| A fresh decode-time budget on the current build | the map for the GPU-side ideas below (the one above predates patches 18-21) | S | yes | per-kernel device time and effective GB/s per kernel family |
| Expert residency | the largest decode lever: ~2.2 misses/token cost expert I/O and most of the host time, more at long context | M-L | yes | misses per token at 20-100K; keep the most-used experts resident |
| Find the start-to-start expert-read variance | identical reads vary +-9% between starts; cleaner comparisons | S | yes | background discard (TRIM) and the drive's host-memory buffer are left; ruled out: filesystem compression, the I/O scheduler, drive temperature, CPU clock, page-cache state |
| Prefill upload overlap on a dedicated copy engine | ~8-15% prefill | M | yes | route uploads to a separate copy engine; keep the graph's small copies off that queue |
| Expert matvec kernels (IQ4_NL down, Q2_0 gate/up) closer to bandwidth | ~0.5-1 ms/token | M | yes | effective GB/s of each in the new budget |
| The remaining BF16 one-token matvecs | up to ~0.6 ms/token if the tensors can take Q8_0 | S-M | maybe | which tensors they are; the indexer projections are BF16 in Unsloth's quants too |
| Deterministic greedy decode | reproducibility; cheaper quality comparisons (NLL varies ~0.04 between identical runs) | M | hygiene | find the op whose result varies between runs |
| Prefill DeltaNet | a few % prefill | M | yes | per-kernel profile at 40K |
| XMX grouped MoE kernel past 20 TFLOP/s | a few seconds per 40K prefill | L | yes | tile and SLM layout profile |
| GSQ-RCO IQ3_S (3.50 bpw) as a base model | quality/speed vs v4 unknown | M | maybe | NLL through the decode path + decode speed vs v4 |
| One graph per token (instead of 48 per-layer graphs) | small now that decode is GPU-bound | L | small | only if the host is back on the critical path |
| Patch 17's local-memory table for MXFP4 / IQ4_XS | several-x faster kernels for those types; upstreamable | S-M | no (v4 uses neither) | port the lookup, bench against the CPU with a 64-matmul graph |
| Host RAM large enough for every expert not in VRAM (~22 GB on top of the system) *(Strata's design)* | takes the NVMe out of decode: est. +15-20% short-prompt decode, more at long context, and no read variance | hardware | yes | re-size `--ram` with an A-B-B-A after the upgrade |
| Startup expert profile *(Strata)* | warm first requests after a model switch | S-M | maybe | rank (layer, expert) pairs from a routing dump; pre-fill the VRAM tier at load |
| The CPU experts' per-call overhead | ~0.3 ms per CPU-computed expert of which the dot products are ~0.05-0.1 (thread wake-up, activation quantization, the CPU graph) | M | yes | per-call timing of the CPU MoE path; a persistent worker pool |
| Gated-delta-net writing its rollback snapshot itself at T=2 | ~1 ms per verify step (36 x 3 MB copies and their kernels) | M | MTP | the state and its snapshot in one allocation, so the existing fused cache write applies |
| Prefix cache on disk, kept across restarts *(Pennyroyal's HiCache/NIXL tier)* | a returning long conversation after a restart restores in ~1-2 s instead of a full prefill (~225 s at 100K); the RAM pool can shrink or spill parked sessions | M | when the process restarts mid-conversation | write each pool entry to NVMe in the background while idle; a namespace hash of everything that changes the bytes (weights, engine commit, ggml tree, KV type, context, head, draft vocabulary, determinism switches); byte-exact round trip through disk |
| BF16 recurrent (gated-delta-net) state, F32 compute *(Pennyroyal)* | ~1% decode (113 MB/token less traffic); rollback snapshots and the 118 MB fixed part of every checkpoint halve | M | small | the GDN kernel loading/storing BF16 state; decode-only NLL and a 100K needle check |

## Overlay v4 with the draft head and time sharing (2026-10)

Overlay v4 back as the base on the current stack (patches 01-27, draft head, exclusive expert tiers, time sharing).

| Change | Measured | Status |
|---|---|---|
| Overlay v4 at `--vram 24` | every check passes on one request, but three concurrent sessions run out of device memory (v4's dense tensors leave less headroom than v2's for the time-shared state) | rejected |
| Overlay v4 at `--vram 23` | all checks pass, three concurrent sessions included; 48.9% of expert blocks in VRAM (v2: 66.9% at 24); vs v2 prefill -9-10% (20K 408 vs 446, 100K 450 vs 490 tok/s), short-prompt decode ~-20% (28-32 vs 39-40 tok/s) | adopted (quality: natural-text NLL ~1.45-1.5 vs ~2.02) |
| Reasoning effort xhigh vs medium (the template's system sentence only) | one prompt, three attempts each: ~4x the tokens at xhigh on both overlays; on v4 the xhigh drawings were also far more elaborate | no change needed |
