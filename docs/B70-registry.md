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

Last updated: 2026-10-13.

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
| ~~Gated-delta-net writing its rollback snapshot itself at T=2~~ **tried 2026-10-05: exact, no measurable gain (see below)** | ~1 ms per verify step (36 x 3 MB copies and their kernels) | M | MTP | the state and its snapshot in one allocation, so the existing fused cache write applies |
| Prefix cache on disk, kept across restarts *(Pennyroyal's HiCache/NIXL tier)* | a returning long conversation after a restart restores in ~1-2 s instead of a full prefill (~225 s at 100K); the RAM pool can shrink or spill parked sessions | M | when the process restarts mid-conversation | write each pool entry to NVMe in the background while idle; a namespace hash of everything that changes the bytes (weights, engine commit, ggml tree, KV type, context, head, draft vocabulary, determinism switches); byte-exact round trip through disk |
| BF16 recurrent (gated-delta-net) state, F32 compute *(Pennyroyal)* | ~1% decode (113 MB/token less traffic); rollback snapshots and the 118 MB fixed part of every checkpoint halve | M | small | the GDN kernel loading/storing BF16 state; decode-only NLL and a 100K needle check |
| Promotion settle one step later | the ~3 ms per step the per-layer promotion settle costs (a second queue alone did not overlap it): up to ~6% decode | M | yes | an event per layer after its promotions; settle layer L's at its next fetch instead of after its MoE |
| Promotion reorder off the GPU | the SOA reorder kernel per promoted part: ~1-1.5 ms per verify step | M | yes | keep RAM slots in the device layout, so a promotion is a plain copy |
| Staged Q8_0 experts through patch 31's kernel | ~0.3 ms per verify step (they still take the stock kernel) | S | yes | convert them to Q8_0_SOA in the staging upload |
| IQ4_NL expert down closer to bandwidth | ~1 ms per verify step (now ~290 GB/s) | S-M | yes | more rows per work-group to amortize the LUT build, or the LUT in registers |
| Layer-chain timeline (first step of the four below) | decides which of them is worth doing | S | yes | trace one decode layer: kernel durations, idle gaps between kernels, work-groups per kernel against the card's capacity; a layer costs ~0.45 ms of GPU time against a ~0.18 ms bandwidth floor and host launch is already hidden **Done 2026-10-14:** under the tracer decode runs ~5x slower, so the gaps are unusable; per token 3,402 kernels, 20.3 ms of kernel time (mean 6.0 us) in a ~33 ms step (device ~60% busy), 67% of kernels launch < 64 work-groups (the small elementwise ones; the matvecs launch 320-800 groups and fill the card), 397 q8_0 matvec launches take 7.4 ms, 579 quantize launches; a 2-token step costs 2.9x one token. Supports fusing the small kernels; a gap-accurate trace needs a lighter tracer. |
| Persistent-kernel bandwidth probe (follows the grid-barrier probe) | decides whether a megakernel can lose bandwidth | S | no | **Done 2026-10-14:** 1 GiB read in 64/16/4 MiB phases; one persistent launch with a bounded software grid barrier between phases reaches 585 / 570 / 502 GB/s at 32 groups x 1024 items (separate launches: 580 / 555 / 484); under 32 groups x 256 it cannot saturate (16 groups = 160, 32 = 275, 64 = 437 GB/s). Barrier ~0.9 us vs ~1.3 us per queued kernel boundary, so the win comes from widening small ops, not the barrier. |
| Warm start of the expert tiers | the first ~1000 tokens after every start decode at 49 ms vs ~32 warm (27 NVMe reads per token vs 7): ~17 s extra per start | M | yes | persist the per-expert counts at shutdown or take them from a profile run and preload the hot set before the first request; a seeded tier serves 91% of the first 100 tokens from VRAM vs 75% from empty, steady after ~200 tokens either way |
| Offload analysis (policy, prediction, swapping) | **Done 2026-10-14, mostly negative.** A simulator of the cache rule on recorded routing traces (validated against the engine's counters): LRU, full-scan LFU and the sampled rule are within 0.1% of each other; an oracle that knows the future saves ~1 NVMe miss per token; static or profile-seeded VRAM tiers serve 53-56% (cross-workload) or 92% (in-sample) against 97% for the dynamic tier; cheap predictors (history, cross-layer co-occurrence) catch at most 5-15% of the lookups that miss VRAM and ~0% of the NVMe misses; random expert-size reads saturate the drive at 3.2-3.3 GB/s from 3 reads in flight and splitting them is worse. Left: cheaper promotions (the settle and reorder items above; 12-16 promotions per token are inherent), the warm start, and VRAM capacity (~0.7 ms per token per GB) |
| Fuse more of the layer chain | fewer dependency barriers and kernel ramp-ups; earlier fusions paid 1-3% each | M | yes | chains of several ops from the timeline's gap list (norm, mixer, router, top-k), each exact against the unfused path **Step 1 done 2026-10-14 (patch 34, `GGML_SYCL_Q8_PROD`, opt-in, not in any config):** one-token UNARY+MUL / SWIGLU producers read next by a direct-path Q8_0 matvec also write the q8_1 copy into the reuse cache (about 96 of ~579 quantize launches per token gone). Bit-exact (replay NLL identical in 8 A-B-B-A arms, `Q8_PROD` backend tests 10/10, MUL_MAT q8_0 43/43) but no measurable speed gain (+0.5% +-1.5%): a removed launch is worth under ~2 us here. For a megakernel that puts the ceiling at ~10-20% before barrier cost. |
| Concurrent compute branches | the shared expert, routed experts and projections are independent but run back to back on one queue | M | yes | check what ggml-sycl already overlaps; then two in-order queues with events at the joins **Measured 2026-10-14, dropped:** the card exposes one compute queue and runs one kernel at a time. A two-queue and an out-of-order-with-events microbenchmark of the layer pattern (a big streaming read beside a chain of 3-12 tiny dependent kernels) shows no overlap even when the big kernel uses 32 of ~256 work-group slots (130.3 us serial vs 126.9 / 128.7 us), and each fork+join costs ~6 us. Raw Level Zero command lists without barriers are the only untested route. |
| Small expert kernels that fill the card *(generalizes the two kernel items above)* | expert matvecs run at 85-290 GB/s against ~470 for the best dense kernel; ~1 ms per step on IQ4_NL down alone | S-M | yes | more rows per work-group, split-K, the lookup table in registers via sub-group shuffles |
| Decode batched across sessions | aggregate throughput with several clients (not single-stream speed): the layer chain costs about the same per step for 1 or 3 sequences | L | yes, for concurrent clients | per-sequence state in one step; measure how fast the union of routed experts grows with the batch |
| Grid-wide software barrier (per-layer persistent kernel) -- **probed 2026-10-13: feasible** | the driver advertises cooperative kernels on the compute queue group and suggests 1024 / 256 / 64 co-resident work-groups at 64 / 256 / 1024 items; a bounded-spin barrier passed up to those sizes only, with no stale reads and no hang; ~1 us per barrier at 32 groups against 2.6-5.8 us per empty kernel boundary | L | yes, if the layer timeline shows gaps | SYCL itself gives no co-residency guarantee: size the launch to the driver's number or launch through Level Zero; whether 32 groups saturate bandwidth is not measured |

## Overlay v4 with the draft head and time sharing (2026-10)

Overlay v4 back as the base on the current stack (patches 01-27, draft head, exclusive expert tiers, time sharing).

| Change | Measured | Status |
|---|---|---|
| Overlay v4 at `--vram 24` | every check passes on one request, but three concurrent sessions run out of device memory (v4's dense tensors leave less headroom than v2's for the time-shared state) | rejected |
| Overlay v4 at `--vram 23` | all checks pass, three concurrent sessions included; 48.9% of expert blocks in VRAM (v2: 66.9% at 24); vs v2 prefill -9-10% (20K 408 vs 446, 100K 450 vs 490 tok/s), short-prompt decode ~-20% (28-32 vs 39-40 tok/s) | adopted (quality: natural-text NLL ~1.45-1.5 vs ~2.02) |
| Reasoning effort xhigh vs medium (the template's system sentence only) | one prompt, three attempts each: ~4x the tokens at xhigh on both overlays; on v4 the xhigh drawings were also far more elaborate | no change needed |

## Decode on overlay v4: budget and fixes (2026-10-05)

Measured through the server with a per-request split of the engine's decode counters: short prompts, and an agent-style
conversation (a ~52K-token first prompt, then turns of ~1.7K new tokens and a short answer). One server per arm,
A B B A, 120 W.

| Change | Measured | Status |
|---|---|---|
| Budget | short prompts: the layer graphs are ~70% of a step (35.5 of ~50 ms per verify step). After any streamed prompt the reply ran at 20-23 tok/s with the same graphs: ~25 experts per step not in VRAM against ~3, because the prefill borrows a third of the expert tier as staging and, with exclusive tiers, the experts those layers held end up on disk | the map |
| Restore the borrowed layers' VRAM experts before the reply (`QWFN_LEND_RESTORE=1`; batched reads, hottest first, 3 s budget) | agent turns at 55K context 22.6 -> 28.7 tok/s (+28%), after a 52K prompt +14-20%; experts per step not in VRAM 24 -> 7-9; 1.6-3 s more on each streamed prompt | **adopted** |
| Keep the borrowed layers inclusive in RAM (`QWFN_RAM_LENT_INCLUSIVE=1`) | turns +13%, short prompts -3%, still 11-20 experts per step not in VRAM | rejected |
| Turns through the cache-batched prompt path (`--prefill-decode-max 2048`, no product cap) | out of device resources on a 1.7K-token turn at 52K context | rejected |
| Patch 28: 2-4-column wide Q8_0 matvec (`GGML_SYCL_Q8W_NCOLS=1`) | the verify step's dense Q8_0 was on the stock multi-column kernel: 11.8 -> 9.3 ms per step; short-prompt decode +9% (32.6 -> 35.5 tok/s) | **adopted** |
| Rollback snapshots in the recurrent state's own allocation (the fused state write at T = 2) | exact, no measurable gain (31.5 vs 31.6 ms per step) | rejected |
| Wide one-token Q8_0 MoE matvec for the Q8_0 down layers | no gain (the 34-byte blocks only allow 16-bit loads) | rejected |
| The draft head over the trunk's selected cells (`QWFN_MTP_SHARED_CELLS=1`) | the head's dense attention grew with context (1.4 ms per step short, 5.8-6.7 ms at 52-60K); over the ~2K cells the trunk's last attention layer chose: 1.9-2.0 ms; agent turns +7%, short prompts and acceptance unchanged | **adopted** |

| A streamed prompt takes VRAM-resident experts from the tier (`QWFN_PF_VRAM=1`, patch 30) | a 1.7K-token turn at 55K context: disk reads 39.4 -> 24.6 GB, prompt 17.3 -> 13.6 s (-21%); a 52K prompt -6.5%; per-layer prefill hashes bit-identical | **adopted** |
| Q8_0 expert-down parts in the SOA layout with their own one-token MoE matvec (`QWFN_Q8_SOA=1`, patch 31) | the stock kernel ran one 16-lane work-group per 680-byte row (~85 GB/s): 2.17 -> 0.8 ms per verify step | **adopted** |
| 8 lanes per row for the IQ4_NL / Q2_0 expert kernels (`GGML_SYCL_IQ4_SOA_LPR=8`, `GGML_SYCL_Q2_LPR=8`, patch 32) | 20- and 40-block rows left 38% / 17% of 16 lanes idle: IQ4_NL 3.64 -> 3.0, Q2_0 gate/up 2.61 -> 2.4 ms per verify step | **adopted** |
| Vectorized greedy argmax (temperature 0) | 0.69 -> 0.03 ms per position on the host, same token; together with the two rows above: short-prompt decode +11% (34.0 -> 37.9 tok/s), agent turns at ~55K +6.5% | **adopted** |
| Expert promotions on a second in-order queue (`QWFN_PROMOTE_SIDE=1`, patch 33) | identical output; the ~3 ms per step of promotion settle unchanged: no overlap on this driver | rejected (switch off) |
| Victim read-back staged on device and read back on a side queue (`QWFN_PROMOTE_D2D=1`, patch 35) | identical placement and replay NLL, swap check clean; promotion settle ~2.8 -> ~1.7 ms per token, wall clock ~-0.5 ms per token (~1.5%, at the noise floor of an in-process off/on alternation), none with two drafts. A 1.6 MB upload takes 125 us and a read-back 116 us, they do not overlap each other, a copy on a second queue hides under a kernel | on in the launcher |
| Warm start of the expert tiers (`QWFN_WARM_COUNTS`) | use counts persist across restarts; the next start fills the tiers with the hot experts first (9 s); interleaved A-B-B-A x2: first 500 tokens 16 -> 27 tok/s, a cold 2,500-token run 100.0 -> 87.3 s, later tokens unchanged; VRAM contents verified against the file | on in the launcher |
| Pipelined restore after a streamed prompt | next layer's reads overlap this batch's uploads: 2.43 -> 2.0 s for 2,590 experts | on (`QWFN_LEND_RESTORE_SYNC=1` restores the old order) |
| Stale window 256 tokens (`QWFN_STALE_TOKENS`) | promotions -18..-25%, swaps -27..-35% (text, code, streamed prompt, MTP), replay NLL equal or better; interleaved pairs show no decode difference (-0.6%) | knob only, default 24 |
| IQ4_NL expert matvec work-group size; staged Q8_0 experts; host time outside the graphs | the IQ4 kernel runs at ~600 GB/s whatever the work-group size; no stock Q8_0 expert kernel in decode; nothing unaccounted outside the graphs | dropped |
| Power cap 160 W vs 120 W on this configuration | prefill +16-23% (20K 434 -> 534, 100K 458 -> 531 tok/s), short decode +5-11% (noisy), decode after long prompts ~equal | information (the cap is the operator's choice) |

Open from this round (before `QWFN_PF_VRAM`): a short streamed turn's prefill still swept every expert from disk (a 1.7K-token turn at 55K
context takes 14-17 s); the restore reads at ~2 GB/s and could overlap reads with uploads; ~2 ms per short-prompt step
is spent outside the decode timers.

Profile of a 20K prefill on this configuration: the MoE step is launch-bound -- ~35 s of wall time for ~11 s of
kernels, because every expert's rows of every chunk go through a separate dequantize -> convert -> gemm (~98K expert
matmuls for 20K tokens). A grouped MoE kernel (one launch per layer and chunk) is the largest prefill lever left;
promotions per layer per token stay at 2 (1 and 3 measured slower).

## Expert-down mid tiers and Swift 1.5 (2026-10-06)

Overlay v4 keeps expert down at IQ4_NL on 43 layers and Q8_0 on five. The mid tiers keep 4-bit down only where ISTA's
GSQ-RCO IQ3_XXS search does (18 layers) and use the base Q2_0 elsewhere, so more experts fit in VRAM. Same engine and
flags, 160 W, one server per arm. Replay NLL: a fixed 8K-token text, 128 replayed tokens.

| Layout | Experts in VRAM | Short decode | Agent turns at ~55K | Replay NLL | Status |
|---|---|---|---|---|---|
| v4 | 48.9% | 40.0 tok/s | 32.5 tok/s | 1.76 | reference |
| M2: IQ4_NL x13 + Q8_0 x5 + Q2_0 x30 | 59.5% | 44.1 (+10%) | 38.9 (+20%) | 2.09 | faster, large quality loss |
| M1: IQ4_NL x18 + Q2_0 x30 | 61.0% | 46.2 (+15%) | 41.2 (+27%) | 2.01 | faster, large quality loss |

Swift 1.5 (a fine-tune of the same model that targets shorter reasoning) runs unchanged on this engine with the
original model's draft head (acceptance unchanged, ~0.66) and tokenizer (identical). In the same three layouts it was
6-9% slower in short decode than the original weights (its routers are F32; not yet profiled). Thinking tokens on one
drawing prompt at xhigh, three samples each: v4 layout 53K (original 47K), M2 18.7K (24.5K), M1 12.3K (24.6K).

## Slimming (2026-10-14)

Tag `v0.4.0` keeps every experimental switch. Removed since, with their measurements: `docs/dropped-features.md`. Every behaviour this fork adds is now behind a switch, default off; the launcher turns the adopted ones on. Verified against `v0.4.0`: greedy output identical on three prompts, replay NLL identical, decode speed equal (24.83 / 25.02 vs 24.99 / 25.17 tok/s), backend tests and the server smoke and stress checks all pass.

## Upstream merge and llama.cpp rebase (2026-10-15)

The engine is merged with upstream QwFNfer master (2026-10-03: Windows support, the io worker pool, io_uring fixes, CI, attached files) and the ggml-sycl series is rebased onto llama.cpp master `a0e6dc7` (2026-10-14); patch 23 is gone (llama.cpp has its own AVX2 Q2_0 dot product). Against the previous build: backend tests pass, decode speed equal, replay NLL over 8 texts +0.1% (within noise), server smoke and 3-session stress pass. Upstream SYCL work that overlaps this series or its ideas (grouped MoE GEMM, Q8_0 wide-load MMVQ, more fusions, sparse FA) is still to be compared.


## Upstream's ops in place of ours (2026-10-15)

| Experiment | Result | Status |
|---|---|---|
| llama.cpp's Q8_0 kernels (#29186) instead of patches 18-21 + 28 | one column: upstream's ESIMD kernel on f32 activations ~1e-7 relative error per matvec against ~1.2e-2 for ours (q8_1 activations), plain decode equal, 16-text NLL +0.12% (SE 0.18) vs a summation-order control's +0.03%. Two columns (the MTP verify step): upstream's wide-load MMVQ makes the layer graphs ~6% slower (623-627 / 918-920 vs 587-588 / 886-887 us); a first "+5% decode" reading was expert-I/O noise | **19-21 dropped; 18 + 28 kept** for 2-4 columns (one column on upstream's kernel: `GGML_SYCL_Q8W=1 GGML_SYCL_Q8W_NCOLS=1`, 19-21 gone) |
| llama.cpp's DSV4_HC_PRE (engine `QWFN_HC_PRE_OP`) instead of patch 14's stream-mean fusion | decode layer graphs identical (recurrent 587-588 vs 587 us, attention 886-887 vs 885 us); 38K prompt dense graphs 27.0 / 26.3 vs 31.5 / 28.0 s; `test-backend-ops` 18/18 | **adopted** (patch 14 keeps only the bf16 matvec's SiLU epilogue) |
| llama.cpp's LIGHTNING_INDEXER (engine `QWFN_LIGHTNING_IDX`) instead of the indexer's matmul + patch 16's head sum | decode / verify steps: attention layer graphs 885 -> 864 us, launch 458 -> 440 us; prompts: +4-5 s dense on 38K, +25 s on 100K (a sub-group per (query, block) pair against the matmul engines); with HC_PRE, NLL over 8 texts -0.03% (SE 0.21) | **adopted at decode (`=1`)**; prompts keep patch 16 |
| llama.cpp's DSV4_HC_POST (engine `QWFN_HC_POST_OP`) instead of OUT_PROD + ADD | loses the fused combine + norm kernel (patches 04/06/11): recurrent 593-611 vs 587 us, launch +12 us per graph | rejected, code removed |
| A streamed prompt lends only what it needs (`QWFN_LEND_FIT`; the lendable tier as one buffer per layer) | 1.7K-token prompt lends 4.1 instead of 7.8 GB and reads 28.5 instead of 31.7 GB; agent turns at 52-61K context through the server (10 per side): prompt 15.0 -> 13.6 s mean, restore ~2.2 -> ~1.0 s, replies 28.5 -> 31.1 tok/s; 38K / 100K prompts in passes without an allocation failure (>= 1.3 GB free at every pass's end) | **adopted** |
| `--vision-gpu` after a long conversation | an image after a 61K-token agent conversation ran the device out of memory (33.56 of 34.24 GB in use; the process exits), on the deployed stack too: the tower's scratch came from the decode reserve. Fixed: every GPU encode borrows 2 GB from the tier first (`QWFN_VISION_LEND_MB`; the SYCL free-memory query said >= 2 GB free at 0.68 GB, so it cannot gate this), the arena is freed after each image | **fixed** |
