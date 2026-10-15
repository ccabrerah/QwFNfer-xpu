# Dropped features

Switches, code paths and ggml-sycl patches that were built, measured and then removed to keep the fork's custom
surface small. Every one of them is still in the tree at tag **`v0.4.0`** (2026-10-14); revive from there.

Read this before building something new on the decode or prefill path: if an idea matches an entry here, start
from its numbers instead of measuring it again under another name. All measurements are on an Arc Pro B70 (32 GB,
160 W unless noted), Ryzen 7 5700, 32 GB DDR4, one PCIe NVMe, Qwen3.8-Flash-Next with exclusive expert tiers
(`--vram 23 --ram 11` unless noted); the B70 registry (`B70-registry.md`) has the surrounding context.

What stays is listed in `PARAMETERS.md`. Every custom feature that stays is behind a switch, default off, and the
launcher (`scripts/b70/qwfn-b70.sh`) turns on the adopted ones.

## Engine switches

### `QWFN_PROMOTE_SIDE` -- decode's expert promotions on a second queue

- **What it did:** put a promotion's copies (the upload of the promoted expert, the swap read-back of the victim, the
  SOA reorder) on a second in-order SYCL queue (ggml-sycl patch 33, `ggml_backend_sycl_init_side_queue`), so they
  would run beside the layer's kernels instead of after them. Off inside lends (prefill, image).
- **Data:** output identical (replay NLL equal), stress test passed; promotion settle unchanged at ~3.0 ms per step;
  short decode 38.0 vs 37.8 tok/s (noise).
- **Why it failed:** the promoted expert is folded into the very next graph, so the upload is on the critical path
  whichever queue carries it. A copy probe later confirmed the hardware does overlap a copy on a second queue with
  a kernel, but host-to-device and device-to-host copies do not overlap each other, and the upload has nothing to
  hide under.
- **What replaced it:** `QWFN_PROMOTE_D2D` (kept): only the read-back leaves the critical path (victim copied aside
  on the device, read back on the side queue a layer later): settle 2.8 -> 1.7 ms/token, decode ~1.5%.
  Patch 33 stays because D2D uses its side queue.

### `QWFN_PROMOTE_D2D_ALT` -- in-process A/B of D2D (qwfn-gen)

- **What it did:** in `qwfn-gen`, switched `QWFN_PROMOTE_D2D` on and off every N decode tokens inside one process
  (a full settle at each switch) and reported ms/token per mode from token 500 on; also the engine's
  `set_promote_d2d()` / the cache's `set_d2d()`.
- **Data / lesson:** separate processes on this box drift by +-4% (layer-graph time moved 55-70 s per 2,500
  tokens between runs of one config), more than most single optimizations. In-process alternation gave plain decode
  -0.33 / -0.84 / -0.51 ms/token for D2D, MTP-1 -0.80 / -0.57, MTP-2 +0.08 / +0.05, and A/A controls +0.18 / -0.80:
  so even this method has a ~0.5 ms/token floor. Interleaved A-B-B-A processes inside one GPU window (scripts in
  the research repo) became the standard instead, because they need no engine support.
- **Why dropped:** measurement-only code in the product binary; the method is kept in the test scripts.

### `QWFN_LEND_RESTORE_SYNC` -- the restore's old one-batch-at-a-time order

- **What it did:** made `QWFN_LEND_RESTORE` (refill the lent layers' VRAM tier after a streamed prompt) read a batch
  of experts, upload it, settle, then read the next, with nothing overlapped.
- **Data:** the pipelined order that replaced it (the next layer's batch read while this one uploads; the reads take
  RAM victims from another layer's pool) restored the same 2,590 experts (4.96 GB) in 2.00 s instead of 2.43-2.45 s
  (-17%, about 0.4 s of time-to-first-token per streamed turn); batch 64 was no different from 32 in either order.
  The reads still run at ~2.5 GB/s against the drive's ~3.3-3.5 GB/s.
- **Why dropped:** superseded; the restore itself stays behind `QWFN_LEND_RESTORE`.

### `QWFN_STALE_TOKENS` -- the VRAM tier's staleness window

- **What it did:** made the engine's fixed rule "a VRAM resident unused for 24 tokens is replaceable whatever its use
  count" (`STALE_FETCHES = 48 * 24`, upstream's constant) tunable.
- **Data:** 256 tokens instead of 24: promotions -18% (natural text, 35.7k -> 29.3k per 2,500 tokens), -21% (free
  generation from a one-token prompt, 28.6k -> 22.7k per 1,500), -25% (MTP-1, 39.9k -> 30.0k); swaps -27..-35%; sweep 24/64/128/256/512 promotions
  35.7k / 34.2k / 31.7k / 29.3k / 28.7k. Decode in within-window runs looked +4..6%, but interleaved A-B-B-A pairs
  (twice) gave 103.2 vs 102.6 s per 2,500 tokens (-0.6%, noise). Replay NLL equal or better in every arm (it moves
  with placement, see the note at the end). An offline simulator on routing traces had predicted <1-2 ms/token.
- **Why it failed:** not established. Fewer promotions mean less PCIe traffic (a promotion's upload is ~0.12 ms),
  but the decode time did not move; a likely reason is that the experts not promoted are then computed on the CPU
  or read again later, which costs about what the promotion saved.
- **Don't redo as:** a hysteresis or budget knob on promotions -- the same simulator showed the engine's sampled
  LFU, pure LRU and full-scan LFU within 0.1% of each other, and an oracle (Belady) saving only ~1 NVMe read per token.

### `QWFN_SWAP_MIN_EF` -- exclusive tiers: don't read back cold victims

- **What it did:** with exclusive tiers (`QWFN_RAM_EXCLUSIVE`), a VRAM victim used fewer than N times (the decayed
  per-expert count) was dropped instead of read back into RAM, saving device-to-host traffic.
- **Data** (tok/s, essay / code / temperature 0.7, `--ram 9`, 120 W, an older stack): read every victim back
  40.9 / 42.2 / 40.7; N=2 38.9 / 41.7 / 40.1; N=4 34.2 / 39.1 / 42.0; N=8 39.1 / 41.5 / 37.7. Read-back traffic fell
  (down per step 15.4 MB -> 8.8 / 5.0 / 1.5 on code), expert I/O rose (0.6 -> 1.5 / 5.3 / 2.5 ms per step on code).
- **Why it failed:** every dropped victim that comes back is an NVMe read (~0.5-1 ms exposed) instead of a ~0.12 ms
  read-back; the trade loses at every threshold.

### `QWFN_RAM_LENT_INCLUSIVE` -- keep the prefill-lent layers inclusive

- **What it did:** with exclusive tiers, the layers whose VRAM tier a streamed prefill borrows kept an inclusive RAM
  tier (sized as if they had no VRAM tier), so a prefill would not drop their VRAM experts to disk.
- **Data:** `--ram 11`: short prompts 40.8 / 45.9 / 38.8 tok/s against 47.7 / 47.9 / 46.2 fully exclusive (less
  coverage), after a 20K prompt 22.1 vs 20.8. Agent turns at ~55K context: 23-26 tok/s against 20-23, short decode
  -3%, still 11-20 experts per step not in VRAM after a turn.
- **Why it failed:** `QWFN_LEND_RESTORE` (kept) solves the same problem better: after the prefill it reads the lent
  layers' former VRAM residents back (hottest first, a ~3 s budget, now pipelined), without costing short prompts.

### `QWFN_PROMOTE_T1` -- the one-token promotion budget at MTP verify steps

- **What it did:** at a verify step of T positions the engine allows `promote_per_layer x T` promotions per layer
  (each position brings its own experts); this switch kept the one-token budget instead.
- **Data:** a unitrace profile had shown the verify step's largest extra over a one-token step was expert uploads
  (pinned host -> device: 155 per step at T=1, 270 at T=2, ~2.5x the copy time). With the one-token budget:
  layer graphs 27.4-27.6 ms per step either way (null), CPU experts slightly up (essay 3.6 / 3.2 ms vs 2.4 / 2.2).
- **Why it failed:** the extra uploads were not on the verify step's critical path; a smaller budget only moved
  experts to the CPU. What actually makes an extra position expensive (~10 ms each on the later stack): the routed
  MoE grows with the unique experts (+5.1 ms), dense Q8_0 +1.0, attention +0.6.

### `QWFN_QSA_CHAIN`, `QWFN_QSA_PROJ_EACH`, `QWFN_QSA_BATCH_EACH` -- the verify step's attention, old shapes

- **What they did:** restored earlier forms of the T-position (MTP verify) sparse attention: T chained one-position
  calls (`CHAIN`), the q/k/v projections per call (`PROJ_EACH`), norms / rope / gate per position (`BATCH_EACH`).
  The kept form computes the projections once for the T positions and then the whole step (cache writes, scoring,
  top-k, masks) in one graph; the chained calls remain only as the structural fallback for a model without an
  indexer.
- **Data:** the one-graph form cut the T=2 attention graph from 155 to 126 kernels (T=1: 117) and the verify step's
  layer graphs from ~28.0 to ~27.4 ms (~2%), with identical greedy answers and identical draft acceptance per request
  across four starts. Tok/s differences were inside the I/O spread.
- **Why dropped:** A/B hooks of an adopted change.

## ggml-sycl patches and backend switches

### Patch 34, `GGML_SYCL_Q8_PROD` -- q8_1 quantization fused into the producer

- **What it did:** a one-token producer (UNARY silu / sigmoid / softplus + MUL, or GLU swiglu) whose f32 output is
  read next, as src1, by a direct-path Q8_0 matvec (patch 20) also wrote that tensor's q8_1 reorder-SoA copy into
  patch 19's reuse cache, so the standalone `quantize_row_q8_1` launch disappeared (~96 of ~579 quantize launches
  per token; 3,840 replaced in a 64-token traced run).
- **Data:** bit-exact (replay NLL identical in 8 of 8 A-B-B-A arms, `test-backend-ops` 10/10 for its own cases
  including an in-place overwrite between the two matvecs, MUL_MAT q8_0 43/43). Decode 18.73 / 18.46 / 18.12 tok/s
  off vs 18.71 / 18.61 / 18.31 on (+0.5% +-1.5%); layer-graph GPU time 11.8-12.7 s either way. A review found a
  stale-cache bug (buffer re-allocation while the producer declined) that a revival must keep fixed.
- **Why it failed:** a removed kernel boundary is worth less than ~2 us here (expected ~4 us). It was step 1 of
  "fuse more of the layer chain"; the result also capped what a full megakernel could buy: at ~1-2 us per boundary,
  removing all ~3,400 boundaries per token is ~3-7 ms of a ~23-33 ms token (~10-20%), not the 1.4x first estimated.
- **Related measurements (no code kept):** a persistent kernel with a grid-wide barrier keeps full bandwidth
  (570-585 GB/s at >= 32 groups x 1024 or 128 x 256; barrier ~0.9 us vs ~1.3 us per queued kernel boundary); two
  in-order queues, an out-of-order queue with events, and Vulkan's separate compute-only queue family all failed to
  overlap a chain of small kernels with a big one (a fork + join costs ~6 us).

### `GGML_SYCL_IQ4_SOA_LUT=0` (patch 17) -- byte-permute lookup in the IQ4_NL_SOA matvec

- **What it did:** decoded IQ4_NL codes with `dpct::byte_level_permute` on the 16-entry table held in four registers
  (the stock kernel's method) instead of patch 17's 256-entry table in local memory (code byte -> both nibbles'
  values, built per work-group).
- **Data** (us per matmul, 640 x 2560, 10 experts): stock IQ4_NL 83.8 (110 GB/s); SOA + byte-permute 84.9 (109 GB/s);
  **SOA + local-memory table 30.4 (303 GB/s)**; SOA without any lookup (a ceiling, wrong results) 19.1 (483 GB/s).
- **Why it failed:** `byte_level_permute` is written with 64-bit variable shifts that this GPU emulates: ALU-bound.
- **Related, also measured:** the IQ4 kernel later runs at ~600 GB/s effective in decode (~15.5 us per call); its
  sub-groups per work-group (4 / 8 / 16 / 32 -> 15.7 / 15.5 / 16.1 / 16.9 us) change nothing, so the per-work-group
  table build is not a bottleneck. Lanes per row matter (rows of 20 blocks): `GGML_SYCL_IQ4_SOA_LPR` 16 / 8 / 4 ->
  1.65 / 1.36 / 1.30 ms per plain step, 3.58 / 2.96 / 3.01 at one draft (8 kept, in the launcher).

### `GGML_SYCL_MOE_GLU_OFF` (patch 08) -- turn off the fused q2_0 MoE GLU

- **What it did:** an A/B hook that disabled patch 08's fusion (one q8_1 quantization and one kernel for the q2_0
  experts' gate + up + SwiGLU at one token, instead of quantize, quantize, matvec, matvec, GLU).
- **Data on the fusion it switched off:** layer graphs 30.96 / 30.95 -> 30.04 / 30.04 ms per token, decode 28.2 ->
  28.9 tok/s; `test-backend-ops` MUL_MAT_VEC_FUSION 1265/1265, MUL_MAT_ID 913/913; replay NLL within run noise.
- **Why dropped:** A/B hook of an adopted change; the fusion itself stays behind `GGML_SYCL_MOE_Q2W`.

### `GGML_SYCL_Q8_SOA_LPR` (patch 31) -- lanes per row of the Q8_0_SOA matvec

- **What it did:** forced 16 / 8 / 4 lanes per row in patch 31's one-token Q8_0_SOA MoE matvec.
- **Data:** the default (the widest that leaves no lane idle: 8 for the 20-block expert-down rows) took the Q8_0
  expert-down from 2.17 ms per step (stock kernel, ~85 GB/s) to 0.33 + 0.47 ms (SOA + the staged stock remainder);
  no other value was ever better.
- **Why dropped:** a tuning knob nobody set.

### Patch 23, `GGML_CPU_Q2_0_AVX2` -- AVX2 `ggml_vec_dot_q2_0_q8_0` (superseded by upstream)

- **What it did:** x86 had only the scalar loop for Q2_0 x Q8_0 (the CPU-computed experts). 2-bit codes unpacked to
  bytes, offset to {-1,0,1,2}, `mul_sum_i8_pairs_float` against each Q8_0 half, FMA-accumulated (the model was
  Strata's AVX2 row kernel).
- **Data:** one expert's gate+up (1280 x 2560) on one core 394 -> 150 us; max error 2.7e-8 of sum|w*a| vs 5.2e-8 scalar.
- **Why dropped:** llama.cpp added its own AVX2 Q2_0 dot product (#30447, two blocks per step in two FMA chains) before
  the 2026-10-14 rebase; the patch no longer applies and is not needed.

## Notes for anyone measuring here again

- **Replay NLL is not a placement-independent number.** Expert placement differs between runs (it depends on
  timing: prefetches landing, promotions), and the NLL over a replayed text moves with it: 1.5568 vs 1.5964 for two
  runs of one configuration, 1.57-1.91 across placements over 2,500 tokens. A per-token comparison shows ~100 of
  2,500 tokens flipping by more than 1 nat in both directions (near-tie routing), not a systematic shift. Bit-exact
  claims need the same placement (the A-B comparisons above that report identical NLL had it), or a check of the
  device contents (`QWFN_TIER_CHECK`, `QWFN_SWAP_CHECK`).
- **Compare speeds inside one GPU window, interleaved (A-B-B-A).** Between windows the layer-graph time drifts by
  +-4%, larger than most single changes.
- **What is still at the hardware's limit** (so new kernels there will not pay): the dense Q8_0 matvecs and the
  IQ4_NL / Q2_0 expert kernels stream at ~560-600 GB/s of the B70's 608; host <-> GPU syncs cost 6-12 us; a layer's
  ~71 kernels are GPU-bound (recorded SYCL graphs would save <= ~1 ms/token); CPU idle states, CPU energy preference
  and pinning the GPU clock (it already runs 2.74-2.79 GHz in decode) change nothing; host-to-device copies run at
  ~13 GB/s and do not overlap device-to-host ones.

## Custom behaviour still without a switch

- ggml-sycl patch 02 (flash attention with quantized, non-contiguous K/V): a correctness fix, not a feature.
- New types and entry points that do nothing until a switch uses them: patches 09, 17, 30, 31 (SOA layouts), 33,
  35 (side queue, device copy). Patch 22's multi-token kernels run only under the switches of patches 05, 07, 08.
- Server features behind command-line flags (`--timeshare`, `--prefix-cache`, `--lookup-drafts`, `--mtp`).
- The fork's engine diff against upstream still has always-on changes from before the fence rule (instrumentation,
  some decode-graph shapes, QSA selection fixes already offered upstream); they are reviewed one by one during the
  upstream merge.
