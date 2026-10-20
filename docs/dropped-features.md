# Dropped features

Switches, code paths and ggml-sycl patches that were built, measured and then removed to keep the fork's custom
surface small. Every one of them up to 2026-10-15 is still in the tree at tag **`v0.4.0`** (2026-10-14) or at the
commit its entry names; revive from there. Later entries that never reached the tree (built and measured on a branch,
then deleted) say so and describe the code closely enough to write it again.

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

### `QWFN_HC_POST_OP` -- the hyper-connection combine as llama.cpp's DSV4_HC_POST (2026-10-15, never shipped)

- **What it did:** `hc_combine` emitted `ggml_dsv4_hc_post(block_out, residual, 2 sigmoid(inject), no comb)` -- the op
  llama.cpp's own qwen4exp graph uses (#28901, SYCL kernel #29132) -- instead of OUT_PROD + ADD.
- **Data** (served layout with the draft head, per decode layer graph): recurrent 587 -> 593-611 us,
  attention 885 -> 893-896 us, host launch 458 -> 469-474 us per graph; `test-backend-ops` DSV4_HC_POST 9/9.
- **Why dropped:** upstream's op covers the product and the add only; our patches 04/06/11 fuse the gate (sigmoid,
  scale), the product, the add and the next mixer's RMS norm + gamma into one kernel, so emitting the op loses that
  fusion (sigmoid, scale, HC_POST, RMS_NORM+MUL = four launches against one). Patches 03/04/06/11 stay.

### `QWFN_HC_MEAN_MULTI` -- the stream mean as one matmul at verify steps (2026-10-15)

- **What it did:** extended upstream's one-token stream mean (sigmoid-gated streams times a [hc] vector of 1/hc, one
  matmul) to the MTP verify step's 2-4 positions (on in the launcher since 2026-10-11).
- **Why dropped:** `QWFN_HC_PRE_OP` (kept, on in the launcher) emits llama.cpp's DSV4_HC_PRE, which does the gate
  and the mean in one kernel at every step width, so the matmul form is no longer built when it is on.

### `QWFN_SETTLE_LATE` -- a layer's promotion settle moved past the next layer graph (2026-10-20, never merged)

- **What it did:** with `QWFN_PROMOTE_D2D`, decode settles each layer's promotions right after the layer's MoE
  (`expert_cache::settle_promotions_layer`): it waits on the compute queue for this layer's promotion uploads (pinned
  RAM -> VRAM, ~140 us each) and the victims' staging copies, then releases the RAM slots the uploads read, maps the
  victims into their RAM slots, queues their read-backs on the side queue and lands the previous layer's. The switch
  skipped that call and ran the same bookkeeping without the wait (`settle_promotions_landed()`) one layer later, in
  the next layer's decode pass after its graph had been waited for and its routing read back, before its fetch: every
  copy still pending had been queued on the same in-order queue before that graph, so it had landed. The last
  layer's promotions settled after layer 0 of the next step, or in any full settle (prompt passes, lends, vision,
  shutdown). Without D2D it was the plain `settle_promotions()`.
- **Data** (1,500 replayed tokens as served -- draft head, promotions on --, 3 interleaved runs per arm in one GPU
  window; the sum of the engine's timed decode parts, 36.40-36.47 s in the base runs): promotion settle 2.92 -> 0.18 s,
  layer graphs 27.24 -> 29.85-29.91 s, net -0.2% (36.42 -> 36.34 s). Next to `GGML_SYCL_FUSE_MULTI_MMV` (kept):
  -1.1% against -0.8% for that alone. Exact: the decode logits hash over 300 replayed tokens with promotions on
  equalled the base's (15,526 promotions in each run; two base runs gave the same hash), and `QWFN_SWAP_CHECK`
  compared 20,480 read-backs with the file, 0 wrong.
- **The first placement was slower, and why:** settled right after the next graph's wait, before its routing read,
  the victims' read-backs (an expert each, ~1.9 MB) went onto the side queue just ahead of the layer's blocking read of the
  routing pack, which then waited behind them: host-to-device and device-to-host copies share one copy engine here.
  Readback 1.01 -> 4.17 s per 1,500 tokens. Anything queued on the side queue must stay out of the way of the
  per-layer routing read.
- **Why it failed:** the experts promoted during layer L's fetch are computed in layer L+1's graph (the "late fold"),
  so their uploads are on the critical path whoever waits for them: of the 2.74 s of host settle wait removed, 2.65 s
  reappeared as layer-graph time. Only the host's own round trip (one sync and the bookkeeping per layer with
  promotions) left the critical path.
- **Don't redo as:** an event per layer, a settle at the layer's next fetch, or the uploads on a second queue
  (`QWFN_PROMOTE_SIDE` above): each moves the same wait. What could pay is fewer bytes or less work on that path (the
  per-part SOA reorder kernel after each upload, or RAM slots kept in the device layout so a promotion is a plain
  copy), not a later wait.

### `QWFN_PACK_ASYNC` -- the decode routing read queued behind the layer graph (2026-10-20, never merged)

- **What it did:** after a replayed decode layer graph the host waits for the graph (`ggml_backend_synchronize`) and
  then reads the readback pack -- the layer output, the routing weights and expert ids, the predictor's next-layer ids
  and scores: T x (n_embd + 2 x n_expert_used + 2 x 16) floats, ~10 KB per position -- with a blocking
  `ggml_backend_tensor_get`, a second round trip. The switch allocated a pinned copy of the pack next to the CPU
  partial's pinned staging and queued the pack's device-to-host copy on the compute queue right behind the graph
  (`ggml_backend_tensor_get_async`), so the one wait covered both; the unpacking then read the pinned copy. Graphs
  built in the same step (first build, a new T) kept the old path.
- **What the idea first asked for, and why not:** skipping the routing read when every expert of the layer was in
  VRAM. The host learns that only from the routing it reads, so a skip cannot be exact: the CPU partial and the
  fetch need the ids. Only the second round trip could go.
- **Data** (same window and method as above): readback 1.01 -> 0.06 s per 1,500 tokens, layer graphs +0.7-1.5 s
  alone (28.06-28.79 vs 27.31-27.37 s in the first window); on top of the other two packages -1.2% against -1.1%
  without it, inside the runs' spread. Exact (logits hash identical, draft head and plain).
- **Why it failed:** the blocking read cost the copy, not the extra wait. A ~20 KB device-to-host transfer at a
  two-position step takes ~25 us (1.01 s over ~40,000 reads), most of it fixed latency, queued behind the graph or
  not; the round trip it saved did not show above the runs' spread.
- **Not tried:** the pack written by the graph's last kernel straight into host-visible memory (no copy at all); it
  would move the transfer into that kernel's writes over PCIe, and it changes a ggml backend contract (graph outputs
  in device memory).

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

### Patches 19-21 -- our one-token Q8_0 matvec path (superseded by upstream, 2026-10-15; 18 and 28 stay)

Switches `GGML_SYCL_Q8_REUSE` (19), `GGML_SYCL_Q8_DIRECT` (20), `GGML_SYCL_Q8_EPILOGUE` (21), in the launcher until
2026-10-15. Last in the tree at commit 06538ee (and `v0.4.0`). Patch 18 (the wide kernel, `GGML_SYCL_Q8W`) and 28 (its
2-4-column form for the MTP verify step, `GGML_SYCL_Q8W_NCOLS`) were dropped with them in 5f01bf0 and restored the same
day (below).

- **What they did:** the dense Q8_0 weights (attention, shared experts, the hyper-connection mixers, the PLE key,
  `ssm_out`, the embedding head) at one token. 19: the last one-token src1 quantized to q8_1 kept for the next matmul
  on the same input. 20: a one-token Q8_0 MUL_MAT straight to the reordered MMVQ kernel (patch 18's wide kernel),
  skipping `ggml_sycl_op_mul_mat`'s generic routine. 21: MUL_MAT -> [SCALE ->] SILU as one launch (overlay v4's mixers).
- **Data when adopted:** 18: dense8 matvec 24.76 -> 21.48 us/call (~405 -> ~467 GB/s), layer graphs -1.0 ms/token.
  19 + 20: -0.32 ms/token, exact. 21: -0.62 ms/token with overlay v4. 28: verify-step dense Q8_0 11.8 -> 9.3 ms/step,
  short-prompt decode with the draft head +9%. A Q8_0 matvec on f32 activations of our own (`GGML_SYCL_Q8F`,
  2026-09-28) was not adopted then: +42 us/graph of GPU wait.
- **Why 19-21 were dropped:** with `GGML_SYCL_Q8_DIRECT` off, llama.cpp's dispatcher (#29186, in the 2026-10-14
  rebase) sends a one-token Q8_0 MUL_MAT to an ESIMD DMMV on f32 activations. Against a double-precision reference on
  every dense Q8_0 matrix of the deployed layout (`q8acc`, research repo `scripts/box/upq8/`): one column, upstream
  ~1e-7 relative RMS error, ours (q8_1 activations) ~1.2e-2 (the CPU backend quantizes the same way: 1.2e-2). Plain
  decode (no draft head) layer graphs equal (recurrent 420-422 us, attention 645-654 us per graph). Replay NLL over
  16 texts (2,000-token prompt + 1,024 replayed, plain decode): +0.12% (SE 0.18), a summation-order control +0.03%
  (SE 0.14); per-text spread 0.70% vs 0.56%: noise. One text is very sensitive to the arithmetic (1,000-2,500 tokens
  replayed after a 5-token prompt: 1.65 with ours, 1.87 with ours in another summation order, 2.05-2.15 on upstream's
  exact kernel); no other text moved like it.
- **Why 18 and 28 came back:** with the draft head almost every step is a 2-column verify step, which upstream runs
  through a wide-load MMVQ on q8_1 activations (same accuracy as ours: ~1.1e-2 at 2 columns): layer graphs 623-627 /
  918-920 us against 587-588 / 886-887 us on patch 28. The first measurement of upstream's path ("decode
  +5% with the draft head", 26.1-26.3 vs 24.6-25.0 tok/s) was expert-I/O noise: io 22.6-23.2 vs 26.5-31.5 s per 2,500
  tokens in those runs, while its layer graphs were already slower. With `GGML_SYCL_Q8W=1 GGML_SYCL_Q8W_NCOLS=1` and
  19-21 gone, one column gives upstream's bits (16/16 matrices) and two columns patch 28's (16/16).

### Patch 14's stream-mean fusion (`GGML_SYCL_FUSE_HC_MIX`, half of the patch; superseded by upstream, 2026-10-15)

- **What it did:** matched the engine's one-token hyper-connection mixer tail -- UNARY(SIGMOID) -> MUL -> RESHAPE ->
  PERMUTE -> CONT -> RESHAPE -> MUL_MAT with the [hc] vector of 1/hc -- and ran it as one kernel
  (`ggml_sycl_hc_mix_mean`, the sum in smallk's order). The patch's other half, the SiLU epilogue of the bf16/f16
  one-token matvec, stays (patch 22 builds on it; upstream has no equivalent).
- **Data when adopted:** part of patches 12-15's "~300 fewer kernels per token", layer graphs -8% together.
- **Why dropped:** llama.cpp added `GGML_OP_DSV4_HC_PRE` (#28901, SYCL #29132): the gated mean in one kernel, at any
  step width. The engine emits it under `QWFN_HC_PRE_OP` (kept, on in the launcher). Measured (served layout with
  the draft head): decode layer graphs identical to the fusion's (recurrent 587-588 vs 587 us, attention 886-887 vs
  885 us); prefill dense graphs on a 38K prompt 27.0 / 26.3 s against 31.5 / 28.0 s, since the op also replaces the
  prefill's sigmoid, multiply, copy, three adds and scale; `test-backend-ops` DSV4_HC_PRE 18/18.

### `QWFN_LIGHTNING_IDX=2` -- llama.cpp's lightning indexer for prompt chunks (2026-10-15; the switch's value 1 stays)

- **What it does:** the indexer's block scores as `GGML_OP_LIGHTNING_INDEXER` (llama.cpp #29825's form) instead of a
  [blocks, 4 heads, T] matmul, a ReLU and patch 16's head sum; it never materializes the per-head scores.
- **Data:** decode / verify steps (value 1, kept): attention layer graphs 885 -> 864 us, host launch 458 -> 440 us per
  graph. Prompt chunks (value 2): dense graphs on a 38K prompt +4-5 s (33.4 / 32.7 vs 27-31 s), on 100K +25 s (111.3
  vs 85.9 s): the op's SYCL kernel scores one (query, block) pair per sub-group on the vector units, where the
  chunk's matmul uses the matrix engines. Its memory saving at the default `--ubatch-kv` (48M cells) is ~0.4 GB of the
  prefill's lent VRAM, not enough to pay for that.
- **Status:** value 2 is kept in the engine for measurement only; prompts keep patch 16.

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
