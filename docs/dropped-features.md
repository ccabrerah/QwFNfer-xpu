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
- **Data:** 256 tokens instead of 24: promotions -18% (natural text, 35.7k -> 29.3k per 2,500 tokens), -21% (code,
  28.6k -> 22.7k per 1,500), -25% (MTP-1, 39.9k -> 30.0k); swaps -27..-35%; sweep 24/64/128/256/512 promotions
  35.7k / 34.2k / 31.7k / 29.3k / 28.7k. Decode in within-window runs looked +4..6%, but interleaved A-B-B-A pairs
  (twice) gave 103.2 vs 102.6 s per 2,500 tokens (-0.6%, noise). Replay NLL equal or better in every arm (it moves
  with placement, see the note at the end). An offline simulator on routing traces had predicted <1-2 ms/token.
- **Why it failed:** fewer promotions save PCIe traffic, but a promotion costs ~0.12 ms of upload and the saved
  ones were mostly off the critical path; nothing else changed.
- **Don't redo as:** a hysteresis or budget knob on promotions -- the same simulator showed the engine's sampled
  LFU, pure LRU and full-scan LFU within 0.1% of each other, and an oracle (Belady) saving only ~1 NVMe read per token.
