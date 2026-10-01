# The B70 configuration

Details of `scripts/b70/qwfn-b70.sh` that the README does not carry: the engine defaults, why the flags have their
values, and the benchmark tools. The configurations (preferred: overlay v4; validated: overlay v2), their switches
and measured effects are in the README; the overlays
are built by `scripts/b70/build-overlay.sh` ([`dense-overlay.md`](dense-overlay.md)); what was tried and rejected
is [`B70-registry.md`](B70-registry.md).

## Switches

Every switch in `scripts/b70/qwfn-b70.sh` is a row of the README's **Major improvements** table (what it changes and
its measured effect); the README's run configuration lists them by group.

Engine defaults in this fork (no switch): the late fold sized to the promotion budget (`QWFN_LATE_FULL=1`
restores), deferred speculative prefetch (`QWFN_PREFETCH_EARLY=1` restores).

## Flags

The flag set is in the README's run configuration. Why these values:

- `--ctx 131072` is the per-session cap; the engine runs one sequence at a time.
- `--vram 25` with overlay v4 (0.6 GB less dense weight than v3): 53% of the expert blocks in VRAM (51% at 24).
  Measured at the limit -- an image plus a 126K-token document at `--ctx 131072`, then the 89K needle -- with no
  GPU memory evicted to system memory and ~0.7 GB still free at the lowest point (1.6 GB at `--vram 24`), on a host
  where nothing else uses the card. On a desktop sharing the card, or with a bigger head, use 24.
  `--ram 8` plus the 3 GB prefix cache plus ~4 GB for the system is what a 31 GB machine can give.
- `--reserve 2048` is not memory held back: the engine only checks, while it sizes the expert tier, that tier plus
  reserve could be allocated, and shrinks the tier until it can. With `--vram` capping the tier first it changes
  nothing (2048, 1536 and 1024 give a byte-identical tier); the headroom that matters is what `--vram` leaves, above.
  (Without the flag the engine's default is 768 MB plus 8 MB per 1K tokens of context beyond 48K, ~1.4 GB at 131K.)
- `--prefix-cache 3`: host-memory checkpoints of conversations switched away from (~2 at 64K, one at 130K).

## MTP

The model's own multi-token-prediction head drafts one token per step and the trunk verifies the pair (exact:
every emitted token is the trunk's). `scripts/b70/qwfn-b70.sh` turns it on when `QWFN_B70_MTP` names the head:

- **The head**: Unsloth's `MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf` with its routed experts re-encoded to Q2_0
  (`tools/overlay/gguf_requant.cpp`, rule `mtpq2`: a per-block scale chosen for the least squared error, since
  ggml's reference never uses the +2 level), assembled as a split head [metadata, Q2_0 experts, the original] so the
  original's Q8_0 experts are shadowed: 0.81 GB, all on the device (`QWFN_MTP_EXPERTS_VRAM=1`).
- **Draft vocabulary** (`QWFN_B70_DRAFT_VOCAB` -> `QWFN_MTP_DRAFT_VOCAB`, int32 token ids): the head's LM head over
  those rows of `output.weight` only. 74,557 ids (English and code, every token with a non-ASCII Latin character,
  the ASCII tokens below id 60,000) halve the head's cost; Spanish keeps ~66% acceptance (78% with the full
  vocabulary, ~46% with English/code only).
- **One draft** per step: a step's layer graphs cost ~1.45x a one-token step's at T=2; a longer window changes T
  from step to step, and keeping a graph per T costs device memory the expert tier needs (`QWFN_GRAPH_STASH_MB`,
  `QWFN_MTP_MIN_P` for the confidence gate, are the knobs for trying it).
- **No layer-0 prefetch pass** (`QWFN_NO_SPEC_L0=1`): with drafts it costs ~1 ms per step and its speculative reads
  crowd the drive.
- **`--vram 24`**: the head takes the GB. At the limit (an image plus a 126K-token document, the 89K needle) ~0.45 GB
  stays free, nothing is evicted.

Measured on short prompts against the same model without the head at `--vram 25`: ~+13% decode (essay ~35, code ~40,
code at temperature 0.7 ~37 tok/s against ~31 / ~35 / ~34), single starts inside a ~10-15% start-to-start spread.
The prefix cache keeps working: a checkpoint holds the head's state too.

## Benchmarks

`tools/perf`: `decab.py` (warm decode with the `/stats` decode split), `ctxdec.py` (prefill and decode after
a long prompt from a fixture: `{"messages": [...]}`), `mtpprobe.py` (tok/s, draft acceptance and step cost),
`pcswitch.py` (two alternating conversations: the prefix cache at work), and the `mv_bench` / `moe_bench`
kernel benches (CMake targets: `mv_bench BACKEND_DIR`).
