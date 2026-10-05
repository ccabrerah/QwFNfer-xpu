# Parameters: command-line flags and environment variables

Everything the engine, its tools and the patched ggml-sycl backend read, in one place. The B70 column says what
`scripts/b70/qwfn-b70.sh` (the production configuration, [B70-config.md](B70-config.md)) sets; blank means the
default. Rules that hold throughout:

- **Flags**: a repeated flag takes its last value, so a supervisor can append overrides to the launcher's.
- **Environment variables are read once, at first use** (most at startup): set them before the process starts.
- **Switches** (`=1`) are on when set to anything unless the table gives values; the ggml-sycl ones are off unless set
  and change nothing when off. "Diagnostic" and "A/B" entries exist for measurement, not for production.

Keeping this file current: every flag is parsed in `tools/qwfn_server.cpp` / `tools/qwfn_gen.cpp` (`a == "--..."`),
every variable is a `getenv("...")` in `src/`, `tools/` or `patches/ggml-sycl/`; a new one gets a row here in the same
commit.

## 1. `qwfn-server` flags

`qwfn-server <shard.gguf> [options]` -- OpenAI and Anthropic APIs on one engine.

| Flag | Default | B70 | What it does |
|---|---|---|---|
| `--host HOST` | 127.0.0.1 | | Bind address. |
| `--port N` | 8080 | | Port. |
| `--alias NAME` | qwen3.8-flash-next | set | Model id reported by `/v1/models`. |
| `--mmproj PATH` | off | set by the deployment | Vision projector GGUF: enables image input. Runs on the CPU (no VRAM): a 1400x1000 screenshot ~15 s, 320x240 ~0.3 s. |
| `--vision-threads N` | `--threads` | | Threads for the image encode. |
| `--ctx N` | 32768 | 131072 | Context length. |
| `--kv f16\|q8_0` | q8_0 | q8_0 | KV cache type. |
| `--batch N` | 4096 | 16384 | Tokens per prefill pass (one expert sweep per pass). Also the time-sharing yield granularity of a prefill. |
| `--prefill-chunk N` | 2048 | 6144 | Tokens per attention chunk inside a pass (bounds the graph arena). |
| `--ubatch-kv M` | 48 | | Cap on n_kv x chunk tokens, in millions (bounds the per-call input arena at long contexts). |
| `--prefill-decode-max N` | 512 | | Prompts up to N new tokens take the cache-batched path (experts through the tiers) instead of the full sweep, while tokens x context stays under 12M. |
| `--no-prefill-overlap` | overlap on | | One prefill staging buffer instead of two: saves ~1.8 GB of RAM, loses the read/compute overlap. |
| `--ram GB` | 8 | 8 (deployments raise it) | Expert RAM tier size. |
| `--ram-frac F` | 0.75 | | MemAvailable share the RAM tier may take (a backstop on `--ram`). |
| `--vram GB` | 12 | 25, 24 with `--mtp` | Expert VRAM tier size (self-tunes down to what the device can spare; 0 disables). |
| `--reserve MB` | auto (sized from `--batch`) | 2048 | VRAM left free after the tier is sized, for the prefill arenas and graph allocators. |
| `--threads N` | 8 | | CPU threads (CPU-computed experts, vision). |
| `--cpu` | GPU | | Run on the CPU only. |
| `--state-host none\|idx\|kv,idx` | none | | Attention state in pinned host memory; its VRAM goes to the expert tier. Costs ~0.35 ms/token (idx) or ~2 ms/token (kv,idx) of PCIe reads. |
| `--no-qsa` | QSA on | | Dense attention instead of the sparse (QSA) path. |
| `--indexer-top-k N` | the GGUF's (2048) | | Cells each query attends to. 0 keeps the checkpoint's value; past ~150K context 4096 is the upstream recommendation. |
| `--think LEVEL` | xhigh | | Default reasoning effort: xhigh, medium, low, off. |
| `--think-budget N` | 0 (unlimited) | | Most reasoning tokens per answer; also `POST /props {"reasoning_budget": N}` or per request. |
| `--prefix-cache GB` | 0 (off) | 3 (deployments 5) | Host memory for checkpoints of conversations switched away from, so alternating clients do not re-prefill. ~16 KB/token at q8_0 + ~118 MB per entry. Time sharing parks requests here too. |
| `--prefix-cache-min N` | 1024 | | Shorter sequences are not saved. |
| `--prefix-cache-boundary N` | 4096 | | Also checkpoint a chat prompt at the end of its history (before the assistant opener) when at least N tokens; 0 = off. |
| `--timeshare N` | 0 (off) | set by the deployment | Time-share the engine between up to N open requests; request N+1 gets 503. One runs per quantum, then parks (at a decode step or between prefill passes) for an open request whose session used the engine less lately. Needs `--prefix-cache`. |
| `--quantum S` | 30 | | Seconds a request holds the engine before it may yield. |
| `--linger S` | 1 | | After a request completes, seconds its session's next request goes first (and continues that session's quantum). |
| `--mtp PATH` | off | from `QWFN_B70_MTP` | The draft head (nextn GGUF): each step carries its draft, verified by the trunk (exact). |
| `--mtp-drafts N` | 1 | 1 | Most head drafts per step (1-3). |
| `--lookup-drafts N` | 0 (off) | | With `--mtp`: prompt-lookup drafts (0-3) when the reply copies its context; each draft beyond `--mtp-drafts` adds a ~76 MB rollback snapshot. |
| `--spec-ahead N` | 1 | | Layers ahead the expert predictor looks (1 or 2). |
| `--spec-depth N` | 10 | | Predicted experts prefetched per layer (0 disables prefetch). |
| `--spec-depth2 N` | 4 | | Prefetch depth for the L+2 prediction (with `--spec-ahead 2`). |
| `--spec-margin F` | 0 (off) | | Skip a predicted read whose router-logit margin from the routing cut-off is below F. |
| `--spec-gate-inflight N` | 0 (always) | | Apply `--spec-margin` only while at least N speculative reads are in flight. |
| `--spec-block` / `--spec-block-layers L` | off | | Predict layer L+1's routing by running its token mixer inside layer L's graph (all layers, or a list like `1,2,5-9`). Experiment. |
| `--predictor PATH` | the next router | | Learned routing predictor file (`scripts/train_predictor.py`). |
| `--gate-drop F` | 0 (off) | | Decode only: drop routed experts whose normalised gate is below F and renormalise. An approximation (price it with replay NLL). |
| `--skip-miss` | off | | Decode without waiting for expert misses (the token uses the resident experts, gates renormalised). An approximation; disables drafts. |

Per request (not flags): `reasoning_effort`, `reasoning_budget`, `max_tokens`, `temperature` and the sampler fields,
`stop`, `tools` / `tool_choice`, `stream`, `timings_per_token` (live tok/s per chunk). `X-Session-Id` (header), `user`
or Anthropic `metadata.user_id` name the session for time sharing; without them the conversation's opening does.
`GET/POST /props` changes the defaults at runtime (reasoning effort, max tokens, sampling presets, `dump_requests`).

## 2. `scripts/b70/qwfn-b70.sh` (the B70 launcher)

Execs `qwfn-server` with the B70 flags above and exports the switches marked in sections 4-6. Its inputs:

| Variable | Default | What it is |
|---|---|---|
| `QWFN_B70_GSQ` | required | Directory with the GSQ-RCO Q2_0 shards (the stock model; its first shard gives the tokenizer). |
| `QWFN_B70_HEAD` | the stock first shard | An overlay head (`scripts/b70/build-overlay.sh`) to load instead. |
| `QWFN_B70_LLAMA` | `$HOME/src/llama-b70` | The patched ggml tree (`scripts/b70/build-llama-sycl.sh`); its `build-sycl/bin` is the backend dir. |
| `QWFN_B70_MTP` | unset | The MTP draft head. When set: `--vram 24 --mtp ... --mtp-drafts 1`, `QWFN_MTP_EXPERTS_VRAM=1`, `QWFN_NO_SPEC_L0=1`. |
| `QWFN_B70_DRAFT_VOCAB` | unset | With `QWFN_B70_MTP`: the token ids the head may draft (`QWFN_MTP_DRAFT_VOCAB`). |
| `ONEAPI` | `/opt/intel/oneapi/setvars.sh` | oneAPI environment script. |

Extra arguments are appended to the server's command line (later values win).

## 3. `qwfn-gen` flags (benchmark and test tool)

`qwfn-gen <shard.gguf> [options]`: one prompt, then a decode or a replay; prints rates, tier statistics and, with
`--ppl`, the replay NLL. Shares the engine flags of section 1 (`--ctx --kv --batch --prefill-chunk --ubatch-kv
--prefill-decode-max --no-prefill-overlap --ram --ram-frac --vram --reserve --threads --cpu --state-host --no-qsa
--indexer-top-k --spec-* --predictor --gate-drop --skip-miss --mtp --mtp-drafts`); `--vram-reserve MB` is its name for
`--reserve`. Its own:

| Flag | What it does |
|---|---|
| `--prompt TEXT` / `--prompt-file F` | The prompt, as text or as a file of token ids. |
| `--gen N` | Tokens to decode. |
| `--replay-file F` | Feed these token ids instead of sampling (teacher forcing); with `--ppl`, report their NLL. |
| `--ppl` | Replay NLL per token (`QWFN_PPL_VERBOSE=1` prints each). |
| `--save-replay F` | Save the generated ids (for a later `--replay-file`, or to compare runs). |
| `--io-threads N` / `--io-uring` | Expert reads on N worker threads (default 16) / through io_uring instead. |
| `--no-reuse` | Rebuild the layer graphs every token instead of replaying them. |
| `--no-speculate` | No expert prefetch. |
| `--promote N` | VRAM promotions per layer per token (default 2). |
| `--evict lru\|lfu\|hybrid` | Expert-cache eviction policy (default lru). |
| `--prefill-cpu` / `--prefill-gpu` | Prefill MoE on the CPU / GPU (default GPU). |
| `--cold F` | A second, colder expert file as an extra tier. |
| `--pair-test`, `--multi-test K` | Decode the replay 2 / K tokens per step (verify-step correctness). |
| `--rollback-test` | Feed a wrong draft (`QWFN_RB_AT`, `QWFN_RB_WRONG`) and check the rollback. |
| `--ckpt-test F` | Checkpoint, run another prompt (F), restore, and check the restored sequence matches. |

## 4. Engine environment variables (`QWFN_*`)

### Production configuration

| Variable | Values | B70 | What it does |
|---|---|---|---|
| `QWFN_GGML_BACKENDS` | dir | set | Where the ggml backend modules are (overrides the built-in path). |
| `QWFN_REQUIRE_GPU` | 1 | 1 | Fail instead of silently running on the CPU when no GPU backend loads. |
| `QWFN_VOCAB_MODEL` | GGUF path | stock shard | Take the tokenizer from another GGUF of the same model (overlay heads carry none). |
| `QWFN_LOCK_HOST` | 1 / 0 | 1 | Lock host buffers (RAM tier, prefill staging) and register them with the driver. Needs the memlock limit raised. |
| `QWFN_DEV_MASK` | 1 | 1 | Build the prefill attention mask on the device. |
| `QWFN_QSA_PACK` | 1 | 1 | Pack the QSA block tables into one upload per chunk. |
| `QWFN_Q2_SOA` | 1 | 1 | VRAM tier's q2_0 experts in the SOA layout (needs ggml-sycl patch 09). |
| `QWFN_IQ4_SOA` | 1 | 1 | VRAM tier's iq4_nl experts in the SOA layout (patch 17). |
| `QWFN_PREDICT_CUR2` | 1 | 1 | Predict layer L+1's routing from this layer's FFN-mix input through L+1's router (cheaper, as accurate). |
| `QWFN_RAM_EXCLUSIVE` | 1 | set by the deployment | Exclusive expert tiers: RAM holds what VRAM does not; a promotion reads the VRAM victim back into RAM. |
| `QWFN_RAM_LENT_INCLUSIVE` | 1 | | With exclusive tiers, keep the layers lent to the prefill inclusive. |
| `QWFN_LEND_RESTORE` | 1 | 1 | After a streamed prompt (or an image's projector staging), read back the VRAM experts of the layers the prefill borrowed, hottest first, before the next decode. Without it, exclusive tiers leave those experts on disk and the reply refills them through misses. |
| `QWFN_LEND_RESTORE_MS` | ms | 3000 | The restore's time budget. |
| `QWFN_LEND_RESTORE_BATCH` | 1-64 | 32 | Experts per batched read in the restore. |
| `QWFN_PF_VRAM` | 1 | 1 | A streamed prefill takes the experts the VRAM tier holds from the tier (a device copy; ggml-sycl patch 30 converts the SOA layouts back) instead of reading them from disk and uploading them. Exact; device staging only. |
| `QWFN_SWAP_MIN_EF` | N | | Exclusive tiers: do not read back victims used fewer than N times (measured slower; off). |
| `QWFN_MTP_EXPERTS_VRAM` | 1 | with MTP | The draft head's experts on the device (else in host memory, computed on the CPU). |
| `QWFN_MTP_DRAFT_VOCAB` | file | with MTP | Token ids the head may draft (a smaller LM head for the draft). |
| `QWFN_NO_SPEC_L0` | 1 | with MTP | No layer-0 prefetch pass ahead of each step. |
| `QWFN_MTP_SHARED_CELLS` | 1 | with MTP | The draft head attends over the cells the trunk's last attention layer selected for the same position (QSA), with that layer's mask, instead of densely over its whole cache. Drafts only; positions the last decode step did not cover stay dense. |

### Drafting and prompt lookup (server)

| Variable | Default | What it does |
|---|---|---|
| `QWFN_DRAFT_COST` | 0.7 | Cost of one extra verify position relative to a one-token step, in the draft-length model. |
| `QWFN_MTP_MIN_P` | 0 (off) | Confidence gate: draft only while the head's own probability is at least this. |
| `QWFN_MTP_ARGMAX_DRAFT` | off | At temperature > 0, draft the head's argmax instead of sampling it (A/B). |
| `QWFN_LOOKUP_ENTER` | 12 | Prompt lookup: match length that switches a step into the lookup window. |
| `QWFN_LOOKUP_PATIENCE` | 2 | Steps the lookup window may not pay before switching back. |
| `QWFN_LOOKUP_MARGIN` | 0.15 | How much faster (tokens/ms) the lookup window must be to switch in. |
| `QWFN_LOOKUP_FORCE` | off | Always take lookup steps (measurement). |
| `QWFN_LOOKUP_DEBUG` | off | Log the lookup policy after each request. |

### Time sharing (server, `--timeshare`)

| Variable | What it does |
|---|---|
| `QWFN_TS_ROUNDTRIP=S` | Test switch: every S seconds, at a yield point, checkpoint and restore the request's own sequence (the path a park takes), with nothing else running. On a fresh server the reply must not change: an exactness test of park/resume. |

### Expert cache, IO and memory (A/B and tuning)

| Variable | What it does |
|---|---|
| `QWFN_SYNC_PROMOTE` | Expert promotions to VRAM complete synchronously (default: asynchronous). |
| `QWFN_VRAM_LRU` | VRAM tier replacement by pure recency (default: frequency with staleness). |
| `QWFN_PROMOTE_T1` | Verify steps keep the one-token promotion budget. |
| `QWFN_LATE_FULL` | Full-width late fold of experts promoted mid-step (default: `promote_per_layer` rows per position). |
| `QWFN_PREFETCH_EARLY` | Submit the next layer's speculative reads before this layer's demand reads land (default: after). |
| `QWFN_PREDICT_SHARED` / `QWFN_PREDICT_PLAIN` | Predict routing from the residual with the shared expert / the bare residual (A/B). |
| `QWFN_READ_SPLIT=k` | Issue every expert read as k pieces (1-8). |
| `QWFN_DIO_512` | 512-byte direct-IO alignment instead of the page. |
| `QWFN_SEPARATE_STAGING` | Prefill staging in its own VRAM instead of lent from the expert tier. |
| `QWFN_SWEEP_FILE_ONLY` | The prefill sweep reads every expert from the file, not from the RAM tier. |
| `QWFN_PAGEABLE_ARENA` / `QWFN_PREFILL_PAGEABLE` | RAM tier / prefill staging in pageable instead of pinned memory. |
| `QWFN_CBATCH_KV=N` | Override the cache-batched prompt path's tokens x context cap (0 disables the rule). |
| `QWFN_NO_CACHE_BATCH` | Short prompts as repeated decodes instead of one cache-batched step. |
| `QWFN_GRAPH_STASH_MB=N` / `QWFN_NO_GRAPH_STASH` | Budget for stashed decode graphs of other step widths (default 256 MB) / none. |

### Decode and prefill graph (A/B against older paths)

| Variable | What it restores |
|---|---|
| `QWFN_LEGACY_MOE` | The per-expert MoE path (instead of `mul_mat_id`). |
| `QWFN_MOE_SEPARATE` | VRAM experts as a separate graph instead of inside each layer's graph. |
| `QWFN_LEGACY_PREFILL` | The per-ubatch expert sweep instead of layer-major passes. |
| `QWFN_LEGACY_PREFILL_MOE` | The grouped prefill MoE instead of `mul_mat_id` on the device. |
| `QWFN_LEGACY_QSA_DECODE` | The older one-token sparse attention graph. |
| `QWFN_LEGACY_UPLOAD` | Prefill expert uploads on the default stream. |
| `QWFN_QSA_CHAIN` / `QWFN_QSA_PROJ_EACH` / `QWFN_QSA_BATCH_EACH` | A verify step's attention as chained one-position calls / projections per call / norms and rope per position. |
| `QWFN_HC_MEAN_T1_ONLY` / `QWFN_PACK_T1_ONLY` | Stream mean / output packing at one-token steps only. |
| `QWFN_NO_FUSE` | No graph-level GPU fusions. |
| `QWFN_NO_INPLACE` | No in-place writes of layer outputs. |
| `QWFN_NO_PACK` | No packed readback of routing results. |
| `QWFN_NO_UPLOAD_SKIP` | Upload zero partials even when the device already holds zeros. |
| `QWFN_NO_INJECT_FOLD` / `QWFN_NO_DOWN_FOLD` | Do not fold the hyper-connection inject / down scales into the weights at load. |
| `QWFN_FORCE_BATCHED` | Run one-token steps through the batched path (unit test of its permutation). |

### Diagnostics

| Variable | What it does |
|---|---|
| `QWFN_CHECK_MOE` | Run the `mul_mat_id` MoE and the per-expert path on the same inputs and report differing values. |
| `QWFN_CHECK_STRIDE` | Verify the expert-slice stride assumption of the prefill reader. |
| `QWFN_VERIFY_UPLOAD` | Byte-compare every uploaded prefill expert slice with the file. |
| `QWFN_SWAP_CHECK` | Compare every exclusive-tier read-back with the file. |
| `QWFN_NAN_CHECK` | Report where a non-finite value first appears in decode. |
| `QWFN_COLD_DEBUG` | Cold-tier offsets and which expert makes a partial non-finite. |
| `QWFN_REVERSE_GROUPS` | Reverse the prefill expert-group order (isolates an order-dependent bug). |
| `QWFN_TRACE_PREFILL` | Per-layer statistics of the prefill activations. |
| `QWFN_ROUTE_DUMP=dir` (`QWFN_ROUTE_STRIDE=k`) / `QWFN_ROUTE_DUMP_DECODE=dir` | Dump prefill routing (every k-th token) / decode routing. |
| `QWFN_GRAPH_STATS=1\|2` | Op histogram of the first layer graphs (2: the op sequence). |
| `QWFN_PF_PROFILE=layer` | Per-node timing of one layer's first prefill chunk graph. |
| `QWFN_PROFILE_MIN_US=N` | Smallest node delta the profilers print (default 8 us). |
| `QWFN_VRAM_AUDIT` | Every device buffer the engine holds after init. |
| `QWFN_HASH_PREFILL=1\|2\|3` (`QWFN_HASH_LAYER=L`) | Run-to-run determinism: 1 hashes each prefill layer's intermediates and the logits; 2 hashes every node of layer 0's first chunk and reruns it in-process, then exits; 3 hashes every node of every chunk of layer L, then exits (compare two processes). |

### Test tools only

| Variable | Tool | What it does |
|---|---|---|
| `QWFN_PPL_VERBOSE` | qwfn-gen | Print each replayed token's NLL. |
| `QWFN_SEGMENTS=N` | qwfn-gen | Rates and tier shares per N decoded tokens. |
| `QWFN_PROFILE_LAYERS=list\|all` | qwfn-gen | Per-node timing of those layers' cached decode graphs. |
| `QWFN_IO_PROFILE` | qwfn-gen | Where decode waits for reads, per layer. |
| `QWFN_MTP_NOVERIFY` | qwfn-gen | Load the head but decode without it. |
| `QWFN_MTP_DRAFT2_TEST` | qwfn-gen | Score a second draft after each draft. |
| `QWFN_RB_AT=k`, `QWFN_RB_WRONG=d` | qwfn-gen | `--rollback-test`: which draft is wrong, and by how much its id is shifted. |
| `QWFN_CKPT_BREAK=ov\|state` | qwfn-gen | `--ckpt-test` negative controls (the check must fail). |
| `QWFN_DUMP`, `QWFN_PROBE_LAYER` | qwfn-logits | Dump intermediate tensors (llama.cpp names) / which layer. |
| `QWFN_LOOKUP_*`, `QWFN_SIM_HEAD_P` | qwfn-lookup-test | Policy parameters; simulated head acceptance (default 0.80). |
| `QWFN_VISION_FA=0\|1`, `QWFN_VISION_WTYPE=f16\|bf16\|q8_0` | vision | Projector attention and weight type (experiments). |

## 5. Patched ggml-sycl switches (`GGML_SYCL_*`, `patches/ggml-sycl/`)

Off unless set; details per patch in [B70-SYCL.md](B70-SYCL.md).

| Variable | Patch | B70 | What it does |
|---|---|---|---|
| `GGML_SYCL_OUTPROD_NATIVE_MAX_K=K` | 03 | | Small OUT_PROD as a native kernel up to K (0 keeps GEMM for every shape). |
| `GGML_SYCL_FUSE_HC=1` | 04, 11 | 1 | Hyper-connection combine + norm fused (prefill). 2 also logs declines; 3 forces two passes (test). |
| `GGML_SYCL_FUSE_HC_DECODE=1` | 06 | 1 | The same fusion at decode sizes. |
| `GGML_SYCL_FUSE_HC_GATE=1` | 11 | 1 | The combine gate evaluated inside the fused kernel. |
| `GGML_SYCL_MMVW=1` | 05, 22 | 1 | Wide-load bf16/f16 matvec for 1-4 tokens. |
| `GGML_SYCL_SMALLK=1` | 05, 22 | 1 | Small-K f32 matmul without oneMKL's per-call cost. |
| `GGML_SYCL_MOE_Q2W=1` | 07, 08, 22 | 1 | q2_0 MoE matvec, a 64-weight block per lane; with it the fused gate+up+SwiGLU. |
| `GGML_SYCL_MOE_GLU_OFF=1` | 08 | | Disable the fused MoE GLU (A/B). |
| `GGML_SYCL_FUSE_SPARSE_DECODE=1` | 10 | 1 | One-token sparse attention straight from the q8_0 cache. |
| `GGML_SYCL_TOPK_DIV_OUT=0` | 12 | | The router fusion's older match (default on). |
| `GGML_SYCL_TOPK_WG=1` | 13 | 1 | Work-group-wide top-k for the router and the predictor. |
| `GGML_SYCL_FUSE_HC_MIX=1` | 14 | 1 | Hyper-connection mixer fusions. |
| `GGML_SYCL_FUSE_ADDCHAIN=1` | 15 | 1 | Chains of 2-4 ADDs as one kernel. |
| `GGML_SYCL_FUSE_MOESUM=1` | 15 | 1 | MoE weighted sum reading the expert rows in place. |
| `GGML_SYCL_FUSE_CONV=1` | 15 | 1 | One-token DeltaNet conv as one kernel. |
| `GGML_SYCL_FUSE_IDX=1` | 16 | 1 | Prefill indexer per-head score sum as one kernel (bit-identical). |
| `GGML_SYCL_IQ4_SOA_LPR=32\|16\|8` | 17 | | Force the IQ4_NL_SOA matvec's lanes per row (tuning). |
| `GGML_SYCL_IQ4_SOA_LUT=mode` | 17 | | IQ4_NL_SOA lookup mode (1, default: a 256-entry table in local memory). |
| `GGML_SYCL_Q8W=1` | 18 | 1 | One-token Q8_0 matvec, a whole block per lane. |
| `GGML_SYCL_Q8_REUSE=1` | 19 | 1 | Reuse a one-token q8_1 activation across matmuls sharing their input. |
| `GGML_SYCL_Q8_DIRECT=1` | 20 | 1 | One-token Q8_0 MUL_MAT dispatched straight to its kernel. |
| `GGML_SYCL_Q8_EPILOGUE=1` | 21 | 1 | Q8_0 MUL_MAT -> [SCALE ->] SILU as one launch (needs `GGML_SYCL_Q8W`). |
| `GGML_SYCL_DMMV_NCOLS=1` | 24 | 1 | 2-4-column Q5_K/Q6_K matvecs through the ESIMD kernel (the MTP verify step). |
| `GGML_SYCL_FUSE_SPARSE_PREFILL=1` | 25 | 1 | Long prompts: QSA attention gathered per query on DPAS (also changes the engine's graph order; =2 is the order only). |
| `GGML_SYCL_SPF_MIN_KV=N` | 25 | 45056 | Below n_kv N the dense path runs. |
| `GGML_SYCL_SPF_MIN_T=N` | 25 | (256) | Chunks of fewer queries (decode, verify steps, short turns) take the dense / decode path. |
| `GGML_SYCL_SPF_CHECK=1` / `GGML_SYCL_SPF_DEBUG=1` | 25 | | Run dense and gathered and compare per layer / report where the matcher stops. |
| `GGML_SYCL_DNNL_DETERMINISTIC=1` | 26 | 1 | oneDNN matmuls in deterministic mode (no split-K atomic accumulation): the same bits every run. |
| `GGML_SYCL_TOPK_DETERMINISTIC=1` | 27 | 1 | Radix top-k emits in column order, ties to the lowest index, no atomics (the QSA indexer's block selection reproducible). With 26: bit-identical runs. |
| `GGML_SYCL_Q8W_NCOLS=1` | 28 | 1 | 2-4-column Q8_0 matvecs (the MTP verify step) through patch 18's wide kernel, each weight block loaded once for all columns; needs `GGML_SYCL_Q8W`. A column's result equals the one-column kernel's. |
| (no switch) | 30 | | CPY from Q2_0_SOA / IQ4_NL_SOA to the canonical blocks (used by `QWFN_PF_VRAM`). Patch 29 was tried and is not in the series. |

## 6. Upstream ggml and runtime variables the launcher sets

| Variable | B70 | What it does |
|---|---|---|
| `GGML_SYCL_FA_ONEDNN=1`, `GGML_SYCL_ENABLE_MKL_FA=0` | set | Flash attention through oneDNN (not oneMKL). |
| `ONEAPI_DEVICE_SELECTOR=level_zero:0` | set | Use the first Level Zero GPU only. |
| `UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=1` | set | Allow device allocations above the default per-allocation limit. |
| `SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1` | set | Immediate command lists (lower launch latency). |
| `KMP_BLOCKTIME=0` | set | OpenMP threads sleep at once after a parallel region (no spinning next to the GPU work). |
| `NEOReadDebugKeys=1 EnableSharedSystemUsmSupport=0` | deployment | Shared-system USM off in the compute runtime (measured at no cost). |
