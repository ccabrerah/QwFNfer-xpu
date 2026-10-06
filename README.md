
# QwFNfer-xpu

**All the fun of running qwen38-flash-next fast, but on the Intel ARC**

An engine for running qwen38-flash-next as fast as possible on a B70. A derivative of the [QwFNfer](https://github.com/Apolog1ze-Dev/QwFNfer) engine that adds B70 support and miscellaneour QoL functionality for usage as your main LLM provider.

In a nutshell:

* Uses custom kernels and has engine work that improves performance over a direct port.
* Meant for single 32GB B70 GPU, 32GB system ram machines. The not-really-poor-neither-rich man's AI workstation.
* Adds timesharing and supports quickly resuming multiple sessions as a mitigation for single prompt processing.
* Incorporates optimizations coming from multiple inspirations from other engines and CUDA paths.

**New here?** [`docs/BUILDING.md`](docs/BUILDING.md) (build the engine, download and prepare the model), then
[`docs/RUNNING.md`](docs/RUNNING.md) (the command line, the knobs worth tuning, the power cap).

## Is it faster?

There is no other way to use this GPU on other similar engines, which makes us infinite% faster than the competition.

## On perf

For the V4 quant (see recipe, a "Q3-level" set), on a Ryzen 5700, 32GB dual channel system ram, on a dedicated linux system with a B70 power limited to 120W you get:

* 30 TP decode
* 400-500 prefill

A weaker "Q2 level" quant gets you to 40TPS, but with a noticeable downgrade on inference.

## Major improvements

Against the first B70 enablement (2026-09-15: stock llama.cpp SYCL backend, Unsloth UD-Q3_K_XL, no switches):
**15-16 tok/s warm chat and 222-248 tok/s prefill at 21K**. Now, with the validated run configuration below, at a
110 W power cap: **29-31 tok/s short-prompt decode, 27 tok/s decode at 40K context, 430-450 tok/s prefill at
20-40K, ~386 tok/s at 89K**. The [preferred config](#preferred-config) (overlay v4) spends some of that on quality --
natural-text NLL ~1.5 against 2.02 for the validated overlay -- and with patches 18-21 and `--vram 25` runs at
**28-33 tok/s short-prompt decode, ~510 / ~600 / ~500-515 tok/s prefill at 20K / 40K / 89K**.

**2026-10: the MTP draft head.** The model's own multi-token-prediction head now drafts one token per step and the
trunk verifies the pair (exact: the output is the trunk's). The recipe -- a small all-on-GPU head with 2-bit experts,
a draft vocabulary, cheap verify steps, an AVX2 CPU kernel for Q2_0 -- comes from
[Strata](https://github.com/Niko1221/Strata), a CUDA/HIP engine for this model; the rows marked *(Strata)* below are
those ideas ported to this engine and the B70. On the stock-quant overlay v2 (`--vram 24` + the head against
`--vram 25` without it): **~+13% short-prompt decode** (essay ~35, code ~40 tok/s), ~39.5 tok/s on long answers, the
prefix cache and vision unchanged. Configuration: [`docs/B70-config.md`](docs/B70-config.md), "MTP".

| Change | Where | Measured effect |
|---|---|---|
| q2_0 experts stored as [codes][scales] so the MoE kernel uses aligned vector loads | patch 09 + engine (`QWFN_Q2_SOA`) | decode layer graphs -18%, 40K decode +20% |
| wide-load bf16/f16 one-token matvec (16-byte loads, f32 accumulate) | patch 05 (`GGML_SYCL_MMVW`) | decode +23% |
| one-token sparse attention fused into one kernel over the q8_0 cache | patch 10 (`GGML_SYCL_FUSE_SPARSE_DECODE`) | decode layer graphs -8.5% |
| small-K f32 matmul without oneMKL's per-call host cost | patch 05 (`GGML_SYCL_SMALLK`) | decode layer graphs -4.9% |
| wide q2_0 MoE matvec + fused gate/up/SwiGLU | patches 07-08 (`GGML_SYCL_MOE_Q2W`) | decode layer graphs -4% |
| hyper-connection combine + norm (and its gate) as one kernel | patches 04, 06, 11 | decode layer graphs -2.3%; with the device-built inputs, 89K prefill -14% |
| small-kernel fusions: router top-k, hyper-connection mixer, ADD chains, MoE weighted sum, DeltaNet conv | patches 12-15 | decode layer graphs -8% (~300 fewer kernels per token), decode ~+5% |
| IQ4_NL experts in an aligned layout, decoded through a local-memory table (overlay v3's expert down) | patch 17 + engine (`QWFN_IQ4_SOA`) | one-token IQ4_NL MoE matvec 2.75x; v3 decode graph -12% |
| one-token Q8_0 matvec with a whole block per lane (16-byte loads of quants and activations) for the dense Q8_0 weights | patch 18 (`GGML_SYCL_Q8W`) | dense Q8_0 matvecs -13% (~405 -> ~467 GB/s), v3 decode graph -1.0 ms/token |
| overlay v4's hyper-connection mixers at Q8_0, their down projection's scale and SiLU as the Q8_0 matvec's epilogue | overlay v4 + patch 21 (`GGML_SYCL_Q8_EPILOGUE`) | v4 vs v3: decode layer graphs -0.62 ms/token, dense weights -0.6 GB, NLL unchanged |
| one-token Q8_0 matvecs without the generic matmul routine's per-call host work, and one q8_1 quantization per shared input (q/k/v, qkv/gate, shared-expert gate/up) | patches 19-20 (`GGML_SYCL_Q8_REUSE`, `GGML_SYCL_Q8_DIRECT`) | 108 fewer quantize launches per token, host launch time -20 us per layer graph, v3 decode graph -0.32 ms/token; bit-identical |
| the prefill sparse-attention indexer's per-head score sum as one kernel | patch 16 (`GGML_SYCL_FUSE_IDX`) | attention -10% at 89K (78 -> 70 s), -1.8 s at 40K; bit-identical |
| OpenMP pool stops spinning next to the launch thread | `KMP_BLOCKTIME=0` | decode +5% |
| next-layer expert prediction from the FFN input | engine (`QWFN_PREDICT_CUR2`) | decode +4.7% |
| oneDNN flash attention for prefill | `GGML_SYCL_FA_ONEDNN` | 89K prefill 90-192 -> ~300 tok/s |
| causal mask and sparse-attention bias built on the device | engine (`QWFN_DEV_MASK`, `QWFN_QSA_PACK`) | 89K prefill 361 -> 389 tok/s |
| locked, driver-registered host memory for the RAM tier and prefill staging | engine (`QWFN_LOCK_HOST`) | 89K prefill 425 -> 482 tok/s |
| fix: the prefill's sparse-attention selection no longer spends its slots on blocks after the query (backend-neutral; offered upstream) | engine | usable cells per query 823 -> 1,998 of 2,052 (40K); 118K prompt, 8 notes to list in order: 0-1/8 placed right in 5 of 5 runs without it, 8/8 in 7 of 7 with it; no speed cost |
| dense overlay v2: Unsloth's bits for the tensors GSQ-RCO cut to 2 bits | `scripts/b70/build-overlay.sh` | natural-text NLL -0.034 (about 10x the run spread) |
| *(Strata)* MTP draft head with its routed experts re-encoded to Q2_0 (per-block least-squares scale), all 0.81 GB on the device, one draft per step | `tools/overlay/gguf_requant.cpp` (rule `mtpq2`), engine (`--mtp`, `QWFN_MTP_EXPERTS_VRAM`) | decode ~+13% against the same model without it; ~80% of drafts accepted (greedy and at temperature 0.7) |
| *(Strata)* draft vocabulary: the head projects onto a subset of the LM head's rows (English and code, Latin-script tokens) | engine (`QWFN_MTP_DRAFT_VOCAB`) | the head's cost halved (2.6 -> 1.3-1.7 ms per step); Spanish keeps ~66% acceptance (78% with the full vocabulary) |
| *(Strata)* cheaper verify steps: the head's logits read back only when sampling, no layer-0 prefetch pass with drafts, the confidence gate as an option | engine + server (`QWFN_NO_SPEC_L0`, `QWFN_MTP_MIN_P`) | host time per step ~9.5 -> ~4.4 ms |
| verify-step kernels and graph: wide bf16 matvec, fused q2_0 MoE GLU and small-K matmul for 1-4 tokens; the attention's cache writes, scoring, top-k and masks built once for the step's positions | patch 22, engine (`QWFN_QSA_CHAIN=1` restores the per-position calls) | verify-step layer graphs 28.4 -> 27.4 ms; identical answers and acceptance |
| prefix cache with the draft head: checkpoints carry the head's caches and last residual | engine | 20K-token restores in ~110 ms with drafting going on after them (before: `--prefix-cache` refused `--mtp`) |
| *(Strata)* AVX2 Q2_0 x Q8_0 dot product for the CPU-computed experts (x86 had only the scalar loop) | patch 23 | 2.6x per core (394 -> 150 us for one expert's gate+up); CPU expert time -12% end to end; NLL unchanged |
| *(Strata)* exclusive expert tiers: RAM holds what VRAM does not; a promotion reads the VRAM victim back into a free RAM slot (asynchronous, ahead of the upload on the in-order queue), kept in the device layout until a CPU use | engine (`QWFN_RAM_EXCLUSIVE`, `QWFN_RAM_LENT_INCLUSIVE`; `QWFN_SWAP_CHECK` compares every read-back with the file) | at `--ram 8`: expert I/O per step 11.6-14.4 -> 0.7-2.5 ms, decode +17-38% (essay 32.5 -> 44.9, code 37.6 -> 46.8 tok/s); 11,264 read-backs byte-exact |
| *(Strata)* prompt-lookup drafts: when the reply copies its context, a step of 1+N positions drafted from the earlier occurrence (suffix index of the history), switched in on a long match and kept while the copy lasts; off by default | server (`--lookup-drafts N`, needs `--mtp`), `src/qwfn_lookup.*`, `qwfn-lookup-test` | N=2: +1..+8% on replies that copy their input (mean ~+4%), ordinary text unchanged, ~230 MiB of device memory; small here because the draft head already drafts copied text at ~99% and a 3-position step costs ~1.25-1.3x |
| multi-column ESIMD matvec for the verify step: a 2-4-column Q5_K / Q6_K MUL_MAT (the MTP verify step's positions) dequantizes each block pair once and MACs it against every column, instead of leaving the one-column ESIMD kernel for the generic n-column MMVQ | patch 24 (`GGML_SYCL_DMMV_NCOLS=1`) | verify-step layer graphs 32.0 -> 29.9 ms, decode +5-7% (essay 41.3 -> 44.2, code 42.0 -> 44.1 tok/s); identical answers |
| 2-4-column Q8_0 matvec for the verify step: the dense Q8_0 weights (overlay v4's mixers and hyper-connections) at the verify step's positions through patch 18's wide kernel, each block loaded once for every column, instead of the stock reorder ncols kernel | patch 28 (`GGML_SYCL_Q8W_NCOLS=1`) | verify-step dense Q8_0 11.8 -> 9.3 ms/step, layer graphs 35.6 -> 31.9 ms/step, short-prompt decode +9% (32.5 -> 35.5 tok/s); replay NLL not worse |
| expert tier restored after a streamed prompt: the layers a prefill borrows as staging get their VRAM experts back (batched reads, hottest first, 3 s budget) before the reply, instead of refilling through demand misses | engine (`QWFN_LEND_RESTORE=1`) | replies after a 1.7K-token turn at 55K context 22.4 -> 28.6 tok/s (+28%), after a 52K prompt 19-21 -> 22-25 tok/s; non-VRAM experts per step 24 -> 7-9; +1.6-3 s on the prompt |
| the draft head over the trunk's selected cells: its attention gathers the ~2K cells the last attention layer chose for the same position, instead of attending densely over its whole cache (whose cost grows with context) | engine (`QWFN_MTP_SHARED_CELLS=1`) | head 5.8-6.7 -> 1.9-2.0 ms per step at 52-60K context, agent turns +7% (28.0 -> 29.9 tok/s), short prompts unchanged, acceptance unchanged |
| a streamed prompt takes VRAM-resident experts from the tier: the prefill copies them on the device into its staging instead of re-reading them from disk and uploading them | engine (`QWFN_PF_VRAM=1`) + patch 30 | a 1.7K-token agent turn at 55K context: prompt 17.3 -> 13.6 s (-21%), disk reads 39.4 -> 24.6 GB; a 52K prompt -6.5%; prefill hashes bit-identical |
| gathered sparse prefill attention: a long prompt chunk's QSA attention (selection mask + dense FA over every cached cell) becomes one kernel per (query, kv head) over that query's ~2,052 chosen cells straight from the q8_0 cache, the GQA group's heads as DPAS matrix rows (after vLLM's QSA kernel); dense below 45K cached cells or 256 queries | patch 25 + engine graph order (`GGML_SYCL_FUSE_SPARSE_PREFILL=1 GGML_SYCL_SPF_MIN_KV=45056`) | prefill +12% at ~100K and at 126K + image (432 -> 485, 419 -> 468 tok/s), 124.5K alone 295 -> 254 s; short prompts and decode unchanged; max per-layer difference vs dense ~1e-3 relative |
| deterministic GPU arithmetic: oneDNN matmuls without split-K atomics and a radix top-k that emits in index order (ties to the lowest index), so two runs of the same prompt give bit-identical hidden states and identical greedy text | patches 26-27 (`GGML_SYCL_DNNL_DETERMINISTIC=1 GGML_SYCL_TOPK_DETERMINISTIC=1`) | reproducible runs; at 100K context decode +13% (the top-k no longer serialises on tied blocks), prefill -2.5% |
| time-shared engine: up to N open requests take turns (a quantum each, parking at decode steps and between prefill passes, least-used session first), checkpointed through the prefix cache | server (`--timeshare N --quantum S --linger S`) | a short request behind a 100K prefill: first token 4 s instead of 174 s; parking exact; see [Two serving features](#two-serving-features-time-sharing-and-the-ram-prefix-cache) |
| one-token expert kernels with no idle lanes: the five Q8_0 expert-down layers in the VRAM tier's SOA layout with their own MoE matvec (16-byte half blocks), and the IQ4_NL / Q2_0 expert kernels at 8 lanes per row (this model's 20- and 40-block rows left 38% / 17% of 16 lanes idle); greedy (temperature 0) sampling's argmax vectorized | patches 31-32 + engine (`QWFN_Q8_SOA=1 GGML_SYCL_IQ4_SOA_LPR=8 GGML_SYCL_Q2_LPR=8`), server | Q8_0 expert down 2.17 -> 0.8 ms, IQ4_NL 3.64 -> 3.0, Q2_0 gate/up 2.61 -> 2.4 ms per verify step; argmax 0.69 -> 0.03 ms per position; short-prompt decode +11% (34.0 -> 37.9 tok/s), agent turns at ~55K context +6.5% (30.9 -> 32.9 tok/s) |

Every change is off by default in the patched ggml tree and checked with `test-backend-ops`; the details and the
rejected alternatives are in [`docs/B70-SYCL.md`](docs/B70-SYCL.md) and [`docs/B70-config.md`](docs/B70-config.md), and
everything tried so far (adopted, rejected, and ideas not yet tried) in [`docs/B70-registry.md`](docs/B70-registry.md). Every command-line flag and environment variable: [`docs/PARAMETERS.md`](docs/PARAMETERS.md).

## Two serving features: time sharing and the RAM prefix cache

The engine holds one sequence at a time: its KV cache, the DeltaNet layers' recurrent state and the draft head's
state. Upstream serves one request after another. Two additions make it behave well with several clients.

### Prefix cache in RAM (`--prefix-cache GB`)

A conversation the engine switches away from is checkpointed whole into host memory -- the KV rows it filled, the
recurrent state, the draft head's caches -- and restored when that conversation comes back, instead of re-prefilling
it. Reusing a KV prefix alone is not enough here: the recurrent layers carry state that cannot be rebuilt from a
prefix of the cache. Entries are evicted least recently used first.

- Cost: ~16 KB per token at `--kv q8_0` plus ~118 MB of recurrent state per entry (a 21K-token conversation is
  ~460 MB). A 20K-token restore takes ~110 ms; a 100K-token checkpoint (1.9 GB) restores in ~0.75 s, where
  re-prefilling it takes ~3.5 minutes.
- `--prefix-cache-boundary N` also checkpoints a chat prompt at the end of its history, so a client that re-renders
  the previous reply differently (token boundaries, whitespace) still restores everything before it.
  `--prefix-cache-min N` skips short one-off requests.

### Time sharing (`--timeshare N --quantum S --linger S`)

Up to N requests are open at once and share the engine in turns, so a short request no longer waits for another
client's whole answer:

- One request runs at a time, for a quantum (default 30 s). Then, at its next yield point, it parks if another open
  request should run: its sequence is checkpointed into the prefix-cache budget, the engine is released, and it
  resumes later exactly where it stopped. Yield points are a decode step's start and the end of a prefill pass
  (`--batch` tokens), so a long prompt does not block the others either.
- The next request is the one whose session has used the engine least lately (decayed usage; ties by arrival). A
  session is an `X-Session-Id` header, OpenAI's `user` or Anthropic's `metadata.user_id`, or else the conversation's
  opening. When a request finishes, its session's next turn gets a short linger (default 1 s) and continues that
  session's quantum, so an agent loop of quick turns still yields.
- Request N+1 gets 503. A request whose client disconnects -- running, parked or waiting -- gives up its slot.
  `/slots` lists the open requests with their state and engine time; `/metrics` has `qwfn:timeshare_*` counters.

Measured on the B70: a short request sent during a 100K-token prefill got its first token after **4 s instead of
174 s**; parking costs 40-115 ms at short contexts and ~1.8 s round trip at 100K. Parking is exact: with the
deterministic arithmetic switches (patches 26-27), a reply with park/resume round trips mid-decode and mid-prefill
is byte-identical to one without. In front of a proxy that limits concurrency per backend, allow N requests to this
one. Every flag and variable: [`docs/PARAMETERS.md`](docs/PARAMETERS.md).

## Preferred config

**Overlay v4 with vision**: the heaviest overlay this card runs at a usable speed. It is the validated
configuration below with more bits where the stock quantization is thinnest: expert down and the dense tensors.
Same build, flags and launcher, a different head. Its IQ4_NL expert down runs through patch 17's kernel
(`QWFN_IQ4_SOA`), which cuts the GPU time per decode token by 12%.

## Acknowledgment

Built on [ggml](https://github.com/ggml-org/ggml) (quantized kernels, CUDA backend) and [llama.cpp](https://github.com/ggml-org/llama.cpp) (tokenizer, and the bit-exact reference the forward pass is validated against). Model: Qwen3.8-Flash-Next by the [Qwen](https://huggingface.co/Qwen) team; quantized GGUFs by [Unsloth](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF), whose Studio served as the harness for testing. The MTP recipe (rows marked *(Strata)*) follows
[Strata](https://github.com/Niko1221/Strata) by Niko1221, MIT: its paper and code showed what makes a draft head pay
on a small GPU; its AVX2 Q2_0 row kernel is the model for patch 23, and its English/code draft vocabulary the base of
ours. The engine itself is [QwFNfer](https://github.com/Apolog1ze-Dev/QwFNfer); this fork only adds the Intel
work on top of it. Base weights: [GSQ-RCO Q2_0](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF)
by ISTA-DASLab. The SYCL backend is ggml's, built with Intel oneAPI and oneDNN.

## License

[Apache License 2.0](LICENSE).
