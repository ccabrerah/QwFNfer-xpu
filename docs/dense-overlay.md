# Dense overlay: Unsloth's bits where GSQ-RCO cut too deep

GSQ-RCO Q2_0 stores most of the model at 2 bits, including tensors that Unsloth's own low tiers never cut
that far. An overlay puts those tensors back at the bits Unsloth uses, without copying the 66 GB model: a
metadata head with `split.count` raised, the replacement GGUFs, then the stock shards as symlinks. The engine
keeps the **first** tensor of a name across shards, so the replacements win.

`scripts/b70/build-overlay.sh GSQ_DIR OUT_DIR [v1|v2]` fetches the replacement tensors from
`unsloth/Qwen3.8-Flash-Next-GGUF` by HTTP range (`tools/overlay/gguf_fetch.py`, only those tensors' bytes)
and assembles the head (`gguf-requant meta`). Run the server on the new head with
`QWFN_VOCAB_MODEL=<stock head shard>`: llama.cpp's split loader, which reads the tokenizer, rejects shards
that carry no `split.no`.

| Overlay | Tensors | Size | Measured |
|---|---|---:|---|
| v1 | attention and shared-expert gate/up at Q5_K, `ssm_out` at Q6_K (from UD-Q2_K_XL) | +2.0 GB on disk, +565 MB dense core | replay NLL -1.2%, no speed cost |
| v2 = v1 + | shared-expert down Q8_0 on all 48 layers, layer-1 `ple_key` Q8_0, layer-2 expert gate/up IQ3_XXS (UD-Q2_K_XL); `token_embd` Q8_0, `output` Q6_K, expert down Q8_0 on layers 2, 4, 30, 46, 47 (UD-Q3_K_XL) | +5.7 GB on disk, dense core 4.00 -> 4.15 GB | natural-text NLL -0.034 (about 10x the run spread); warm decode -7%, prefill -8% (20K) to -2% (100K) |

v2's cost is its heavier per-token reads (Q8_0 expert down on five layers, Q8_0 shared-expert down on all,
Q6_K output), not the expert tier: the VRAM tier holds 16437 instead of 17328 blocks, but the share of
expert reads served from VRAM barely moves.

`tools/overlay/quant_compare.py GGML_H HEAD [TIER ...]` prints the effective type of every tensor role of an
assembled model next to Unsloth's published tiers (headers read by range, nothing downloaded): the check that
an overlay took. `gguf_fetch_verify.py` spot-checks a fetched file against its source.

`gguf-requant conv` re-encodes tensors locally (for formats no published quant ships); quantizing
already-quantized weights keeps their error, so it is for speed experiments, and `gguf-requant err`
measures what the second rounding adds.
