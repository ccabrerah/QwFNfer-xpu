#!/bin/bash
# Assemble the dense overlays over the stock GSQ-RCO Q2_0 shards (docs/dense-overlay.md). Nothing of the 66 GB
# model is copied: an overlay is a metadata head with split.count raised, the replacement GGUFs (tensors fetched
# from Unsloth's published quants by HTTP range), then the stock shards as symlinks. The engine keeps the first
# tensor of a name across shards, so the replacements win.
#   scripts/b70/build-overlay.sh GSQ_DIR OUT_DIR [v1|v2]     (default v2)
# GSQ_DIR holds Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-0000{1,2}-of-00002.gguf; needs build/gguf-requant.
# Run the server with QWFN_VOCAB_MODEL=GSQ_DIR/...-00001-of-00002.gguf (the tokenizer comes from the stock head).
set -euo pipefail
HERE=$(cd "$(dirname "$0")/../.." && pwd)
GSQ=$1; OUT=$2; VER=${3:-v2}
S1=$GSQ/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf
S2=$GSQ/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf
FETCH="python3 $HERE/tools/overlay/gguf_fetch.py --jobs 8"
REQUANT=${REQUANT:-$HERE/build/gguf-requant}
mkdir -p "$OUT"
# v1: Unsloth's dense formats (Q5_K attention and shared-expert gate/up, Q6_K ssm_out) from UD-Q2_K_XL
[ -f "$OUT/dense5u.gguf" ] || $FETCH UD-Q2_K_XL "$OUT/dense5u.gguf" \
  'attn_qkv\.weight$' 'attn_gate\.weight$' 'attn_q\.weight$' 'attn_k\.weight$' 'attn_v\.weight$' \
  'attn_output\.weight$' 'ffn_gate_shexp\.weight$' 'ffn_up_shexp\.weight$' 'ssm_out\.weight$'
parts=("$OUT/dense5u.gguf")
if [ "$VER" = v2 ]; then
  # v2: the tensors Unsloth never cuts, at their bits. From UD-Q2_K_XL: shared-expert down Q8_0 (all layers),
  # layer 1 ple_key Q8_0, layer 2 gate/up IQ3_XXS. From UD-Q3_K_XL: token_embd Q8_0, output Q6_K, expert down
  # Q8_0 on layers 2, 4, 30, 46, 47.
  [ -f "$OUT/q2kxl.gguf" ] || $FETCH UD-Q2_K_XL "$OUT/q2kxl.gguf" \
    'ffn_down_shexp\.weight$' '^blk\.1\.ple_key\.weight$' '^blk\.2\.ffn_(gate|up)_exps\.weight$'
  [ -f "$OUT/q3kxl.gguf" ] || $FETCH UD-Q3_K_XL "$OUT/q3kxl.gguf" \
    '^token_embd\.weight$' '^output\.weight$' '^blk\.(2|4|30|46|47)\.ffn_down_exps\.weight$'
  parts=("$OUT/q2kxl.gguf" "$OUT/q3kxl.gguf" "$OUT/dense5u.gguf")
fi
parts+=("$S1" "$S2")
N=$(( ${#parts[@]} + 1 ))
D=$OUT/$VER; mkdir -p "$D"
NAME=Qwen3.8-Flash-Next-GSQ-RCO-Q2_0
"$REQUANT" meta "$S1" "$D/$NAME-00001-of-0000$N.gguf" $N
i=2; for p in "${parts[@]}"; do ln -sf "$(readlink -f "$p")" "$D/$NAME-0000$i-of-0000$N.gguf"; i=$((i + 1)); done
ls -la "$D"
echo "head: $D/$NAME-00001-of-0000$N.gguf"
