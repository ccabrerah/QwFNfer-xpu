#!/bin/bash
# qwfn-server with the B70 configuration (docs/B70-config.md). Extra arguments are appended, so a supervisor
# can add --host/--port (the server takes the last value of a repeated flag) or --mmproj.
#   QWFN_B70_LLAMA=~/src/llama-b70 QWFN_B70_GSQ=~/models/.../gsq-rco QWFN_B70_HEAD=<overlay head> scripts/b70/qwfn-b70.sh
# exec, deliberately: a supervisor that treats this process exiting as "VRAM released" must see qwfn-server
# itself, not a wrapper that could outlive it. QWFN_LOCK_HOST needs the memlock limit raised
# (LimitMEMLOCK=infinity under systemd); check the log for "locked (VmLck".
set -e
HERE=$(cd "$(dirname "$0")/../.." && pwd)
LLAMA=${QWFN_B70_LLAMA:-$HOME/src/llama-b70}
GSQ=${QWFN_B70_GSQ:?set QWFN_B70_GSQ to the directory with the GSQ-RCO Q2_0 shards}
STOCK=$GSQ/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf
HEAD=${QWFN_B70_HEAD:-$STOCK}                 # an overlay head (scripts/b70/build-overlay.sh), or the stock model
set +u; source ${ONEAPI:-/opt/intel/oneapi/setvars.sh} --force >/dev/null 2>&1; set -u
export QWFN_GGML_BACKENDS=$LLAMA/build-sycl/bin QWFN_REQUIRE_GPU=1 QWFN_VOCAB_MODEL=$STOCK
export ONEAPI_DEVICE_SELECTOR=level_zero:0 UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=1 SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1
export GGML_SYCL_FA_ONEDNN=1 GGML_SYCL_ENABLE_MKL_FA=0          # oneDNN flash attention
export QWFN_DEV_MASK=1 QWFN_QSA_PACK=1 QWFN_LOCK_HOST=1          # prefill inputs on the device; locked host memory
export GGML_SYCL_FUSE_IDX=1                                      # prefill: the sparse-attention indexer's head sum as one kernel (patch 16)
export GGML_SYCL_FUSE_HC=1 GGML_SYCL_FUSE_HC_DECODE=1            # hyper-connection combine+norm fusion
export KMP_BLOCKTIME=0 GGML_SYCL_MMVW=1 GGML_SYCL_SMALLK=1 GGML_SYCL_MOE_Q2W=1 QWFN_PREDICT_CUR2=1   # decode
export QWFN_Q2_SOA=1 GGML_SYCL_FUSE_SPARSE_DECODE=1 GGML_SYCL_FUSE_HC_GATE=1                         # decode (patches 09-11)
export QWFN_IQ4_SOA=1                                                                                # iq4_nl experts in VRAM as IQ4_NL_SOA (patch 17)
export GGML_SYCL_Q8W=1 GGML_SYCL_Q8W_NCOLS=1                                                        # 2-4-column Q8_0 matvecs (the MTP verify step) on patch 18's wide kernel (patch 28); one token stays on llama.cpp's exact ESIMD kernel
export GGML_SYCL_DMMV_NCOLS=1                                                                        # 2-4-column Q5_K/Q6_K matvecs through the ESIMD kernel (patch 24; the MTP verify step)
export QWFN_LEND_RESTORE=1                                                                           # after a streamed prompt, read the lent layers' VRAM experts back before decode
export QWFN_LEND_FIT=1                                                                               # a streamed prompt lends only what it needs, per layer: agent turns -1.2 s, replies after them faster
export QWFN_MTP_SHARED_CELLS=1                                                                       # the draft head attends over the trunk's selected cells (last attention layer), not its whole cache
export QWFN_PF_VRAM=1                                                                                # a streamed prefill copies VRAM-resident experts on the device instead of reading them (needs patch 30)
export QWFN_Q8_SOA=1                                                                                 # q8_0 experts in VRAM as Q8_0_SOA, their own one-token MoE matvec (patch 31)
export QWFN_PROMOTE_D2D=1                                                         # a promotion's victim staged on device, its read-back on a side queue (patches 33+35): settle ~2.8 -> ~1.7 ms/token, placement identical
export QWFN_HC_PRE_OP=1 QWFN_LIGHTNING_IDX=1                                                      # llama.cpp's DSV4_HC_PRE for the mixer's gate + stream mean (all steps; patch 14's mean fusion is gone) and its LIGHTNING_INDEXER at decode / verify steps
export QWFN_UPLOAD_SKIP=1 QWFN_DOWN_FOLD=1 QWFN_GRAPH_STASH=1 QWFN_LATE_NARROW=1 QWFN_PACK_MULTI=1   # decode: skip zero uploads, fold the HC down scale, keep per-T layer graphs, late fold sized to the promotion budget, readback pack at verify steps (all ours, opt-in)
export GGML_SYCL_MMID_SMALL_T=1 GGML_SYCL_TOPK_DIV_OUT=1   # patches 01, 12: MUL_MAT_ID of <= 8 tokens per token on the fused kernel, router top-k writing the DIV
export QWFN_WARM_COUNTS="${QWFN_WARM_COUNTS:-$HOME/.cache/qwfnfer/expert-counts-$(basename "$HEAD" .gguf).bin}"   # warm start: per-expert use counts persist here; the next start fills the tiers with the hot experts before the first request (~9 s)
mkdir -p "$(dirname "$QWFN_WARM_COUNTS")" 2>/dev/null || { echo "qwfn-b70.sh: cannot create $(dirname "$QWFN_WARM_COUNTS"): warm start off" >&2; unset QWFN_WARM_COUNTS; }
export GGML_SYCL_IQ4_SOA_LPR=8 GGML_SYCL_Q2_LPR=8                                                    # 8 lanes per expert row: no idle lane on 20-/40-block rows (patches 17, 32)
export GGML_SYCL_FUSE_SPARSE_PREFILL=1 GGML_SYCL_SPF_MIN_KV=45056                                    # long prompts: QSA attention gathered per query on DPAS past n_kv 45K (patch 25)
export GGML_SYCL_DNNL_DETERMINISTIC=1 GGML_SYCL_TOPK_DETERMINISTIC=1                                 # bit-identical runs: oneDNN matmuls and the radix top-k without atomics (patches 26-27)
export GGML_SYCL_TOPK_WG=1 GGML_SYCL_FUSE_HC_MIX=1 GGML_SYCL_FUSE_ADDCHAIN=1 GGML_SYCL_FUSE_MOESUM=1 GGML_SYCL_FUSE_CONV=1   # decode fusions (patches 12-15)
export GGML_SYCL_FUSE_SCALE_SILU=1 GGML_SYCL_FUSE_GATE_BCAST=1 GGML_SYCL_FUSE_ROUTE_GATHER=1 GGML_SYCL_FUSE_PREDICT=1 GGML_SYCL_FUSE_ADDCHAIN_VIEWS=1   # decode glue as merged kernels (patch 37, bit-identical): layer graphs -3%
export GGML_SYCL_FUSE_CONV_T=1 GGML_SYCL_FUSE_POOL=1 GGML_SYCL_FUSE_MASK=1                                     # decode glue, round 2 (patch 38, bit-identical): layer graphs -2.3%
export GGML_SYCL_FUSE_MULTI_MMV=1 QWFN_MMV_ADJACENT=1                                              # a verify step's q8_0 projections of one input as one launch (patch 39 + the engine's node order, bit-identical): layer graphs -1.3%
export GGML_SYCL_MMID_SOA_T=1 GGML_SYCL_FUSE_MULTI_MMV_GLU=1 GGML_SYCL_LIDX_DECODE=1 GGML_SYCL_TOPK_SCAN=1 GGML_SYCL_FUSE_MOESUM_W=1 GGML_SYCL_FUSE_PACK_IDS=1   # decode round 4 (patch 40, bit-identical): short -1.3%, 52K context -2.1%
# The MTP draft head (docs/B70-config.md, "MTP"): QWFN_B70_MTP = the head (its experts in Q2_0, all on the device),
# QWFN_B70_DRAFT_VOCAB = the token ids it may draft (optional). One draft per step, no layer-0 prefetch pass, and the
# expert tier one GB smaller for the head.
MTP=()
if [ -n "${QWFN_B70_MTP:-}" ]; then
  export QWFN_MTP_EXPERTS_VRAM=1 QWFN_NO_SPEC_L0=1
  [ -n "${QWFN_B70_DRAFT_VOCAB:-}" ] && export QWFN_MTP_DRAFT_VOCAB=$QWFN_B70_DRAFT_VOCAB
  MTP=(--vram 24 --mtp "$QWFN_B70_MTP" --mtp-drafts 1)
fi
exec "$HERE/build/qwfn-server" "$HEAD" \
    --ctx 131072 --kv q8_0 --vram 25 --ram 8 --batch 16384 --prefill-chunk 6144 --reserve 2048 --prefix-cache 3 \
    "${MTP[@]}" "$@"
