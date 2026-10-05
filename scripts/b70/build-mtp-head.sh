#!/bin/bash
# The MTP draft head with its experts in Q2_0 (docs/BUILDING.md): Unsloth's shared-Q8_0 MTP head, its experts
# re-encoded by gguf-requant (rule mtpq2), as a 3-part split [metadata, Q2_0 experts, original]. The loader keeps
# the first tensor of a name, so the original's Q8_0 experts are shadowed and never loaded. Nothing is copied.
#   scripts/b70/build-mtp-head.sh MTP_Q8_0.gguf OUT_DIR        -> OUT_DIR/mtp-q2-00001-of-00003.gguf (QWFN_B70_MTP)
# Needs build/gguf-requant (built with the engine). ~0.7 GB written, a few minutes on 8 cores.
set -euo pipefail
HERE=$(cd "$(dirname "$0")/../.." && pwd)
SRC=$(readlink -f "$1"); OUT=$2
REQUANT=${REQUANT:-$HERE/build/gguf-requant}
mkdir -p "$OUT"; OUT=$(readlink -f "$OUT")
if [ ! -f "$OUT/experts-q2.gguf" ]; then       # written aside, so an interrupted run is redone, not reused
  "$REQUANT" conv mtpq2 "$OUT/experts-q2.part.gguf" "$(nproc)" "$SRC"
  mv "$OUT/experts-q2.part.gguf" "$OUT/experts-q2.gguf"
fi
"$REQUANT" meta "$SRC" "$OUT/mtp-q2-00001-of-00003.gguf" 3
ln -sf "$OUT/experts-q2.gguf" "$OUT/mtp-q2-00002-of-00003.gguf"
ln -sf "$SRC" "$OUT/mtp-q2-00003-of-00003.gguf"
echo "head: $OUT/mtp-q2-00001-of-00003.gguf"
