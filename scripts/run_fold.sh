#!/usr/bin/env bash
# Train and predict one fold: scripts/run_fold.sh <fold>
set -euo pipefail
cd "$(dirname "$0")/.."
FOLD=${1:?fold}; shift
DATA=data/fold$FOLD
OUT=runs/fold$FOLD/kevbio
mkdir -p "$OUT"
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

if [ ! -f "$OUT/head.pt" ] || [ -f "$OUT/resume.pt" ]; then
    python -m kevbio.train --train "$DATA/train.jsonl" --val "$DATA/val.jsonl" --out "$OUT" "$@" 2>&1 | tee -a "$OUT/train.log"
fi
python -m kevbio.predict --run "$OUT" --data "$DATA/val.jsonl" --layout_train "$DATA/train.jsonl" --out "$OUT/predictions.json" \
    2>&1 | tee "$OUT/predict.log"
