#!/usr/bin/env bash
set -Eeuo pipefail
BASE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${DATA_ROOT:?DATA_ROOT must point at token_embeddings/}"
: "${CACHE_ROOT:?CACHE_ROOT must point to directory containing train_bases.npy and test_bases.npy}"
: "${OUT_ROOT:?OUT_ROOT is required}"
SEEDS="${SEEDS:-42}"
GPUS="${GPUS:-2}"
mkdir -p "$OUT_ROOT"
for seed in $SEEDS; do
  out="$OUT_ROOT/seed_${seed}"
  if [[ -s "$out/metrics.json" && -s "$out/test_predictions.csv" ]]; then
    echo "[SKIP] seed=$seed completed"; continue
  fi
  mkdir -p "$out"
  cmd=(torchrun --standalone --nproc_per_node="$GPUS" "$BASE/train_onehot_rescnn.py"
    --train-prefix "$DATA_ROOT/ntv3_train_all/ntv3_train_all"
    --test-prefix "$DATA_ROOT/ntv3_test_all/ntv3_test_all"
    --train-cache "$CACHE_ROOT/train_bases.npy"
    --test-cache "$CACHE_ROOT/test_bases.npy"
    --output-dir "$out" --seed "$seed" --split-seed 42
    --positions 128 --batch-size 128 --eval-batch-size 256
    --hidden-dim 256 --num-blocks 6 --kernel-size 7 --head flatten
    --dropout 0.1 --loss huber --huber-delta 1.0
    --lr 3e-4 --weight-decay 1e-4 --epochs 50 --patience 8
    --amp --num-workers 2 --chunk-size 65536 --resume)
  printf '%q ' "${cmd[@]}" > "$out/command.txt";echo >> "$out/command.txt"
  echo "[START] $(date -Is) seed=$seed host=$(hostname)"
  PYTHONUNBUFFERED=1 OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" "${cmd[@]}" 2>&1 | tee -a "$out/train.log"
  echo "[DONE] $(date -Is) seed=$seed"
done
