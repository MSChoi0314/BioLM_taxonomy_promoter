#!/usr/bin/env bash
set -Eeuo pipefail
BASE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="${DATA_ROOT:?Set DATA_ROOT to token_embeddings directory}"
OUT_ROOT="${OUT_ROOT:?Set OUT_ROOT to results directory}"
SEEDS="${SEEDS:-123 2024}"
GPUS="${GPUS:-2}"
mkdir -p "$OUT_ROOT"
for seed in $SEEDS; do
  run="$OUT_ROOT/seed_${seed}"
  if [[ -s "$run/metrics.json" ]]; then echo "[SKIP] completed $seed"; continue; fi
  mkdir -p "$run"
  echo "[START] $(date -Is) host=$(hostname) seed=$seed"
  cmd=(torchrun --standalone --nproc_per_node="$GPUS" "$BASE/train_token_cnn_multiseed_v3.py"
    --train-prefix "$DATA_ROOT/ntv3_train_all/ntv3_train_all"
    --test-prefix "$DATA_ROOT/ntv3_test_all/ntv3_test_all"
    --output-dir "$run" --seed "$seed" --split-seed 42
    --batch-size 128 --eval-batch-size 256 --hidden-dim 256
    --num-blocks 6 --kernel-size 7 --head flatten --dropout 0.1
    --loss huber --lr 3e-4 --weight-decay 1e-4
    --epochs 50 --patience 8 --amp --num-workers 2 --chunk-size 65536 --resume)
  printf '%q ' "${cmd[@]}" > "$run/command.txt"; echo >> "$run/command.txt"
  PYTHONUNBUFFERED=1 OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" "${cmd[@]}" 2>&1 | tee -a "$run/train.log"
  echo "[DONE] $(date -Is) seed=$seed"
done
