#!/usr/bin/env bash
set -euo pipefail

# LF_DNN | SCL90_1231 | FULL 5-fold (KFold on original train) + evaluate on original test

FEATURE="/root/autodl-tmp/data/scl90_{mode}_1231_converted.pkl"

bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_experiment.sh" \
  --model lf_dnn \
  --dataset scl90_1231 \
  --kfold \
  --gpu 0 \
  --feature-path "${FEATURE}" \
  --max-epochs 50 \
  --batch 512 \
  --workers 8 \
  --prefetch-factor 1 \
  --no-pin-memory \
  --no-persistent-workers \
  --tag lf_dnn_scl90_1231_kfold_full \
  --exp-name lf_dnn_scl90_1231_kfold_full \
  --foreground

