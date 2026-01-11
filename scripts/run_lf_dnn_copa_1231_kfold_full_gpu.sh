#!/usr/bin/env bash
set -euo pipefail

# LF_DNN | COPA_1231 | FULL 5-fold (KFold on original train) + evaluate on original test

bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_experiment.sh" \
  --model lf_dnn \
  --dataset copa_1231 \
  --kfold \
  --gpu 0 \
  --max-epochs 50 \
  --batch 512 \
  --workers 8 \
  --tag lf_dnn_copa_1231_kfold_full \
  --exp-name lf_dnn_copa_1231_kfold_full \
  --foreground

