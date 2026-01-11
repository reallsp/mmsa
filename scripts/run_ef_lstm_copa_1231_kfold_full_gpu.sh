#!/usr/bin/env bash
set -euo pipefail

# EF_LSTM | COPA_1231 | FULL 5-fold (KFold on original train) + evaluate on original test
#
# Note: EF_LSTM can be memory-hungry in this setup; keep batch conservative to avoid OOM/Killed.

bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_experiment.sh" \
  --model ef_lstm \
  --dataset copa_1231 \
  --kfold \
  --gpu 0 \
  --max-epochs 50 \
  --batch 128 \
  --workers 8 \
  --tag ef_lstm_copa_1231_kfold_full \
  --exp-name ef_lstm_copa_1231_kfold_full \
  --foreground

