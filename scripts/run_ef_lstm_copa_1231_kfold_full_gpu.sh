#!/usr/bin/env bash
set -euo pipefail

# EF_LSTM | COPA_1231 | FULL 5-fold (KFold on original train) + evaluate on original test
#
# Note: EF_LSTM can be host-memory-hungry (may get OOM-killed). If you see 'Killed',
# try lowering prefetch/pinning rather than changing batch/workers first.

bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_experiment.sh" \
  --model ef_lstm \
  --dataset copa_1231 \
  --kfold \
  --gpu 0 \
  --max-epochs 50 \
  --batch 64 \
  --workers 8 \
  --prefetch-factor 1 \
  --no-pin-memory \
  --no-persistent-workers \
  --use-all-features \
  --tag ef_lstm_copa_1231_kfold_full \
  --exp-name ef_lstm_copa_1231_kfold_full \
  --foreground

