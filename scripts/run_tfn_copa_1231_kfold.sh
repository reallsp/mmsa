#!/usr/bin/env bash
set -euo pipefail

# TFN | COPA_1231 | KFold(train-only) + evaluate on original test

bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_experiment.sh" \
  --model tfn \
  --dataset copa_1231 \
  --kfold \
  --gpu 0 \
  --max-epochs 50 \
  --batch 64 \
  --workers 8 \
  --tag tfn_copa_1231_kfold_full \
  --exp-name tfn_copa_1231_kfold_full \
  --foreground

