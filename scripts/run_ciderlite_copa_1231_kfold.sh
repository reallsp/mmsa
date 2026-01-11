#!/usr/bin/env bash
set -euo pipefail

# CIDerLite | COPA_1231 | KFold(train-only) + evaluate on original test

bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_experiment.sh" \
  --model cider_lite \
  --dataset copa_1231 \
  --kfold \
  --gpu 0 \
  --max-epochs 50 \
  --batch 128 \
  --workers 8 \
  --tag ciderlite_copa_1231_kfold_full \
  --exp-name ciderlite_copa_1231_kfold_full \
  --foreground

