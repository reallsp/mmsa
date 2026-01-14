#!/usr/bin/env bash
set -euo pipefail

# LMF | COPA_1231 | FULL 5-fold (KFold on original train) + evaluate on original test

bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_experiment.sh" \
  --model lmf \
  --dataset copa_1231 \
  --kfold \
  --gpu 0 \
  --max-epochs 50 \
  --batch 64 \
  --workers 8 \
  --use-all-features \
  --tag lmf_copa_1231_kfold_full \
  --exp-name lmf_copa_1231_kfold_full \
  --foreground

