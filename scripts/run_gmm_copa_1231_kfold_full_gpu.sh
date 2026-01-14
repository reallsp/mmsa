#!/usr/bin/env bash
set -euo pipefail

# GMM | COPA_1231 | FULL 5-fold (KFold on original train) + evaluate on original test
# Using num_workers=0 to avoid memory issues with large dataset

bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_experiment.sh" \
  --model gmm \
  --dataset copa_1231 \
  --kfold \
  --gpu 0 \
  --max-epochs 50 \
  --batch 32 \
  --workers 0 \
  --use-all-features \
  --pretrained /root/autodl-tmp/models/bert-base-uncased \
  --tag gmm_copa_1231_kfold_full \
  --exp-name gmm_copa_1231_kfold_full \
  --foreground
