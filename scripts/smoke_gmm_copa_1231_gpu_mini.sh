#!/usr/bin/env bash
set -euo pipefail

# GMM | COPA_1231 | Mini smoke test (1 epoch, 1 fold, very small batch)

bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_experiment.sh" \
  --model gmm \
  --dataset copa_1231 \
  --kfold \
  --gpu 0 \
  --smoke \
  --only-fold 1 \
  --batch 4 \
  --workers 2 \
  --use-all-features \
  --pretrained /root/autodl-tmp/models/bert-base-uncased \
  --tag gmm_copa_1231_smoke_mini \
  --exp-name gmm_copa_1231_smoke_mini \
  --foreground
