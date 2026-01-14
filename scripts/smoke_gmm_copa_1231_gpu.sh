#!/usr/bin/env bash
set -euo pipefail

# GMM | COPA_1231 | Smoke test (1 epoch, 1 fold)

bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_experiment.sh" \
  --model gmm \
  --dataset copa_1231 \
  --kfold \
  --gpu 0 \
  --smoke \
  --only-fold 1 \
  --batch 8 \
  --workers 4 \
  --use-all-features \
  --pretrained /root/autodl-tmp/models/bert-base-uncased \
  --tag gmm_copa_1231_smoke \
  --exp-name gmm_copa_1231_smoke \
  --foreground
