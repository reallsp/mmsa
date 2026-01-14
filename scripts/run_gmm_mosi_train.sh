#!/usr/bin/env bash
set -euo pipefail

# GMM | MOSI | Training with local BERT

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

python3 train_any.py \
  --model-name gmm \
  --dataset-name mosi \
  --pretrained /root/autodl-tmp/models/bert-base-uncased \
  --max-epochs 50 \
  --batch-size 16 \
  --num-workers 8 \
  --exp-name gmm_mosi_localbert \
  --foreground
