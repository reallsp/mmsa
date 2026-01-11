#!/usr/bin/env bash
set -euo pipefail

# Fast GPU smoke: TFN | COPA_1231 | KFold(train-only) + eval original test

bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_experiment.sh" \
  --model tfn \
  --dataset copa_1231 \
  --kfold \
  --smoke \
  --gpu 0 \
  --tag smoke_tfn_copa_1231_gpu

