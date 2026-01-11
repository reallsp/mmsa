#!/usr/bin/env bash
set -euo pipefail

# Fast GPU smoke: CIDerLite | COPA_1231 | KFold(train-only) + eval original test

bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_experiment.sh" \
  --model cider_lite \
  --dataset copa_1231 \
  --kfold \
  --smoke \
  --gpu 0 \
  --tag smoke_ciderlite_copa_1231_gpu

