#!/usr/bin/env bash
set -euo pipefail

# Run 5 compatible models on SCL90_1231 sequentially on ONE GPU.
# KFold on original train only -> evaluate once on original test -> append to global CSV.

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "START SCL90_1231 5-model pipeline @ $(date)"
echo "Logs: ${DIR}/../logs/experiments"
echo

bash "${DIR}/run_tfn_scl90_1231_kfold_full_gpu.sh"
bash "${DIR}/run_ciderlite_scl90_1231_kfold_full_gpu.sh"
bash "${DIR}/run_lf_dnn_scl90_1231_kfold_full_gpu.sh"
bash "${DIR}/run_ef_lstm_scl90_1231_kfold_full_gpu.sh"
bash "${DIR}/run_lmf_scl90_1231_kfold_full_gpu.sh"

echo
echo "DONE SCL90_1231 5-model pipeline @ $(date)"

