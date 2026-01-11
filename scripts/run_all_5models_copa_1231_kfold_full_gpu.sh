#!/usr/bin/env bash
set -euo pipefail

# Run ALL supported models (that currently pass kfold-all) sequentially on ONE GPU.
#
# Models (currently kfold-all compatible):
# - tfn
# - cider_lite
# - lf_dnn
# - ef_lstm
# - lmf
#
# This script runs in FOREGROUND to avoid multiple concurrent jobs fighting for GPU memory.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== START: all 5 models | copa_1231 | kfold-all | full 5-fold ==="

bash "${SCRIPT_DIR}/run_tfn_copa_1231_kfold.sh"
bash "${SCRIPT_DIR}/run_ciderlite_copa_1231_kfold.sh"
bash "${SCRIPT_DIR}/run_lf_dnn_copa_1231_kfold_full_gpu.sh"
bash "${SCRIPT_DIR}/run_ef_lstm_copa_1231_kfold_full_gpu.sh"
bash "${SCRIPT_DIR}/run_lmf_copa_1231_kfold_full_gpu.sh"

echo "=== DONE: all 5 models ==="

