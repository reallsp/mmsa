#!/usr/bin/env bash
set -euo pipefail

# Unified experiment runner for MMSA (train_any.py / test_any.py)
#
# Examples:
#   bash mmsa/scripts/run_experiment.sh --model tfn --dataset copa_1231 --kfold --gpu 0 --max-epochs 50 --batch 64 --workers 8
#   bash mmsa/scripts/run_experiment.sh --model cider_lite --dataset copa_1231 --kfold --gpu 0 --max-epochs 50 --batch 128 --workers 8
#   bash mmsa/scripts/run_experiment.sh --model tfn --dataset copa_1231 --kfold --smoke
#
# Notes:
# - KFold here means: KFold on ORIGINAL train only, then evaluate on ORIGINAL test (no extra final training).

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

MODEL=""
DATASET="copa_1231"
GPU_ID="0"
CPU="0"
KFOLD="0"
SMOKE="0"
ONLY_FOLD=""
MAX_EPOCHS=""
BATCH=""
WORKERS=""
NOHUP_LOG_DIR="${ROOT_DIR}/logs/experiments"
PY_LOG_DIR="${ROOT_DIR}/logs/experiments_py"
RESULTS_CSV="${ROOT_DIR}/results/all_final_test_results.csv"
TAG=""
EXP_NAME=""
FOREGROUND="0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL="$2"; shift 2 ;;
    --dataset) DATASET="$2"; shift 2 ;;
    --gpu) GPU_ID="$2"; CPU="0"; shift 2 ;;
    --cpu) CPU="1"; shift 1 ;;
    --kfold) KFOLD="1"; shift 1 ;;
    --smoke) SMOKE="1"; shift 1 ;;
    --only-fold) ONLY_FOLD="$2"; shift 2 ;;
    --max-epochs) MAX_EPOCHS="$2"; shift 2 ;;
    --batch) BATCH="$2"; shift 2 ;;
    --workers) WORKERS="$2"; shift 2 ;;
    --nohup-log-dir) NOHUP_LOG_DIR="$2"; shift 2 ;;
    --py-log-dir) PY_LOG_DIR="$2"; shift 2 ;;
    --results-csv) RESULTS_CSV="$2"; shift 2 ;;
    --tag) TAG="$2"; shift 2 ;;
    --exp-name) EXP_NAME="$2"; shift 2 ;;
    --foreground) FOREGROUND="1"; shift 1 ;;
    *) echo "Unknown arg: $1"; exit 2 ;;
  esac
done

if [[ -z "${MODEL}" ]]; then
  echo "ERROR: --model is required (e.g. tfn, cider_lite)"
  exit 2
fi

mkdir -p "${NOHUP_LOG_DIR}"
mkdir -p "${PY_LOG_DIR}"
mkdir -p "$(dirname "${RESULTS_CSV}")"

STAMP="$(date +%Y%m%d-%H%M%S)"
TAG_PART=""
if [[ -n "${TAG}" ]]; then TAG_PART="__${TAG}"; fi
RUN_NAME="${MODEL}__${DATASET}__$( [[ "${KFOLD}" == "1" ]] && echo kfold || echo standard )${TAG_PART}__${STAMP}"
LOG_FILE="${NOHUP_LOG_DIR}/${RUN_NAME}.log"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
if [[ "${CPU}" == "1" ]]; then
  export CUDA_VISIBLE_DEVICES=""
else
  export CUDA_VISIBLE_DEVICES="${GPU_ID}"
fi

CMD=(python3 -u "${ROOT_DIR}/train_any.py" --model-name "${MODEL}" --dataset-name "${DATASET}")
if [[ "${KFOLD}" == "1" ]]; then CMD+=(--kfold-all); fi
if [[ "${SMOKE}" == "1" ]]; then CMD+=(--smoke); fi
if [[ -n "${ONLY_FOLD}" ]]; then CMD+=(--only-fold "${ONLY_FOLD}"); fi
if [[ -n "${MAX_EPOCHS}" ]]; then CMD+=(--max-epochs "${MAX_EPOCHS}"); fi
if [[ -n "${BATCH}" ]]; then CMD+=(--batch-size "${BATCH}"); fi
if [[ -n "${WORKERS}" ]]; then CMD+=(--num-workers "${WORKERS}"); fi
if [[ "${CPU}" == "1" ]]; then CMD+=(--cpu); fi
# python-side logging + results
if [[ -z "${EXP_NAME}" ]]; then
  if [[ -n "${TAG}" ]]; then EXP_NAME="${TAG}"; else EXP_NAME="${MODEL}_${DATASET}"; fi
fi
CMD+=(--exp-name "${EXP_NAME}" --log-dir "${PY_LOG_DIR}" --results-csv "${RESULTS_CSV}")

echo "RUN_NAME=${RUN_NAME}"
echo "LOG_FILE=${LOG_FILE}"
echo "CMD=${CMD[*]}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<cpu>}"
echo "PY_LOG_DIR=${PY_LOG_DIR}"
echo "RESULTS_CSV=${RESULTS_CSV}"
echo

if [[ "${FOREGROUND}" == "1" ]]; then
  echo "Running in FOREGROUND (sequential-friendly)..."
  env PYTHONUNBUFFERED=1 "${CMD[@]}" 2>&1 | tee "${LOG_FILE}"
else
  nohup env PYTHONUNBUFFERED=1 "${CMD[@]}" > "${LOG_FILE}" 2>&1 & echo $! | tee "${LOG_FILE}.pid"
  echo "Started. Tail logs with:"
  echo "  tail -f ${LOG_FILE}"
fi

