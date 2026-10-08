#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "${BASH_SOURCE[0]%/*}/.." && pwd)"
export REPO_ROOT
CONFIG_PATH="${1:-${REPO_ROOT}/configs/local.sh}"
if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "Config missing: ${CONFIG_PATH}. Copy configs/example.sh to configs/local.sh." >&2
  exit 1
fi
source "${CONFIG_PATH}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
PYTHON="${PYTHON:-python}"
TORCHRUN="${TORCHRUN:-torchrun}"
DRY_RUN="${DRY_RUN:-0}"
IFS=',' read -r -a VISIBLE_GPUS <<< "${GPU_IDS}"
NPROC_PER_NODE="${NPROC_PER_NODE:-${#VISIBLE_GPUS[@]}}"
MASTER_PORT="${MASTER_PORT:-29501}"

run() {
  if [[ "${DRY_RUN}" == 1 ]]; then
    printf '%q ' "$@"
    printf '\n'
  else
    "$@"
  fi
}
require_file() {
  [[ "${DRY_RUN}" == 1 ]] && return 0
  [[ -f "$1" ]] || { echo "File missing: $1" >&2; exit 1; }
}
require_dir() {
  [[ "${DRY_RUN}" == 1 ]] && return 0
  [[ -d "$1" ]] || { echo "Directory missing: $1" >&2; exit 1; }
}
