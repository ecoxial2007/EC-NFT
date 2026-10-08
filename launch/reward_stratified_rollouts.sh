#!/usr/bin/env bash
set -euo pipefail
source "${BASH_SOURCE[0]%/*}/common.sh" "$@"
require_file "${TRAIN_ANNOTATION}"
require_dir "${ROLLOUT_MODEL}"
require_dir "${DATA_ROOT}"
if [[ "${DRY_RUN}" != 1 ]]; then
  cd "${DATA_ROOT}"
fi
# One visible GPU collects one dataset; training supports multiple visible GPUs.
run "${PYTHON}" "${REPO_ROOT}/scripts/check_transformers_attention_contract.py"
run "${PYTHON}" "${REPO_ROOT}/scripts/collect_rollouts.py" \
  --base-model "${ROLLOUT_MODEL}" --annotation-path "${TRAIN_ANNOTATION}" \
  --dataset "${DATASET}" --output "${ROLLOUT_DB}" --device cuda:0 \
  --dtype bf16 --attn-impl eager --perturbations clean \
  --batch-size "${ROLLOUT_BATCH_SIZE}" --rollout-micro-batch-size "${ROLLOUT_MICRO_BATCH_SIZE}" \
  --rollouts-per-prompt "${G}" --max-new-tokens "${MAX_NEW_TOKENS}" \
  --temperature "${TEMPERATURE}" --top-p "${TOP_P}" --top-k 0 \
  --min-pixels "${MIN_PIXELS}" --max-pixels "${MAX_PIXELS}" \
  --record-evidence-topk 1 --evidence-attention-layers "${ATTENTION_LAYERS}" \
  --evidence-top-ratio "${TOP_RATIO}" --evidence-border-ratio "${BORDER_RATIO}" \
  --evidence-layer-weighting uniform --answer-matching "${ANSWER_MATCHING}" \
  --limit "${ROLLOUT_LIMIT}" --seed "${SEED}" --resume 1
run "${PYTHON}" "${REPO_ROOT}/scripts/validate_rollout_attention.py" \
  --sqlite "${ROLLOUT_DB}" --dataset "${DATASET}"
