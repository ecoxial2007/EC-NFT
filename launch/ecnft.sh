#!/usr/bin/env bash
set -euo pipefail
source "${BASH_SOURCE[0]%/*}/common.sh" "$@"
require_file "${ROLLOUT_DB}"
require_dir "${ROLLOUT_MODEL}"
require_dir "${DATA_ROOT}"
if [[ "${DRY_RUN}" != 1 ]]; then
  cd "${DATA_ROOT}"
fi
run "${PYTHON}" "${REPO_ROOT}/scripts/score_evidence.py" \
  --sqlite "${ROLLOUT_DB}" --output "${EVIDENCE_CACHE}" \
  --top-ratio "${TOP_RATIO}" --border-ratio "${BORDER_RATIO}" --beta "${BETA}" \
  --weight-min "${WEIGHT_MIN}" --weight-max "${WEIGHT_MAX}"
export ATTN_IMPL=sdpa
run "${TORCHRUN}" --nproc_per_node="${NPROC_PER_NODE}" --master_port="${MASTER_PORT}" \
  "${REPO_ROOT}/scripts/train_ecnft.py" \
  --model_name_or_path "${ROLLOUT_MODEL}" --output_dir "${ECNFT_OUTPUT}" \
  --nft_sqlite_path "${ROLLOUT_DB}" --evidence_cache_path "${EVIDENCE_CACHE}" \
  --nft_perturbations clean --nft_neg_weight 1.0 --nft_normalize 1 \
  --nft_clamp_negative 1.0 --nft_clamp_positive 0.0 --nft_log_ratio_max 10.0 \
  --nft_loss_scale 1000.0 --nft_min_r_hat 0.02 --nft_max_r_hat 0.98 \
  --nft_validate_old_logprobs True --nft_log_components True \
  --tune_mm_vision False --tune_mm_mlp False --tune_mm_llm True \
  --lora_enable True --lora_r 256 --lora_alpha 512 --lora_dropout 0.05 \
  --bf16 True --fp16 False --learning_rate "${ECNFT_LR}" --num_train_epochs "${ECNFT_EPOCHS}" \
  --per_device_train_batch_size "${ECNFT_BATCH_SIZE}" \
  --gradient_accumulation_steps "${ECNFT_GRAD_ACCUM}" \
  --model_max_length "${MODEL_MAX_LENGTH}" --gradient_checkpointing True \
  --min_pixels "${MIN_PIXELS}" --max_pixels "${MAX_PIXELS}" \
  --eval_strategy no --save_strategy steps --save_steps 100 --save_total_limit 3 \
  --warmup_steps 10 --weight_decay 0.1 --max_grad_norm 1.0 \
  --logging_steps 10 --report_to none --seed "${SEED}"
