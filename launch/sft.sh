#!/usr/bin/env bash
set -euo pipefail
source "${BASH_SOURCE[0]%/*}/common.sh" "$@"
require_file "${TRAIN_ANNOTATION}"
require_dir "${INITIAL_MODEL}"
require_dir "${DATA_ROOT}"
EXTRA_ARGS=()
if [[ -n "${DEEPSPEED_CONFIG:-}" ]]; then
  require_file "${DEEPSPEED_CONFIG}"
  EXTRA_ARGS+=(--deepspeed "${DEEPSPEED_CONFIG}")
fi
export ATTN_IMPLEMENTATION=sdpa
run "${TORCHRUN}" --nproc_per_node="${NPROC_PER_NODE}" --master_port="${MASTER_PORT}" \
  "${REPO_ROOT}/qwenvl/train/train_qwen_local_cot_sa.py" \
  --local_dataset_name "${DATASET}" --local_annotation_path "${TRAIN_ANNOTATION}" \
  --local_data_path "${DATA_ROOT}" --dataset_use "${DATASET}" \
  --model_name_or_path "${INITIAL_MODEL}" --output_dir "${SFT_ADAPTER}" \
  --tune_mm_vision False --tune_mm_mlp False --tune_mm_llm True \
  --lora_enable True --lora_r 64 --lora_alpha 128 --lora_dropout 0.05 \
  --cot_loss_mode "${COT_LOSS_MODE}" --data_flatten False --data_packing False \
  --bf16 True --fp16 False --learning_rate "${SFT_LR}" --num_train_epochs "${SFT_EPOCHS}" \
  --per_device_train_batch_size "${SFT_BATCH_SIZE}" \
  --gradient_accumulation_steps "${SFT_GRAD_ACCUM}" \
  --min_pixels "${MIN_PIXELS}" --max_pixels "${MAX_PIXELS}" \
  --model_max_length "${MODEL_MAX_LENGTH}" --gradient_checkpointing True \
  --eval_strategy no --save_strategy steps --save_steps 100 --save_total_limit 3 \
  --weight_decay 0 --warmup_ratio 0.03 --lr_scheduler_type cosine \
  --logging_steps 10 --report_to none --seed "${SEED}" "${EXTRA_ARGS[@]}"

# Merge SFT LoRA before sampling from the fixed medically adapted policy.
run "${PYTHON}" "${REPO_ROOT}/scripts/merge_lora.py" \
  --base_model_path "${INITIAL_MODEL}" --lora_path "${SFT_ADAPTER}" \
  --output_path "${ROLLOUT_MODEL}" --load_dtype bf16 --yes
