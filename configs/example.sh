#!/usr/bin/env bash
# Copy to configs/local.sh. Use absolute paths for external data/checkpoints.
export DATASET="${DATASET:-MIMIC}"
export DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/data}"
export TRAIN_ANNOTATION="${TRAIN_ANNOTATION:-${DATA_ROOT}/annotations/${DATASET}_train.json}"
export INITIAL_MODEL="${INITIAL_MODEL:-${REPO_ROOT}/checkpoints/Qwen3-VL-4B-Instruct}"
export SFT_ADAPTER="${SFT_ADAPTER:-${REPO_ROOT}/outputs/${DATASET}/sft}"
export ROLLOUT_MODEL="${ROLLOUT_MODEL:-${REPO_ROOT}/checkpoints/qwen3-vl-${DATASET}-sft}"
export ROLLOUT_DB="${ROLLOUT_DB:-${REPO_ROOT}/outputs/${DATASET}/rollouts.sqlite}"
export EVIDENCE_CACHE="${EVIDENCE_CACHE:-${REPO_ROOT}/outputs/${DATASET}/evidence.json}"
export ECNFT_OUTPUT="${ECNFT_OUTPUT:-${REPO_ROOT}/outputs/${DATASET}/ecnft}"
export GPU_IDS="${GPU_IDS:-0}"
export SEED="${SEED:-42}"

# Use identical pixel bounds for rollout and EC-NFT token likelihood ratios.
export MIN_PIXELS="${MIN_PIXELS:-1048576}"
export MAX_PIXELS="${MAX_PIXELS:-1048576}"
export MODEL_MAX_LENGTH="${MODEL_MAX_LENGTH:-4096}"

# SFT: full assistant-response supervision; answer_only is also supported.
export SFT_LR="${SFT_LR:-1e-5}"
export SFT_EPOCHS="${SFT_EPOCHS:-1}"
export SFT_BATCH_SIZE="${SFT_BATCH_SIZE:-2}"
export SFT_GRAD_ACCUM="${SFT_GRAD_ACCUM:-16}"
export COT_LOSS_MODE="${COT_LOSS_MODE:-full}"
# Optional: pip install deepspeed; export DEEPSPEED_CONFIG="${REPO_ROOT}/configs/zero2.json"

# Reward-Stratified Rollouts.
export G="${G:-8}"
export ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-1}"
export ROLLOUT_MICRO_BATCH_SIZE="${ROLLOUT_MICRO_BATCH_SIZE:-2}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
export TEMPERATURE="${TEMPERATURE:-0.6}"
export TOP_P="${TOP_P:-0.7}"
export ANSWER_MATCHING="${ANSWER_MATCHING:-exact}"
export ATTENTION_LAYERS="${ATTENTION_LAYERS:-4}"
export ROLLOUT_LIMIT="${ROLLOUT_LIMIT:-0}"

# EC-NFT. These are editable launch defaults, not a frozen benchmark recipe.
export TOP_RATIO="${TOP_RATIO:-0.10}"
export BORDER_RATIO="${BORDER_RATIO:-0.05}"
export BETA="${BETA:-0.25}"
export WEIGHT_MIN="${WEIGHT_MIN:-0.5}"
export WEIGHT_MAX="${WEIGHT_MAX:-1.5}"
export ECNFT_LR="${ECNFT_LR:-1e-6}"
export ECNFT_EPOCHS="${ECNFT_EPOCHS:-1}"
export ECNFT_BATCH_SIZE="${ECNFT_BATCH_SIZE:-4}"
export ECNFT_GRAD_ACCUM="${ECNFT_GRAD_ACCUM:-8}"
