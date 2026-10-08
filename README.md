# Beyond Answer Correctness: Evidence Calibration for Supervised Post-Training of Medical Vision-Language Models

## Overview

**EC-NFT** combines evidence-calibrated positive supervision with negative-aware learning from self-generated medical VQA responses. It reuses response-to-image attention to distinguish correct trajectories by a spatial proxy for evidence utilization, while preserving negative supervision for incorrect responses. Calibration requires no regional annotations, counterfactual generations, or external visual-evidence scoring.

## Data

Experiments cover **MIMIC, SLAKE, OBScan, and OmniMedVQA**; PubMedVision supports medical adaptation. OmniMedVQA is used for held-out evaluation.

**Downloads:** Hugging Face — *Coming soon* · Google Drive — *Coming soon* · BaiduYun — *Coming soon*.

Use the conversation format in [`examples/train.json`](examples/train.json). Image paths are relative to `DATA_ROOT`, or absolute. Download the corresponding images separately; access follows each source dataset's terms.

## Model Weights

Backbone: **Qwen3-VL-4B-Instruct**. Configure a local checkpoint in `configs/local.sh`. SFT produces a merged rollout policy; EC-NFT saves a LoRA adapter for that policy.

**Downloads:** Hugging Face — *Coming soon* · Google Drive — *Coming soon* · BaiduYun — *Coming soon*.

## Setup

Use Linux, Python 3.10+, and a CUDA GPU supporting BF16.

```bash
pip install -r requirements.txt
cp configs/example.sh configs/local.sh
```

Edit the data, checkpoint, and GPU paths in `configs/local.sh`. The launch defaults are configurable; see [`docs/IMPLEMENTATION.md`](docs/IMPLEMENTATION.md) for scoring and reward details.

## File Layout

```text
EC-NFT/
├── configs/       # Paths, hyperparameters, optional DeepSpeed config
├── launch/        # SFT, Reward-Stratified Rollouts, EC-NFT
├── ec_nft/        # CPU evidence scoring and group calibration
├── qwenvl/        # Qwen-VL SFT and multimodal data utilities
├── scripts/       # Collection, training, merging, evaluation, visualization
├── examples/      # Annotation format
└── tests/         # Calibration and cache validation
```

## Quick Start

### 1. Supervised Fine-Tuning (SFT)

Adapt the backbone to medical instructions and merge the LoRA adapter. For broad adaptation followed by task-specific SFT, run this stage with separate configurations, using the first merged checkpoint as the second stage's `INITIAL_MODEL`.

```bash
bash launch/sft.sh
```

### 2. Reward-Stratified Rollouts

Sample `G` responses per input from the fixed adapted policy and record token log-probabilities and visual attention. Training retains complete groups containing both correct and incorrect answers.

```bash
bash launch/reward_stratified_rollouts.sh
```

The collector defaults to normalized exact answer matching. Open-ended responses need appropriately verified correctness labels before EC-NFT training; visual calibration itself does not use an external judge.

### 3. Evidence-Calibrated Negative-Aware Fine-Tuning (EC-NFT)

Compute attention scores on CPU, calibrate positive weights within each input group, and train with the original negative term.

```bash
bash launch/ecnft.sh
```

Each launcher accepts a configuration path, e.g. `bash launch/ecnft.sh configs/local.sh`. Set `DRY_RUN=1` to inspect commands without starting training.
