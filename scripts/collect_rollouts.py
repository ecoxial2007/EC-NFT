#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Collect reward-stratified Qwen3-VL rollouts from a fixed medical policy.

This is a data-collection/scoring script, not the optimizer itself.
For every (sample, perturbation) prompt it:
  1. samples K responses from the current frozen policy (pi_old),
  2. verifies every response with a binary correctness reward r in {0, 1},
  3. computes r_hat = mean(r_1, ..., r_K) separately for that prompt,
  4. records exact generated token ids and pi_old token log-probabilities,
  5. records sequence logP, mean logP, NLL and PPL,
  6. marks groups with 0 < r_hat < 1 as NFT-eligible.

Important: clean/black/noise/blur are different multimodal prompts, so each
perturbation gets its own K-rollout group and its own r_hat.
"""

import argparse
import json
import math
import os
import random
import re
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

os.environ.setdefault("NCCL_IB_DISABLE", "1")
os.environ.setdefault("NCCL_P2P_DISABLE", "1")
os.environ.setdefault("NCCL_SHM_DISABLE", "1")

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from qwen_vl_utils import process_vision_info


ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.IGNORECASE | re.DOTALL)
ANSWER_MATCHING = "exact"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--base-model", required=True)
    parser.add_argument("--annotation-path", required=True)
    parser.add_argument(
        "--dataset",
        required=True,
        help=(
            "Dataset label stored in SQLite. The release launcher uses "
            "MIMIC, SLAKE, OBSCAN, or PUBMEDVISION."
        ),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dtype",
        choices=["bf16", "fp16", "fp32"],
        default="bf16",
    )
    parser.add_argument("--attn-impl", default="eager")
    parser.add_argument("--output", required=True)

    parser.add_argument(
        "--perturb-root",
        default="./data/perturbations",
    )
    parser.add_argument("--manifest-name", default="manifest.json")
    parser.add_argument("--perturbations", default="clean")
    parser.add_argument("--answer-matching", choices=["exact", "legacy"], default="exact",
                        help="Normalized exact match, or the original substring matcher.")

    parser.add_argument("--batch-size", type=int, default=2,
                        help="Number of prompts generated together.")
    parser.add_argument("--score-batch-size", type=int, default=4,
                        help="Number of sampled responses teacher-forced together.")
    parser.add_argument(
        "--rollout-micro-batch-size",
        type=int,
        default=2,
        help=(
            "Number of responses sampled per prompt in one generation call. "
            "Responses are explicitly expanded before multimodal processing, "
            "avoiding unsafe num_return_sequences expansion of pixel_values."
        ),
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", type=int, choices=[0, 1], default=1)
    parser.add_argument("--save-every", type=int, default=50,
                        help="Save after this many completed prompt/perturbation groups.")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument(
        "--rollouts-per-prompt",
        "--num-samples",
        dest="rollouts_per_prompt",
        type=int,
        default=8,
    )

    parser.add_argument("--min-pixels", type=int, default=1048576)
    parser.add_argument("--max-pixels", type=int, default=4194304)
    parser.add_argument(
        "--save-token-text",
        action="store_true",
        help="Also save raw/decoded token strings. Token ids and token logprobs are always saved.",
    )
    parser.add_argument(
        "--record-evidence-topk",
        type=int,
        choices=[0, 1],
        default=0,
        help=(
            "Record sampled-token logP and CoT-to-visual Top-K indices online "
            "inside model.generate. This removes the post-generation old-logP "
            "teacher-forcing pass and lets Evidence-Calibrated NFT run only the "
            "masked intervention forward. Requires --attn-impl eager."
        ),
    )
    parser.add_argument("--evidence-top-ratio", type=float, default=0.10)
    parser.add_argument("--evidence-border-ratio", type=float, default=0.05)
    parser.add_argument("--evidence-border-tokens", type=int, default=-1)
    parser.add_argument("--evidence-attention-layers", type=int, default=4)
    parser.add_argument(
        "--evidence-layer-weighting",
        choices=["uniform", "depth_linear"],
        default="uniform",
    )

    return parser.parse_args()

def clean_text(value: Any) -> str:
    text = str(value or "")
    text = text.replace("<|im_end|>", "")
    text = text.replace("<|endoftext|>", "")
    text = text.replace("\r", " ").replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_answer(value: Any) -> str:
    if isinstance(value, list):
        value = ", ".join(str(x).strip() for x in value)

    text = clean_text(value).lower()
    text = re.sub(r"</?answer>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"</?think>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()

    if len(text) > 1 and text.endswith("."):
        text = text[:-1].strip()

    return text


def extract_answer(text: str) -> str:
    match = ANSWER_RE.search(str(text or ""))
    if match:
        return normalize_answer(match.group(1))
    return normalize_answer(text)


def is_correct(prediction: str, gold: str) -> bool:
    pred = normalize_answer(prediction)
    target = normalize_answer(gold)

    if not pred or not target:
        return False

    if ANSWER_MATCHING == "legacy":
        return pred == target or pred in target or target in pred
    return pred == target


def load_json_list(path: str) -> List[Dict[str, Any]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))

    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        for key in ["data", "annotations", "items", "samples", "results"]:
            if isinstance(data.get(key), list):
                return data[key]

    raise ValueError(f"Unsupported annotation JSON format: {path}")


def get_question(item: Dict[str, Any]) -> str:
    if item.get("question") not in [None, ""]:
        return clean_text(str(item["question"]).replace("<image>", ""))

    conversations = item.get("conversations") or []
    for message in conversations:
        if not isinstance(message, dict):
            continue

        role = str(
            message.get("from", message.get("role", ""))
        ).lower()

        if role in {"human", "user"}:
            content = message.get("value", message.get("content", ""))
            return clean_text(str(content).replace("<image>", ""))

    return ""


def get_gt_value(item: Dict[str, Any]) -> str:
    conversations = item.get("conversations") or []

    if isinstance(conversations, list):
        for message in reversed(conversations):
            if not isinstance(message, dict):
                continue

            role = str(
                message.get("from", message.get("role", ""))
            ).lower()

            if role in {"gpt", "assistant"}:
                value = message.get("value", message.get("content", ""))
                return str(value).replace("<|im_end|>", "").strip()

    for key in ["CoT", "gt_value", "original_cot", "explanation"]:
        if item.get(key) not in [None, ""]:
            value = str(item[key]).replace("<|im_end|>", "").strip()
            if "<answer>" in value.lower():
                return value

    answer = item.get("answer", item.get("gold_answer", ""))
    if isinstance(answer, list):
        answer = ", ".join(str(x) for x in answer)

    return (
        "<think>\n"
        "My task is to answer the medical visual question based on the image.\n"
        "</think>\n"
        "<answer>\n"
        f"{answer}\n"
        "</answer>"
    )


def get_gold_answer(item: Dict[str, Any], gt_value: str) -> str:
    if item.get("answer") not in [None, ""]:
        return clean_text(item["answer"])

    if item.get("gold_answer") not in [None, ""]:
        return clean_text(item["gold_answer"])

    match = ANSWER_RE.search(gt_value or "")
    if match:
        return clean_text(match.group(1))

    return clean_text(gt_value)


def as_path_list(value: Any) -> List[str]:
    if value is None or value == "":
        return []

    if isinstance(value, list):
        return [
            str(x).strip()
            for x in value
            if str(x).strip()
        ]

    return [str(value).strip()]


def get_image_paths(item: Dict[str, Any]) -> List[str]:
    for key in [
        "image",
        "image_path",
        "image_paths",
        "img_path",
        "path",
    ]:
        if item.get(key):
            return as_path_list(item[key])

    return []


def check_single_image_samples(
    data: Sequence[Dict[str, Any]],
) -> None:
    invalid = []

    for index, item in enumerate(data):
        image_paths = get_image_paths(item)
        if len(image_paths) != 1:
            invalid.append((index, len(image_paths)))

    if invalid:
        raise ValueError(
            "This batched script requires exactly one image per sample. "
            f"Found {len(invalid)} invalid samples. "
            f"First examples: {invalid[:10]}"
        )


def load_manifest(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Perturbation manifest not found: {path}"
        )

    manifest = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(manifest, dict):
        raise ValueError(f"Invalid manifest format: {path}")

    has_samples = isinstance(manifest.get("samples"), list)
    has_items = isinstance(manifest.get("items"), list)

    if not has_samples and not has_items:
        raise ValueError(
            "Manifest must contain either list field "
            f"'samples' or 'items': {path}"
        )

    return manifest


def build_manifest_index(
    manifest: Dict[str, Any],
) -> Dict[int, Dict[str, Any]]:
    """
    Normalize both supported manifest formats into:

    {
        sample_index: {
            "sample_index": int,
            "images": [
                {
                    "image_index": int,
                    "source_path": str,
                    "black_path": str,
                    "noise_path": str,
                    "blur_path": str,
                }
            ],
        }
    }
    """
    index: Dict[int, Dict[str, Any]] = {}

    if isinstance(manifest.get("samples"), list):
        for sample in manifest["samples"]:
            if not isinstance(sample, dict):
                continue

            sample_index = sample.get("sample_index")
            if sample_index is None:
                continue

            images = sample.get("images", [])
            if not isinstance(images, list):
                images = []

            normalized_images = []

            for fallback_image_index, image_record in enumerate(images):
                if not isinstance(image_record, dict):
                    continue

                normalized_images.append(
                    {
                        "image_index": int(
                            image_record.get(
                                "image_index",
                                fallback_image_index,
                            )
                        ),
                        "source_path": str(
                            image_record.get("source_path", "") or ""
                        ),
                        "black_path": str(
                            image_record.get("black_path", "") or ""
                        ),
                        "noise_path": str(
                            image_record.get("noise_path", "") or ""
                        ),
                        "blur_path": str(
                            image_record.get("blur_path", "") or ""
                        ),
                    }
                )

            index[int(sample_index)] = {
                "sample_index": int(sample_index),
                "images": normalized_images,
            }

    elif isinstance(manifest.get("items"), list):
        for item in manifest["items"]:
            if not isinstance(item, dict):
                continue

            sample_index = item.get("sample_index")
            if sample_index is None:
                continue

            source_images = item.get("source_images", [])
            outputs = item.get("outputs", {})

            if not isinstance(source_images, list):
                source_images = []

            if not isinstance(outputs, dict):
                outputs = {}

            black_paths = outputs.get("black", [])
            noise_paths = outputs.get("noise", [])
            blur_paths = outputs.get("blur", [])

            if not isinstance(black_paths, list):
                black_paths = []
            if not isinstance(noise_paths, list):
                noise_paths = []
            if not isinstance(blur_paths, list):
                blur_paths = []

            number_of_images = max(
                len(source_images),
                len(black_paths),
                len(noise_paths),
                len(blur_paths),
            )

            normalized_images = []

            for image_index in range(number_of_images):
                normalized_images.append(
                    {
                        "image_index": image_index,
                        "source_path": (
                            str(source_images[image_index])
                            if image_index < len(source_images)
                            else ""
                        ),
                        "black_path": (
                            str(black_paths[image_index])
                            if image_index < len(black_paths)
                            else ""
                        ),
                        "noise_path": (
                            str(noise_paths[image_index])
                            if image_index < len(noise_paths)
                            else ""
                        ),
                        "blur_path": (
                            str(blur_paths[image_index])
                            if image_index < len(blur_paths)
                            else ""
                        ),
                    }
                )

            index[int(sample_index)] = {
                "sample_index": int(sample_index),
                "images": normalized_images,
            }

    return index


def get_prepared_image_paths(
    manifest_index: Dict[int, Dict[str, Any]],
    sample_index: int,
    perturbation: str,
    expected_number_of_images: int,
) -> List[str]:
    if perturbation not in {"black", "noise", "blur"}:
        raise ValueError(
            f"Unsupported prepared perturbation: {perturbation}"
        )

    sample = manifest_index.get(int(sample_index))

    if sample is None:
        raise KeyError(
            f"sample_index={sample_index} not found in manifest"
        )

    images = sample.get("images", [])

    if not isinstance(images, list):
        raise ValueError(
            f"Manifest sample {sample_index} has no image list"
        )

    images = sorted(
        images,
        key=lambda x: int(x.get("image_index", 0)),
    )

    path_key = f"{perturbation}_path"
    image_paths = []

    for image_record in images:
        image_path = str(
            image_record.get(path_key, "") or ""
        ).strip()

        if not image_path:
            raise KeyError(
                f"Missing {path_key} for "
                f"sample_index={sample_index}, "
                f"image_index={image_record.get('image_index')}"
            )

        image_paths.append(image_path)

    if len(image_paths) != expected_number_of_images:
        raise ValueError(
            "Image-count mismatch for "
            f"sample_index={sample_index}, "
            f"perturbation={perturbation}: "
            f"annotation={expected_number_of_images}, "
            f"manifest={len(image_paths)}"
        )

    missing_paths = [
        path
        for path in image_paths
        if not Path(path).is_file()
    ]

    if missing_paths:
        raise FileNotFoundError(
            "Prepared perturbation images are missing for "
            f"sample_index={sample_index}, "
            f"perturbation={perturbation}: "
            f"{missing_paths[:5]}"
        )

    return image_paths


def resolve_image_paths(
    clean_image_paths: List[str],
    manifest_index: Dict[int, Dict[str, Any]],
    sample_index: int,
    perturbation: str,
) -> List[str]:
    if perturbation == "clean":
        missing_paths = [
            path
            for path in clean_image_paths
            if not Path(path).is_file()
        ]

        if missing_paths:
            raise FileNotFoundError(
                f"Missing clean images: {missing_paths[:5]}"
            )

        return clean_image_paths

    return get_prepared_image_paths(
        manifest_index=manifest_index,
        sample_index=sample_index,
        perturbation=perturbation,
        expected_number_of_images=len(clean_image_paths),
    )


def make_messages(
    image_paths: List[str],
    question: str,
) -> List[Dict[str, Any]]:
    content = [
        {"type": "image", "image": image_path}
        for image_path in image_paths
    ]
    content.append({"type": "text", "text": question})

    return [{"role": "user", "content": content}]


def set_padding_side(processor, side: str) -> str:
    old_side = processor.tokenizer.padding_side
    processor.tokenizer.padding_side = side
    return old_side


def restore_padding_side(processor, old_side: str) -> None:
    processor.tokenizer.padding_side = old_side



def get_gold_answer_candidates(
    item: Dict[str, Any],
    gt_value: str,
) -> Tuple[Any, List[str]]:
    """Return the raw gold object and normalized accepted answers."""
    raw_gold: Any = None

    for key in ["answer", "gold_answer", "gt_answer"]:
        if item.get(key) not in [None, ""]:
            raw_gold = item[key]
            break

    if raw_gold is None:
        match = ANSWER_RE.search(gt_value or "")
        raw_gold = match.group(1) if match else gt_value

    if isinstance(raw_gold, list):
        candidates = [
            normalize_answer(value)
            for value in raw_gold
            if normalize_answer(value)
        ]
    else:
        normalized = normalize_answer(raw_gold)
        candidates = [normalized] if normalized else []

    # Deduplicate while preserving order.
    seen = set()
    candidates = [
        answer for answer in candidates
        if not (answer in seen or seen.add(answer))
    ]
    return raw_gold, candidates


def prediction_is_correct(
    prediction: str,
    gold_candidates: Sequence[str],
) -> bool:
    return any(
        is_correct(prediction, gold)
        for gold in gold_candidates
    )


def trim_generated_token_ids(
    token_ids: Sequence[int],
    tokenizer,
) -> List[int]:
    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None:
        eos_ids = set()
    elif isinstance(eos_token_id, (list, tuple, set)):
        eos_ids = {int(x) for x in eos_token_id}
    else:
        eos_ids = {int(eos_token_id)}

    pad_token_id = tokenizer.pad_token_id
    output: List[int] = []

    for token_id_value in token_ids:
        token_id = int(token_id_value)

        # If PAD is distinct from EOS, it is generated padding and not a token.
        if (
            pad_token_id is not None
            and token_id == int(pad_token_id)
            and token_id not in eos_ids
        ):
            break

        output.append(token_id)

        # Keep the first EOS token because pi_old assigned probability to it.
        if token_id in eos_ids:
            break

    return output


def model_visual_token_ids(model: Any, tokenizer: Any) -> List[int]:
    output: List[int] = []
    config = getattr(model, "config", None)
    for name in ("image_token_id", "video_token_id"):
        value = getattr(config, name, None)
        if isinstance(value, int) and value >= 0:
            output.append(int(value))
    for token in ("<|image_pad|>", "<|video_pad|>"):
        token_id = tokenizer.convert_tokens_to_ids(token)
        if (
            isinstance(token_id, int)
            and token_id >= 0
            and token_id != tokenizer.unk_token_id
        ):
            output.append(int(token_id))
    return list(dict.fromkeys(output))


def get_spatial_merge_size(model: Any) -> int:
    vision_config = getattr(getattr(model, "config", None), "vision_config", None)
    if isinstance(vision_config, Mapping):
        value = vision_config.get("spatial_merge_size", 2)
    else:
        value = getattr(vision_config, "spatial_merge_size", 2)
    return max(1, int(value))


def evidence_candidate_indices(
    model: Any,
    grid_thw: Sequence[int],
    expected_visual_tokens: int,
    border_ratio: float,
    border_tokens: int,
) -> List[int]:
    if len(grid_thw) != 3:
        raise ValueError(f"Expected one image grid [t,h,w], got {grid_thw}")
    grid_t, grid_h, grid_w = [int(value) for value in grid_thw]
    merge = get_spatial_merge_size(model)
    if grid_h % merge != 0 or grid_w % merge != 0:
        raise ValueError(
            "image_grid_thw is not divisible by spatial_merge_size: "
            f"grid={(grid_t, grid_h, grid_w)}, merge={merge}"
        )
    merged_h = grid_h // merge
    merged_w = grid_w // merge
    if int(border_tokens) >= 0:
        border_h = border_w = int(border_tokens)
    else:
        border_h = int(round(merged_h * float(border_ratio)))
        border_w = int(round(merged_w * float(border_ratio)))
    border_h = max(0, min(border_h, max(0, (merged_h - 1) // 2)))
    border_w = max(0, min(border_w, max(0, (merged_w - 1) // 2)))

    candidates: List[int] = []
    for temporal_index in range(grid_t):
        temporal_start = temporal_index * merged_h * merged_w
        for row in range(border_h, merged_h - border_h):
            row_start = temporal_start + row * merged_w
            for col in range(border_w, merged_w - border_w):
                candidates.append(row_start + col)

    actual_visual_tokens = grid_t * merged_h * merged_w
    if actual_visual_tokens != int(expected_visual_tokens):
        raise ValueError(
            "Rollout-time visual-token count does not match image grid: "
            f"tokens={expected_visual_tokens}, grid_tokens={actual_visual_tokens}"
        )
    if not candidates:
        raise ValueError("Evidence border removed every visual-token candidate")
    return candidates


class OnlineRolloutEvidenceRecorder:
    """Stream old-policy logP and response-to-visual attention during generate.

    Qwen3VLTextAttention already returns ``(attn_output, attn_weights)`` under
    the eager backend, although its decoder layer discards the second value.
    Hooks on the selected attention modules read that existing value directly;
    model/generate never receives ``output_attentions=True`` and therefore
    never retains an all-layer/all-step attention output. Decode calls have
    query length 1, so each hook only sees [B,H,1,K] and immediately reduces it
    to the persistent [B,V] accumulator.

    Raw sampled-token logP is gathered from each generation step's logits on
    the next call. This replaces the collector's later teacher-forced scoring
    pass. For the first response token, the attention hook consumes only the
    final prompt-query row from the prefill layer output. The full prefill map
    is never added to generate's retained output history.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        prompt_input_ids: torch.Tensor,
        image_grid_thw: torch.Tensor,
        max_new_tokens: int,
        top_ratio: float,
        border_ratio: float,
        border_tokens: int,
        attention_layers: int,
        layer_weighting: str,
        min_pixels: int,
        max_pixels: int,
    ) -> None:
        if prompt_input_ids.ndim != 2:
            raise ValueError("prompt_input_ids must be [batch, sequence]")
        if not (0.0 < float(top_ratio) <= 1.0):
            raise ValueError("--evidence-top-ratio must be in (0,1]")
        if int(attention_layers) < 1:
            raise ValueError("--evidence-attention-layers must be >= 1")

        self.model = model
        self.tokenizer = tokenizer
        self.batch_size = int(prompt_input_ids.shape[0])
        self.max_new_tokens = int(max_new_tokens)
        self.top_ratio = float(top_ratio)
        self.border_ratio = float(border_ratio)
        self.border_tokens = int(border_tokens)
        self.attention_layers = int(attention_layers)
        self.layer_weighting = str(layer_weighting)
        self.min_pixels = int(min_pixels)
        self.max_pixels = int(max_pixels)
        self.device = prompt_input_ids.device

        visual_ids = model_visual_token_ids(model, tokenizer)
        if not visual_ids:
            raise ValueError("Could not resolve Qwen visual token ids")
        visual_positions: List[torch.Tensor] = []
        for row in prompt_input_ids:
            mask = torch.zeros_like(row, dtype=torch.bool)
            for token_id in visual_ids:
                mask |= row.eq(int(token_id))
            positions = torch.nonzero(mask, as_tuple=False).flatten()
            if positions.numel() < 1:
                raise ValueError("No visual language tokens found in rollout prompt")
            visual_positions.append(positions)

        max_visual = max(int(value.numel()) for value in visual_positions)
        self.max_visual_position = max(
            int(value.max().detach().cpu()) for value in visual_positions
        )
        self.visual_index = torch.zeros(
            (self.batch_size, max_visual), dtype=torch.long, device=self.device
        )
        self.visual_valid = torch.zeros(
            (self.batch_size, max_visual), dtype=torch.float32, device=self.device
        )
        self.visual_counts: List[int] = []
        for row_index, positions in enumerate(visual_positions):
            count = int(positions.numel())
            self.visual_index[row_index, :count] = positions.to(self.device)
            self.visual_valid[row_index, :count] = 1.0
            self.visual_counts.append(count)

        grid = image_grid_thw.detach().cpu()
        if grid.ndim == 1:
            grid = grid.reshape(1, 3)
        if grid.ndim != 2 or grid.shape[-1] != 3:
            raise ValueError(f"Unexpected image_grid_thw={tuple(grid.shape)}")
        if int(grid.shape[0]) != self.batch_size:
            raise ValueError(
                "Online evidence recording currently requires one image per "
                f"expanded rollout: grids={grid.shape[0]}, batch={self.batch_size}"
            )
        self.grid_rows = [tuple(int(v) for v in row) for row in grid.tolist()]

        attention_modules: List[Tuple[int, Any]] = []
        for module in model.modules():
            class_name = module.__class__.__name__
            is_qwen_text_attention = class_name in {
                "Qwen3VLTextAttention",
                "Qwen3VLMoeTextAttention",
            }
            structural_match = (
                "TextAttention" in class_name
                and hasattr(module, "layer_idx")
                and all(
                    hasattr(module, name)
                    for name in ("q_proj", "k_proj", "v_proj", "o_proj")
                )
            )
            if is_qwen_text_attention or structural_match:
                attention_modules.append((int(module.layer_idx), module))
        attention_modules.sort(key=lambda pair: pair[0])
        if not attention_modules:
            raise RuntimeError("Could not find Qwen3-VL text-attention modules")

        selected_count = max(1, min(self.attention_layers, len(attention_modules)))
        self.selected_attention_modules = attention_modules[-selected_count:]
        self.layer_indices = [index for index, _ in self.selected_attention_modules]
        if self.layer_weighting == "uniform":
            weights = torch.ones(
                selected_count, dtype=torch.float32, device=self.device
            )
        elif self.layer_weighting == "depth_linear":
            weights = torch.tensor(
                [index + 1 for index in self.layer_indices],
                dtype=torch.float32,
                device=self.device,
            )
        else:
            raise ValueError(
                f"Unsupported evidence layer weighting: {self.layer_weighting}"
            )
        self.layer_weights = weights / weights.sum().clamp_min(1e-8)
        self.attention_layers = selected_count

        self.active = torch.ones(self.batch_size, dtype=torch.bool, device=self.device)
        self.logprob_buffer = torch.zeros(
            (self.batch_size, self.max_new_tokens),
            dtype=torch.float32,
            device=self.device,
        )
        self.logprob_counts = torch.zeros(
            self.batch_size, dtype=torch.long, device=self.device
        )
        self.attention_accumulator = torch.zeros(
            (self.batch_size, max_visual), dtype=torch.float32, device=self.device
        )
        self.attended_counts = torch.zeros(
            self.batch_size, dtype=torch.long, device=self.device
        )
        self.pending_logits: Optional[torch.Tensor] = None
        self.current_is_decode = False
        self.current_capture_attention = False
        self.current_attention_layers_seen = 0
        self._pre_handle = None
        self._post_handle = None
        self._attention_handles: List[Any] = []
        self._finished = False
        self._logprobs_cpu: Optional[torch.Tensor] = None
        self._counts_cpu: Optional[torch.Tensor] = None
        self._attention_cpu: Optional[torch.Tensor] = None
        self._attended_cpu: Optional[torch.Tensor] = None

        eos = tokenizer.eos_token_id
        if eos is None:
            self.eos_ids: List[int] = []
        elif isinstance(eos, (tuple, list, set)):
            self.eos_ids = [int(value) for value in eos]
        else:
            self.eos_ids = [int(eos)]
        self.pad_token_id = (
            int(tokenizer.pad_token_id)
            if tokenizer.pad_token_id is not None
            else None
        )

    def __enter__(self) -> "OnlineRolloutEvidenceRecorder":
        self._pre_handle = self.model.register_forward_pre_hook(
            self._pre_forward,
            with_kwargs=True,
        )
        self._post_handle = self.model.register_forward_hook(self._post_forward)
        for local_index, (_, module) in enumerate(self.selected_attention_modules):
            self._attention_handles.append(
                module.register_forward_hook(
                    self._make_attention_hook(local_index)
                )
            )
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self._pre_handle is not None:
            self._pre_handle.remove()
        if self._post_handle is not None:
            self._post_handle.remove()
        for handle in self._attention_handles:
            handle.remove()
        self._attention_handles.clear()

    def _make_attention_hook(self, local_index: int):
        def hook(module, args, output):
            del module, args
            if not self.current_capture_attention:
                return output
            attention = None
            if isinstance(output, (tuple, list)) and len(output) >= 2:
                attention = output[1]
            if not isinstance(attention, torch.Tensor) or attention.ndim != 4:
                layer_index = self.layer_indices[local_index]
                raise RuntimeError(
                    "Qwen3-VL text-attention layer did not expose a 4-D "
                    f"attention tensor at layer {layer_index}. The active "
                    "Transformers attention backend likely returns None."
                )

            key_count = int(attention.shape[-1])
            if self.max_visual_position >= key_count:
                raise RuntimeError("A visual key index exceeds generation attention length")
            mean_heads = attention[:, :, -1, :].float().mean(dim=1)
            gathered = torch.gather(mean_heads, 1, self.visual_index)
            active_float = self.active.float()[:, None]
            self.attention_accumulator += (
                self.layer_weights[local_index]
                * gathered
                * self.visual_valid
                * active_float
            )
            self.current_attention_layers_seen += 1
            return output

        return hook

    def _record_selected_tokens(self, selected_ids: torch.Tensor) -> None:
        if self.pending_logits is None:
            return
        selected_ids = selected_ids.to(self.device, dtype=torch.long)
        selected_logits = self.pending_logits.gather(
            1, selected_ids[:, None]
        ).squeeze(1).float()
        log_normalizer = torch.logsumexp(self.pending_logits.float(), dim=-1)
        token_logps = selected_logits - log_normalizer

        valid = self.active.clone()
        if self.pad_token_id is not None and self.pad_token_id not in self.eos_ids:
            valid &= selected_ids.ne(int(self.pad_token_id))
        rows = torch.nonzero(valid, as_tuple=False).flatten()
        if rows.numel() > 0:
            columns = self.logprob_counts.index_select(0, rows)
            self.logprob_buffer[rows, columns] = token_logps.index_select(0, rows)
            self.logprob_counts[rows] += 1

        stop = torch.zeros_like(self.active)
        for token_id in self.eos_ids:
            stop |= selected_ids.eq(int(token_id))
        if self.pad_token_id is not None and self.pad_token_id not in self.eos_ids:
            stop |= selected_ids.eq(int(self.pad_token_id))
        self.active &= ~stop
        self.pending_logits = None

    def _pre_forward(self, module, args, kwargs):
        del module
        input_ids = kwargs.get("input_ids")
        if input_ids is None and args:
            input_ids = args[0]
        if not isinstance(input_ids, torch.Tensor):
            self.current_is_decode = False
            self.current_capture_attention = False
            return args, kwargs

        self.current_is_decode = int(input_ids.shape[-1]) == 1
        self.current_capture_attention = True
        self.current_attention_layers_seen = 0
        if self.current_is_decode:
            self._record_selected_tokens(input_ids[:, -1])
        # Selected Qwen attention-module hooks read the layer's existing second
        # return value. Do not ask model.generate to collect/retain attentions.
        kwargs["output_attentions"] = False
        return args, kwargs

    def _post_forward(self, module, args, outputs):
        del module, args
        logits = getattr(outputs, "logits", None)
        if isinstance(logits, torch.Tensor):
            self.pending_logits = logits[:, -1, :].detach().clone()

        if self.current_capture_attention:
            expected = len(self.selected_attention_modules)
            if self.current_attention_layers_seen != expected:
                raise RuntimeError(
                    "Did not observe every selected Qwen3-VL attention layer "
                    f"during generation forward: seen={self.current_attention_layers_seen}, "
                    f"expected={expected}"
                )
            self.attended_counts += self.active.long()
        return outputs

    def finish(self, generated_sequences: torch.Tensor) -> None:
        if self._finished:
            return
        if generated_sequences.ndim != 2:
            raise ValueError("generated_sequences must be [batch, sequence]")
        # The final sampled token has no subsequent model call on which to be
        # gathered, so finalize it explicitly from the last pending logits.
        self._record_selected_tokens(generated_sequences[:, -1])
        self._logprobs_cpu = self.logprob_buffer.detach().cpu()
        self._counts_cpu = self.logprob_counts.detach().cpu()
        self._attention_cpu = self.attention_accumulator.detach().cpu()
        self._attended_cpu = self.attended_counts.detach().cpu()
        self.pending_logits = None
        self._finished = True

    def token_logprobs(self, row_index: int, expected_tokens: int) -> List[float]:
        if not self._finished or self._logprobs_cpu is None or self._counts_cpu is None:
            raise RuntimeError("Online rollout recorder has not been finalized")
        count = int(self._counts_cpu[row_index])
        if count != int(expected_tokens):
            raise RuntimeError(
                "Online sampled-token/logP length mismatch: "
                f"row={row_index}, logps={count}, tokens={expected_tokens}"
            )
        return [float(value) for value in self._logprobs_cpu[row_index, :count].tolist()]

    def evidence_trace(self, row_index: int) -> Dict[str, Any]:
        if (
            not self._finished
            or self._attention_cpu is None
            or self._attended_cpu is None
        ):
            raise RuntimeError("Online rollout recorder has not been finalized")
        visual_count = int(self.visual_counts[row_index])
        attended = int(self._attended_cpu[row_index])
        raw_map = self._attention_cpu[row_index, :visual_count].float()
        total = float(raw_map.sum())
        if attended <= 0 or total <= 0.0:
            return {
                "version": 2,
                "valid": False,
                "reason": "no_decode_attention",
                "top_visual_indices": [],
                "visual_attention_map": [],
                "visual_tokens": visual_count,
                "attended_response_tokens": attended,
                "image_grid_thw": list(self.grid_rows[row_index]),
                "min_pixels": self.min_pixels,
                "max_pixels": self.max_pixels,
            }

        attention_map = raw_map / raw_map.sum().clamp_min(1e-8)
        candidates = evidence_candidate_indices(
            model=self.model,
            grid_thw=self.grid_rows[row_index],
            expected_visual_tokens=visual_count,
            border_ratio=self.border_ratio,
            border_tokens=self.border_tokens,
        )
        candidate_tensor = torch.tensor(candidates, dtype=torch.long)
        candidate_attention = attention_map.index_select(0, candidate_tensor)
        selected_count = max(
            1,
            min(
                len(candidates),
                int(math.ceil(self.top_ratio * len(candidates))),
            ),
        )
        local_top = torch.topk(
            candidate_attention,
            k=selected_count,
            largest=True,
            sorted=True,
        ).indices
        top_indices = candidate_tensor.index_select(0, local_top).tolist()
        return {
            "version": 2,
            "valid": True,
            "selection": "online_decode_attention_topk",
            "top_visual_indices": [int(value) for value in top_indices],
            # Persist the single aggregated vector, not per-token/per-layer
            # attention matrices. This lets later Top-ratio/border grid points
            # choose a different mask on CPU without another model replay.
            "visual_attention_map": [
                round(float(value), 10) for value in attention_map.tolist()
            ],
            "visual_tokens": visual_count,
            "selected_visual_tokens": selected_count,
            "attended_response_tokens": attended,
            "attention_mass_selected": float(
                attention_map.index_select(
                    0,
                    torch.tensor(top_indices, dtype=torch.long),
                ).sum()
            ),
            "top_ratio": self.top_ratio,
            "border_ratio": self.border_ratio,
            "border_tokens": self.border_tokens,
            "attention_layers": self.attention_layers,
            "attention_layer_indices": list(self.layer_indices),
            "layer_weighting": self.layer_weighting,
            "attention_capture": "qwen_text_attention_forward_hook",
            "first_response_token_attention": "prefill_last_query_row",
            "image_grid_thw": list(self.grid_rows[row_index]),
            "min_pixels": self.min_pixels,
            "max_pixels": self.max_pixels,
        }


def generate_rollout_batch(
    model,
    processor,
    device: str,
    batch_image_paths: List[List[str]],
    batch_questions: List[str],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    rollouts_per_prompt: int,
    rollout_micro_batch_size: int,
    save_token_text: bool,
    record_evidence_topk: bool,
    evidence_top_ratio: float,
    evidence_border_ratio: float,
    evidence_border_tokens: int,
    evidence_attention_layers: int,
    evidence_layer_weighting: str,
    min_pixels: int,
    max_pixels: int,
) -> List[List[Dict[str, Any]]]:
    """
    Sample K exact token sequences per multimodal prompt.

    We explicitly duplicate each multimodal prompt before calling the processor
    instead of relying on generate(num_return_sequences=K). Generic HF input
    expansion is unsafe for flattened multimodal pixel_values tensors because
    dim-0 is usually the patch dimension, not the prompt batch dimension.
    """
    if rollouts_per_prompt < 1:
        raise ValueError("--rollouts-per-prompt must be >= 1")
    if rollout_micro_batch_size < 1:
        raise ValueError("--rollout-micro-batch-size must be >= 1")
    if rollouts_per_prompt > 1 and temperature <= 0:
        raise ValueError(
            "K > 1 requires stochastic sampling. Set --temperature > 0."
        )

    base_messages = [
        make_messages(image_paths, question)
        for image_paths, question in zip(batch_image_paths, batch_questions)
    ]
    grouped_rollouts: List[List[Dict[str, Any]]] = [
        [] for _ in batch_questions
    ]

    for rollout_start in range(
        0,
        rollouts_per_prompt,
        rollout_micro_batch_size,
    ):
        responses_this_call = min(
            rollout_micro_batch_size,
            rollouts_per_prompt - rollout_start,
        )

        expanded_messages: List[List[Dict[str, Any]]] = []
        owners: List[Tuple[int, int]] = []
        for prompt_index, messages in enumerate(base_messages):
            for local_rollout_index in range(responses_this_call):
                expanded_messages.append(messages)
                owners.append(
                    (
                        prompt_index,
                        rollout_start + local_rollout_index,
                    )
                )

        prompt_texts = [
            processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            for messages in expanded_messages
        ]
        image_inputs, video_inputs = process_vision_info(expanded_messages)

        old_padding_side = set_padding_side(processor, "left")
        try:
            model_inputs = processor(
                text=prompt_texts,
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
        finally:
            restore_padding_side(processor, old_padding_side)

        model_inputs = {
            key: value.to(device) if hasattr(value, "to") else value
            for key, value in model_inputs.items()
        }

        do_sample = temperature > 0
        generation_kwargs: Dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "num_return_sequences": 1,
            "use_cache": True,
        }
        if do_sample:
            generation_kwargs["temperature"] = temperature
            generation_kwargs["top_p"] = top_p
            if top_k > 0:
                generation_kwargs["top_k"] = top_k

        recorder: Optional[OnlineRolloutEvidenceRecorder] = None
        if record_evidence_topk:
            image_grid_thw = model_inputs.get("image_grid_thw")
            if not isinstance(image_grid_thw, torch.Tensor):
                raise ValueError(
                    "--record-evidence-topk requires image_grid_thw from processor"
                )
            recorder = OnlineRolloutEvidenceRecorder(
                model=model,
                tokenizer=processor.tokenizer,
                prompt_input_ids=model_inputs["input_ids"],
                image_grid_thw=image_grid_thw,
                max_new_tokens=max_new_tokens,
                top_ratio=evidence_top_ratio,
                border_ratio=evidence_border_ratio,
                border_tokens=evidence_border_tokens,
                attention_layers=evidence_attention_layers,
                layer_weighting=evidence_layer_weighting,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
            )

        with torch.inference_mode():
            if recorder is None:
                generated_sequences = model.generate(
                    **model_inputs,
                    **generation_kwargs,
                )
            else:
                with recorder:
                    generated_sequences = model.generate(
                        **model_inputs,
                        **generation_kwargs,
                    )
                recorder.finish(generated_sequences)

        padded_prompt_length = int(model_inputs["input_ids"].shape[1])
        if len(generated_sequences) != len(owners):
            raise RuntimeError(
                "Unexpected number of generated sequences: "
                f"got {len(generated_sequences)}, expected {len(owners)}"
            )

        for generated_row, (sequence, (prompt_index, rollout_index)) in enumerate(
            zip(generated_sequences, owners)
        ):
            token_ids = trim_generated_token_ids(
                sequence[padded_prompt_length:].detach().cpu().tolist(),
                processor.tokenizer,
            )
            raw_output = processor.tokenizer.decode(
                token_ids,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            rollout_record: Dict[str, Any] = {
                "rollout_index": rollout_index,
                "generated_token_ids": token_ids,
                "raw_output": raw_output,
            }
            if recorder is not None:
                token_logprobs = recorder.token_logprobs(
                    row_index=generated_row,
                    expected_tokens=len(token_ids),
                )
                rollout_record["old_policy_score"] = summarize_old_policy_logprobs(
                    token_ids=token_ids,
                    token_logprobs=token_logprobs,
                    tokenizer=processor.tokenizer,
                    save_token_text=save_token_text,
                )
                evidence_trace = recorder.evidence_trace(generated_row)
                if not evidence_trace.get("valid"):
                    raise RuntimeError(
                        "Direct Qwen3-VL decode attention capture produced an "
                        f"invalid trace: {evidence_trace.get('reason')}"
                    )
                rollout_record["evidence_trace"] = evidence_trace
            grouped_rollouts[prompt_index].append(rollout_record)

        del model_inputs, generated_sequences

    for group in grouped_rollouts:
        group.sort(key=lambda rollout: int(rollout["rollout_index"]))

    return grouped_rollouts


def summarize_old_policy_logprobs(
    token_ids: List[int],
    token_logprobs: List[float],
    tokenizer,
    save_token_text: bool,
) -> Dict[str, Any]:
    if len(token_ids) != len(token_logprobs):
        raise ValueError(
            f"Token/logprob length mismatch: {len(token_ids)} vs {len(token_logprobs)}"
        )

    if not token_logprobs:
        output: Dict[str, Any] = {
            "n_tokens": 0,
            "logprob_sum": None,
            "logP": None,
            "logprob_mean": None,
            "nll": None,
            "ppl": None,
            "geometric_mean_token_probability": None,
            "token_ids": [],
            "token_logprobs": [],
        }
        if save_token_text:
            output["tokens"] = []
        return output

    logprob_sum = float(sum(token_logprobs))
    logprob_mean = float(logprob_sum / len(token_logprobs))
    nll = float(-logprob_mean)
    ppl = float(math.exp(min(nll, 50.0)))

    output = {
        "n_tokens": len(token_logprobs),
        "logprob_sum": logprob_sum,
        "logP": logprob_sum,
        "logprob_mean": logprob_mean,
        "nll": nll,
        "ppl": ppl,
        "geometric_mean_token_probability": float(math.exp(logprob_mean)),
        "token_ids": [int(x) for x in token_ids],
        "token_logprobs": [float(x) for x in token_logprobs],
        "scoring_policy": "pi_old: same frozen checkpoint used for online rollout",
    }

    if save_token_text:
        raw_tokens = tokenizer.convert_ids_to_tokens(token_ids)
        output["tokens"] = [
            {
                "token_id": int(token_id),
                "raw_token": raw_token,
                "decoded_token": tokenizer.decode(
                    [token_id],
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                ),
                "logprob": float(token_logprob),
            }
            for token_id, raw_token, token_logprob in zip(
                token_ids, raw_tokens, token_logprobs
            )
        ]

    return output


def score_generated_token_ids_batch(
    model,
    processor,
    device: str,
    batch_image_paths: List[List[str]],
    batch_questions: List[str],
    batch_generated_token_ids: List[List[int]],
    save_token_text: bool,
) -> List[Dict[str, Any]]:
    """
    Teacher-force the exact sampled token ids under pi_old.

    This avoids output_scores=True during generation, which can consume several
    GB because it stores a full-vocabulary score tensor for every decoding step.
    """
    if not (
        len(batch_image_paths)
        == len(batch_questions)
        == len(batch_generated_token_ids)
    ):
        raise ValueError("Scoring batch fields have different lengths")

    messages_batch = [
        make_messages(image_paths, question)
        for image_paths, question in zip(batch_image_paths, batch_questions)
    ]
    prompt_texts = [
        processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        for messages in messages_batch
    ]
    image_inputs, video_inputs = process_vision_info(messages_batch)

    old_padding_side = set_padding_side(processor, "right")
    try:
        prompt_inputs = processor(
            text=prompt_texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
    finally:
        restore_padding_side(processor, old_padding_side)

    prompt_lengths = prompt_inputs["attention_mask"].sum(dim=1).tolist()
    pad_token_id = processor.tokenizer.pad_token_id
    if pad_token_id is None:
        raise ValueError("Tokenizer pad_token_id is None")

    full_sequences: List[torch.Tensor] = []
    for row_index, generated_ids in enumerate(batch_generated_token_ids):
        prompt_length = int(prompt_lengths[row_index])
        prompt_ids = prompt_inputs["input_ids"][row_index, :prompt_length]
        generated_tensor = torch.tensor(
            generated_ids,
            dtype=prompt_ids.dtype,
        )
        full_sequences.append(torch.cat([prompt_ids.cpu(), generated_tensor], dim=0))

    max_length = max(sequence.numel() for sequence in full_sequences)
    batch_size = len(full_sequences)
    full_input_ids = torch.full(
        (batch_size, max_length),
        fill_value=int(pad_token_id),
        dtype=prompt_inputs["input_ids"].dtype,
    )
    full_attention_mask = torch.zeros(
        (batch_size, max_length),
        dtype=prompt_inputs["attention_mask"].dtype,
    )

    for row_index, sequence in enumerate(full_sequences):
        sequence_length = int(sequence.numel())
        full_input_ids[row_index, :sequence_length] = sequence
        full_attention_mask[row_index, :sequence_length] = 1

    model_inputs: Dict[str, Any] = {
        "input_ids": full_input_ids.to(device),
        "attention_mask": full_attention_mask.to(device),
    }
    for key, value in prompt_inputs.items():
        if key in {"input_ids", "attention_mask"}:
            continue
        model_inputs[key] = value.to(device) if hasattr(value, "to") else value

    with torch.inference_mode():
        logits = model(**model_inputs).logits

    shifted_logits = logits[:, :-1, :]
    shifted_labels = model_inputs["input_ids"][:, 1:]

    # Cross entropy returns -log p(target) without materializing a full
    # log-softmax tensor, which is substantially more memory efficient.
    token_nll = F.cross_entropy(
        shifted_logits.transpose(1, 2),
        shifted_labels,
        reduction="none",
    )
    selected_logprobs = -token_nll

    batch_scores: List[Dict[str, Any]] = []
    for row_index, generated_ids in enumerate(batch_generated_token_ids):
        prompt_length = int(prompt_lengths[row_index])
        generated_length = len(generated_ids)
        shifted_start = max(prompt_length - 1, 0)
        shifted_end = shifted_start + generated_length

        logprobs = (
            selected_logprobs[row_index, shifted_start:shifted_end]
            .detach()
            .float()
            .cpu()
            .tolist()
        )
        batch_scores.append(
            summarize_old_policy_logprobs(
                token_ids=generated_ids,
                token_logprobs=logprobs,
                tokenizer=processor.tokenizer,
                save_token_text=save_token_text,
            )
        )

    del logits, shifted_logits, token_nll, selected_logprobs
    return batch_scores


def score_flat_rollouts(
    model,
    processor,
    device: str,
    flat_rollouts: List[Dict[str, Any]],
    score_batch_size: int,
    save_token_text: bool,
) -> None:
    for batch_start in range(0, len(flat_rollouts), score_batch_size):
        chunk = flat_rollouts[batch_start:batch_start + score_batch_size]
        scores = score_generated_token_ids_batch(
            model=model,
            processor=processor,
            device=device,
            batch_image_paths=[entry["image_paths"] for entry in chunk],
            batch_questions=[entry["question"] for entry in chunk],
            batch_generated_token_ids=[
                entry["rollout"]["generated_token_ids"] for entry in chunk
            ],
            save_token_text=save_token_text,
        )
        for entry, score in zip(chunk, scores):
            entry["rollout"]["old_policy_score"] = score


def finalize_perturbation_record(
    item: Dict[str, Any],
    record: Dict[str, Any],
) -> None:
    rollouts = record.get("rollouts", [])
    for rollout in rollouts:
        predicted_answer = extract_answer(rollout.get("raw_output", ""))
        correct = prediction_is_correct(
            predicted_answer,
            item.get("gold_answers", []),
        )
        rollout["pred_answer"] = predicted_answer
        rollout["reward"] = int(correct)
        rollout["is_correct"] = bool(correct)

    rewards = [int(rollout["reward"]) for rollout in rollouts]
    number_positive = sum(rewards)
    number_rollouts = len(rewards)
    r_hat = (
        float(number_positive / number_rollouts)
        if number_rollouts > 0
        else None
    )

    record["num_rollouts"] = number_rollouts
    record["num_positive"] = number_positive
    record["num_negative"] = number_rollouts - number_positive
    record["r_hat"] = r_hat
    record["correctness_rate"] = r_hat
    record["positive_rollout_indices"] = [
        rollout["rollout_index"]
        for rollout in rollouts
        if rollout["reward"] == 1
    ]
    record["negative_rollout_indices"] = [
        rollout["rollout_index"]
        for rollout in rollouts
        if rollout["reward"] == 0
    ]

    requested = int(record.get("num_requested_rollouts", 0))
    complete = number_rollouts == requested and all(
        isinstance(rollout.get("old_policy_score"), dict)
        and rollout["old_policy_score"].get("logprob_mean") is not None
        for rollout in rollouts
    )

    if not complete:
        record["nft_prompt_eligible"] = False
        record["filter_reason"] = "incomplete_rollouts_or_scores"
    elif r_hat == 0.0:
        record["nft_prompt_eligible"] = False
        record["filter_reason"] = "all_negative"
    elif r_hat == 1.0:
        record["nft_prompt_eligible"] = False
        record["filter_reason"] = "all_positive"
    else:
        record["nft_prompt_eligible"] = True
        record["filter_reason"] = None


def perturbation_is_complete(
    record: Any,
    expected_rollouts: int,
) -> bool:
    if not isinstance(record, dict):
        return False
    if record.get("missing_image") or record.get("generation_error"):
        return False
    rollouts = record.get("rollouts")
    if not isinstance(rollouts, list) or len(rollouts) != expected_rollouts:
        return False
    if record.get("r_hat") is None:
        return False
    return all(
        isinstance(rollout.get("old_policy_score"), dict)
        and rollout["old_policy_score"].get("logprob_mean") is not None
        and len(rollout["old_policy_score"].get("token_logprobs", []))
        == len(rollout.get("generated_token_ids", []))
        for rollout in rollouts
    )


def make_base_record(
    item: Dict[str, Any],
    sample_index: int,
    dataset: str,
) -> Dict[str, Any]:
    question = get_question(item)
    gt_value = get_gt_value(item)
    image_paths = get_image_paths(item)
    gold_raw, gold_answers = get_gold_answer_candidates(item, gt_value)

    return {
        "dataset": dataset,
        "mode": "CoT",
        "collection_type": "online_nft_rollout",
        "sample_index": sample_index,
        "idx": item.get("idx"),
        "question": question,
        "gold_answer_raw": gold_raw,
        "gold_answers": gold_answers,
        "gold_answer": gold_answers[0] if gold_answers else "",
        "gt_value": gt_value,
        "image_paths": image_paths,
        "image_path": image_paths[0] if image_paths else "",
        "num_images": len(image_paths),
        "image_ids": item.get("image_ids"),
        "question_type": item.get("question_type"),
        "semantic_type": item.get("semantic_type"),
        "answer_type": item.get("answer_type"),
        "subject_id": item.get("subject_id"),
        "template": item.get("template"),
        "template_arguments": item.get("template_arguments"),
        "template_program": item.get("template_program"),
        "perturbation_results": {},
    }


def build_collection_config(
    args: argparse.Namespace,
    perturbations: List[str],
) -> Dict[str, Any]:
    return {
        "base_model": str(Path(args.base_model).resolve()),
        "annotation_path": str(Path(args.annotation_path).resolve()),
        "dataset": args.dataset,
        "perturbations": perturbations,
        "rollouts_per_prompt": args.rollouts_per_prompt,
        "rollout_micro_batch_size": args.rollout_micro_batch_size,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "max_new_tokens": args.max_new_tokens,
        "min_pixels": args.min_pixels,
        "max_pixels": args.max_pixels,
        "record_evidence_topk": bool(args.record_evidence_topk),
        "evidence_trace_version": 2 if bool(args.record_evidence_topk) else 0,
        "evidence_top_ratio": args.evidence_top_ratio,
        "evidence_border_ratio": args.evidence_border_ratio,
        "evidence_border_tokens": args.evidence_border_tokens,
        "evidence_attention_layers": args.evidence_attention_layers,
        "evidence_layer_weighting": args.evidence_layer_weighting,
        "seed": args.seed,
        "reward_policy": f"binary {args.answer_matching} match against normalized gold candidates",
        "grouping_policy": "separate K-rollout group for each (sample, perturbation)",
    }


def build_summary(
    results: List[Dict[str, Any]],
    dataset: str,
    perturbations: List[str],
) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "dataset": dataset,
        "mode": "CoT",
        "collection_type": "online_nft_rollout",
        "total_items": len(results),
        "perturbations": {},
    }

    for perturbation in perturbations:
        records = [
            item.get("perturbation_results", {}).get(perturbation)
            for item in results
        ]
        records = [record for record in records if isinstance(record, dict)]
        completed = [
            record for record in records
            if record.get("r_hat") is not None
            and not record.get("generation_error")
            and not record.get("missing_image")
        ]
        all_rollouts = [
            rollout
            for record in completed
            for rollout in record.get("rollouts", [])
        ]
        total_correct = sum(int(x.get("reward", 0)) for x in all_rollouts)
        ppls = [
            float(x["old_policy_score"]["ppl"])
            for x in all_rollouts
            if isinstance(x.get("old_policy_score"), dict)
            and x["old_policy_score"].get("ppl") is not None
        ]
        nlls = [
            float(x["old_policy_score"]["nll"])
            for x in all_rollouts
            if isinstance(x.get("old_policy_score"), dict)
            and x["old_policy_score"].get("nll") is not None
        ]

        summary["perturbations"][perturbation] = {
            "prompt_groups_total": len(records),
            "prompt_groups_completed": len(completed),
            "prompt_groups_nft_eligible": sum(
                1 for record in completed if record.get("nft_prompt_eligible")
            ),
            "prompt_groups_all_positive": sum(
                1 for record in completed if record.get("filter_reason") == "all_positive"
            ),
            "prompt_groups_all_negative": sum(
                1 for record in completed if record.get("filter_reason") == "all_negative"
            ),
            "rollouts_total": len(all_rollouts),
            "rollouts_correct": total_correct,
            "rollout_accuracy": (
                total_correct / len(all_rollouts) if all_rollouts else None
            ),
            "mean_r_hat": (
                sum(float(record["r_hat"]) for record in completed) / len(completed)
                if completed else None
            ),
            "mean_rollout_ppl": sum(ppls) / len(ppls) if ppls else None,
            "mean_rollout_nll": sum(nlls) / len(nlls) if nlls else None,
            "missing_or_error": sum(
                1 for record in records
                if record.get("missing_image") or record.get("generation_error")
            ),
        }

    return summary


def atomic_dump(
    path: Path,
    config: Dict[str, Any],
    summary: Dict[str, Any],
    results: List[Dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(
            {
                "collection_config": config,
                "summary": summary,
                "results": results,
            },
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_path, path)


def restore_existing_results(
    output_path: Path,
    new_results: List[Dict[str, Any]],
    collection_config: Dict[str, Any],
    expected_rollouts: int,
) -> Tuple[int, int]:
    if not output_path.is_file():
        return 0, 0

    try:
        old_data = json.loads(output_path.read_text(encoding="utf-8"))
    except Exception as error:
        print(f"[WARN] Failed to read existing output for resume: {error}")
        return 0, 0

    old_config = old_data.get("collection_config")
    if old_config != collection_config:
        print("[WARN] Existing output config differs; resume is disabled for safety.")
        return 0, 0

    old_results = old_data.get("results", [])
    old_by_sample_index = {
        int(item["sample_index"]): item
        for item in old_results
        if isinstance(item, dict) and item.get("sample_index") is not None
    }

    matched_records = 0
    restored_groups = 0
    for new_item in new_results:
        old_item = old_by_sample_index.get(int(new_item["sample_index"]))
        if old_item is None:
            continue
        matched_records += 1
        for perturbation, record in old_item.get("perturbation_results", {}).items():
            if perturbation_is_complete(record, expected_rollouts):
                new_item["perturbation_results"][perturbation] = record
                restored_groups += 1

    return matched_records, restored_groups


def load_model(args: argparse.Namespace):
    dtype_map = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.base_model,
        torch_dtype=dtype_map[args.dtype],
        attn_implementation=args.attn_impl,
        device_map=args.device,
    )
    model.eval()
    return model


def main() -> None:
    args = parse_args()

    if (
        args.batch_size < 1
        or args.score_batch_size < 1
        or args.rollout_micro_batch_size < 1
    ):
        raise ValueError("Batch sizes must be >= 1")
    if args.rollouts_per_prompt < 1:
        raise ValueError("--rollouts-per-prompt must be >= 1")
    if args.rollouts_per_prompt > 1 and args.temperature <= 0:
        raise ValueError("K > 1 requires --temperature > 0")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    perturbations = [
        name.strip()
        for name in args.perturbations.split(",")
        if name.strip()
    ]
    allowed_perturbations = {"clean", "black", "noise", "blur"}
    unsupported = [name for name in perturbations if name not in allowed_perturbations]
    if unsupported:
        raise ValueError(
            f"Unsupported perturbations: {unsupported}. Allowed: {sorted(allowed_perturbations)}"
        )

    data = load_json_list(args.annotation_path)
    if args.limit > 0:
        data = data[:args.limit]
    check_single_image_samples(data)

    manifest_path = Path(args.perturb_root) / args.dataset / args.manifest_name
    manifest = load_manifest(manifest_path)
    manifest_index = build_manifest_index(manifest)

    results = [
        make_base_record(item, sample_index, args.dataset)
        for sample_index, item in enumerate(data)
    ]
    output_path = Path(args.output)
    collection_config = build_collection_config(args, perturbations)

    if args.resume:
        matched, restored = restore_existing_results(
            output_path=output_path,
            new_results=results,
            collection_config=collection_config,
            expected_rollouts=args.rollouts_per_prompt,
        )
        print(f"[RESUME] matched_records={matched}, restored_groups={restored}")

    print(f"[INFO] model={args.base_model}")
    print(f"[INFO] dataset={args.dataset}")
    print(f"[INFO] samples={len(results)}")
    print(f"[INFO] perturbations={perturbations}")
    print(f"[INFO] K={args.rollouts_per_prompt}")
    print(f"[INFO] temperature={args.temperature}, top_p={args.top_p}, top_k={args.top_k}")
    print(f"[INFO] generation_prompt_batch_size={args.batch_size}")
    print(f"[INFO] rollout_micro_batch_size={args.rollout_micro_batch_size}")
    print(f"[INFO] score_batch_size={args.score_batch_size}")
    print(f"[INFO] manifest={manifest_path}")
    print("[INFO] pi_old token logprobs: exact generated token ids, teacher-forced before any update")

    model = load_model(args)
    processor = AutoProcessor.from_pretrained(args.base_model)
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    processor.tokenizer.padding_side = "right"

    try:
        processor.image_processor.min_pixels = args.min_pixels
        processor.image_processor.max_pixels = args.max_pixels
        processor.image_processor.size["shortest_edge"] = args.min_pixels
        processor.image_processor.size["longest_edge"] = args.max_pixels
    except Exception as error:
        print(f"[WARN] Could not set image processor pixel limits: {error}")

    completed_since_save = 0

    for perturbation in perturbations:
        pending_indices = [
            sample_index
            for sample_index, item in enumerate(results)
            if not perturbation_is_complete(
                item.get("perturbation_results", {}).get(perturbation),
                args.rollouts_per_prompt,
            )
        ]

        print(
            f"[INFO] perturbation={perturbation}, "
            f"pending={len(pending_indices)}/{len(results)}"
        )
        progress = tqdm(
            total=len(pending_indices),
            desc=f"{args.dataset}:{perturbation}",
            unit="prompt",
            dynamic_ncols=True,
        )

        for batch_start in range(0, len(pending_indices), args.batch_size):
            batch_indices = pending_indices[batch_start:batch_start + args.batch_size]
            valid_indices: List[int] = []
            batch_image_paths: List[List[str]] = []
            batch_questions: List[str] = []

            for sample_index in batch_indices:
                item = results[sample_index]
                record = {
                    "perturbation": perturbation,
                    "used_image_paths": [],
                    "num_requested_rollouts": args.rollouts_per_prompt,
                    "num_rollouts": 0,
                    "num_positive": 0,
                    "num_negative": 0,
                    "r_hat": None,
                    "correctness_rate": None,
                    "nft_prompt_eligible": False,
                    "filter_reason": "not_processed",
                    "positive_rollout_indices": [],
                    "negative_rollout_indices": [],
                    "rollouts": [],
                    "missing_image": False,
                    "generation_error": None,
                }
                item["perturbation_results"][perturbation] = record

                try:
                    used_image_paths = resolve_image_paths(
                        clean_image_paths=item["image_paths"],
                        manifest_index=manifest_index,
                        sample_index=sample_index,
                        perturbation=perturbation,
                    )
                    record["used_image_paths"] = used_image_paths
                    valid_indices.append(sample_index)
                    batch_image_paths.append(used_image_paths)
                    batch_questions.append(item["question"])
                except FileNotFoundError as error:
                    record["missing_image"] = True
                    record["generation_error"] = repr(error)
                    record["filter_reason"] = "missing_image"
                    completed_since_save += 1
                except Exception as error:
                    record["generation_error"] = repr(error)
                    record["filter_reason"] = "input_resolution_error"
                    completed_since_save += 1

            if valid_indices:
                try:
                    grouped_rollouts = generate_rollout_batch(
                        model=model,
                        processor=processor,
                        device=args.device,
                        batch_image_paths=batch_image_paths,
                        batch_questions=batch_questions,
                        max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        top_k=args.top_k,
                        rollouts_per_prompt=args.rollouts_per_prompt,
                        rollout_micro_batch_size=args.rollout_micro_batch_size,
                        save_token_text=args.save_token_text,
                        record_evidence_topk=bool(args.record_evidence_topk),
                        evidence_top_ratio=args.evidence_top_ratio,
                        evidence_border_ratio=args.evidence_border_ratio,
                        evidence_border_tokens=args.evidence_border_tokens,
                        evidence_attention_layers=args.evidence_attention_layers,
                        evidence_layer_weighting=args.evidence_layer_weighting,
                        min_pixels=args.min_pixels,
                        max_pixels=args.max_pixels,
                    )

                    flat_for_scoring: List[Dict[str, Any]] = []
                    for local_index, sample_index in enumerate(valid_indices):
                        item = results[sample_index]
                        record = item["perturbation_results"][perturbation]
                        record["rollouts"] = grouped_rollouts[local_index]
                        if not bool(args.record_evidence_topk):
                            for rollout in record["rollouts"]:
                                flat_for_scoring.append(
                                    {
                                        "sample_index": sample_index,
                                        "image_paths": batch_image_paths[local_index],
                                        "question": batch_questions[local_index],
                                        "rollout": rollout,
                                    }
                                )

                    if not bool(args.record_evidence_topk):
                        score_flat_rollouts(
                            model=model,
                            processor=processor,
                            device=args.device,
                            flat_rollouts=flat_for_scoring,
                            score_batch_size=args.score_batch_size,
                            save_token_text=args.save_token_text,
                        )

                    for sample_index in valid_indices:
                        item = results[sample_index]
                        record = item["perturbation_results"][perturbation]
                        finalize_perturbation_record(item, record)
                        completed_since_save += 1

                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    raise RuntimeError(
                        "CUDA out of memory. Reduce BATCH_SIZE, SCORE_BATCH_SIZE, "
                        "ROLLOUTS_PER_PROMPT, MAX_NEW_TOKENS, or image pixel limits."
                    )
                except Exception as error:
                    error_text = repr(error)
                    for sample_index in valid_indices:
                        record = results[sample_index]["perturbation_results"][perturbation]
                        record["generation_error"] = error_text
                        record["filter_reason"] = "generation_or_scoring_error"
                        completed_since_save += 1

            progress.update(len(batch_indices))

            if completed_since_save >= args.save_every:
                summary = build_summary(results, args.dataset, perturbations)
                atomic_dump(output_path, collection_config, summary, results)
                completed_since_save = 0

        progress.close()
        summary = build_summary(results, args.dataset, perturbations)
        atomic_dump(output_path, collection_config, summary, results)

    summary = build_summary(results, args.dataset, perturbations)
    atomic_dump(output_path, collection_config, summary, results)
    print(json.dumps(summary, ensure_ascii=False, indent=2))



# ---------------------------------------------------------------------------
# Streaming SQLite storage
# ---------------------------------------------------------------------------

SCHEMA_VERSION = 3


def json_compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def open_rollout_db(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(path), timeout=120.0)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute("PRAGMA temp_store=MEMORY")
    connection.execute("PRAGMA busy_timeout=120000")
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS meta (
            key TEXT PRIMARY KEY,
            value_json TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS samples (
            sample_index INTEGER PRIMARY KEY,
            base_record_json TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS rollout_groups (
            sample_index INTEGER NOT NULL,
            perturbation TEXT NOT NULL,
            complete INTEGER NOT NULL DEFAULT 0,
            missing_image INTEGER NOT NULL DEFAULT 0,
            generation_error TEXT,
            r_hat REAL,
            num_rollouts INTEGER NOT NULL DEFAULT 0,
            num_positive INTEGER NOT NULL DEFAULT 0,
            num_negative INTEGER NOT NULL DEFAULT 0,
            nft_prompt_eligible INTEGER NOT NULL DEFAULT 0,
            filter_reason TEXT,
            rollout_correct INTEGER NOT NULL DEFAULT 0,
            ppl_sum REAL NOT NULL DEFAULT 0.0,
            ppl_count INTEGER NOT NULL DEFAULT 0,
            nll_sum REAL NOT NULL DEFAULT 0.0,
            nll_count INTEGER NOT NULL DEFAULT 0,
            record_json TEXT NOT NULL,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (sample_index, perturbation)
        );

        CREATE INDEX IF NOT EXISTS idx_rollout_groups_complete
            ON rollout_groups(perturbation, complete, sample_index);
        """
    )
    return connection


def get_meta(connection: sqlite3.Connection, key: str) -> Any:
    row = connection.execute(
        "SELECT value_json FROM meta WHERE key = ?",
        (key,),
    ).fetchone()
    if row is None:
        return None
    return json.loads(row[0])


def set_meta(connection: sqlite3.Connection, key: str, value: Any) -> None:
    connection.execute(
        """
        INSERT INTO meta(key, value_json) VALUES(?, ?)
        ON CONFLICT(key) DO UPDATE SET value_json = excluded.value_json
        """,
        (key, json_compact(value)),
    )


def initialize_rollout_db(
    connection: sqlite3.Connection,
    collection_config: Dict[str, Any],
    data: Sequence[Dict[str, Any]],
    dataset: str,
    resume: bool,
) -> None:
    existing_config = get_meta(connection, "collection_config")

    if not resume:
        connection.execute("DELETE FROM rollout_groups")
        connection.execute("DELETE FROM samples")
        connection.execute("DELETE FROM meta")
        existing_config = None

    def semantic_config(config: Dict[str, Any]) -> Dict[str, Any]:
        # Execution-only knobs may safely change across resume runs. They do
        # not change the sampled distribution or the meaning of stored rows.
        ignored_keys = {
            "rollout_micro_batch_size",
        }
        return {
            key: value
            for key, value in config.items()
            if key not in ignored_keys
        }

    if (
        existing_config is not None
        and semantic_config(existing_config) != semantic_config(collection_config)
    ):
        raise ValueError(
            "Existing SQLite collection_config differs in data-semantic "
            "parameters. Use a new output path, restore the original model/"
            "sampling parameters, or run with --resume 0 to reset this database."
        )

    if existing_config is None:
        set_meta(connection, "schema_version", SCHEMA_VERSION)
        set_meta(connection, "collection_config", collection_config)
        set_meta(connection, "dataset", dataset)
        set_meta(connection, "total_items", len(data))
    else:
        set_meta(connection, "collection_config", collection_config)
        set_meta(connection, "total_items", len(data))

    existing_sample_count = int(
        connection.execute("SELECT COUNT(*) FROM samples").fetchone()[0]
    )
    if existing_sample_count != len(data):
        def iter_sample_rows():
            for sample_index, annotation_item in enumerate(data):
                base_record = make_base_record(
                    annotation_item, sample_index, dataset
                )
                base_record.pop("perturbation_results", None)
                yield sample_index, json_compact(base_record)

        connection.executemany(
            """
            INSERT INTO samples(sample_index, base_record_json) VALUES(?, ?)
            ON CONFLICT(sample_index) DO UPDATE SET
                base_record_json = excluded.base_record_json
            """,
            iter_sample_rows(),
        )

    connection.commit()


def record_storage_metrics(
    record: Dict[str, Any],
    expected_rollouts: int,
) -> Tuple[Any, ...]:
    rollouts = record.get("rollouts", [])
    if not isinstance(rollouts, list):
        rollouts = []

    ppl_values: List[float] = []
    nll_values: List[float] = []
    rollout_correct = 0
    for rollout in rollouts:
        rollout_correct += int(rollout.get("reward", 0))
        score = rollout.get("old_policy_score")
        if not isinstance(score, dict):
            continue
        if score.get("ppl") is not None:
            ppl_values.append(float(score["ppl"]))
        if score.get("nll") is not None:
            nll_values.append(float(score["nll"]))

    complete = int(perturbation_is_complete(record, expected_rollouts))
    generation_error = record.get("generation_error")
    if generation_error is not None:
        generation_error = str(generation_error)

    return (
        complete,
        int(bool(record.get("missing_image"))),
        generation_error,
        record.get("r_hat"),
        int(record.get("num_rollouts", len(rollouts)) or 0),
        int(record.get("num_positive", rollout_correct) or 0),
        int(record.get("num_negative", len(rollouts) - rollout_correct) or 0),
        int(bool(record.get("nft_prompt_eligible"))),
        record.get("filter_reason"),
        rollout_correct,
        float(sum(ppl_values)),
        len(ppl_values),
        float(sum(nll_values)),
        len(nll_values),
    )


def upsert_rollout_groups(
    connection: sqlite3.Connection,
    records: Sequence[Tuple[int, str, Dict[str, Any]]],
    expected_rollouts: int,
) -> None:
    if not records:
        return

    rows = []
    for sample_index, perturbation, record in records:
        metrics = record_storage_metrics(record, expected_rollouts)
        rows.append(
            (
                int(sample_index),
                perturbation,
                *metrics,
                json_compact(record),
            )
        )

    with connection:
        connection.executemany(
            """
            INSERT INTO rollout_groups(
                sample_index,
                perturbation,
                complete,
                missing_image,
                generation_error,
                r_hat,
                num_rollouts,
                num_positive,
                num_negative,
                nft_prompt_eligible,
                filter_reason,
                rollout_correct,
                ppl_sum,
                ppl_count,
                nll_sum,
                nll_count,
                record_json,
                updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(sample_index, perturbation) DO UPDATE SET
                complete = excluded.complete,
                missing_image = excluded.missing_image,
                generation_error = excluded.generation_error,
                r_hat = excluded.r_hat,
                num_rollouts = excluded.num_rollouts,
                num_positive = excluded.num_positive,
                num_negative = excluded.num_negative,
                nft_prompt_eligible = excluded.nft_prompt_eligible,
                filter_reason = excluded.filter_reason,
                rollout_correct = excluded.rollout_correct,
                ppl_sum = excluded.ppl_sum,
                ppl_count = excluded.ppl_count,
                nll_sum = excluded.nll_sum,
                nll_count = excluded.nll_count,
                record_json = excluded.record_json,
                updated_at = CURRENT_TIMESTAMP
            """,
            rows,
        )


def get_completed_indices(
    connection: sqlite3.Connection,
    perturbation: str,
) -> set:
    return {
        int(row[0])
        for row in connection.execute(
            """
            SELECT sample_index
            FROM rollout_groups
            WHERE perturbation = ? AND complete = 1
            """,
            (perturbation,),
        )
    }


def build_sqlite_summary(
    connection: sqlite3.Connection,
    dataset: str,
    perturbations: Sequence[str],
    total_items: int,
) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "dataset": dataset,
        "mode": "CoT",
        "collection_type": "online_nft_rollout",
        "storage_format": "sqlite",
        "total_items": total_items,
        "perturbations": {},
    }

    for perturbation in perturbations:
        row = connection.execute(
            """
            SELECT
                COUNT(*),
                COALESCE(SUM(complete), 0),
                COALESCE(SUM(CASE WHEN complete = 1 THEN nft_prompt_eligible ELSE 0 END), 0),
                COALESCE(SUM(CASE WHEN complete = 1 AND filter_reason = 'all_positive' THEN 1 ELSE 0 END), 0),
                COALESCE(SUM(CASE WHEN complete = 1 AND filter_reason = 'all_negative' THEN 1 ELSE 0 END), 0),
                COALESCE(SUM(CASE WHEN complete = 1 THEN num_rollouts ELSE 0 END), 0),
                COALESCE(SUM(CASE WHEN complete = 1 THEN rollout_correct ELSE 0 END), 0),
                AVG(CASE WHEN complete = 1 THEN r_hat END),
                COALESCE(SUM(CASE WHEN complete = 1 THEN ppl_sum ELSE 0 END), 0.0),
                COALESCE(SUM(CASE WHEN complete = 1 THEN ppl_count ELSE 0 END), 0),
                COALESCE(SUM(CASE WHEN complete = 1 THEN nll_sum ELSE 0 END), 0.0),
                COALESCE(SUM(CASE WHEN complete = 1 THEN nll_count ELSE 0 END), 0),
                COALESCE(SUM(CASE WHEN missing_image = 1 OR generation_error IS NOT NULL THEN 1 ELSE 0 END), 0)
            FROM rollout_groups
            WHERE perturbation = ?
            """,
            (perturbation,),
        ).fetchone()

        (
            groups_total,
            groups_completed,
            groups_eligible,
            groups_all_positive,
            groups_all_negative,
            rollouts_total,
            rollouts_correct,
            mean_r_hat,
            ppl_sum,
            ppl_count,
            nll_sum,
            nll_count,
            missing_or_error,
        ) = row

        summary["perturbations"][perturbation] = {
            "prompt_groups_total": int(groups_total),
            "prompt_groups_completed": int(groups_completed),
            "prompt_groups_pending": int(total_items - groups_completed),
            "prompt_groups_nft_eligible": int(groups_eligible),
            "prompt_groups_all_positive": int(groups_all_positive),
            "prompt_groups_all_negative": int(groups_all_negative),
            "rollouts_total": int(rollouts_total),
            "rollouts_correct": int(rollouts_correct),
            "rollout_accuracy": (
                float(rollouts_correct / rollouts_total)
                if rollouts_total
                else None
            ),
            "mean_r_hat": float(mean_r_hat) if mean_r_hat is not None else None,
            "mean_rollout_ppl": (
                float(ppl_sum / ppl_count) if ppl_count else None
            ),
            "mean_rollout_nll": (
                float(nll_sum / nll_count) if nll_count else None
            ),
            "missing_or_error": int(missing_or_error),
        }

    return summary


def main_sqlite() -> None:
    global ANSWER_MATCHING
    args = parse_args()
    ANSWER_MATCHING = args.answer_matching

    if (
        args.batch_size < 1
        or args.score_batch_size < 1
        or args.rollout_micro_batch_size < 1
    ):
        raise ValueError("Batch sizes must be >= 1")
    if args.rollouts_per_prompt < 1:
        raise ValueError("--rollouts-per-prompt must be >= 1")
    if args.rollouts_per_prompt > 1 and args.temperature <= 0:
        raise ValueError("K > 1 requires --temperature > 0")
    if bool(args.record_evidence_topk):
        if str(args.attn_impl).lower() != "eager":
            raise ValueError(
                "--record-evidence-topk requires --attn-impl eager so decode "
                "attention tensors are available"
            )
        if not (0.0 < float(args.evidence_top_ratio) <= 1.0):
            raise ValueError("--evidence-top-ratio must be in (0,1]")
        if int(args.evidence_attention_layers) < 1:
            raise ValueError("--evidence-attention-layers must be >= 1")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    perturbations = [
        name.strip()
        for name in args.perturbations.split(",")
        if name.strip()
    ]
    allowed_perturbations = {"clean", "black", "noise", "blur"}
    unsupported = [
        name for name in perturbations
        if name not in allowed_perturbations
    ]
    if unsupported:
        raise ValueError(
            f"Unsupported perturbations: {unsupported}. "
            f"Allowed: {sorted(allowed_perturbations)}"
        )

    data = load_json_list(args.annotation_path)
    if args.limit > 0:
        data = data[:args.limit]
    check_single_image_samples(data)

    manifest_path = Path(args.perturb_root) / args.dataset / args.manifest_name
    manifest = (load_manifest(manifest_path) if any(name != "clean" for name in perturbations)
                else {"samples": []})
    manifest_index = build_manifest_index(manifest)
    output_path = Path(args.output)
    collection_config = build_collection_config(args, perturbations)

    connection = open_rollout_db(output_path)
    try:
        initialize_rollout_db(
            connection=connection,
            collection_config=collection_config,
            data=data,
            dataset=args.dataset,
            resume=bool(args.resume),
        )

        restored_groups = int(
            connection.execute(
                "SELECT COUNT(*) FROM rollout_groups WHERE complete = 1"
            ).fetchone()[0]
        )
        print(f"[RESUME] restored_complete_groups={restored_groups}")
        print(f"[INFO] storage=SQLite WAL: {output_path}")
        print("[INFO] completed groups are committed after every outer batch")
        print(f"[INFO] model={args.base_model}")
        print(f"[INFO] dataset={args.dataset}")
        print(f"[INFO] samples={len(data)}")
        print(f"[INFO] perturbations={perturbations}")
        print(f"[INFO] K={args.rollouts_per_prompt}")
        print(
            f"[INFO] temperature={args.temperature}, "
            f"top_p={args.top_p}, top_k={args.top_k}"
        )
        print(f"[INFO] generation_prompt_batch_size={args.batch_size}")
        print(f"[INFO] rollout_micro_batch_size={args.rollout_micro_batch_size}")
        print(f"[INFO] score_batch_size={args.score_batch_size}")
        print(
            "[INFO] online_evidence_topk="
            f"{bool(args.record_evidence_topk)}, "
            f"top_ratio={args.evidence_top_ratio}, "
            f"attention_layers={args.evidence_attention_layers}"
        )
        print(f"[INFO] manifest={manifest_path}")

        model = load_model(args)
        processor = AutoProcessor.from_pretrained(args.base_model)
        if processor.tokenizer.pad_token_id is None:
            processor.tokenizer.pad_token = processor.tokenizer.eos_token
        processor.tokenizer.padding_side = "right"

        try:
            processor.image_processor.min_pixels = args.min_pixels
            processor.image_processor.max_pixels = args.max_pixels
            processor.image_processor.size["shortest_edge"] = args.min_pixels
            processor.image_processor.size["longest_edge"] = args.max_pixels
        except Exception as error:
            print(f"[WARN] Could not set image processor pixel limits: {error}")

        completed_since_summary = 0

        for perturbation in perturbations:
            completed_indices = get_completed_indices(connection, perturbation)
            pending_indices = [
                sample_index
                for sample_index in range(len(data))
                if sample_index not in completed_indices
            ]

            print(
                f"[INFO] perturbation={perturbation}, "
                f"pending={len(pending_indices)}/{len(data)}"
            )
            progress = tqdm(
                total=len(pending_indices),
                desc=f"{args.dataset}:{perturbation}",
                unit="prompt",
                dynamic_ncols=True,
            )

            for batch_start in range(0, len(pending_indices), args.batch_size):
                batch_indices = pending_indices[
                    batch_start:batch_start + args.batch_size
                ]
                valid_indices: List[int] = []
                batch_image_paths: List[List[str]] = []
                batch_questions: List[str] = []
                base_items: Dict[int, Dict[str, Any]] = {}
                records: Dict[int, Dict[str, Any]] = {}

                for sample_index in batch_indices:
                    item = make_base_record(
                        data[sample_index], sample_index, args.dataset
                    )
                    base_items[sample_index] = item
                    record: Dict[str, Any] = {
                        "perturbation": perturbation,
                        "used_image_paths": [],
                        "num_requested_rollouts": args.rollouts_per_prompt,
                        "num_rollouts": 0,
                        "num_positive": 0,
                        "num_negative": 0,
                        "r_hat": None,
                        "correctness_rate": None,
                        "nft_prompt_eligible": False,
                        "filter_reason": "not_processed",
                        "positive_rollout_indices": [],
                        "negative_rollout_indices": [],
                        "rollouts": [],
                        "missing_image": False,
                        "generation_error": None,
                    }
                    records[sample_index] = record

                    try:
                        used_image_paths = resolve_image_paths(
                            clean_image_paths=item["image_paths"],
                            manifest_index=manifest_index,
                            sample_index=sample_index,
                            perturbation=perturbation,
                        )
                        record["used_image_paths"] = used_image_paths
                        valid_indices.append(sample_index)
                        batch_image_paths.append(used_image_paths)
                        batch_questions.append(item["question"])
                    except FileNotFoundError as error:
                        record["missing_image"] = True
                        record["generation_error"] = repr(error)
                        record["filter_reason"] = "missing_image"
                    except Exception as error:
                        record["generation_error"] = repr(error)
                        record["filter_reason"] = "input_resolution_error"

                if valid_indices:
                    try:
                        grouped_rollouts = generate_rollout_batch(
                            model=model,
                            processor=processor,
                            device=args.device,
                            batch_image_paths=batch_image_paths,
                            batch_questions=batch_questions,
                            max_new_tokens=args.max_new_tokens,
                            temperature=args.temperature,
                            top_p=args.top_p,
                            top_k=args.top_k,
                            rollouts_per_prompt=args.rollouts_per_prompt,
                            rollout_micro_batch_size=args.rollout_micro_batch_size,
                            save_token_text=args.save_token_text,
                            record_evidence_topk=bool(args.record_evidence_topk),
                            evidence_top_ratio=args.evidence_top_ratio,
                            evidence_border_ratio=args.evidence_border_ratio,
                            evidence_border_tokens=args.evidence_border_tokens,
                            evidence_attention_layers=args.evidence_attention_layers,
                            evidence_layer_weighting=args.evidence_layer_weighting,
                            min_pixels=args.min_pixels,
                            max_pixels=args.max_pixels,
                        )

                        flat_for_scoring: List[Dict[str, Any]] = []
                        for local_index, sample_index in enumerate(valid_indices):
                            record = records[sample_index]
                            record["rollouts"] = grouped_rollouts[local_index]
                            if not bool(args.record_evidence_topk):
                                for rollout in record["rollouts"]:
                                    flat_for_scoring.append(
                                        {
                                            "sample_index": sample_index,
                                            "image_paths": batch_image_paths[local_index],
                                            "question": batch_questions[local_index],
                                            "rollout": rollout,
                                        }
                                    )

                        if not bool(args.record_evidence_topk):
                            score_flat_rollouts(
                                model=model,
                                processor=processor,
                                device=args.device,
                                flat_rollouts=flat_for_scoring,
                                score_batch_size=args.score_batch_size,
                                save_token_text=args.save_token_text,
                            )

                        for sample_index in valid_indices:
                            finalize_perturbation_record(
                                base_items[sample_index],
                                records[sample_index],
                            )

                    except torch.cuda.OutOfMemoryError:
                        torch.cuda.empty_cache()
                        raise RuntimeError(
                            "CUDA out of memory. Reduce BATCH_SIZE, "
                            "SCORE_BATCH_SIZE, ROLLOUTS_PER_PROMPT, "
                            "MAX_NEW_TOKENS, or image pixel limits."
                        )
                    except Exception as error:
                        error_text = repr(error)
                        if bool(args.record_evidence_topk) and (
                            "attention" in error_text.lower()
                            or "online rollout recorder" in error_text.lower()
                        ):
                            raise RuntimeError(
                                "Direct rollout decode-attention capture failed; "
                                "aborting immediately before training."
                            ) from error
                        for sample_index in valid_indices:
                            record = records[sample_index]
                            record["generation_error"] = error_text
                            record["filter_reason"] = "generation_or_scoring_error"

                # One transaction per outer batch. The full dataset is never held
                # in memory and the whole output file is never rewritten.
                upsert_rollout_groups(
                    connection=connection,
                    records=[
                        (sample_index, perturbation, records[sample_index])
                        for sample_index in batch_indices
                    ],
                    expected_rollouts=args.rollouts_per_prompt,
                )

                completed_since_summary += len(batch_indices)
                progress.update(len(batch_indices))

                if completed_since_summary >= args.save_every:
                    summary = build_sqlite_summary(
                        connection,
                        args.dataset,
                        perturbations,
                        len(data),
                    )
                    set_meta(connection, "summary", summary)
                    connection.commit()
                    completed_since_summary = 0

            progress.close()
            summary = build_sqlite_summary(
                connection,
                args.dataset,
                perturbations,
                len(data),
            )
            set_meta(connection, "summary", summary)
            connection.commit()

        summary = build_sqlite_summary(
            connection,
            args.dataset,
            perturbations,
            len(data),
        )
        set_meta(connection, "summary", summary)
        connection.commit()
        print(json.dumps(summary, ensure_ascii=False, indent=2))

    finally:
        connection.close()


if __name__ == "__main__":
    main_sqlite()
