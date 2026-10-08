#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Token-level NFT trainer for Qwen-VL using SQLite rollouts.

The SQLite database is produced by collect_qwen3vl_nft_rollouts_sqlite.py.
Only complete groups with 0 < r_hat < 1 and nft_prompt_eligible=1 are used.

For each generated response token:
    ratio = pi_theta(token) / pi_old(token)

Positive rollout:
    loss_pos = -log(ratio)

Negative rollout:
    negative_ratio = (1 - r_hat * ratio) / (1 - r_hat)
    loss_neg = -log(ST-clamp(negative_ratio, min=clamp_negative))

The default configuration follows the practical NFT implementation:
    token-level ratio, neg_weight=1, normalize=1,
    clamp_negative=1, clamp_positive=0, loss scale=1000.
"""

from __future__ import annotations

import json
import logging
import math
import os
import pathlib
import sqlite3
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
import transformers
from torch.utils.data import Dataset
from transformers import (
    AutoConfig,
    AutoProcessor,
    AutoTokenizer,
    HfArgumentParser,
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    Qwen3VLForConditionalGeneration,
    Qwen3VLMoeForConditionalGeneration,
    Trainer,
)

from qwen_vl_utils import process_vision_info

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(
    os.environ.get("QWEN_VL_FINETUNE_ROOT", str(SCRIPT_DIR.parent))
).resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from qwenvl.train.argument import (  # noqa: E402
    DataArguments,
    ModelArguments,
    TrainingArguments,
)
from qwenvl.data.data_processor import (  # noqa: E402
    get_rope_index_2,
    get_rope_index_25,
    get_rope_index_3,
    update_processor_pixels,
)

LOGGER = logging.getLogger(__name__)
local_rank = -1


def rank0_print(*values: Any) -> None:
    if local_rank in (-1, 0):
        print(*values, flush=True)


@dataclass
class NFTSQLiteArguments(TrainingArguments):
    nft_sqlite_path: str = field(
        default="",
        metadata={"help": "SQLite rollout database produced by the NFT collector."},
    )
    nft_perturbations: str = field(
        default="clean",
        metadata={
            "help": (
                "Comma-separated prompt groups to train on. For the original "
                "clean-image NFT setting use 'clean'."
            )
        },
    )
    nft_neg_weight: float = field(default=1.0)
    nft_normalize: int = field(
        default=1,
        metadata={"help": "0: none, 1: (1-r_hat), 2: sqrt((1-r_hat)/r_hat)."},
    )
    nft_clamp_negative: float = field(default=1.0)
    nft_clamp_positive: float = field(default=0.0)
    nft_log_ratio_max: float = field(default=10.0)
    nft_loss_scale: float = field(default=1000.0)
    nft_min_r_hat: float = field(default=0.02)
    nft_max_r_hat: float = field(default=0.98)
    nft_validate_old_logprobs: bool = field(default=True)
    nft_max_rollouts: int = field(
        default=0,
        metadata={"help": "Optional rollout limit for smoke tests; 0 means all."},
    )
    nft_log_components: bool = field(default=True)


def parse_perturbations(value: str) -> List[str]:
    names = [part.strip() for part in value.split(",") if part.strip()]
    if not names:
        raise ValueError("--nft_perturbations cannot be empty")
    allowed = {"clean", "black", "noise", "blur"}
    unsupported = sorted(set(names) - allowed)
    if unsupported:
        raise ValueError(
            f"Unsupported perturbations: {unsupported}; allowed={sorted(allowed)}"
        )
    return list(dict.fromkeys(names))


def open_readonly_sqlite(path: Path) -> sqlite3.Connection:
    resolved = path.resolve()
    connection = sqlite3.connect(
        f"file:{resolved.as_posix()}?mode=ro",
        uri=True,
        timeout=120.0,
        check_same_thread=False,
    )
    connection.execute("PRAGMA query_only=ON")
    connection.execute("PRAGMA busy_timeout=120000")
    return connection


def validate_sqlite_schema(connection: sqlite3.Connection) -> None:
    required_tables = {"samples", "rollout_groups"}
    existing = {
        str(row[0])
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
    }
    missing = required_tables - existing
    if missing:
        raise ValueError(f"SQLite database is missing tables: {sorted(missing)}")

    group_columns = {
        str(row[1])
        for row in connection.execute("PRAGMA table_info(rollout_groups)")
    }
    required_group_columns = {
        "sample_index",
        "perturbation",
        "complete",
        "r_hat",
        "nft_prompt_eligible",
        "record_json",
    }
    missing_columns = required_group_columns - group_columns
    if missing_columns:
        raise ValueError(
            "rollout_groups is missing columns: "
            f"{sorted(missing_columns)}"
        )


def make_messages(image_paths: Sequence[str], question: str) -> List[Dict[str, Any]]:
    content: List[Dict[str, Any]] = [
        {"type": "image", "image": str(image_path)}
        for image_path in image_paths
    ]
    content.append({"type": "text", "text": str(question)})
    return [{"role": "user", "content": content}]


def as_image_paths(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    return [text] if text else []


def straight_through_clamp_min(
    value: torch.Tensor,
    minimum: float,
) -> torch.Tensor:
    clamped = torch.clamp(value, min=float(minimum))
    return value + (clamped - value).detach()


class NFTSQLiteDataset(Dataset):
    """
    A flattened rollout view over eligible SQLite prompt groups.

    Only lightweight references are kept in memory. Each DataLoader process opens
    its own read-only SQLite connection and loads record_json lazily.
    """

    def __init__(
        self,
        sqlite_path: str,
        processor: Any,
        model_type: str,
        model_max_length: int,
        perturbations: Sequence[str],
        validate_old_logprobs: bool,
        max_rollouts: int = 0,
    ) -> None:
        self.sqlite_path = Path(sqlite_path).expanduser().resolve()
        if not self.sqlite_path.is_file():
            raise FileNotFoundError(f"NFT SQLite database not found: {self.sqlite_path}")

        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.model_type = model_type
        self.model_max_length = int(model_max_length)
        self.perturbations = list(perturbations)
        self.validate_old_logprobs = bool(validate_old_logprobs)
        self.max_rollouts = int(max_rollouts)
        self._connection: Optional[sqlite3.Connection] = None

        if self.model_type == "qwen3vl":
            self.get_rope_index = get_rope_index_3
        elif self.model_type == "qwen2.5vl":
            self.get_rope_index = get_rope_index_25
        else:
            self.get_rope_index = get_rope_index_2

        self.merge_size = int(self.processor.image_processor.merge_size)
        self.references: List[Tuple[int, str, int]] = []
        self.group_count = 0
        self._build_index()

    def __getstate__(self) -> Dict[str, Any]:
        state = dict(self.__dict__)
        state["_connection"] = None
        return state

    def __del__(self) -> None:
        connection = getattr(self, "_connection", None)
        if connection is not None:
            try:
                connection.close()
            except Exception:
                pass

    def _get_connection(self) -> sqlite3.Connection:
        if self._connection is None:
            self._connection = open_readonly_sqlite(self.sqlite_path)
        return self._connection

    def _build_index(self) -> None:
        placeholders = ",".join("?" for _ in self.perturbations)
        query = f"""
            SELECT sample_index, perturbation, r_hat, record_json
            FROM rollout_groups
            WHERE complete = 1
              AND nft_prompt_eligible = 1
              AND r_hat > 0.0
              AND r_hat < 1.0
              AND perturbation IN ({placeholders})
            ORDER BY sample_index, perturbation
        """

        invalid_groups = 0
        invalid_rollouts = 0
        connection = open_readonly_sqlite(self.sqlite_path)
        try:
            validate_sqlite_schema(connection)
            for sample_index, perturbation, db_r_hat, record_json in connection.execute(
                query, tuple(self.perturbations)
            ):
                try:
                    record = json.loads(record_json)
                    rollouts = record.get("rollouts", [])
                    r_hat = float(record.get("r_hat", db_r_hat))
                except Exception:
                    invalid_groups += 1
                    continue

                if not (0.0 < r_hat < 1.0) or not isinstance(rollouts, list):
                    invalid_groups += 1
                    continue

                valid_in_group = 0
                for rollout_position, rollout in enumerate(rollouts):
                    if not isinstance(rollout, dict):
                        invalid_rollouts += 1
                        continue
                    token_ids = rollout.get("generated_token_ids", [])
                    old_score = rollout.get("old_policy_score", {})
                    old_logprobs = (
                        old_score.get("token_logprobs", [])
                        if isinstance(old_score, dict)
                        else []
                    )
                    reward = rollout.get("reward")
                    if (
                        reward not in (0, 1, False, True)
                        or not isinstance(token_ids, list)
                        or not token_ids
                        or not isinstance(old_logprobs, list)
                        or len(token_ids) != len(old_logprobs)
                    ):
                        invalid_rollouts += 1
                        continue

                    self.references.append(
                        (int(sample_index), str(perturbation), int(rollout_position))
                    )
                    valid_in_group += 1
                    if self.max_rollouts > 0 and len(self.references) >= self.max_rollouts:
                        break

                if valid_in_group > 0:
                    self.group_count += 1
                if self.max_rollouts > 0 and len(self.references) >= self.max_rollouts:
                    break
        finally:
            connection.close()

        if not self.references:
            raise ValueError(
                "No NFT-eligible rollouts found. Expected complete groups with "
                "nft_prompt_eligible=1 and 0<r_hat<1 for perturbations="
                f"{self.perturbations}."
            )

        rank0_print(
            "[NFT SQLite] "
            f"groups={self.group_count}, rollouts={len(self.references)}, "
            f"perturbations={self.perturbations}, "
            f"invalid_groups={invalid_groups}, invalid_rollouts={invalid_rollouts}"
        )

    def __len__(self) -> int:
        return len(self.references)

    def _load_record(
        self,
        sample_index: int,
        perturbation: str,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        row = self._get_connection().execute(
            """
            SELECT s.base_record_json, g.record_json
            FROM rollout_groups AS g
            JOIN samples AS s ON s.sample_index = g.sample_index
            WHERE g.sample_index = ?
              AND g.perturbation = ?
              AND g.complete = 1
              AND g.nft_prompt_eligible = 1
            """,
            (int(sample_index), str(perturbation)),
        ).fetchone()
        if row is None:
            raise KeyError(
                f"Eligible rollout group disappeared: "
                f"sample_index={sample_index}, perturbation={perturbation}"
            )
        return json.loads(row[0]), json.loads(row[1])

    def _position_ids(
        self,
        input_ids: torch.Tensor,
        image_grid_thw: Optional[torch.Tensor],
        video_grid_thw: Optional[torch.Tensor],
    ) -> torch.Tensor:
        second_per_grid_ts = None
        if video_grid_thw is not None and int(video_grid_thw.numel()) > 0:
            number_of_videos = int(video_grid_thw.shape[0])
            temporal_patch_size = getattr(
                self.processor.video_processor,
                "temporal_patch_size",
                2,
            )
            fps = getattr(self.processor.video_processor, "fps", 2)
            second_per_grid_ts = [
                float(temporal_patch_size) / float(fps)
            ] * number_of_videos

        position_ids, _ = self.get_rope_index(
            self.merge_size,
            input_ids,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            second_per_grid_ts=second_per_grid_ts,
        )
        return position_ids

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample_index, perturbation, rollout_position = self.references[index]
        base_record, group_record = self._load_record(sample_index, perturbation)

        rollouts = group_record.get("rollouts", [])
        if rollout_position >= len(rollouts):
            raise IndexError(
                f"rollout_position={rollout_position} out of range for "
                f"sample_index={sample_index}, perturbation={perturbation}"
            )
        rollout = rollouts[rollout_position]

        image_paths = as_image_paths(group_record.get("used_image_paths"))
        if not image_paths:
            image_paths = as_image_paths(base_record.get("image_paths"))
        if not image_paths:
            image_paths = as_image_paths(base_record.get("image_path"))
        if not image_paths:
            raise ValueError(
                f"No image path for sample_index={sample_index}, "
                f"perturbation={perturbation}"
            )

        missing_images = [path for path in image_paths if not Path(path).is_file()]
        if missing_images:
            raise FileNotFoundError(
                f"Missing image(s) for sample_index={sample_index}: "
                f"{missing_images[:3]}"
            )

        question = str(base_record.get("question", "")).strip()
        if not question:
            raise ValueError(f"Empty question at sample_index={sample_index}")

        generated_token_ids = [
            int(value) for value in rollout.get("generated_token_ids", [])
        ]
        old_score = rollout.get("old_policy_score", {})
        if not isinstance(old_score, dict):
            raise ValueError(
                f"Missing old_policy_score at sample_index={sample_index}, "
                f"rollout={rollout_position}"
            )
        old_token_logprobs = [
            float(value) for value in old_score.get("token_logprobs", [])
        ]
        if len(generated_token_ids) != len(old_token_logprobs):
            raise ValueError(
                "Stored token/logprob length mismatch at "
                f"sample_index={sample_index}, rollout={rollout_position}: "
                f"{len(generated_token_ids)} vs {len(old_token_logprobs)}"
            )
        stored_score_token_ids = old_score.get("token_ids")
        if (
            self.validate_old_logprobs
            and isinstance(stored_score_token_ids, list)
            and [int(value) for value in stored_score_token_ids]
            != generated_token_ids
        ):
            raise ValueError(
                "old_policy_score.token_ids do not match generated_token_ids at "
                f"sample_index={sample_index}, rollout={rollout_position}"
            )
        if not generated_token_ids:
            raise ValueError(
                f"Empty generated response at sample_index={sample_index}, "
                f"rollout={rollout_position}"
            )
        if self.validate_old_logprobs and not all(
            math.isfinite(value) for value in old_token_logprobs
        ):
            raise ValueError(
                f"Non-finite old log-prob at sample_index={sample_index}, "
                f"rollout={rollout_position}"
            )

        reward = float(int(bool(rollout.get("reward", 0))))
        r_hat = float(group_record.get("r_hat"))
        if not (0.0 < r_hat < 1.0):
            raise ValueError(
                f"Invalid r_hat={r_hat} at sample_index={sample_index}"
            )

        messages = make_messages(image_paths, question)
        prompt_text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        image_inputs, video_inputs = process_vision_info(messages)

        previous_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "right"
        try:
            prompt_inputs = self.processor(
                text=[prompt_text],
                images=image_inputs,
                videos=video_inputs,
                padding=False,
                return_tensors="pt",
            )
        finally:
            self.tokenizer.padding_side = previous_padding_side

        prompt_ids = prompt_inputs["input_ids"][0]
        prompt_length = int(prompt_ids.numel())
        maximum_response_length = self.model_max_length - prompt_length
        if maximum_response_length <= 0:
            raise ValueError(
                f"Prompt length {prompt_length} exceeds model_max_length="
                f"{self.model_max_length} at sample_index={sample_index}"
            )

        if len(generated_token_ids) > maximum_response_length:
            generated_token_ids = generated_token_ids[:maximum_response_length]
            old_token_logprobs = old_token_logprobs[:maximum_response_length]

        response_ids = torch.tensor(
            generated_token_ids,
            dtype=prompt_ids.dtype,
        )
        input_ids = torch.cat([prompt_ids, response_ids], dim=0).unsqueeze(0)
        sequence_length = int(input_ids.shape[1])

        attention_mask = torch.ones(
            (1, sequence_length),
            dtype=torch.long,
        )
        response_mask = torch.zeros(
            (1, sequence_length),
            dtype=torch.float32,
        )
        response_mask[:, prompt_length:sequence_length] = 1.0

        old_logprobs = torch.zeros(
            (1, sequence_length),
            dtype=torch.float32,
        )
        old_logprobs[:, prompt_length:sequence_length] = torch.tensor(
            old_token_logprobs,
            dtype=torch.float32,
        )

        image_grid_thw = prompt_inputs.get("image_grid_thw")
        video_grid_thw = prompt_inputs.get("video_grid_thw")
        position_ids = self._position_ids(
            input_ids=input_ids,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
        )

        item: Dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "old_logprobs": old_logprobs,
            "response_mask": response_mask,
            "reward": torch.tensor([reward], dtype=torch.float32),
            "r_hat": torch.tensor([r_hat], dtype=torch.float32),
        }

        for key in (
            "pixel_values",
            "image_grid_thw",
            "pixel_values_videos",
            "video_grid_thw",
        ):
            if key in prompt_inputs and prompt_inputs[key] is not None:
                item[key] = prompt_inputs[key]

        return item


class NFTSQLiteDataCollator:
    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer
        if self.tokenizer.pad_token_id is None:
            raise ValueError("Tokenizer pad_token_id must be set")

    @staticmethod
    def _pad_2d(
        tensors: Sequence[torch.Tensor],
        maximum_length: int,
        value: float,
    ) -> torch.Tensor:
        padded = []
        for tensor in tensors:
            pad_length = maximum_length - int(tensor.shape[-1])
            padded.append(F.pad(tensor, (0, pad_length), value=value))
        return torch.cat(padded, dim=0)

    def __call__(self, features: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        maximum_length = max(int(item["input_ids"].shape[-1]) for item in features)
        batch: Dict[str, Any] = {
            "input_ids": self._pad_2d(
                [item["input_ids"] for item in features],
                maximum_length,
                float(self.tokenizer.pad_token_id),
            ).long(),
            "attention_mask": self._pad_2d(
                [item["attention_mask"] for item in features],
                maximum_length,
                0.0,
            ).long(),
            "old_logprobs": self._pad_2d(
                [item["old_logprobs"] for item in features],
                maximum_length,
                0.0,
            ).float(),
            "response_mask": self._pad_2d(
                [item["response_mask"] for item in features],
                maximum_length,
                0.0,
            ).float(),
            "reward": torch.cat([item["reward"] for item in features], dim=0).float(),
            "r_hat": torch.cat([item["r_hat"] for item in features], dim=0).float(),
        }

        position_ids = []
        for item in features:
            position = item["position_ids"]
            pad_length = maximum_length - int(position.shape[-1])
            position_ids.append(F.pad(position, (0, pad_length), value=1))
        batch["position_ids"] = torch.cat(position_ids, dim=1)

        pixel_values = [
            item["pixel_values"]
            for item in features
            if item.get("pixel_values") is not None
        ]
        if pixel_values:
            batch["pixel_values"] = torch.cat(pixel_values, dim=0)

        image_grid_thw = [
            item["image_grid_thw"]
            for item in features
            if item.get("image_grid_thw") is not None
        ]
        if image_grid_thw:
            batch["image_grid_thw"] = torch.cat(image_grid_thw, dim=0)

        pixel_values_videos = [
            item["pixel_values_videos"]
            for item in features
            if item.get("pixel_values_videos") is not None
        ]
        if pixel_values_videos:
            batch["pixel_values_videos"] = torch.cat(
                pixel_values_videos,
                dim=0,
            )

        video_grid_thw = [
            item["video_grid_thw"]
            for item in features
            if item.get("video_grid_thw") is not None
        ]
        if video_grid_thw:
            batch["video_grid_thw"] = torch.cat(video_grid_thw, dim=0)

        return batch


class NFTSQLiteTrainer(Trainer):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.model_accepts_loss_kwargs = False
        self._component_steps = 0
        self._component_sums = {
            "nft_positive": 0.0,
            "nft_negative": 0.0,
            "nft_ratio_mean": 0.0,
        }

    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: Dict[str, Any],
        return_outputs: bool = False,
        num_items_in_batch: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Any:
        del num_items_in_batch, kwargs

        old_logprobs = inputs.pop("old_logprobs")
        response_mask = inputs.pop("response_mask")
        reward = inputs.pop("reward")
        r_hat = inputs.pop("r_hat")
        evidence_weight = inputs.pop("evidence_weight", None)

        outputs = model(**inputs)
        logits = outputs.logits

        shifted_logits = logits[:, :-1, :]
        shifted_target_ids = inputs["input_ids"][:, 1:]
        shifted_response_mask = response_mask[:, 1:].to(logits.dtype)
        shifted_old_logprobs = old_logprobs[:, 1:].to(logits.dtype)

        token_nll = F.cross_entropy(
            shifted_logits.transpose(1, 2),
            shifted_target_ids,
            reduction="none",
        )
        new_logprobs = -token_nll

        log_ratio = new_logprobs - shifted_old_logprobs
        clipped_log_ratio = torch.clamp(
            log_ratio,
            max=float(self.args.nft_log_ratio_max),
        )
        ratio = torch.exp(clipped_log_ratio)

        bounded_r_hat = torch.clamp(
            r_hat.to(logits.dtype),
            min=float(self.args.nft_min_r_hat),
            max=float(self.args.nft_max_r_hat),
        )
        reward = reward.to(logits.dtype)

        if float(self.args.nft_clamp_positive) > 0.0:
            positive_ratio = straight_through_clamp_min(
                ratio,
                float(self.args.nft_clamp_positive),
            )
            positive_loss = -torch.log(
                positive_ratio.clamp_min(torch.finfo(ratio.dtype).tiny)
            )
        else:
            # Exact -log(exp(log_ratio)), avoiding numerical underflow.
            positive_loss = -clipped_log_ratio

        # EC-NFT calibrates the positive term only. Plain NFT supplies no weight.
        if evidence_weight is not None:
            positive_loss = positive_loss * evidence_weight.to(logits.dtype)[:, None]

        negative_ratio = (
            1.0 - bounded_r_hat[:, None] * ratio
        ) / (
            1.0 - bounded_r_hat[:, None]
        )
        negative_ratio = straight_through_clamp_min(
            negative_ratio,
            abs(float(self.args.nft_clamp_negative)),
        )
        negative_loss = -torch.log(
            negative_ratio.clamp_min(torch.finfo(ratio.dtype).tiny)
        )

        token_loss = (
            reward[:, None] * positive_loss
            + (1.0 - reward[:, None])
            * float(self.args.nft_neg_weight)
            * negative_loss
        )

        normalize_mode = int(self.args.nft_normalize)
        if normalize_mode == 1:
            prompt_weight = 1.0 - bounded_r_hat
        elif normalize_mode == 2:
            prompt_weight = torch.sqrt(
                (1.0 - bounded_r_hat) / bounded_r_hat
            )
        elif normalize_mode == 0:
            prompt_weight = torch.ones_like(bounded_r_hat)
        else:
            raise ValueError(
                f"Unsupported nft_normalize={normalize_mode}; use 0, 1, or 2"
            )

        weighted_token_loss = (
            token_loss
            * prompt_weight[:, None]
            * shifted_response_mask
        )
        loss = weighted_token_loss.sum() / float(self.args.nft_loss_scale)

        if bool(self.args.nft_log_components):
            with torch.no_grad():
                valid_tokens = shifted_response_mask.sum().clamp_min(1.0)
                positive_mask = reward[:, None] * shifted_response_mask
                negative_mask = (1.0 - reward[:, None]) * shifted_response_mask
                positive_value = (
                    (positive_loss * positive_mask).sum()
                    / positive_mask.sum().clamp_min(1.0)
                )
                negative_value = (
                    (negative_loss * negative_mask).sum()
                    / negative_mask.sum().clamp_min(1.0)
                )
                ratio_value = (
                    (ratio * shifted_response_mask).sum() / valid_tokens
                )
                self._component_sums["nft_positive"] += float(
                    positive_value.detach().float().cpu()
                )
                self._component_sums["nft_negative"] += float(
                    negative_value.detach().float().cpu()
                )
                self._component_sums["nft_ratio_mean"] += float(
                    ratio_value.detach().float().cpu()
                )
                self._component_steps += 1

        return (loss, outputs) if return_outputs else loss

    def log(
        self,
        logs: Dict[str, float],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        if self._component_steps > 0:
            for key, total in self._component_sums.items():
                logs[key] = total / self._component_steps
            self._component_steps = 0
            for key in self._component_sums:
                self._component_sums[key] = 0.0
        super().log(logs, *args, **kwargs)


def set_module_trainable(module: Any, trainable: bool) -> None:
    if module is None:
        return
    for parameter in module.parameters():
        parameter.requires_grad = trainable


def set_non_lora_trainability(model_args: ModelArguments, model: Any) -> None:
    visual = getattr(model, "visual", None)
    language_model = getattr(model, "language_model", None)
    lm_head = getattr(model, "lm_head", None)

    set_module_trainable(visual, bool(model_args.tune_mm_vision))
    merger = getattr(visual, "merger", None) if visual is not None else None
    set_module_trainable(merger, bool(model_args.tune_mm_mlp))
    set_module_trainable(language_model, bool(model_args.tune_mm_llm))
    set_module_trainable(lm_head, bool(model_args.tune_mm_llm))


def load_model(
    model_args: ModelArguments,
    training_args: NFTSQLiteArguments,
    attn_implementation: str,
) -> Tuple[Any, str]:
    load_dtype = None
    if training_args.bf16:
        load_dtype = torch.bfloat16
    elif training_args.fp16:
        load_dtype = torch.float16

    common_kwargs = {
        "cache_dir": training_args.cache_dir,
        "attn_implementation": attn_implementation,
        "dtype": load_dtype,
    }

    checkpoint_type = AutoConfig.from_pretrained(
        model_args.model_name_or_path, cache_dir=training_args.cache_dir
    ).model_type
    if checkpoint_type == "qwen3_vl_moe":
        model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            **common_kwargs,
        )
        model_type = "qwen3vl"
    elif checkpoint_type == "qwen3_vl":
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            **common_kwargs,
        )
        model_type = "qwen3vl"
    elif checkpoint_type == "qwen2_5_vl":
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            **common_kwargs,
        )
        model_type = "qwen2.5vl"
    elif checkpoint_type == "qwen2_vl":
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            **common_kwargs,
        )
        model_type = "qwen2vl"

    else:
        raise ValueError(f"Unsupported model type: {checkpoint_type}")

    return model, model_type


def maybe_enable_lora(
    model: Any,
    training_args: NFTSQLiteArguments,
    model_args: ModelArguments,
) -> Any:
    if not bool(training_args.lora_enable):
        set_non_lora_trainability(model_args, model)
        return model

    from peft import LoraConfig, TaskType, get_peft_model

    for parameter in model.parameters():
        parameter.requires_grad = False

    lora_config = LoraConfig(
        r=int(training_args.lora_r or 64),
        lora_alpha=int(training_args.lora_alpha or 128),
        lora_dropout=float(training_args.lora_dropout or 0.05),
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    rank0_print("[NFT SQLite] LoRA enabled")
    model.print_trainable_parameters()
    return model


def enable_gradient_checkpointing_inputs(model: Any) -> None:
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
        return

    embeddings = model.get_input_embeddings()

    def make_inputs_require_grad(
        module: torch.nn.Module,
        inputs: Tuple[Any, ...],
        output: torch.Tensor,
    ) -> None:
        del module, inputs
        output.requires_grad_(True)

    embeddings.register_forward_hook(make_inputs_require_grad)


def train(attn_implementation: str = "flash_attention_2") -> None:
    global local_rank

    parser = HfArgumentParser(
        (ModelArguments, DataArguments, NFTSQLiteArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = int(training_args.local_rank)

    if not training_args.nft_sqlite_path:
        raise ValueError("--nft_sqlite_path is required")
    if float(training_args.nft_loss_scale) <= 0:
        raise ValueError("--nft_loss_scale must be > 0")
    if float(training_args.nft_clamp_negative) <= 0:
        raise ValueError("--nft_clamp_negative must be > 0")

    perturbations = parse_perturbations(training_args.nft_perturbations)

    model, model_type = load_model(
        model_args=model_args,
        training_args=training_args,
        attn_implementation=attn_implementation,
    )
    data_args.model_type = model_type

    processor = AutoProcessor.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    processor.tokenizer = tokenizer
    processor = update_processor_pixels(processor, data_args)

    training_args.remove_unused_columns = False
    os.makedirs(training_args.output_dir, exist_ok=True)

    model.config.use_cache = False
    if bool(training_args.gradient_checkpointing):
        enable_gradient_checkpointing_inputs(model)

    model = maybe_enable_lora(model, training_args, model_args)

    train_dataset = NFTSQLiteDataset(
        sqlite_path=training_args.nft_sqlite_path,
        processor=processor,
        model_type=model_type,
        model_max_length=int(training_args.model_max_length),
        perturbations=perturbations,
        validate_old_logprobs=bool(
            training_args.nft_validate_old_logprobs
        ),
        max_rollouts=int(training_args.nft_max_rollouts),
    )

    rank0_print(
        "[NFT SQLite config] "
        f"neg_weight={training_args.nft_neg_weight}, "
        f"normalize={training_args.nft_normalize}, "
        f"clamp_negative={training_args.nft_clamp_negative}, "
        f"clamp_positive={training_args.nft_clamp_positive}, "
        f"log_ratio_max={training_args.nft_log_ratio_max}, "
        f"loss_scale={training_args.nft_loss_scale}"
    )

    trainer = NFTSQLiteTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,
        data_collator=NFTSQLiteDataCollator(tokenizer),
    )

    output_directory = Path(training_args.output_dir)
    checkpoints = sorted(output_directory.glob("checkpoint-*"))
    if checkpoints:
        rank0_print(
            f"[NFT SQLite] Resuming from latest checkpoint under "
            f"{output_directory}"
        )
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    trainer.save_state()
    model.config.use_cache = True
    trainer.save_model(training_args.output_dir)
    processor.save_pretrained(training_args.output_dir)
    rank0_print(f"[NFT SQLite] Saved to {training_args.output_dir}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    train(
        attn_implementation=os.environ.get(
            "ATTN_IMPL",
            "flash_attention_2",
        )
    )
