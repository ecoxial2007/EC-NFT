# Adopted from https://github.com/lm-sys/FastChat.
# CoT/SA masked-loss version.

import argparse
import logging
import os
import pathlib
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import transformers
from transformers import (
    AutoProcessor,
    Trainer,
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    Qwen3VLForConditionalGeneration,
    Qwen3VLMoeForConditionalGeneration,
)

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from qwenvl.data.data_processor import make_supervised_data_module
from qwenvl.train.argument import (
    ModelArguments,
    DataArguments,
    TrainingArguments,
)

local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def parse_cot_args():
    """Parse extra CoT/SA args, then remove them before HfArgumentParser sees sys.argv."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--cot_loss_mode",
        type=str,
        default=os.environ.get("COT_LOSS_MODE", "answer_only"),
        choices=["full", "answer_only", "full_plus_answer"],
    )
    parser.add_argument(
        "--cot_full_loss_weight",
        type=float,
        default=float(os.environ.get("COT_FULL_LOSS_WEIGHT", "1.0")),
    )
    parser.add_argument(
        "--cot_answer_loss_weight",
        type=float,
        default=float(os.environ.get("COT_ANSWER_LOSS_WEIGHT", "1.0")),
    )
    parser.add_argument(
        "--cot_mask_fallback",
        type=str,
        default=os.environ.get("COT_MASK_FALLBACK", "full"),
        choices=["full", "none"],
        help="If <answer> tags are not found: full keeps original labels; none masks all labels.",
    )

    cot_args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    return cot_args


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dumps to disk."""
    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def set_model(model_args, model):
    if model_args.tune_mm_vision:
        for _, p in model.visual.named_parameters():
            p.requires_grad = True
    else:
        for _, p in model.visual.named_parameters():
            p.requires_grad = False

    if model_args.tune_mm_mlp:
        for _, p in model.visual.merger.named_parameters():
            p.requires_grad = True
    else:
        for _, p in model.visual.merger.named_parameters():
            p.requires_grad = False

    if model_args.tune_mm_llm:
        for _, p in model.language_model.named_parameters():
            p.requires_grad = True
        model.lm_head.requires_grad = True
    else:
        for _, p in model.language_model.named_parameters():
            p.requires_grad = False
        model.lm_head.requires_grad = False


class CotAnswerMaskCollator:
    """Wrap the original data collator and modify labels for CoT/SA training.

    Modes:
      full:
        Keep original labels. Equivalent to normal SFT.

      answer_only:
        Only tokens inside <answer>...</answer> keep labels.
        Everything else, including <think>...</think>, is set to -100.

      full_plus_answer:
        batch["labels"] becomes answer-only labels.
        batch["full_labels"] stores original full labels.
        CotSATrainer computes both losses from one forward pass.
    """

    def __init__(
        self,
        base_collator,
        tokenizer,
        mode: str = "answer_only",
        fallback: str = "full",
    ):
        self.base_collator = base_collator
        self.tokenizer = tokenizer
        self.mode = mode
        self.fallback = fallback

        self.answer_open_ids = self._encode_no_special("<answer>")
        self.answer_close_ids = self._encode_no_special("</answer>")

        self._warned_missing = False

    def _encode_no_special(self, text: str) -> List[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    @staticmethod
    def _find_subsequence(seq: List[int], sub: List[int], start: int = 0) -> int:
        if not sub:
            return -1
        limit = len(seq) - len(sub) + 1
        for i in range(start, max(start, limit)):
            if seq[i : i + len(sub)] == sub:
                return i
        return -1

    def _is_whitespace_token(self, token_id: int) -> bool:
        text = self.tokenizer.decode([token_id], skip_special_tokens=False)
        return text.strip() == ""

    def _trim_answer_span(self, ids: List[int], start: int, end: int):
        while start < end and self._is_whitespace_token(ids[start]):
            start += 1
        while end > start and self._is_whitespace_token(ids[end - 1]):
            end -= 1
        return start, end

    def _make_answer_only_labels(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
    ):
        answer_labels = torch.full_like(labels, -100)
        found = 0

        for b in range(labels.shape[0]):
            ids = input_ids[b].tolist()

            open_pos = self._find_subsequence(ids, self.answer_open_ids, start=0)
            if open_pos < 0:
                if self.fallback == "full":
                    answer_labels[b] = labels[b]
                continue

            content_start = open_pos + len(self.answer_open_ids)
            close_pos = self._find_subsequence(
                ids,
                self.answer_close_ids,
                start=content_start,
            )
            content_end = close_pos if close_pos >= 0 else len(ids)

            content_start, content_end = self._trim_answer_span(
                ids, content_start, content_end
            )

            if content_start < content_end:
                answer_labels[b, content_start:content_end] = labels[
                    b, content_start:content_end
                ]
                answer_labels[b, labels[b] == -100] = -100
                found += 1
            elif self.fallback == "full":
                answer_labels[b] = labels[b]

        if found < labels.shape[0] and not self._warned_missing:
            print(
                f"[WARN] <answer> span found in {found}/{labels.shape[0]} samples "
                f"in one batch. fallback={self.fallback}"
            )
            self._warned_missing = True

        return answer_labels

    def __call__(self, features):
        batch = self.base_collator(features)

        if self.mode == "full":
            return batch

        if "input_ids" not in batch or "labels" not in batch:
            return batch

        input_ids = batch["input_ids"]
        labels = batch["labels"]

        if not isinstance(input_ids, torch.Tensor) or not isinstance(labels, torch.Tensor):
            return batch

        full_labels = labels.clone()
        answer_labels = self._make_answer_only_labels(input_ids, labels)

        batch["labels"] = answer_labels

        if self.mode == "full_plus_answer":
            batch["full_labels"] = full_labels

        return batch


class CotSATrainer(Trainer):
    def __init__(
        self,
        *args,
        cot_loss_mode: str = "answer_only",
        cot_full_loss_weight: float = 1.0,
        cot_answer_loss_weight: float = 1.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.cot_loss_mode = cot_loss_mode
        self.cot_full_loss_weight = cot_full_loss_weight
        self.cot_answer_loss_weight = cot_answer_loss_weight

    @staticmethod
    def _masked_lm_loss(logits: torch.Tensor, labels: torch.Tensor):
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        shift_logits = shift_logits.view(-1, shift_logits.size(-1))
        shift_labels = shift_labels.view(-1).to(shift_logits.device)

        return loss_fct(shift_logits, shift_labels)

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs: bool = False,
        num_items_in_batch: Optional[int] = None,
    ):
        if self.cot_loss_mode != "full_plus_answer":
            try:
                return super().compute_loss(
                    model,
                    inputs,
                    return_outputs=return_outputs,
                    num_items_in_batch=num_items_in_batch,
                )
            except TypeError:
                return super().compute_loss(
                    model,
                    inputs,
                    return_outputs=return_outputs,
                )

        full_labels = inputs.pop("full_labels", None)
        answer_labels = inputs.pop("labels")

        outputs = model(**inputs)
        logits = outputs.logits

        answer_loss = self._masked_lm_loss(logits, answer_labels)

        if full_labels is not None:
            full_loss = self._masked_lm_loss(logits, full_labels)
            loss = (
                self.cot_answer_loss_weight * answer_loss
                + self.cot_full_loss_weight * full_loss
            )
        else:
            loss = answer_loss

        return (loss, outputs) if return_outputs else loss


def train(attn_implementation="flash_attention_2"):
    global local_rank

    cot_args = parse_cot_args()

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    local_rank = training_args.local_rank
    os.makedirs(training_args.output_dir, exist_ok=True)

    load_dtype = None
    if training_args.bf16:
        load_dtype = torch.bfloat16
    elif training_args.fp16:
        load_dtype = torch.float16

    checkpoint_type = transformers.AutoConfig.from_pretrained(
        model_args.model_name_or_path, cache_dir=training_args.cache_dir
    ).model_type

    if checkpoint_type == "qwen3_vl_moe":
        model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            dtype=load_dtype,
        )
        data_args.model_type = "qwen3vl"
    elif checkpoint_type == "qwen3_vl":
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            dtype=load_dtype,
        )
        data_args.model_type = "qwen3vl"
    elif checkpoint_type == "qwen2_5_vl":
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            dtype=load_dtype,
        )
        data_args.model_type = "qwen2.5vl"
    elif checkpoint_type == "qwen2_vl":
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            dtype=load_dtype,
        )
        data_args.model_type = "qwen2vl"

    else:
        raise ValueError(f"Unsupported model type: {checkpoint_type}")

    print(
        f"the initialized model is {model_args.model_name_or_path} "
        f"the class is {model.__class__.__name__}"
    )
    print(
        f"[INFO] cot_loss_mode={cot_args.cot_loss_mode}, "
        f"cot_full_loss_weight={cot_args.cot_full_loss_weight}, "
        f"cot_answer_loss_weight={cot_args.cot_answer_loss_weight}"
    )

    processor = AutoProcessor.from_pretrained(model_args.model_name_or_path)

    if data_args.data_flatten or data_args.data_packing:
        from qwenvl.train.trainer import replace_qwen2_vl_attention_class

        replace_qwen2_vl_attention_class()

    model.config.use_cache = False

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:

            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    if training_args.lora_enable:
        from peft import LoraConfig, TaskType, get_peft_model

        print("LoRA enabled")

        for p in model.parameters():
            p.requires_grad = False

        lora_config = LoraConfig(
            r=training_args.lora_r or 64,
            lora_alpha=training_args.lora_alpha or 128,
            lora_dropout=training_args.lora_dropout or 0.05,
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
    else:
        set_model(model_args, model)

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            if torch.distributed.get_rank() == 0:
                model.visual.print_trainable_parameters()
                model.model.print_trainable_parameters()
        else:
            model.visual.print_trainable_parameters()
            model.model.print_trainable_parameters()

    data_module = make_supervised_data_module(processor, data_args=data_args)

    if "data_collator" in data_module and data_module["data_collator"] is not None:
        data_module["data_collator"] = CotAnswerMaskCollator(
            base_collator=data_module["data_collator"],
            tokenizer=tokenizer,
            mode=cot_args.cot_loss_mode,
            fallback=cot_args.cot_mask_fallback,
        )

    trainer = CotSATrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        cot_loss_mode=cot_args.cot_loss_mode,
        cot_full_loss_weight=cot_args.cot_full_loss_weight,
        cot_answer_loss_weight=cot_args.cot_answer_loss_weight,
        **data_module,
    )

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        logging.info("checkpoint found, resume training")
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    trainer.save_state()

    model.config.use_cache = True

    safe_save_model_for_hf_trainer(
        trainer=trainer,
        output_dir=training_args.output_dir,
    )

    processor.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")
