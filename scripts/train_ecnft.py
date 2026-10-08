#!/usr/bin/env python3
"""Train EC-NFT with attention-based positive weights and unchanged NFT negatives."""

from __future__ import annotations

import json
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import torch
from transformers import AutoProcessor, AutoTokenizer, HfArgumentParser

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import train_nft_sqlite as base
from ec_nft.evidence import load_cache, open_rollouts


@dataclass
class ECNFTArguments(base.NFTSQLiteArguments):
    evidence_cache_path: str = field(default="", metadata={"help": "CPU attention-score JSON cache"})


class EvidenceDataset(base.NFTSQLiteDataset):
    def __init__(self, *args, cache: dict, **kwargs):
        super().__init__(*args, **kwargs)
        self.weights = {(int(row["sample_index"]), str(row["perturbation"]),
                         int(row["rollout_position"])): float(row["weight"])
                        for row in cache["records"]}
        missing = set(self.references).difference(self.weights)
        if missing:
            raise ValueError(f"Evidence cache lacks {len(missing)} training trajectories")

    def __getitem__(self, index):
        item = super().__getitem__(index)
        item["evidence_weight"] = torch.tensor([self.weights[self.references[index]]],
                                               dtype=torch.float32)
        return item


class EvidenceCollator(base.NFTSQLiteDataCollator):
    def __call__(self, features):
        batch = super().__call__(features)
        batch["evidence_weight"] = torch.cat([item["evidence_weight"] for item in features]).float()
        return batch


def train():
    parser = HfArgumentParser((base.ModelArguments, base.DataArguments, ECNFTArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    base.local_rank = int(training_args.local_rank)
    if not training_args.nft_sqlite_path or not training_args.evidence_cache_path:
        raise ValueError("--nft_sqlite_path and --evidence_cache_path are required")
    if base.parse_perturbations(training_args.nft_perturbations) != ["clean"]:
        raise ValueError("The EC-NFT release uses original images only")
    if training_args.nft_loss_scale <= 0 or training_args.nft_clamp_negative <= 0:
        raise ValueError("NFT loss scale and negative clamp must be positive")
    cache = load_cache(training_args.evidence_cache_path, training_args.nft_sqlite_path)
    output = Path(training_args.output_dir)
    provenance = {key: value for key, value in cache.items() if key != "records"}
    previous_config = output / "evidence_config.json"
    if list(output.glob("checkpoint-*")):
        if not previous_config.is_file() or json.loads(previous_config.read_text(encoding="utf-8")) != provenance:
            raise ValueError("Existing training checkpoints use a different evidence configuration; choose a new output_dir")
    connection = open_rollouts(training_args.nft_sqlite_path)
    try:
        row = connection.execute("SELECT value_json FROM meta WHERE key='collection_config'").fetchone()
    finally:
        connection.close()
    if not row:
        raise ValueError("Rollout database lacks collection_config metadata")
    collection = json.loads(row[0])
    for key in ("min_pixels", "max_pixels"):
        if int(collection[key]) != int(getattr(data_args, key)):
            raise ValueError(f"{key} must match the rollout image preprocessing ({collection[key]})")

    model, model_type = base.load_model(model_args, training_args,
                                       os.environ.get("ATTN_IMPL", "sdpa"))
    data_args.model_type = model_type
    processor = AutoProcessor.from_pretrained(model_args.model_name_or_path,
                                              cache_dir=training_args.cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path,
                                              cache_dir=training_args.cache_dir,
                                              model_max_length=training_args.model_max_length,
                                              padding_side="right", use_fast=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    processor.tokenizer = tokenizer
    processor = base.update_processor_pixels(processor, data_args)
    training_args.remove_unused_columns = False
    output.mkdir(parents=True, exist_ok=True)
    # Save calibration provenance beside the adapter for reproducibility.
    if training_args.should_save:
        previous_config.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    model.config.use_cache = False
    if training_args.gradient_checkpointing:
        base.enable_gradient_checkpointing_inputs(model)
    model = base.maybe_enable_lora(model, training_args, model_args)
    dataset = EvidenceDataset(sqlite_path=training_args.nft_sqlite_path, processor=processor,
                              model_type=model_type, model_max_length=training_args.model_max_length,
                              perturbations=["clean"],
                              validate_old_logprobs=training_args.nft_validate_old_logprobs,
                              max_rollouts=training_args.nft_max_rollouts, cache=cache)
    trainer = base.NFTSQLiteTrainer(model=model, processing_class=tokenizer, args=training_args,
                                    train_dataset=dataset, eval_dataset=None,
                                    data_collator=EvidenceCollator(tokenizer))
    trainer.train(resume_from_checkpoint=True if list(output.glob("checkpoint-*")) else None)
    trainer.save_state()
    model.config.use_cache = True
    trainer.save_model(str(output))
    if training_args.should_save:
        processor.save_pretrained(str(output))
    base.rank0_print(f"[EC-NFT] Saved adapter to {output}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    train()
