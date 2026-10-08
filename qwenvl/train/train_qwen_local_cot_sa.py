#!/usr/bin/env python3
"""Register a local dataset, then reuse qwenvl/train/train_qwen_cot+sa.py."""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

import qwenvl.data as qdata


def load_train_func():
    train_file = Path(__file__).resolve().parent / "train_qwen_cot_sa.py"
    spec = importlib.util.spec_from_file_location("train_qwen_cot_sa_runtime", train_file)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load training file: {train_file}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "train"):
        raise RuntimeError(f"No train() function found in: {train_file}")

    return module.train


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--local_dataset_name", type=str, default="merge_train_1k")
    parser.add_argument(
        "--local_annotation_path",
        type=str,
        default="./data/annotations/merged_qwenvl_trainall_with_system.json",
    )
    parser.add_argument("--local_data_path", type=str, default="")

    local_args, remaining = parser.parse_known_args()

    qdata.data_dict[local_args.local_dataset_name] = {
        "annotation_path": local_args.local_annotation_path,
        "data_path": local_args.local_data_path,
    }

    sys.argv = [sys.argv[0]] + remaining

    attn_impl = os.environ.get("ATTN_IMPLEMENTATION", "eager")
    train = load_train_func()
    train(attn_implementation=attn_impl)


if __name__ == "__main__":
    main()