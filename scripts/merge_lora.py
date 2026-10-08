#!/usr/bin/env python3
import argparse
from collections import Counter
from pathlib import Path
from typing import Any, Optional

import torch
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoProcessor, AutoTokenizer


def summarize_parameter_dtypes(model: Any, label: str) -> tuple[str, Counter[str]]:
    dtype_counter: Counter[str] = Counter()
    total_params = 0

    for _, param in model.named_parameters():
        dtype_name = str(param.dtype).replace("torch.", "")
        param_count = param.numel()
        dtype_counter[dtype_name] += param_count
        total_params += param_count

    print(f"\n[{label}] parameter dtype summary:")
    for dtype_name, count in dtype_counter.most_common():
        ratio = count / max(total_params, 1)
        print(f"  - {dtype_name}: {count} params ({ratio:.2%})")

    dominant_dtype = dtype_counter.most_common(1)[0][0] if dtype_counter else "unknown"
    print(f"[{label}] dominant dtype: {dominant_dtype}")
    return dominant_dtype, dtype_counter


def detect_lora_storage_dtype(lora_path: str) -> str:
    lora_dir = Path(lora_path)
    candidates = [
        lora_dir / "adapter_model.safetensors",
        lora_dir / "adapter_model.bin",
        lora_dir / "pytorch_model.bin",
    ]

    state_dict: Optional[dict[str, Any]] = None
    for candidate in candidates:
        if not candidate.exists():
            continue
        if candidate.suffix == ".safetensors":
            from safetensors.torch import load_file

            state_dict = load_file(str(candidate), device="cpu")
        else:
            loaded = torch.load(candidate, map_location="cpu")
            if isinstance(loaded, dict):
                state_dict = loaded
        break

    if not state_dict:
        print("\n[LoRA checkpoint files] unable to detect dtype from checkpoint files.")
        return "unknown"

    dtype_counter: Counter[str] = Counter()
    total_tensors = 0
    for value in state_dict.values():
        if torch.is_tensor(value):
            dtype_name = str(value.dtype).replace("torch.", "")
            dtype_counter[dtype_name] += value.numel()
            total_tensors += value.numel()

    print("\n[LoRA checkpoint files] tensor dtype summary:")
    for dtype_name, count in dtype_counter.most_common():
        ratio = count / max(total_tensors, 1)
        print(f"  - {dtype_name}: {count} params ({ratio:.2%})")

    dominant_dtype = dtype_counter.most_common(1)[0][0] if dtype_counter else "unknown"
    print(f"[LoRA checkpoint files] dominant dtype: {dominant_dtype}")
    return dominant_dtype


def ask_should_merge() -> bool:
    answer = input("\nContinue to merge after inspecting dtypes? [y/N]: ").strip().lower()
    return answer in {"y", "yes"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge LoRA weights into the base model.")
    parser.add_argument(
        "--base_model_path",
        type=str,
        default="./checkpoints/Qwen3-VL-4B-Instruct",
        help="基础模型的路径 (例如: Qwen/Qwen-VL-Chat 或本地路径)",
    )
    parser.add_argument(
        "--lora_path",
        type=str,
        default="./outputs/MIMIC/sft",
        help="你训练好的 LoRA 权重文件夹路径",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./checkpoints/qwen3-vl-MIMIC-sft",
        help="合并后的完整模型保存路径",
    )
    parser.add_argument(
        "--load_dtype",
        choices=["auto", "bf16", "fp16", "fp32"],
        default="auto",
        help="加载 base model 时使用的精度；auto 表示让 transformers 自动决定",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="打印精度信息后，不再二次确认，直接合并",
    )

    args = parser.parse_args()

    dtype_map = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    torch_dtype = dtype_map.get(args.load_dtype)

    print(f"Loading base model from {args.base_model_path}...")
    base_model = AutoModelForImageTextToText.from_pretrained(
        args.base_model_path,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        device_map="cpu",
    )

    summarize_parameter_dtypes(base_model, "Base model")
    detect_lora_storage_dtype(args.lora_path)

    print(f"\nLoading LoRA from {args.lora_path}...")
    peft_model = PeftModel.from_pretrained(
        base_model,
        args.lora_path,
        device_map="cpu",
    )

    summarize_parameter_dtypes(peft_model, "Base model + LoRA loaded")

    if not args.yes and not ask_should_merge():
        print("Merge cancelled by user.")
        return

    print("Merging weights (this might take a while)...")
    merged_model = peft_model.merge_and_unload()

    summarize_parameter_dtypes(merged_model, "Merged model")

    print(f"Saving merged model to {args.output_path}...")
    merged_model.save_pretrained(args.output_path, safe_serialization=True)

    print("Saving tokenizer/processor...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.base_model_path, trust_remote_code=True)
        tokenizer.save_pretrained(args.output_path)
    except Exception as exc:
        print(f"Tokenizer saving skipped or failed: {exc}")

    try:
        processor = AutoProcessor.from_pretrained(args.base_model_path, trust_remote_code=True)
        processor.save_pretrained(args.output_path)
    except Exception as exc:
        print(f"Processor saving skipped or failed: {exc}")

    print("✅ Merge complete! You can now use the model at:", args.output_path)


if __name__ == "__main__":
    main()
