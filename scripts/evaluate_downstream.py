import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Set

os.environ.setdefault("NCCL_IB_DISABLE", "1")
os.environ.setdefault("NCCL_P2P_DISABLE", "1")
os.environ.setdefault("NCCL_SHM_DISABLE", "1")

import torch
from peft import PeftModel
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from qwen_vl_utils import process_vision_info


DATASET_FILES = {
    "MIMIC": "./data/annotations/MIMIC_test.json",
    "OBSCAN": "./data/annotations/OBSCAN_test.json",
    "SLAKE": "./data/annotations/SLAKE_test.json",
    "PathVQA": "./data/annotations/PathVQA_test.json",
    "VQARAD": "./data/annotations/VQARAD_test.json",
}

IMAGE_ROOTS = {
    "MIMIC": "./data/images/MIMIC",
    "OBSCAN": "./data/images/OBSCAN",
    "SLAKE": "./data/images/SLAKE",
    "PathVQA": "./data/images/PathVQA",
    "VQARAD": "./data/images/VQARAD",
}

ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.IGNORECASE | re.DOTALL)

MIN_IMAGE_PIXELS = 1024 * 1024
MAX_IMAGE_PIXELS = 4096 * 4096
MATCHING_POLICY = "relaxed_bidirectional_phrase_v1"


def parse_args():
    parser = argparse.ArgumentParser(description="Batch test local Qwen3-VL base model + LoRA adapter")
    parser.add_argument("--base-model", default="./checkpoints/Qwen3-VL-4B-Instruct")
    parser.add_argument(
        "--adapter",
        default="./output/qwen3vl_4b_instruct_slake_sa_morelayer",
    )
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="bf16")
    parser.add_argument("--attn-impl", default="eager")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help=(
            "Base evaluation seed. Each dataset sample uses seed + sample_index "
            "so resumed evaluation reproduces the same generations."
        ),
    )
    parser.add_argument("--limit", type=int, default=0, help="0 means use all samples")
    parser.add_argument(
        "--datasets",
        default="SLAKE",
        help="Comma-separated dataset names from: MIMIC, OBS, SLAKE, PathVQA, VQARAD",
    )
    parser.add_argument(
        "--output",
        default="./results/test_sa/batch_eval_results_sft_slake_sa_morelayer.json",
        help="Path to save detailed results",
    )
    parser.add_argument(
        "--annotation-path",
        default="",
        help="Optional test annotation override. Use with one --datasets value.",
    )
    parser.add_argument(
        "--image-root",
        default="",
        help="Optional image-root override. Use with one --datasets value.",
    )
    parser.add_argument(
        "--system",
        default=(
            "You are a medical AI assistant. "
            "Answer the question based on the image. "
            "Put the final short answer inside <answer></answer>."
        ),
    )
    parser.add_argument("--merge-and-unload", action="store_true")
    return parser.parse_args()


def resolve_dtype(dtype_name: str) -> torch.dtype:
    if dtype_name == "bf16":
        return torch.bfloat16
    if dtype_name == "fp32":
        return torch.float32
    return torch.float16


def load_processor(base_model: str, adapter_path: str):
    adapter_dir = Path(adapter_path)
    if adapter_dir.is_dir() and (adapter_dir / "preprocessor_config.json").exists():
        return AutoProcessor.from_pretrained(adapter_path)
    return AutoProcessor.from_pretrained(base_model)


def normalize_answer(value: Any) -> str:
    if isinstance(value, list):
        if len(value) > 1:
            value = ", ".join(str(x).strip() for x in value)
        else:
            value = value[0] if value else ""
    return str(value).strip().lower()


# Commas are first treated as label separators. Punctuation is then removed
# inside each label, so multi-label subset matching remains possible.
ANSWER_SPLIT_RE = re.compile(r"\s*(?:,|，|;|；|\n|\r)\s*")
PERIOD_COMMA_RE = re.compile(r"[\.,，。]+")
MULTISPACE_RE = re.compile(r"\s+")


def normalize_answer_label(value: Any) -> str:
    """Normalize one atomic answer label for comparison.

    Rules requested for evaluation:
    - lowercase and trim whitespace;
    - remove periods and commas;
    - map ``top`` to ``upper`` and ``bottom`` to ``lower``;
    - preserve slash expressions such as ``pulmonary edema/hazy opacity``.
    """
    text = str(value or "").replace("<|im_end|>", "").strip().lower()
    text = PERIOD_COMMA_RE.sub(" ", text)
    text = MULTISPACE_RE.sub(" ", text).strip()

    # Canonicalize vertical-position words without changing left/right.
    text = re.sub(r"\btop\b", "upper", text)
    text = re.sub(r"\bbottom\b", "lower", text)
    return MULTISPACE_RE.sub(" ", text).strip()


def split_answer_labels(value: Any) -> List[str]:
    """Split an answer into normalized comma-separated labels."""
    if isinstance(value, (list, tuple, set)):
        raw_parts: List[str] = []
        for item in value:
            raw_parts.extend(ANSWER_SPLIT_RE.split(str(item or "")))
    else:
        raw_parts = ANSWER_SPLIT_RE.split(str(value or ""))

    labels: List[str] = []
    for part in raw_parts:
        label = normalize_answer_label(part)
        if label and label not in labels:
            labels.append(label)
    return labels


def labels_equivalent(pred_label: str, gold_label: str) -> bool:
    """Relaxed comparison between two atomic answer labels.

    In addition to exact equality and the upper/lower hierarchy, a shorter
    complete token phrase may match a longer phrase. Raw character substring
    matching is deliberately avoided, so ``no`` cannot match ``nodule``.
    """
    if pred_label == gold_label:
        return True

    pred_tokens = pred_label.split()
    gold_tokens = gold_label.split()
    pred_token_set = set(pred_tokens)
    gold_token_set = set(gold_tokens)

    for vertical_word in ("upper", "lower"):
        pred_is_generic = pred_label == vertical_word
        gold_is_generic = gold_label == vertical_word
        if pred_is_generic and vertical_word in gold_token_set:
            return True
        if gold_is_generic and vertical_word in pred_token_set:
            return True

    def is_contiguous_subsequence(
        shorter: List[str],
        longer: List[str],
    ) -> bool:
        if not shorter or len(shorter) > len(longer):
            return False
        width = len(shorter)
        return any(
            longer[index:index + width] == shorter
            for index in range(len(longer) - width + 1)
        )

    return (
        is_contiguous_subsequence(pred_tokens, gold_tokens)
        or is_contiguous_subsequence(gold_tokens, pred_tokens)
    )


def extract_answer_tag(text: str) -> str:
    """
    Extract prediction answer.

    Priority:
    1. If <answer>...</answer> exists, use content inside the tag.
    2. Otherwise, use the whole raw output.
    3. Remove <|im_end|>.
    4. Normalize to lowercase.
    """
    text = str(text or "").replace("<|im_end|>", "")

    m = ANSWER_RE.search(text)
    if m:
        answer = m.group(1)
    else:
        answer = text

    answer = answer.replace("\r", "").replace("\n", " ")
    return answer.strip().lower()


def build_image_path(dataset_name: str, item: Dict[str, Any]) -> str:
    image_path = str(item.get("image_path", "")).strip()
    image_ids = item.get("image_ids") or []
    if isinstance(image_ids, str):
        image_id = image_ids.strip()
    elif isinstance(image_ids, list) and image_ids:
        image_id = str(image_ids[0]).strip()
    else:
        image_id = ""

    candidate = image_path or image_id
    if not candidate:
        raise ValueError(f"Missing image path info in {dataset_name} sample")

    root = IMAGE_ROOTS[dataset_name]
    candidate_path = Path(candidate)

    if dataset_name == "MIMIC":
        image_stem = candidate_path.stem
        return str(Path(root) / f"{image_stem}.png")
    if dataset_name == "OBSCAN":
        image_stem = candidate_path.stem
        return str(Path(root) / f"{image_stem}.jpg")
    if dataset_name == "PathVQA":
        image_name = candidate_path.name
        if candidate_path.suffix.lower() != ".jpg":
            image_name = f"{image_name}.jpg"
        return str(Path(root) / image_name)
    if dataset_name == "VQARAD":
        return str(Path(root) / candidate_path.name)

    return str(Path(root) / candidate)


def is_answer_correct(generated_text: str, gold_answer: str) -> bool:
    """Relaxed answer evaluation.

    The prediction is accepted when either label set covers the other under
    ``labels_equivalent``. This exactly restores the earlier loose evaluator:

        GT   = atelectasis, lung lesion
        Pred = atelectasis

        GT   = enteric tube
        Pred = enteric tube, lung opacity
    """
    pred_labels = split_answer_labels(generated_text)
    gold_labels = split_answer_labels(gold_answer)
    if not pred_labels or not gold_labels:
        return False

    pred_covered_by_gold = all(
        any(labels_equivalent(pred, gold) for gold in gold_labels)
        for pred in pred_labels
    )
    gold_covered_by_pred = all(
        any(labels_equivalent(pred, gold) for pred in pred_labels)
        for gold in gold_labels
    )

    return pred_covered_by_gold or gold_covered_by_pred


def rescore_existing_results(results: List[Dict[str, Any]]) -> None:
    """Recompute stored metrics with the current evaluator when resuming.

    Without this step, an existing output JSON would keep old ``is_correct``
    values and all completed samples would be skipped with stale scores.
    """
    for item in results:
        gold_answer = item.get("gold_answer", "")
        pred_answers = item.get("pred_answers")
        if not isinstance(pred_answers, list):
            single = item.get("pred_answer", "")
            pred_answers = [single] if str(single or "").strip() else []

        flags = [is_answer_correct(pred, gold_answer) for pred in pred_answers]
        item["sample_correct_flags"] = flags
        item["is_correct"] = bool(flags[0]) if flags else False
        item["pass_at_k"] = any(flags)
        item["num_correct_samples"] = sum(flags)
        if pred_answers:
            item["pred_answer"] = pred_answers[0]


def generate_samples(
    model,
    processor,
    device: str,
    system_prompt: str,
    image_path: str,
    question: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    num_samples: int,
) -> List[str]:
    messages = []
    system_prompt = """You are a medical AI. 
    Output short reasoning inside <think>...</think> and the final short answer inside <answer>...</answer>.
    Your <think> block MUST strictly follow 4 steps:
    1. Question Interpretation
    2. Anatomical Localization
    3. Visual Observation & KG Alignment (Format: `(Entity - Attribute - Finding)`)
    4. Clinical Reasoning"""

    # messages.append({"role": "system", "content": system_prompt})
    messages.append(
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": question},
            ],
        }
    )

    text_prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text_prompt],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}

    do_sample = temperature > 0
    k = max(1, num_samples) if do_sample else 1
    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            num_return_sequences=k,
        )

    prompt_len = inputs["input_ids"].shape[1]
    trimmed = [out_ids[prompt_len:] for out_ids in generated_ids]
    return processor.batch_decode(
        trimmed,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )


def load_existing_results(output_path: Path) -> Dict[str, Any]:
    if not output_path.is_file() or output_path.stat().st_size == 0:
        return {"summary": {}, "results": []}

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in existing output file: {output_path}")

    summary = payload.get("summary") or {}
    results = payload.get("results") or []
    if not isinstance(summary, dict):
        raise ValueError(f"Expected 'summary' to be an object in {output_path}")
    if not isinstance(results, list):
        raise ValueError(f"Expected 'results' to be an array in {output_path}")

    return {"summary": summary, "results": results}


def make_resume_key(dataset_name: str, sample_index: int) -> str:
    return f"{dataset_name}::{sample_index}"


def build_dataset_summary(results: List[Dict[str, Any]], dataset_name: str, k: int) -> Dict[str, Any]:
    dataset_results = [item for item in results if item.get("dataset") == dataset_name]
    total = len(dataset_results)
    correct = sum(1 for item in dataset_results if item.get("is_correct"))
    pass_hits = sum(1 for item in dataset_results if item.get("pass_at_k"))
    total_correct_samples = sum(int(item.get("num_correct_samples", 0)) for item in dataset_results)
    missing_images = sum(1 for item in dataset_results if not item.get("raw_outputs"))

    return {
        "total": total,
        "correct": correct,
        "accuracy": correct / max(total, 1),
        "pass_at_k_hits": pass_hits,
        "pass_at_k": pass_hits / max(total, 1),
        "total_correct_samples": total_correct_samples,
        "total_accuracy_denominator": k * total,
        "total_accuracy_over_k": total_correct_samples / max(k * total, 1),
        "missing_images": missing_images,
        "k": k,
    }


def build_output_payload(
    args: argparse.Namespace,
    summary: Dict[str, Dict[str, Any]],
    results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    return {
        "evaluation": {
            "matching_policy": MATCHING_POLICY,
            "max_new_tokens": int(args.max_new_tokens),
            "temperature": float(args.temperature),
            "top_p": float(args.top_p),
            "num_samples": int(args.num_samples),
            "seed": int(args.seed),
            "sample_seed_policy": "base_seed_plus_zero_based_sample_index",
            "min_image_pixels": MIN_IMAGE_PIXELS,
            "max_image_pixels": MAX_IMAGE_PIXELS,
        },
        "summary": summary,
        "results": results,
    }


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if not os.path.isdir(args.base_model):
        raise FileNotFoundError(f"Base model directory not found: {args.base_model}")
    if not os.path.isdir(args.adapter):
        raise FileNotFoundError(f"Adapter directory not found: {args.adapter}")

    selected_datasets = [x.strip() for x in args.datasets.split(",") if x.strip()]
    bad = [x for x in selected_datasets if x not in DATASET_FILES]
    if bad:
        raise ValueError(f"Unsupported datasets: {bad}")
    if (args.annotation_path or args.image_root) and len(selected_datasets) != 1:
        raise ValueError(
            "--annotation-path/--image-root overrides require exactly one dataset"
        )
    if args.annotation_path:
        DATASET_FILES[selected_datasets[0]] = args.annotation_path
    if args.image_root:
        IMAGE_ROOTS[selected_datasets[0]] = args.image_root

    torch_dtype = resolve_dtype(args.dtype)

    print(f"[INFO] base_model={args.base_model}")
    print(f"[INFO] adapter={args.adapter}")
    print(f"[INFO] device={args.device}")
    print(f"[INFO] dtype={args.dtype}")
    print(f"[INFO] attn_impl={args.attn_impl}")
    print(f"[INFO] datasets={selected_datasets}")
    print(f"[INFO] num_samples={args.num_samples}")
    print(f"[INFO] matching_policy={MATCHING_POLICY}")
    print(f"[INFO] evaluation_seed={args.seed} (per-sample deterministic)")

    print("[INFO] Loading base model...")
    base_model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.base_model,
        torch_dtype=torch_dtype,
        attn_implementation=args.attn_impl,
        device_map=args.device,
    )

    print("[INFO] Loading LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, args.adapter)
    for name, param in model.named_parameters():
        if "lora" in name:
            print(f"LoRA Layer: {name} | Sum: {param.sum().item()}")
            break

    if args.merge_and_unload:
        print("[INFO] Merging LoRA weights into base model...")
        model = model.merge_and_unload()

    model.eval()

    processor = load_processor(args.base_model, args.adapter)
    processor.image_processor.min_pixels = MIN_IMAGE_PIXELS
    processor.image_processor.max_pixels = MAX_IMAGE_PIXELS
    print(
        f"[PIXELS] eval dynamic bounds: "
        f"min={MIN_IMAGE_PIXELS} (1024x1024), "
        f"max={MAX_IMAGE_PIXELS} (4096x4096)"
    )
    output_path = Path(args.output)
    k = max(1, args.num_samples) if args.temperature > 0 else 1

    existing_payload = load_existing_results(output_path)
    results: List[Dict[str, Any]] = list(existing_payload["results"])

    # Always refresh old results with the current relaxed evaluator before
    # resume keys are built. This rescoring does not regenerate model outputs.
    rescore_existing_results(results)
    summary: Dict[str, Dict[str, Any]] = {}
    for existing_dataset in sorted(
        {str(item.get("dataset")) for item in results if item.get("dataset")}
    ):
        summary[existing_dataset] = build_dataset_summary(
            results, existing_dataset, k
        )

    completed_keys: Set[str] = set()

    for item in results:
        dataset_name = item.get("dataset")
        sample_index = item.get("sample_index")
        if isinstance(dataset_name, str) and isinstance(sample_index, int):
            completed_keys.add(make_resume_key(dataset_name, sample_index))

    if results:
        print(f"[INFO] Resuming from existing output: {output_path}")
        print(f"[INFO] Existing completed samples: {len(completed_keys)}")

    for dataset_name in selected_datasets:
        data_path = Path(DATASET_FILES[dataset_name])
        if not data_path.is_file():
            raise FileNotFoundError(f"Dataset file not found: {data_path}")

        data = json.loads(data_path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError(f"Expected JSON array in {data_path}")
        if args.limit > 0:
            data = data[: args.limit]

        dataset_existing_results = [item for item in results if item.get("dataset") == dataset_name]
        correct = sum(1 for item in dataset_existing_results if item.get("is_correct"))
        pass_hits = sum(1 for item in dataset_existing_results if item.get("pass_at_k"))
        total = len(dataset_existing_results)
        missing_images = sum(1 for item in dataset_existing_results if not item.get("raw_outputs"))
        total_correct_samples = sum(int(item.get("num_correct_samples", 0)) for item in dataset_existing_results)

        print(f"\n[INFO] Evaluating {dataset_name} on {len(data)} samples...")
        if total:
            print(f"[INFO] Found {total} completed samples for {dataset_name}, will skip them.")

        for idx, item in enumerate(data, start=1):
            resume_key = make_resume_key(dataset_name, idx - 1)
            if resume_key in completed_keys:
                continue

            image_path = build_image_path(dataset_name, item)
            question = str(item.get("question", "")).strip()
            gold_answer = normalize_answer(item.get("answer"))

            if not os.path.exists(image_path):
                missing_images += 1
                raw_outputs: List[str] = []
                pred_answers: List[str] = []
                flags: List[bool] = []
            else:
                # Per-sample seeding makes a resumed run identical to a clean
                # run even when earlier samples already exist in the JSON.
                sample_seed = int(args.seed) + idx - 1
                torch.manual_seed(sample_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(sample_seed)
                raw_outputs = generate_samples(
                    model=model,
                    processor=processor,
                    device=args.device,
                    system_prompt=args.system,
                    image_path=image_path,
                    question=question,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_samples=args.num_samples,
                )

                pred_answers = [extract_answer_tag(text) for text in raw_outputs]

                # Use cleaned pred_answers for correctness evaluation.
                flags = [is_answer_correct(pred, gold_answer) for pred in pred_answers]

            pred_answer = pred_answers[0] if pred_answers else ""
            raw_output = raw_outputs[0] if raw_outputs else ""
            is_correct = flags[0] if flags else False
            pass_at_k = any(flags)
            num_correct_samples = sum(flags)

            total += 1
            correct += int(is_correct)
            pass_hits += int(pass_at_k)
            total_correct_samples += num_correct_samples

            result = {
                "dataset": dataset_name,
                "sample_index": idx - 1,
                "question": question,
                "image_path": image_path,
                "gold_answer": gold_answer,
                "pred_answer": pred_answer,
                "is_correct": is_correct,
                "raw_output": raw_output,
                "num_samples": k,
                "pred_answers": pred_answers,
                "sample_correct_flags": flags,
                "raw_outputs": raw_outputs,
                "pass_at_k": pass_at_k,
                "num_correct_samples": num_correct_samples,
            }

            if "idx" in item:
                result["idx"] = item["idx"]
            if "subject_id" in item:
                result["subject_id"] = item["subject_id"]
            if "image_ids" in item:
                result["image_ids"] = item["image_ids"]

            results.append(result)
            completed_keys.add(resume_key)

            if idx % 10 == 0 or idx == len(data):
                summary[dataset_name] = build_dataset_summary(results, dataset_name, k)
                print(
                    f"[{dataset_name}] {idx}/{len(data)} | "
                    f"acc={correct}/{total}={correct / max(total, 1):.4f} | "
                    f"pass@{k}={pass_hits}/{total}={pass_hits / max(total, 1):.4f} | "
                    f"total_acc@{k}={total_correct_samples}/{k * total}="
                    f"{total_correct_samples / max(k * total, 1):.4f} | "
                    f"missing_images={missing_images}"
                )
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text(
                    json.dumps(
                        build_output_payload(args, summary, results),
                        ensure_ascii=False,
                        indent=2,
                    ) + "\n",
                    encoding="utf-8",
                )

        summary[dataset_name] = build_dataset_summary(results, dataset_name, k)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            build_output_payload(args, summary, results),
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )

    print("\n" + "=" * 80)
    print("Summary")
    for dataset_name in selected_datasets:
        stats = summary[dataset_name]
        print(
            f"{dataset_name}: total={stats['total']}, correct={stats['correct']}, "
            f"accuracy={stats['accuracy']:.4f}, pass@{stats['k']}={stats['pass_at_k']:.4f}, "
            f"total_acc@{stats['k']}={stats['total_accuracy_over_k']:.4f}, "
            f"total_correct_samples={stats['total_correct_samples']}/{stats['total_accuracy_denominator']}, "
            f"missing_images={stats['missing_images']}"
        )
    print(f"Detailed results written to: {output_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
