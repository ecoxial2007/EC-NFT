#!/usr/bin/env python3
"""Export rollout evidence maps without another model forward.

The exporter reads the normalized ``visual_attention_map`` saved during the
Qwen3-VL decode.  It never reloads the model and never recomputes attention.
For each sample it averages valid correct-rollout maps, reshapes visual tokens
to their merged image grid, and writes the original image, an overlay, and QA
text under ``visualization/<dataset>/<sample_id>/``.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sqlite3
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
from PIL import Image


def safe_name(value: Any, fallback: str) -> str:
    text = str(value or "").strip()
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("._")
    return text or fallback


def choose_sample_id(base: Dict[str, Any], sample_index: int) -> str:
    for key in ("idx", "sample_id", "question_id", "subject_id"):
        value = base.get(key)
        if value not in (None, ""):
            return safe_name(value, f"sample_{sample_index:06d}")
    return f"sample_{sample_index:06d}"


def choose_image_path(base: Dict[str, Any], record: Dict[str, Any]) -> Path:
    used = record.get("used_image_paths") or []
    if isinstance(used, str):
        used = [used]
    candidates: List[str] = [str(x) for x in used if str(x).strip()]
    if not candidates:
        paths = base.get("image_paths") or []
        if isinstance(paths, str):
            paths = [paths]
        candidates.extend(str(x) for x in paths if str(x).strip())
    if not candidates and base.get("image_path"):
        candidates.append(str(base["image_path"]))
    if not candidates:
        raise ValueError("sample has no image path")
    path = Path(candidates[0])
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def valid_rollout_maps(record: Dict[str, Any], positive_only: bool) -> List[Dict[str, Any]]:
    traces: List[Dict[str, Any]] = []
    for rollout in record.get("rollouts") or []:
        if not isinstance(rollout, dict):
            continue
        if positive_only and int(rollout.get("reward", 0)) != 1:
            continue
        trace = rollout.get("evidence_trace")
        if not isinstance(trace, dict) or not trace.get("valid"):
            continue
        values = trace.get("visual_attention_map")
        if isinstance(values, list) and values:
            traces.append({"trace": trace, "rollout": rollout})
    return traces


def infer_grid_shape(trace: Dict[str, Any], vector_size: int) -> Tuple[int, int, int]:
    grid = trace.get("image_grid_thw") or []
    if isinstance(grid, list) and len(grid) >= 3:
        t, h, w = (max(1, int(grid[0])), max(1, int(grid[1])), max(1, int(grid[2])))
        merge_sq = (t * h * w) / max(vector_size, 1)
        merge = max(1, int(round(math.sqrt(merge_sq))))
        merged_h = max(1, h // merge)
        merged_w = max(1, w // merge)
        if t * merged_h * merged_w == vector_size:
            return t, merged_h, merged_w

    # Defensive fallback: choose a factor pair close to the source aspect.
    target_ratio = 1.0
    if isinstance(grid, list) and len(grid) >= 3 and int(grid[2]) > 0:
        target_ratio = float(grid[1]) / float(grid[2])
    best = (1, vector_size)
    best_error = float("inf")
    for h in range(1, int(math.sqrt(vector_size)) + 1):
        if vector_size % h:
            continue
        w = vector_size // h
        error = abs((h / w) - target_ratio)
        if error < best_error:
            best = (h, w)
            best_error = error
    return 1, best[0], best[1]


def average_map(entries: Sequence[Dict[str, Any]]) -> Tuple[np.ndarray, Dict[str, Any]]:
    first = entries[0]["trace"]
    size = len(first["visual_attention_map"])
    compatible = [
        np.asarray(entry["trace"]["visual_attention_map"], dtype=np.float32)
        for entry in entries
        if len(entry["trace"]["visual_attention_map"]) == size
    ]
    if not compatible:
        raise ValueError("no compatible attention maps")
    vector = np.mean(np.stack(compatible, axis=0), axis=0)
    t, h, w = infer_grid_shape(first, vector.size)
    grid = vector.reshape(t, h, w).mean(axis=0)
    return grid, first


def normalize_map(values: np.ndarray) -> np.ndarray:
    low = float(np.percentile(values, 5))
    high = float(np.percentile(values, 99))
    if high <= low:
        low = float(values.min())
        high = float(values.max())
    return np.clip((values - low) / max(high - low, 1e-12), 0.0, 1.0)


def colorize(values: np.ndarray) -> np.ndarray:
    # A compact blue-cyan-yellow-red map implemented without matplotlib.
    points = np.asarray([0.00, 0.33, 0.66, 1.00], dtype=np.float32)
    colors = np.asarray(
        [[14, 22, 66], [0, 153, 255], [255, 224, 64], [210, 24, 32]],
        dtype=np.float32,
    )
    channels = [np.interp(values, points, colors[:, i]) for i in range(3)]
    return np.stack(channels, axis=-1).astype(np.uint8)


def qa_text(
    dataset: str,
    sample_id: str,
    base: Dict[str, Any],
    image_path: Path,
    entries: Sequence[Dict[str, Any]],
) -> str:
    lines = [
        f"dataset: {dataset}",
        f"sample_id: {sample_id}",
        f"sample_index: {base.get('sample_index', '')}",
        f"image_path: {image_path}",
        f"question: {base.get('question', '')}",
        f"gold_answer: {base.get('gold_answer', '')}",
        f"correct_rollouts_visualized: {len(entries)}",
        "",
    ]
    for entry in entries:
        rollout = entry["rollout"]
        lines.extend(
            [
                f"--- rollout_{rollout.get('rollout_index', '')} ---",
                f"reward: {rollout.get('reward', '')}",
                f"pred_answer: {rollout.get('pred_answer', '')}",
                f"raw_output: {rollout.get('raw_output', '')}",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def rows(connection: sqlite3.Connection) -> Iterable[Tuple[int, str, str]]:
    return connection.execute(
        """
        SELECT s.sample_index, s.base_record_json, g.record_json
        FROM samples AS s
        JOIN rollout_groups AS g ON g.sample_index = s.sample_index
        WHERE g.perturbation = 'clean' AND g.complete = 1
        ORDER BY s.sample_index
        """
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sqlite", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--max-samples", type=int, default=0, help="0 exports all")
    parser.add_argument("--positive-only", type=int, choices=(0, 1), default=1)
    parser.add_argument("--overlay-alpha", type=float, default=0.48)
    args = parser.parse_args()

    dataset_key = safe_name(args.dataset.lower(), "dataset")
    dataset_root = Path(args.output_root) / dataset_key
    dataset_root.mkdir(parents=True, exist_ok=True)
    manifest: List[Dict[str, Any]] = []
    used_ids = set()

    connection = sqlite3.connect(args.sqlite)
    try:
        for sample_index, base_json, record_json in rows(connection):
            if args.max_samples > 0 and len(manifest) >= args.max_samples:
                break
            base = json.loads(base_json)
            record = json.loads(record_json)
            entries = valid_rollout_maps(record, bool(args.positive_only))
            if not entries:
                continue
            try:
                image_path = choose_image_path(base, record)
                heatmap, trace = average_map(entries)
            except (OSError, ValueError) as error:
                print(f"[WARN] sample {sample_index}: {error}")
                continue

            sample_id = choose_sample_id(base, int(sample_index))
            if sample_id in used_ids:
                sample_id = f"{sample_id}_{int(sample_index):06d}"
            used_ids.add(sample_id)
            sample_dir = dataset_root / sample_id
            sample_dir.mkdir(parents=True, exist_ok=True)

            image_id = safe_name(image_path.stem, f"image_{int(sample_index):06d}")
            original_path = sample_dir / f"{image_id}_original.jpg"
            visualization_path = sample_dir / f"{image_id}_visualization.jpg"
            qa_path = sample_dir / "qa.txt"

            original = Image.open(image_path).convert("RGB")
            original.save(original_path, quality=95)
            normalized = normalize_map(heatmap)
            resampling = getattr(Image, "Resampling", Image)
            resized = Image.fromarray((normalized * 255).astype(np.uint8)).resize(
                original.size, resampling.BILINEAR
            )
            normalized_full = np.asarray(resized, dtype=np.float32) / 255.0
            colors = colorize(normalized_full).astype(np.float32)
            base_pixels = np.asarray(original, dtype=np.float32)
            alpha = np.clip(args.overlay_alpha * (0.25 + 0.75 * normalized_full), 0.0, 1.0)
            overlay = base_pixels * (1.0 - alpha[..., None]) + colors * alpha[..., None]
            Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8)).save(
                visualization_path, quality=95
            )
            qa_path.write_text(
                qa_text(args.dataset, sample_id, base, image_path, entries),
                encoding="utf-8",
            )
            manifest.append(
                {
                    "dataset": args.dataset,
                    "sample_index": int(sample_index),
                    "sample_id": sample_id,
                    "image_path": str(image_path),
                    "original": str(original_path),
                    "visualization": str(visualization_path),
                    "qa": str(qa_path),
                    "rollouts_averaged": len(entries),
                    "attention_capture": trace.get("attention_capture"),
                }
            )
    finally:
        connection.close()

    manifest_path = dataset_root / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"[VISUALIZATION] exported {len(manifest)} samples -> {dataset_root}")


if __name__ == "__main__":
    main()
