"""CPU-only evidence scoring and positive-group weight calibration.

Attention is normalized over the full image before border exclusion. The
remaining attention values are NOT renormalized after excluding the border.
"""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from pathlib import Path
from typing import Sequence


def open_rollouts(path: str | Path) -> sqlite3.Connection:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    connection = sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True)
    connection.execute("PRAGMA query_only=ON")
    connection.execute("BEGIN")
    return connection


def eligible_groups(connection: sqlite3.Connection):
    return connection.execute(
        """
        SELECT g.sample_index, g.perturbation, g.r_hat, g.record_json,
               s.base_record_json
        FROM rollout_groups AS g
        JOIN samples AS s ON s.sample_index = g.sample_index
        WHERE g.complete = 1 AND g.nft_prompt_eligible = 1
          AND g.r_hat > 0 AND g.r_hat < 1 AND g.perturbation = 'clean'
        ORDER BY g.sample_index, g.perturbation
        """
    )


def source_fingerprint(connection: sqlite3.Connection) -> str:
    """Hash actual mixed-group contents, including labels and old log-probs."""
    digest = hashlib.sha256()
    for row in eligible_groups(connection):
        digest.update(json.dumps(list(row), ensure_ascii=False).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def load_cache(cache_path: str | Path, sqlite_path: str | Path) -> dict:
    cache = json.loads(Path(cache_path).read_text(encoding="utf-8"))
    if cache.get("version") != 1 or cache.get("score_type") != "interior_topk_attention_mass":
        raise ValueError("Unsupported evidence cache; run scripts/score_evidence.py")
    connection = open_rollouts(sqlite_path)
    try:
        current = source_fingerprint(connection)
    finally:
        connection.close()
    if current != cache.get("source_fingerprint"):
        raise ValueError("Evidence cache is stale or belongs to a different rollout database")
    seen = set()
    for record in cache.get("records", []):
        key = (record["sample_index"], record["perturbation"], record["rollout_position"])
        weight = float(record["weight"])
        if key in seen or not math.isfinite(weight) or weight <= 0:
            raise ValueError("Duplicate or invalid evidence-cache record")
        seen.add(key)
        if record["reward"] not in (0, 1):
            raise ValueError("Non-binary reward in evidence cache")
        if record["reward"] == 0 and weight != 1.0:
            raise ValueError("Negative trajectories must retain unit evidence weights")
    return cache


def evidence_score(trace: dict, top_ratio: float = 0.1,
                   border_ratio: float = 0.05) -> float:
    if not 0 < top_ratio <= 1:
        raise ValueError("top_ratio must be in (0, 1]")
    if not 0 <= border_ratio < 0.5:
        raise ValueError("border_ratio must be in [0, 0.5)")
    if not trace.get("valid"):
        raise ValueError("Missing valid rollout-time attention; recollect rollouts")
    values = [float(value) for value in trace.get("visual_attention_map", [])]
    count = int(trace.get("visual_tokens", 0))
    if count <= 0 or len(values) != count:
        raise ValueError("Attention length does not match visual-token count")
    if any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError("Attention values must be finite and nonnegative")
    grid = trace.get("image_grid_thw", [])
    if len(grid) == 1 and isinstance(grid[0], list):
        grid = grid[0]
    if len(grid) != 3:
        raise ValueError("Expected one image grid [t, h, w]")
    t, h, w = map(int, grid)
    # The stored grid is before Qwen's spatial merge. Infer the merge size
    # from the recorded token count instead of hard-coding a checkpoint value.
    if min(t, h, w) <= 0:
        raise ValueError("Image-grid dimensions must be positive")
    merge_squared, remainder = divmod(t * h * w, count)
    merge = math.isqrt(merge_squared)
    if remainder or merge < 1 or merge * merge != merge_squared:
        raise ValueError("Image grid and visual-token count are inconsistent")
    if h % merge or w % merge:
        raise ValueError("Image grid is not divisible by the inferred merge size")
    h, w = h // merge, w // merge
    bh = max(0, min(round(h * border_ratio), (h - 1) // 2))
    bw = max(0, min(round(w * border_ratio), (w - 1) // 2))
    interior = [values[z * h * w + row * w + col]
                for z in range(t) for row in range(bh, h - bh)
                for col in range(bw, w - bw)]
    total = math.fsum(values)
    if total <= 0:
        return 0.0
    k = max(1, math.ceil(top_ratio * len(interior)))
    return math.fsum(sorted(interior, reverse=True)[:k]) / total


def positive_weights(scores: Sequence[float], beta: float = 0.25,
                     minimum: float = 0.5, maximum: float = 1.5,
                     eps: float = 1e-8) -> list[float]:
    """Relative scores -> clip -> renormalize -> clip -> blend with one.

This preserves the existing calibration implementation's order of operations.
The bounds apply to relative weights before the beta blend. Scores near zero
are floored for numerical stability. A single positive gets a unit weight.
"""
    if not 0 <= beta <= 1 or not 0 < minimum <= 1 <= maximum or eps <= 0:
        raise ValueError("Invalid calibration strength, weight bounds, or epsilon")
    raw = [float(value) for value in scores]
    if any(not math.isfinite(value) or value < 0 for value in raw):
        raise ValueError("Evidence scores must be finite and nonnegative")
    if not raw:
        return []
    values = [max(eps, value) for value in raw]
    mean = math.fsum(values) / len(values)
    relative = [min(maximum, max(minimum, value / mean)) for value in values]
    mean = math.fsum(relative) / len(relative)
    relative = [min(maximum, max(minimum, value / mean)) for value in relative]
    return [(1 - beta) + beta * value for value in relative]


def build_cache(sqlite_path: str | Path, *, top_ratio: float = 0.1,
                border_ratio: float = 0.05, beta: float = 0.25,
                minimum: float = 0.5, maximum: float = 1.5) -> dict:
    records = []
    groups = 0
    connection = open_rollouts(sqlite_path)
    try:
        fingerprint = source_fingerprint(connection)
        for sample, perturbation, db_mean, record_json, _ in eligible_groups(connection):
            group = json.loads(record_json)
            rollouts = group.get("rollouts", [])
            rewards = [rollout.get("reward") for rollout in rollouts]
            if not rewards or any(reward not in (0, 1) for reward in rewards):
                raise ValueError(f"Group {sample}: invalid binary rewards")
            mean = sum(rewards) / len(rewards)
            if not 0 < mean < 1 or not math.isclose(mean, float(db_mean), abs_tol=1e-6):
                raise ValueError(f"Group {sample}: rewards and r_hat disagree")
            if not math.isclose(mean, float(group.get("r_hat", db_mean)), abs_tol=1e-6):
                raise ValueError(f"Group {sample}: stale r_hat in record_json")
            for rollout in rollouts:
                tokens = rollout.get("generated_token_ids", [])
                logprobs = rollout.get("old_policy_score", {}).get("token_logprobs", [])
                if not tokens or len(tokens) != len(logprobs):
                    raise ValueError(f"Group {sample}: missing or misaligned token log-probs")
                if any(not math.isfinite(float(value)) for value in logprobs):
                    raise ValueError(f"Group {sample}: non-finite old log-probs")
            scores = [evidence_score(rollout.get("evidence_trace", {}),
                                     top_ratio, border_ratio)
                      if reward == 1 else 0.0
                      for rollout, reward in zip(rollouts, rewards)]
            weights = iter(positive_weights([score for score, reward in zip(scores, rewards)
                                             if reward == 1], beta, minimum, maximum))
            for position, (reward, score) in enumerate(zip(rewards, scores)):
                records.append({"sample_index": int(sample), "perturbation": perturbation,
                                "rollout_position": position, "reward": int(reward),
                                "score": score, "weight": next(weights) if reward == 1 else 1.0})
            groups += 1
    finally:
        connection.close()
    if not groups:
        raise ValueError("No complete mixed reward groups found in the rollout database")
    return {"version": 1, "score_type": "interior_topk_attention_mass",
            "source_fingerprint": fingerprint, "group_count": groups,
            "parameters": {"top_ratio": top_ratio, "border_ratio": border_ratio,
                           "beta": beta, "weight_min": minimum, "weight_max": maximum},
            "records": records}
