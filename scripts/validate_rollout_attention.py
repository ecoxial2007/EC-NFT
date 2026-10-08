#!/usr/bin/env python3
"""Fail fast when a rollout DB lacks directly recorded decode attention."""

import argparse
import json
import sqlite3


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sqlite", required=True)
    parser.add_argument("--dataset", required=True)
    args = parser.parse_args()

    connection = sqlite3.connect(args.sqlite)
    completed_groups = 0
    total_groups = 0
    rollout_count = 0
    valid_count = 0
    problems = []
    incomplete_examples = []
    missing_image_groups = 0
    generation_error_groups = 0
    try:
        all_rows = connection.execute(
            """
            SELECT sample_index, complete, record_json
            FROM rollout_groups
            WHERE perturbation = 'clean'
            ORDER BY sample_index
            """
        )
        for sample_index, complete, record_json in all_rows:
            total_groups += 1
            if int(complete) == 1:
                continue
            record = json.loads(record_json)
            missing_image = bool(record.get("missing_image"))
            generation_error = record.get("generation_error")
            missing_image_groups += int(missing_image)
            generation_error_groups += int(bool(generation_error))
            if len(incomplete_examples) < 5:
                incomplete_examples.append(
                    {
                        "sample_index": sample_index,
                        "missing_image": missing_image,
                        "generation_error": generation_error,
                        "filter_reason": record.get("filter_reason"),
                        "original_image_paths": record.get("original_image_paths"),
                    }
                )

        rows = connection.execute(
            """
            SELECT sample_index, record_json
            FROM rollout_groups
            WHERE perturbation = 'clean' AND complete = 1
            ORDER BY sample_index
            """
        )
        for sample_index, record_json in rows:
            completed_groups += 1
            record = json.loads(record_json)
            for rollout in record.get("rollouts") or []:
                rollout_count += 1
                rollout_index = rollout.get("rollout_index")
                token_ids = rollout.get("generated_token_ids") or []
                score = rollout.get("old_policy_score") or {}
                token_logprobs = score.get("token_logprobs") or []
                trace = rollout.get("evidence_trace") or {}
                attention_map = trace.get("visual_attention_map") or []
                visual_tokens = int(trace.get("visual_tokens") or 0)
                valid = (
                    bool(trace.get("valid"))
                    and len(attention_map) == visual_tokens
                    and visual_tokens > 0
                    and len(token_ids) == len(token_logprobs)
                    and len(token_ids) > 0
                )
                if valid:
                    valid_count += 1
                elif len(problems) < 10:
                    problems.append(
                        {
                            "sample_index": sample_index,
                            "rollout_index": rollout_index,
                            "trace_reason": trace.get("reason"),
                            "tokens": len(token_ids),
                            "token_logprobs": len(token_logprobs),
                            "visual_tokens": visual_tokens,
                            "attention_values": len(attention_map),
                        }
                    )
    finally:
        connection.close()

    if completed_groups == 0 or rollout_count == 0:
        raise RuntimeError(
            f"{args.dataset}: rollout database has no completed clean groups; "
            f"total_clean_groups={total_groups}, "
            f"missing_image_groups={missing_image_groups}, "
            f"generation_error_groups={generation_error_groups}, "
            f"examples={incomplete_examples}. Check rollout.log."
        )
    if valid_count != rollout_count:
        raise RuntimeError(
            f"{args.dataset}: direct decode attention validation failed: "
            f"valid={valid_count}/{rollout_count}; examples={problems}"
        )
    print(
        f"[ATTENTION-CHECK] {args.dataset}: "
        f"valid={valid_count}/{rollout_count}, completed_groups={completed_groups}"
    )


if __name__ == "__main__":
    main()
