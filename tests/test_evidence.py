import copy
from contextlib import contextmanager
import json
import math
import os
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ec_nft.evidence import build_cache, evidence_score, load_cache, positive_weights


def trace(values, h=4, w=4, merge=2):
    return {"valid": True, "visual_tokens": len(values),
            "visual_attention_map": values, "image_grid_thw": [1, h * merge, w * merge]}


@contextmanager
def writable_db(path):
    connection = sqlite3.connect(path)
    try:
        with connection:
            yield connection
    finally:
        connection.close()


class EvidenceTests(unittest.TestCase):
    def test_border_sink_does_not_become_evidence(self):
        values = [1.0] * 16
        values[0] = 100.0
        values[5] = 2.0
        score = evidence_score(trace(values), top_ratio=0.25, border_ratio=0.25)
        self.assertAlmostEqual(score, 2.0 / sum(values))
        # Re-normalizing after the crop would incorrectly inflate this score.
        self.assertLess(score, 0.02)

    def test_full_image_normalization_is_scale_invariant(self):
        values = list(range(1, 17))
        self.assertAlmostEqual(evidence_score(trace(values)),
                               evidence_score(trace([10 * x for x in values])))

    def test_grid_mismatch_and_invalid_attention_fail(self):
        broken = trace([1.0] * 15)
        with self.assertRaises(ValueError):
            evidence_score(broken)
        for value in (math.nan, -1.0, math.inf):
            with self.assertRaises(ValueError):
                evidence_score(trace([value] + [1.0] * 15))
        with self.assertRaises(ValueError):
            evidence_score({"valid": False})

    def test_beta_zero_and_single_positive_are_neutral(self):
        self.assertEqual(positive_weights([0.01, 0.99], beta=0), [1.0, 1.0])
        self.assertEqual(positive_weights([0.03]), [1.0])
        self.assertEqual(positive_weights([0.0, 0.0]), [1.0, 1.0])

    def test_weight_order_and_bounds(self):
        weights = positive_weights([0.001, 0.4, 0.9], beta=0.25)
        self.assertLess(weights[0], weights[-1])
        self.assertTrue(all(0.875 <= weight <= 1.125 for weight in weights))
        first, second = positive_weights([0.2, 0.6], minimum=0.1, maximum=5, beta=1)
        self.assertAlmostEqual(first, 0.5)
        self.assertAlmostEqual(second, 1.5)


class CacheTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(dir=os.environ.get("ECNFT_TEST_TEMP"))
        self.root = Path(self.directory.name)
        self.db = self.root / "rollouts.sqlite"
        positive_a = {"reward": 1, "generated_token_ids": [1, 2],
                      "old_policy_score": {"token_logprobs": [-1, -2]},
                      "evidence_trace": trace([10.0] + [1.0] * 15)}
        positive_b = copy.deepcopy(positive_a)
        positive_b["evidence_trace"] = trace([1.0] * 16)
        negative = copy.deepcopy(positive_a)
        negative["reward"] = 0
        negative.pop("evidence_trace")  # Negatives must not need spatial scores.
        self.group = {"r_hat": 2 / 3, "rollouts": [positive_a, positive_b, negative]}
        with writable_db(self.db) as connection:
            connection.executescript("""
                CREATE TABLE samples(sample_index INTEGER, base_record_json TEXT);
                CREATE TABLE rollout_groups(sample_index INTEGER, perturbation TEXT,
                    complete INTEGER, nft_prompt_eligible INTEGER, r_hat REAL, record_json TEXT);
            """)
            connection.execute("INSERT INTO samples VALUES(0, ?)", (json.dumps({"question": "Q"}),))
            connection.execute("INSERT INTO rollout_groups VALUES(0, 'clean', 1, 1, ?, ?)",
                               (2 / 3, json.dumps(self.group)))
            # Homogeneous and incomplete groups must not enter calibration.
            for index, complete, eligible, mean in ((1, 1, 0, 1.0), (2, 0, 1, 0.5)):
                connection.execute("INSERT INTO samples VALUES(?, '{}')", (index,))
                connection.execute("INSERT INTO rollout_groups VALUES(?, 'clean', ?, ?, ?, '{}')",
                                   (index, complete, eligible, mean))

    def tearDown(self):
        self.directory.cleanup()

    def test_mixed_groups_positive_calibration_and_unchanged_negative(self):
        cache = build_cache(self.db, border_ratio=0)
        self.assertEqual(cache["group_count"], 1)
        a, b, negative = cache["records"]
        self.assertGreater(a["weight"], b["weight"])
        self.assertEqual(negative["weight"], 1.0)
        self.assertEqual(negative["score"], 0.0)

    def test_stale_labels_invalidate_cache(self):
        cache = build_cache(self.db)
        path = self.root / "cache.json"
        path.write_text(json.dumps(cache), encoding="utf-8")
        self.assertEqual(len(load_cache(path, self.db)["records"]), 3)
        with writable_db(self.db) as connection:
            connection.execute("UPDATE samples SET base_record_json='{}' WHERE sample_index=0")
        with self.assertRaisesRegex(ValueError, "stale"):
            load_cache(path, self.db)

    def test_stale_group_mean_fails(self):
        with writable_db(self.db) as connection:
            connection.execute("UPDATE rollout_groups SET r_hat=0.5 WHERE sample_index=0")
        with self.assertRaisesRegex(ValueError, "disagree"):
            build_cache(self.db)

    def test_missing_positive_attention_fails(self):
        self.group["rollouts"][0]["evidence_trace"]["valid"] = False
        with writable_db(self.db) as connection:
            connection.execute("UPDATE rollout_groups SET record_json=? WHERE sample_index=0",
                               (json.dumps(self.group),))
        with self.assertRaisesRegex(ValueError, "attention"):
            build_cache(self.db)

    def test_cli_end_to_end(self):
        output = self.root / "evidence.json"
        script = Path(__file__).resolve().parents[1] / "scripts" / "score_evidence.py"
        result = subprocess.run([sys.executable, str(script), "--sqlite", str(self.db),
                                 "--output", str(output), "--beta", "0"],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        cache = load_cache(output, self.db)
        self.assertTrue(all(row["weight"] == 1.0 for row in cache["records"]))


if __name__ == "__main__":
    unittest.main()
