"""Check final-answer parsing and avoid substring-based false positives."""

import ast
import re
import unittest
from pathlib import Path
from typing import Any


class RewardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = Path(__file__).resolve().parents[1] / "scripts" / "collect_rollouts.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                     and node.name in {"clean_text", "normalize_answer", "extract_answer", "is_correct"}]
        namespace = {"re": re, "Any": Any, "ANSWER_MATCHING": "exact",
                     "ANSWER_RE": re.compile(r"<answer>(.*?)</answer>", re.I | re.S)}
        exec(compile(ast.Module(body=functions, type_ignores=[]), str(source), "exec"), namespace)
        cls.functions = namespace

    def test_only_final_answer_is_scored(self):
        answer = self.functions["extract_answer"](
            "<think>Consider VD or VH.</think><answer>VD</answer>")
        self.assertEqual(answer, "vd")
        self.assertTrue(self.functions["is_correct"](answer, "VD."))

    def test_short_answer_is_not_a_substring_match(self):
        self.assertFalse(self.functions["is_correct"]("no", "nodule"))
        self.assertFalse(self.functions["is_correct"]("vd", "not vd"))


if __name__ == "__main__":
    unittest.main()
