"""Offline summary regressions; no API calls or model downloads.

Run: python -m unittest discover -s docs/chapter5/code/tests -v
"""
import importlib.util
from pathlib import Path
import unittest

PATH = Path(__file__).resolve().parents[1] / "summarize_eval_results.py"
spec = importlib.util.spec_from_file_location("eval_summary", PATH)
summary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summary)


class SummaryTests(unittest.TestCase):
    def test_request_errors_are_visible_in_end_to_end_denominator(self):
        rows = [{"reference": "cat", "prediction": "CAT", "status": "ok"}]
        rows += [{"reference": "cat", "prediction": "", "status": "error"}] * 9
        stats = summary.summarize(rows)
        self.assertEqual(stats["rough_hit_rate"], 1.0)  # Existing conditional metric.
        self.assertEqual(stats["end_to_end_hit_rate"], 0.1)
        self.assertEqual((stats["ref_den"], stats["ref_total"]), (1, 10))
        report = summary._format_markdown_table(["model"], [stats])
        self.assertIn("仅含 reference 的 ok 行", report)
        self.assertIn("1/10", report)
        self.assertIn("10.0%", report)
        self.assertIn("1/10", summary._format_text_section("model", stats))

    def test_all_errors_with_references_score_zero(self):
        stats = summary.summarize([{"reference": "cat", "status": "error"}])
        self.assertIsNone(stats["rough_hit_rate"])
        self.assertEqual(stats["end_to_end_hit_rate"], 0.0)

    def test_empty_or_missing_references_are_not_scored(self):
        for rows in ([], [{"status": "error"}], [{"status": "ok", "reference": " "}]):
            stats = summary.summarize(rows)
            self.assertEqual(stats["ref_total"], 0)
            self.assertIsNone(stats["end_to_end_hit_rate"])
            self.assertIn("n/a", summary._format_markdown_table(["model"], [stats]))


if __name__ == "__main__":
    unittest.main()
