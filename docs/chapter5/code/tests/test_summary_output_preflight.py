"""Summary reports must not overwrite any input, including compare files."""

import argparse
import contextlib
import importlib.util
import io
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


PATH = Path(__file__).resolve().parents[1] / "summarize_eval_results.py"
spec = importlib.util.spec_from_file_location("summary_output", PATH)
summary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summary)
ROW = '{"id":"1","status":"ok","prediction":"yes","reference":"yes","elapsed_seconds":1,"tags":[]}\n'


class SummaryOutputTests(unittest.TestCase):
    def run_summary(self, primary, compare, output):
        args = argparse.Namespace(input=str(primary), compare=[str(p) for p in compare],
                                  label=[], markdown=False, write=str(output) if output else "")
        with patch.object(summary, "parse_args", return_value=args), patch.object(summary, "_try_utf8_stdio"):
            summary.main()

    def test_rejects_direct_and_relative_alias_of_every_input(self):
        for index in (0, 1, 2):
            for relative_alias in (False, True):
                with self.subTest(index=index, relative_alias=relative_alias), tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    inputs = [root / "primary.jsonl", root / "compare.jsonl", root / "compare2.jsonl"]
                    for path in inputs:
                        path.write_text(ROW, encoding="utf-8")
                    (root / "sub").mkdir()
                    output = root / "sub" / ".." / inputs[index].name if relative_alias else inputs[index]
                    with self.assertRaisesRegex(ValueError, "--write"):
                        self.run_summary(inputs[0], inputs[1:], output)
                    self.assertEqual([p.read_text(encoding="utf-8") for p in inputs], [ROW, ROW, ROW])

    def test_rejects_symlink_and_hardlink_of_every_input(self):
        for index in (0, 1, 2):
            for kind, create_link in (("symlink", os.symlink), ("hardlink", os.link)):
                with self.subTest(index=index, kind=kind), tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    inputs = [root / "primary.jsonl", root / "compare.jsonl", root / "compare2.jsonl"]
                    for path in inputs:
                        path.write_text(ROW, encoding="utf-8")
                    output = root / "report.txt"
                    try:
                        create_link(inputs[index], output)
                    except (OSError, NotImplementedError):
                        self.skipTest(f"{kind} unavailable on this platform")
                    with self.assertRaisesRegex(ValueError, "--write"):
                        self.run_summary(inputs[0], inputs[1:], output)
                    self.assertEqual([p.read_text(encoding="utf-8") for p in inputs], [ROW, ROW, ROW])

    def test_distinct_report_and_stdout_still_work(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            primary = root / "input.jsonl"
            primary.write_text(ROW, encoding="utf-8")
            report = root / "report.txt"
            self.run_summary(primary, [], report)
            self.assertTrue(report.read_text(encoding="utf-8").strip())
            with contextlib.redirect_stdout(io.StringIO()) as output:
                self.run_summary(primary, [], None)
            self.assertEqual(output.getvalue(), report.read_text(encoding="utf-8"))
            self.assertEqual(primary.read_text(encoding="utf-8"), ROW)


if __name__ == "__main__":
    unittest.main()
