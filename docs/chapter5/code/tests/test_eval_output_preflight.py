"""Test output rejection before any API client or output writes, using stubs."""
import argparse
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

PATH = Path(__file__).resolve().parents[1] / "eval_vlm_dataset.py"
spec = importlib.util.spec_from_file_location("eval_runner", PATH)
runner = importlib.util.module_from_spec(spec)
with patch.dict(sys.modules, {"dotenv": types.SimpleNamespace(load_dotenv=lambda: None)}):
    spec.loader.exec_module(runner)


class OutputPreflightTests(unittest.TestCase):
    def assert_rejected_before_requests(self, dataset, output):
        before = dataset.read_bytes()
        client = Mock()
        args = argparse.Namespace(dataset=str(dataset), output=str(output))
        with patch.object(runner, "parse_args", return_value=args), patch.object(runner, "_ensure_dataset_images_exist") as images:
            with patch.dict(sys.modules, {"openai": types.SimpleNamespace(OpenAI=client)}):
                with self.assertRaises((ValueError, NotADirectoryError)):
                    runner.main()
            client.assert_not_called()
            images.assert_not_called()
        self.assertEqual(dataset.read_bytes(), before)

    def test_same_file_and_resolved_alias(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "data.jsonl"
            dataset.write_text('{"question":"keep this data"}\n', encoding="utf-8")
            (root / "sub").mkdir()
            for output in (dataset, root / "sub" / ".." / dataset.name):
                self.assert_rejected_before_requests(dataset, output)

    def test_symlink_and_hardlink_aliases(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "data.jsonl"
            dataset.write_text("original\n", encoding="utf-8")
            for name, link in (("symlink", os.symlink), ("hardlink", os.link)):
                output = root / name
                try:
                    link(dataset, output)
                except OSError:
                    continue  # Some platforms disallow link creation.
                self.assert_rejected_before_requests(dataset, output)

    def test_missing_output_directory_and_directory_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "data.jsonl"
            dataset.write_text("original\n", encoding="utf-8")
            for output in (root / "missing" / "results.jsonl", root):
                self.assert_rejected_before_requests(dataset, output)
            self.assertFalse((root / "missing").exists())

    def test_distinct_output_is_allowed_without_creating_it(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "data.jsonl"
            dataset.write_text("original\n", encoding="utf-8")
            output = root / "results.jsonl"
            runner._validate_output_path(dataset, output)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
