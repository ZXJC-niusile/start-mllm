"""Validate the content image paths actually consumed by the training loader."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

PATH = Path(__file__).resolve().parents[1] / "prepare_dataset_manifest.py"


def record(images, top=""):
    return {"id": "sample", "image": top, "messages": [
        {"role": "user", "content": [*({"type": "image", "image": p} for p in images), {"type": "text", "text": "Question"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "Answer"}]},
    ]}


class SftImagePathTests(unittest.TestCase):
    def run_cli(self, directory, sample, strict=True):
        dataset = Path(directory) / "data.jsonl"
        dataset.write_text(json.dumps(sample) + '\n', encoding="utf-8")
        return subprocess.run([sys.executable, str(PATH), "--dataset", str(dataset), *(["--check-sft-format"] if strict else [])], cwd=directory, capture_output=True, text=True, encoding="utf-8")

    def test_missing_nested_image_rejected_even_with_valid_top_level_index(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "index.png").write_bytes(b"existing")
            result = self.run_cli(directory, record(["missing.png"], "index.png"))
            self.assertEqual(result.returncode, 1, result.stdout)
            self.assertIn("[1]", result.stdout)
            self.assertIn("messages[0].content[0].image", result.stdout)
            self.assertIn("missing.png", result.stdout)
            self.assertNotIn("全部通过", result.stdout)
            self.assertNotIn("Traceback", result.stderr)

    def test_all_nested_images_checked_and_directories_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "exists.png").write_bytes(b"existing")
            (root / "not-image").mkdir()
            result = self.run_cli(directory, record(["exists.png", str(root / "not-image")]))
            self.assertEqual(result.returncode, 1, result.stdout)
            self.assertIn("messages[0].content[1].image", result.stdout)

    def test_valid_text_and_multi_image_samples_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "exists.png").write_bytes(b"existing")
            for sample in (record([]), record(["exists.png", str(root / "exists.png")])):
                result = self.run_cli(directory, sample)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("全部通过", result.stdout)

    def test_basic_report_exposes_missing_content_image(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self.run_cli(directory, record(["missing.png"]), strict=False)
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads(result.stdout)[0]
            self.assertFalse(report["content_images"][0]["image_exists"])
            self.assertEqual(report["content_images"][0]["field"], "messages[0].content[0].image")


if __name__ == "__main__":
    unittest.main()
