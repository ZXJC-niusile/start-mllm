"""Image inputs must survive evaluation output path mistakes (offline stubs)."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

PATH = Path(__file__).resolve().parents[1] / "eval_vlm_dataset.py"
spec = importlib.util.spec_from_file_location("eval_image_runner", PATH)
runner = importlib.util.module_from_spec(spec)
with patch.dict(sys.modules, {"dotenv": types.SimpleNamespace(load_dotenv=lambda: None)}):
    spec.loader.exec_module(runner)


class ImageOutputPreflightTests(unittest.TestCase):
    def test_image_and_alias_outputs_are_rejected_before_client_creation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "source.png"
            original = b"original source image bytes"
            image.write_bytes(original)
            dataset = root / "data.jsonl"
            dataset.write_text(json.dumps({"id": "one", "image": image.name, "question": "Describe"}) + "\n", encoding="utf-8")
            aliases = [image]
            for name, link in (("symbolic.jsonl", os.symlink), ("hard.jsonl", os.link)):
                alias = root / name
                try:
                    link(image, alias)
                except OSError:
                    continue
                aliases.append(alias)
            for output in aliases:
                with self.subTest(output=output.name):
                    client = Mock()
                    client.return_value.chat.completions.create.return_value = types.SimpleNamespace(
                        choices=[types.SimpleNamespace(message=types.SimpleNamespace(content="answer"))])
                    args = argparse.Namespace(dataset=str(dataset), output=str(output), base_url="unused", api_key="unused", model="stub", max_tokens=1)
                    with patch.object(runner, "parse_args", return_value=args), patch.dict(sys.modules, {"openai": types.SimpleNamespace(OpenAI=client)}):
                        with self.assertRaisesRegex(ValueError, "image|图片"):
                            runner.main()
                    client.assert_not_called()
                    self.assertEqual(image.read_bytes(), original)

    def test_later_absolute_image_is_also_protected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "absolute.png"
            image.write_bytes(b"original")
            dataset = root / "data.jsonl"
            dataset.write_text('\n' + json.dumps({"image": ""}) + '\n' + json.dumps({"image": str(image)}) + '\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "3"):
                runner._validate_image_output_path(dataset, image)
            self.assertEqual(image.read_bytes(), b"original")

    def test_distinct_output_is_allowed_without_writing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "image.png"
            image.write_bytes(b"original")
            dataset = root / "data.jsonl"
            dataset.write_text(json.dumps({"image": image.name}) + '\n', encoding="utf-8")
            output = root / "results.jsonl"
            runner._validate_image_output_path(dataset, output)
            self.assertFalse(output.exists())
            self.assertEqual(image.read_bytes(), b"original")


if __name__ == "__main__":
    unittest.main()
