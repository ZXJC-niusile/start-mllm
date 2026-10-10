"""Standard-library regressions for malformed SFT input and CLI failure status."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

PATH = Path(__file__).resolve().parents[1] / "prepare_dataset_manifest.py"
spec = importlib.util.spec_from_file_location("sft_validator", PATH)
validator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validator)


def valid_record():
    return {"id": "text-only", "image": "", "messages": [
        {"role": "user", "content": [{"type": "text", "text": "Question"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "Answer"}]},
    ]}


class ValidationTests(unittest.TestCase):
    def run_cli(self, text):
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "data.jsonl"
            dataset.write_text(text, encoding="utf-8")
            return subprocess.run([sys.executable, str(PATH), "--dataset", str(dataset), "--check-sft-format"], capture_output=True, text=True, encoding="utf-8")

    def test_malformed_shapes_report_line_errors(self):
        invalid = [None, [], {"id": "bad", "messages": None}, {"id": "bad", "messages": "bad"}]
        for message in (None, "bad", {"role": "user", "content": None}, {"role": "user", "content": "bad"}, {"role": "user", "content": [None, "bad"]}):
            record = valid_record()
            record["messages"][0] = message
            invalid.append(record)
        record = valid_record()
        record["image"] = None
        invalid.append(record)
        for kind in ("text", "image"):
            for value in (42, ["bad"], {"bad": "shape"}):
                record = valid_record()
                record["messages"][0]["content"] = [{"type": kind, kind: value}]
                invalid.append(record)
        for record in invalid:
            with self.subTest(record=record):
                errors = validator.validate_sft_format(record, 7)
                self.assertTrue(errors)
                self.assertTrue(all(error.startswith("[7]") for error in errors))
        result = self.run_cli("\n" + "\n".join(json.dumps(r) for r in invalid))
        self.assertEqual(result.returncode, 1)
        self.assertIn("[2]", result.stdout)
        self.assertIn("[11]", result.stdout)
        self.assertNotIn("全部通过", result.stdout)
        self.assertNotIn("Traceback", result.stderr)

    def test_invalid_json_is_reported_and_scan_continues(self):
        result = self.run_cli("{bad json}\nnull\n")
        self.assertEqual(result.returncode, 1)
        self.assertIn("[1] JSON", result.stdout)
        self.assertIn("[2]", result.stdout)
        self.assertNotIn("Traceback", result.stderr)

    def test_incomplete_or_misplaced_system_turn_is_rejected(self):
        record = valid_record()
        record["messages"].pop()
        self.assertTrue(validator.validate_sft_format(record, 1))
        record = valid_record()
        record["messages"].insert(1, {"role": "system", "content": [{"type": "text", "text": "rules"}]})
        self.assertTrue(validator.validate_sft_format(record, 1))

    def test_valid_bundled_records_still_pass(self):
        fixture = PATH.with_name("sample_multimodal_sft.jsonl").read_text(encoding="utf-8")
        for index, line in enumerate(fixture.splitlines(), 1):
            self.assertEqual(validator.validate_sft_format(json.loads(line), index), [])
        result = self.run_cli(json.dumps(valid_record()))
        self.assertEqual(result.returncode, 0)
        self.assertIn("全部通过", result.stdout)


if __name__ == "__main__":
    unittest.main()
