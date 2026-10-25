"""Run the real CLI entry point offline; no SDK, service or API key required."""
import base64
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "openai_compatible_client.py"
RUNNER = r'''
import json
import os
from pathlib import Path
import runpy
import sys
from types import SimpleNamespace


def create(**kwargs):
    Path(os.environ["REQUEST_LOG"]).write_text(json.dumps(kwargs), encoding="utf-8")
    scenario = os.environ["SCENARIO"]
    if scenario == "api_error":
        raise RuntimeError("offline API failure")
    if scenario == "empty_choices":
        return SimpleNamespace(choices=[])
    if scenario == "null_choices":
        return SimpleNamespace(choices=None)
    if scenario == "missing_choices":
        return SimpleNamespace()
    if scenario == "null_message":
        return SimpleNamespace(choices=[SimpleNamespace(message=None)])
    content = {
        "success": "A small synthetic image.",
        "null_content": None,
        "empty_content": "",
        "blank_content": " \n\t",
        "list_content": [],
        "number_content": 42,
    }[scenario]
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


sys.modules["openai"] = SimpleNamespace(
    OpenAI=lambda **kwargs: SimpleNamespace(chat=SimpleNamespace(
        completions=SimpleNamespace(create=create))))
sys.modules["dotenv"] = SimpleNamespace(load_dotenv=lambda: None)
runpy.run_path(sys.argv[1], run_name="__main__")
'''


class APIClientCLITests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.image = self.root / "sample.png"
        self.image.write_bytes(base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/l3sAAAAASUVORK5CYII="))
        self.log = self.root / "request.json"

    def run_cli(self, scenario="success", **overrides):
        self.log.unlink(missing_ok=True)
        env = dict(os.environ)
        env.update(OPENAI_BASE_URL="http://invalid.example/v1", OPENAI_API_KEY="offline-stub",
                   MODEL_ID="offline-model", IMAGE_PATH=str(self.image), QUESTION="Describe",
                   REQUEST_LOG=str(self.log), SCENARIO=scenario)
        env.pop("MAX_TOKENS", None)
        env.update(overrides)
        result = subprocess.run(
            [sys.executable, "-c", RUNNER, str(SCRIPT)], env=env,
            cwd=self.root, text=True, capture_output=True, timeout=10,
        )
        request = json.loads(self.log.read_text()) if self.log.exists() else None
        return result, request

    def assert_cli_failure(self, result):
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertEqual(result.stdout, "")
        self.assertIn("Request failed:", result.stderr)
        self.assertNotIn("Traceback", result.stderr)

    def test_success_prints_answer_and_exits_zero(self):
        result, request = self.run_cli()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "A small synthetic image.\n")
        self.assertEqual(result.stderr, "")
        self.assertEqual(request["max_tokens"], 256)
        self.assertEqual(request["model"], "offline-model")
        self.assertEqual(request["messages"][0]["content"][0]["text"], "Describe")
        self.assertTrue(request["messages"][0]["content"][1]["image_url"]["url"].startswith("data:image/png;base64,"))

    def test_positive_max_tokens_is_forwarded(self):
        result, request = self.run_cli(MAX_TOKENS=" 7 ")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(request["max_tokens"], 7)

    def test_invalid_max_tokens_is_guarded_before_any_request(self):
        for value in ("", "not-a-number", "1.5", "0", "-1"):
            with self.subTest(value=value):
                result, request = self.run_cli(MAX_TOKENS=value)
                self.assert_cli_failure(result)
                self.assertIn("MAX_TOKENS must be a positive integer", result.stderr)
                self.assertIsNone(request)

    def test_missing_image_is_a_failed_process_without_request(self):
        result, request = self.run_cli(IMAGE_PATH=str(self.root / "missing.png"))
        self.assert_cli_failure(result)
        self.assertIsNone(request)

    def test_api_error_is_a_failed_process(self):
        result, request = self.run_cli("api_error")
        self.assert_cli_failure(result)
        self.assertIn("offline API failure", result.stderr)
        self.assertIsNotNone(request)

    def test_empty_and_non_text_content_is_not_reported_as_success(self):
        for scenario in ("null_content", "empty_content", "blank_content", "list_content", "number_content"):
            with self.subTest(scenario=scenario):
                result, request = self.run_cli(scenario)
                self.assert_cli_failure(result)
                self.assertIn("no text answer", result.stderr)
                self.assertIsNotNone(request)

    def test_missing_choices_is_an_explicit_failure(self):
        for scenario in ("empty_choices", "null_choices"):
            with self.subTest(scenario=scenario):
                result, request = self.run_cli(scenario)
                self.assert_cli_failure(result)
                self.assertIn("no choices", result.stderr)
                self.assertIsNotNone(request)

    def test_malformed_response_is_a_failed_process(self):
        for scenario in ("missing_choices", "null_message"):
            with self.subTest(scenario=scenario):
                result, _ = self.run_cli(scenario)
                self.assert_cli_failure(result)


if __name__ == "__main__":
    unittest.main()
