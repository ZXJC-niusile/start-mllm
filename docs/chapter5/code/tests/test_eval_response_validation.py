"""Offline tests for CLI token validation and unusable model responses."""
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

PATH = Path(__file__).resolve().parents[1] / 'eval_vlm_dataset.py'
spec = importlib.util.spec_from_file_location('eval_response_runner', PATH)
runner = importlib.util.module_from_spec(spec)
with patch.dict(sys.modules, {'dotenv': SimpleNamespace(load_dotenv=lambda: None)}):
    spec.loader.exec_module(runner)


class EvalResponseTests(unittest.TestCase):
    def test_help_ignores_invalid_environment_default(self):
        with patch.dict(os.environ, {'EVAL_MAX_TOKENS': 'bad'}), patch.object(sys, 'argv', ['eval', '--help']), contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit) as raised:
                runner.parse_args()
        self.assertEqual(raised.exception.code, 0)

    def test_invalid_token_limits_exit_before_client(self):
        client = Mock()
        for raw in ('0', '-1', 'bad', '1.5'):
            for argv in (['eval'], ['eval', '--max-tokens', raw]):
                with self.subTest(raw=raw, argv=argv), patch.dict(os.environ, {'EVAL_MAX_TOKENS': raw}), patch.object(sys, 'argv', argv), contextlib.redirect_stderr(io.StringIO()), patch.dict(sys.modules, {'openai': SimpleNamespace(OpenAI=client)}):
                    with self.assertRaises(SystemExit) as raised:
                        runner.main()
                    self.assertEqual(raised.exception.code, 2)
        client.assert_not_called()

    def test_explicit_valid_limit_overrides_invalid_default(self):
        with patch.dict(os.environ, {'EVAL_MAX_TOKENS': 'bad'}), patch.object(sys, 'argv', ['eval', '--max-tokens', '7']):
            self.assertEqual(runner.parse_args().max_tokens, 7)

    def test_unusable_answers_become_error_rows(self):
        for answer in (None, '', ' ', [], 2, 'valid answer'):
            for empty_choices in (False, True):
                with self.subTest(answer=answer, empty_choices=empty_choices), tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    (root / 'image.png').write_bytes(b'image')
                    dataset = root / 'data.jsonl'
                    dataset.write_text(json.dumps({'image': 'image.png', 'question': 'describe'}) + '\n')
                    output = root / 'results.jsonl'
                    choices = [] if empty_choices else [SimpleNamespace(message=SimpleNamespace(content=answer))]
                    client = Mock()
                    client.chat.completions.create.return_value = SimpleNamespace(choices=choices)
                    with patch.dict(sys.modules, {'openai': SimpleNamespace(OpenAI=Mock(return_value=client))}), patch.object(sys, 'argv', ['eval', '--dataset', str(dataset), '--output', str(output)]), contextlib.redirect_stdout(io.StringIO()):
                        runner.main()
                    row = json.loads(output.read_text())
                    ok = answer == 'valid answer' and not empty_choices
                    self.assertEqual(row['status'], 'ok' if ok else 'error')
                    self.assertEqual(row['prediction'], answer if ok else '')
                    if not ok:
                        self.assertIn('Model returned no', row['error'])


if __name__ == '__main__':
    unittest.main()
