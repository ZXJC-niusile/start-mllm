"""Offline response and startup checks; no API requests or model downloads."""
import contextlib
import importlib.util
import io
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

DOCS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DOCS))
from learner_runtime import positive_env_int


def load(relative):
    path = DOCS / relative
    spec = importlib.util.spec_from_file_location(path.stem + "_boundary", path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {
        "openai": SimpleNamespace(OpenAI=Mock()),
        "dotenv": SimpleNamespace(load_dotenv=lambda: None),
        "gradio": Mock(), "torch": Mock(), "qwen_vl_utils": Mock(), "transformers": Mock(),
    }):
        spec.loader.exec_module(module)
    return module


class RuntimeBoundaries(unittest.TestCase):
    def test_empty_and_invalid_numeric_config(self):
        for value in ("", "  ", "25", "0", "-1", "bad", "65536"):
            with self.subTest(value=value), patch.dict(os.environ, {"TEST_PORT": value}):
                if not value.strip():
                    self.assertEqual(positive_env_int("TEST_PORT", 7860, maximum=65535), 7860)
                elif value == "25":
                    self.assertEqual(positive_env_int("TEST_PORT", 7860, maximum=65535), 25)
                else:
                    with self.assertRaisesRegex(ValueError, "TEST_PORT"):
                        positive_env_int("TEST_PORT", 7860, maximum=65535)

    def test_invalid_generation_length_stops_before_download(self):
        for value in ("bad", "0", "-2"):
            with patch.dict(os.environ, {"MAX_NEW_TOKENS": value}):
                module = load("chapter7/code/transformers_chat.py")
                with contextlib.redirect_stderr(io.StringIO()) as error:
                    self.assertEqual(module.main(), 1)
                self.assertIn("MAX_NEW_TOKENS", error.getvalue())
                module.Qwen2_5_VLForConditionalGeneration.from_pretrained.assert_not_called()

    def test_demo_rejects_config_before_building_ui(self):
        module = load("chapter8/code/app.py")
        module.build_demo = Mock()
        for env in ({"GRADIO_SERVER_PORT": "bad", "DEFAULT_MODE": ""},
                    {"GRADIO_SERVER_PORT": "7860", "DEFAULT_MODE": "unknown"}):
            with patch.dict(os.environ, env), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(module.main(), 1)
        module.build_demo.assert_not_called()
        with patch.dict(os.environ, {"GRADIO_SERVER_PORT": "", "DEFAULT_MODE": ""}), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(module.main(), 0)
        module.build_demo.assert_called_once_with("自由问答")
        self.assertEqual(module.build_demo.return_value.launch.call_args.kwargs["server_port"], 7860)

    def test_demo_reports_empty_choices_as_failed(self):
        module = load("chapter8/code/app.py")
        module.to_data_url = lambda _: "data:image/png;base64,AA=="
        module.OpenAI.return_value.chat.completions.create.return_value = SimpleNamespace(choices=[])
        answer, status = module.ask_vlm("fixture.png", "question", "自由问答")
        self.assertIn("no choices", answer)
        self.assertIn("失败", status)

    def test_agent_request_stages_reject_empty_choices(self):
        for relative, cls_name, methods in [
            ("chapter9/app/agent_workflow_demo.py", "MultimodalAgent", ["perceive"]),
            ("chapter9/app/agent_workflow_v2.py", "MultimodalAgentV2", ["perceive_and_plan", "summarize"]),
        ]:
            module = load(relative)
            module.to_data_url = lambda _: "data:image/png;base64,AA=="
            agent = getattr(module, cls_name)()
            for choices in ([], None):
                agent.client.chat.completions.create.return_value = SimpleNamespace(choices=choices)
                for method in methods:
                    args = ("question", {"messages": []}, []) if method == "summarize" else ("fixture.png", "question")
                    with self.subTest(method=method, choices=choices), self.assertRaisesRegex(ValueError, "no choices"):
                        getattr(agent, method)(*args)
