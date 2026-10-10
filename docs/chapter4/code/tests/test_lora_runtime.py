"""Offline runtime control-flow tests; no real torch/model/GPU execution."""
import ast
from contextlib import redirect_stdout
import io
from pathlib import Path
import types
import unittest
from unittest.mock import Mock

SOURCE = Path(__file__).resolve().parents[1] / "lora_finetune_minimal.py"


def runtime(*, available=True, devices=1, capability=(8, 0), bf16=True, cuda_version="12.1", world_size="1"):
    cuda = Mock()
    cuda.is_available.return_value = available
    cuda.device_count.return_value = devices
    cuda.get_device_capability.return_value = capability
    cuda.is_bf16_supported.return_value = bf16
    cuda.get_device_properties.return_value = types.SimpleNamespace(total_memory=1)
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    ns = {"Path": Path, "torch": types.SimpleNamespace(cuda=cuda, version=types.SimpleNamespace(cuda=cuda_version), bfloat16="bf16", __version__="mock"),
          "os": types.SimpleNamespace(environ={"WORLD_SIZE": world_size}),
          "AutoProcessor": Mock(), "Qwen2_5_VLForConditionalGeneration": Mock(), "MODEL_ID": "mock-model"}
    exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef)], type_ignores=[]), str(SOURCE), "exec"), ns)
    return ns, tree


class RuntimeTests(unittest.TestCase):
    def test_unsupported_devices_fail_before_downloads(self):
        cases = ({"available": False}, {"cuda_version": None},
                 {"capability": (7, 5), "bf16": True},  # T4-like, even if emulation reports True.
                 {"cuda_version": "10.2"}, {"bf16": False})
        for options in cases:
            with self.subTest(options=options):
                ns, _ = runtime(**options)
                with self.assertRaises(RuntimeError):
                    ns["main"]()
                ns["AutoProcessor"].from_pretrained.assert_not_called()
                ns["Qwen2_5_VLForConditionalGeneration"].from_pretrained.assert_not_called()

    def test_multiple_devices_or_processes_fail_before_downloads(self):
        for options in ({"devices": 2}, {"world_size": "2"}):
            ns, _ = runtime(**options)
            with self.assertRaisesRegex(RuntimeError, "CUDA_VISIBLE_DEVICES"):
                ns["main"]()
            ns["AutoProcessor"].from_pretrained.assert_not_called()

    def test_supported_path_loads_bf16_without_inference_dispatch(self):
        ns, tree = runtime()
        class StopAfterLoad(Exception):
            pass
        model = ns["Qwen2_5_VLForConditionalGeneration"]
        model.from_pretrained.side_effect = StopAfterLoad
        with redirect_stdout(io.StringIO()), self.assertRaises(StopAfterLoad):
            ns["main"]()
        self.assertEqual(model.from_pretrained.call_args.kwargs["torch_dtype"], "bf16")
        self.assertNotIn("device_map", model.from_pretrained.call_args.kwargs)
        args = next(n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "TrainingArguments")
        flags = {k.arg: ast.literal_eval(k.value) for k in args.keywords if k.arg in ("fp16", "bf16")}
        self.assertEqual(flags, {"fp16": False, "bf16": True})

    def test_version_floor_is_checked_before_model_class_import(self):
        ns, tree = runtime()
        for version in ("4.45.0", "4.48.3"):
            with self.assertRaisesRegex(ImportError, "4.49.0"):
                ns["_require_transformers_version"](version)
        ns["_require_transformers_version"]("4.49.0")
        ns["_require_transformers_version"]("4.57.1")
        check = next(i for i, n in enumerate(tree.body) if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Name) and n.value.func.id == "_require_transformers_version")
        model_import = next(i for i, n in enumerate(tree.body) if isinstance(n, ast.ImportFrom) and any(a.name == "Qwen2_5_VLForConditionalGeneration" for a in n.names))
        self.assertLess(check, model_import)


if __name__ == "__main__":
    unittest.main()
