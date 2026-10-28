"""Offline interface tests, not a real Qwen tokenizer/model or GPU training run.

Run: python -m unittest discover -s docs/chapter4/code/tests -v
Only the preprocessing function definitions are loaded, so torch/model downloads
are not needed. The tiny processor expands every image marker to three tokens.
"""
import ast
import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, Mock


SOURCE = Path(__file__).resolve().parents[1] / "lora_finetune_minimal.py"


def load_functions():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    spec = importlib.util.spec_from_file_location("sft_rules", SOURCE.with_name("prepare_dataset_manifest.py"))
    rules = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rules)
    namespace = {"Path": Path, "DATA_PATH": SOURCE, "MAX_SEQ_LEN": 512,
                 "validate_sft_format": rules.validate_sft_format}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(SOURCE), "exec"), namespace)
    return namespace


class Scalar:
    def __init__(self, value):
        self.value = value

    def item(self):
        return self.value


class Vector:
    def __init__(self, values):
        self.values = values

    def tolist(self):
        return self.values[:]


class Tensor:
    def __init__(self, rows):
        self.rows = rows

    def clone(self):
        return Tensor(copy.deepcopy(self.rows))

    def __getitem__(self, index):
        if isinstance(index, tuple):
            i, j = index
            return Scalar(self.rows[i][j])
        return Vector(self.rows[index])

    def __setitem__(self, index, value):
        i, j = index
        if isinstance(j, slice):
            self.rows[i][j] = [value] * len(self.rows[i][j])
        else:
            self.rows[i][j] = value.item() if isinstance(value, Scalar) else value


class Processor:
    def __init__(self, side="right"):
        self.side = side
        self.seen_images = []

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        text = ""
        for message in messages:
            text += "<" + message["role"] + ">"
            text += "".join("@" if p["type"] == "image" else p["text"] for p in message["content"])
            text += "!"
        return text + ("<assistant>" if add_generation_prompt else "")

    def __call__(self, *, text, images, add_special_tokens, truncation, **kwargs):
        assert not truncation and not add_special_tokens
        assert sum(t.count("@") for t in text) == len(images or [])
        self.seen_images.append(list(images or []))
        rows = [[token for c in t for token in ([901, 902, 903] if c == "@" else [ord(c)])] for t in text]
        if kwargs.get("return_tensors") != "pt":
            return {"input_ids": rows}
        masks = [[1] * len(row) for row in rows]
        maximum = max(map(len, rows))
        for row, mask in zip(rows, masks):
            padding = [0] * (maximum - len(row))
            row[:] = row + padding if self.side == "right" else padding + row
            mask[:] = mask + padding if self.side == "right" else padding + mask
        return {"input_ids": Tensor(rows), "attention_mask": Tensor(masks), "pixel_values": "kept"}


def message(role, text, image=None):
    parts = [{"type": "text", "text": text}]
    if image is not None:
        parts.insert(0, {"type": "image", "image": image})
    return {"role": role, "content": parts}


class CollatorTests(unittest.TestCase):
    def setUp(self):
        self.ns = load_functions()

    def test_multiturn_mixed_batch_both_padding_sides(self):
        first = [message("system", "rules"), message("user", "question", "first"),
                 message("assistant", "answer"), message("user", "followup", "second"),
                 message("assistant", "next")]
        second = [message("user", "short"), message("assistant", "yes")]
        for side in ("left", "right"):
            processor = Processor(side)
            result = self.ns["collate_fn"]([{"messages": first}, {"messages": second}], processor)
            self.assertEqual(result["pixel_values"], "kept")
            self.assertEqual(processor.seen_images[0], ["first", "second"])
            self.assertEqual(processor.seen_images[1], ["first"])
            for row, expected in zip(result["labels"].rows, ("answer!next!", "yes!")):
                self.assertEqual("".join(chr(x) for x in row if x != -100), expected)
                self.assertTrue(all(x not in row for x in (0, 901, 902, 903)))

    def test_reject_overlength_without_truncating(self):
        self.ns["MAX_SEQ_LEN"] = 10
        with self.assertRaisesRegex(ValueError, "MAX_SEQ_LEN"):
            self.ns["collate_fn"]([{"messages": [message("user", "Q", "img"), message("assistant", "A")]}], Processor())

    def test_reject_inconsistent_processor_prefix(self):
        class Inconsistent(Processor):
            def __call__(self, **kwargs):
                result = super().__call__(**kwargs)
                if kwargs.get("return_tensors") != "pt":
                    result["input_ids"][0][0] = -1
                return result
        with self.assertRaisesRegex(ValueError, "前缀不一致"):
            self.ns["collate_fn"]([{"messages": [message("user", "Q"), message("assistant", "A")]}], Inconsistent())

    def test_reject_missing_target(self):
        with self.assertRaisesRegex(ValueError, "没有可监督"):
            self.ns["collate_fn"]([{"messages": [message("user", "Q")]}], Processor())

    def test_formatter_keeps_system_and_accepts_text_only(self):
        image_api = Mock()
        self.ns["Image"] = image_api
        sample = {"id": "test", "image": "", "messages": [message("system", "rules"), message("user", "Q"), message("assistant", "A")]}
        self.assertEqual(self.ns["format_for_qwen25vl"](sample)["messages"], sample["messages"])
        image_api.open.assert_not_called()
        with self.assertRaises(ValueError):
            self.ns["format_for_qwen25vl"]({"messages": [message("user", "Q")]})

    def test_formatter_resolves_content_images_and_missing_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "image.png").write_bytes(b"mock image")
            self.ns["DATA_PATH"] = root / "data.jsonl"
            self.ns["Image"] = MagicMock()
            sample = {"id": "test", "image": "image.png", "messages": [message("user", "Q", "image.png"), message("assistant", "A")]}
            self.assertIsNotNone(self.ns["format_for_qwen25vl"](sample))
            self.ns["Image"].open.assert_called_once_with(root / "image.png")
            sample["messages"][0]["content"][0]["image"] = "missing.png"
            sample["image"] = "missing.png"
            self.assertIsNone(self.ns["format_for_qwen25vl"](sample))


if __name__ == "__main__":
    unittest.main()
