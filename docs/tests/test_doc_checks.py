import importlib.util
from pathlib import Path
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "check_docs.py"
spec = importlib.util.spec_from_file_location("doc_checks", SCRIPT)
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)


class DocChecks(unittest.TestCase):
    def test_missing_file_and_fragment_are_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "target.md").write_text('<a id="valid"></a>', encoding="utf-8")
            path = root / "README.md"
            path.write_text("[missing](absent.md)\n[anchor](target.md#bad)", encoding="utf-8")
            count, errors = checker.check_file(path, root)
            self.assertEqual(count, 2)
            self.assertEqual(len(errors), 2)
            self.assertIn("missing local path", errors[0])
            self.assertIn("missing explicit anchor", errors[1])

    def test_docs_root_unicode_path_and_code_example(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "docs").mkdir()
            (root / "docs" / "中文.md").write_text('<a id="valid"></a>', encoding="utf-8")
            path = root / "README.md"
            path.write_text("[ok](/%E4%B8%AD%E6%96%87.md#valid)\n```md\n[example](absent.md)\n```", encoding="utf-8")
            self.assertEqual(checker.check_file(path, root), (1, []))

    def test_unclosed_fence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "README.md"
            path.write_text("```python\nprint(1)\n", encoding="utf-8")
            self.assertIn("unclosed code fence", checker.check_file(path, root)[1][0])
