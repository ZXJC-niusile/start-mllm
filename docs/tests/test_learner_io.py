import base64
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from learner_io import to_data_url


class ImageEncodingTests(unittest.TestCase):
    def test_binary_payload_and_mime(self):
        with tempfile.TemporaryDirectory() as directory:
            for name, mime in [("x.png", "image/png"), ("x.jpg", "image/jpeg"), ("x.unknownext", "image/jpeg")]:
                path = Path(directory) / name
                payload = bytes(range(256))
                path.write_bytes(payload)
                prefix, encoded = to_data_url(str(path)).split(",", 1)
                self.assertEqual(prefix, f"data:{mime};base64")
                self.assertEqual(base64.b64decode(encoded), payload)

    def test_missing_file_and_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            for path in [Path(directory), Path(directory) / "missing.png"]:
                with self.assertRaises(FileNotFoundError):
                    to_data_url(str(path))
