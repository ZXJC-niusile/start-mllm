"""Offline checks: placeholder setup must not destroy learner images."""

import contextlib
import importlib.util
import io
from pathlib import Path
import struct
import tempfile
import unittest
import zlib


DOCS = Path(__file__).resolve().parents[1]


class PlaceholderImagesTests(unittest.TestCase):
    def generators(self):
        for chapter in (4, 5):
            path = DOCS / f"chapter{chapter}" / "code" / "create_placeholder_images.py"
            spec = importlib.util.spec_from_file_location(f"placeholder_{chapter}", path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            yield chapter, module

    def test_new_png_is_valid(self):
        for chapter, module in self.generators():
            with self.subTest(chapter=chapter), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "images" / "new.png"
                module.save_solid_png(path, 2, 3, 10, 20, 30)
                png = path.read_bytes()
                self.assertEqual(png[:8], b"\x89PNG\r\n\x1a\n")
                offset = 8
                chunks = {}
                while offset < len(png):
                    size = struct.unpack("!I", png[offset:offset + 4])[0]
                    kind = png[offset + 4:offset + 8]
                    data = png[offset + 8:offset + 8 + size]
                    crc = struct.unpack("!I", png[offset + 8 + size:offset + 12 + size])[0]
                    self.assertEqual(crc, zlib.crc32(kind + data) & 0xFFFFFFFF)
                    chunks[kind] = data
                    offset += size + 12
                self.assertEqual(struct.unpack("!2I5B", chunks[b"IHDR"]), (2, 3, 8, 2, 0, 0, 0))
                self.assertEqual(zlib.decompress(chunks[b"IDAT"]), bytes([0, 10, 20, 30, 10, 20, 30]) * 3)

    def test_existing_file_is_preserved(self):
        for chapter, module in self.generators():
            with self.subTest(chapter=chapter), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "real.png"
                path.write_bytes(b"learner replacement image")
                with contextlib.redirect_stdout(io.StringIO()) as output:
                    module.save_solid_png(path, 2, 3, 10, 20, 30)
                self.assertEqual(path.read_bytes(), b"learner replacement image")
                self.assertIn("Skipped existing path:", output.getvalue())

    def test_existing_symlinks_are_not_followed(self):
        for chapter, module in self.generators():
            for target_exists in (True, False):
                with self.subTest(chapter=chapter, target_exists=target_exists), tempfile.TemporaryDirectory() as directory:
                    target = Path(directory) / "target.png"
                    if target_exists:
                        target.write_bytes(b"original target")
                    path = Path(directory) / "image.png"
                    try:
                        path.symlink_to(target)
                    except (OSError, NotImplementedError):
                        self.skipTest("Symlinks unavailable on this platform")
                    with contextlib.redirect_stdout(io.StringIO()):
                        module.save_solid_png(path, 2, 3, 10, 20, 30)
                    self.assertTrue(path.is_symlink())
                    self.assertEqual(target.exists(), target_exists)
                    if target_exists:
                        self.assertEqual(target.read_bytes(), b"original target")

    def test_encoding_failure_does_not_create_file(self):
        for chapter, module in self.generators():
            with self.subTest(chapter=chapter), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "invalid.png"
                with self.assertRaises(ValueError):
                    module.save_solid_png(path, 2, 3, 300, 20, 30)
                self.assertFalse(path.exists())

    def test_main_can_be_rerun_after_replacing_one_image(self):
        for chapter, module in self.generators():
            with self.subTest(chapter=chapter), tempfile.TemporaryDirectory() as directory:
                module.__file__ = str(Path(directory) / "create_placeholder_images.py")
                with contextlib.redirect_stdout(io.StringIO()):
                    module.main()
                    files = sorted((Path(directory) / "images").glob("*.png"))
                    self.assertEqual(len(files), 4 if chapter == 4 else 3)
                    files[0].write_bytes(b"learner replacement")
                    files[-1].unlink()
                    module.main()
                self.assertEqual(files[0].read_bytes(), b"learner replacement")
                self.assertTrue(files[-1].read_bytes().startswith(b"\x89PNG"))


if __name__ == "__main__":
    unittest.main()
