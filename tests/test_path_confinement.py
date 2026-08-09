from pathlib import Path
import tempfile
import unittest

from py_utils.paths import resolve_inside_project


class ProjectPathTests(unittest.TestCase):
    def test_relative_path_stays_inside_project(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertEqual(
                resolve_inside_project(root, "src/example.py"),
                (root / "src/example.py").resolve(),
            )

    def test_parent_and_external_absolute_paths_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            outside = root.parent / "outside.py"
            with self.assertRaisesRegex(ValueError, "escapes project root"):
                resolve_inside_project(root, "../outside.py")
            with self.assertRaisesRegex(ValueError, "escapes project root"):
                resolve_inside_project(root, outside)

    def test_symlink_escape_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            outside = root.parent / f"{root.name}-outside"
            outside.mkdir()
            self.addCleanup(lambda: outside.rmdir() if outside.exists() else None)
            link = root / "escape"
            try:
                link.symlink_to(outside, target_is_directory=True)
            except OSError:
                self.skipTest("symlink creation is not permitted")
            with self.assertRaisesRegex(ValueError, "escapes project root"):
                resolve_inside_project(root, "escape/secret.py")


if __name__ == "__main__":
    unittest.main()
