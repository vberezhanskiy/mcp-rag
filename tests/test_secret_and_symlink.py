"""Секретные файлы и симлинки наружу не попадают в индексы.

Проверка `_is_secret_file` существовала, но применялась только к РЕЗУЛЬТАТУ
извлечения — то есть после того, как полный текст файла ушёл в LLM-экстрактор.
А обходчики решали «файл ли это» через ``Path.is_file()``, который идёт по
симлинку и отвечает про цель: ссылка изнутри репозитория на файл снаружи
выглядела обычным файлом проекта.
"""

from pathlib import Path
import os
import tempfile
import unittest

from py_utils.paths import is_inside_project, is_secret_file


class SecretFileTests(unittest.TestCase):
    def test_known_secret_names_are_detected(self) -> None:
        for name in (
            ".env",
            ".env.production",
            "prod.env",
            "credentials.json",
            "secrets.yaml",
            "server.pem",
            "id_rsa",
            "id_ed25519",
            "app.key",
        ):
            self.assertTrue(is_secret_file(name), name)

    def test_lookalike_source_files_are_not_secret(self) -> None:
        # Ловушки: имя лишь похоже на секретное. Ложное срабатывание здесь —
        # это молча выпавший из графа исходник, а такое замечают нескоро.
        for name in (
            "env.ts",
            "environment.ts",
            "keyboard.tsx",
            "keychain.py",
            "monkey.md",
        ):
            self.assertFalse(is_secret_file(name), name)

    def test_path_is_taken_from_the_file_name_only(self) -> None:
        self.assertTrue(is_secret_file("backend/config/.env.local"))
        self.assertTrue(is_secret_file(r"C:\project\certs\server.pem"))
        # Каталог с «секретным» именем сам по себе файл не пятнает.
        self.assertFalse(is_secret_file("secrets/loader.py"))


class SymlinkConfinementTests(unittest.TestCase):
    def test_plain_file_inside_the_project_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inside = root / "app.py"
            inside.write_text("x = 1\n", encoding="utf-8")
            self.assertTrue(is_inside_project(root, inside))

    def test_symlink_pointing_outside_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as outer:
            outside = Path(outer) / "credentials"
            outside.write_text("aws_secret_access_key = leak\n", encoding="utf-8")
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                link = root / "credentials.json"
                try:
                    os.symlink(outside, link)
                except (OSError, NotImplementedError):
                    self.skipTest("создание симлинков в этой системе недоступно")
                # Именно так обходчик и обманывался: цель существует и читается.
                self.assertTrue(link.is_file())
                self.assertFalse(is_inside_project(root, link))

    def test_missing_path_is_rejected_without_raising(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertFalse(is_inside_project(root, root / "nope" / ".." / ".." / "x"))


if __name__ == "__main__":
    unittest.main()
