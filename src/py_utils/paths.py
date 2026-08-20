"""Canonical project-path helpers shared by MCP entrypoints and graph storage."""

from pathlib import Path


def resolve_inside_project(project_root: Path, value: "str | Path") -> Path:
    """Resolve ``value`` inside ``project_root``, following existing links."""
    root = Path(project_root).resolve()
    candidate = Path(value)
    resolved = (candidate if candidate.is_absolute() else root / candidate).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Path escapes project root: {value}") from exc
    return resolved


def is_inside_project(project_root: Path, value: "str | Path") -> bool:
    """Whether ``value`` resolves to something inside ``project_root``.

    Обходчики проверяли файл через ``Path.is_file()``, а он идёт ПО симлинку и
    отвечает про цель. Симлинк ``credentials.json`` → ``~/.aws/credentials``
    внутри репозитория выглядел обычным файлом проекта: его читали, слали
    экстрактору и складывали в индекс. ``os.walk`` бережёт от такого только
    каталоги (followlinks=False), файлы — нет.
    """
    try:
        resolve_inside_project(project_root, value)
    except (ValueError, OSError):
        return False
    return True


# Файлы, содержимое которых не уходит ни в LLM-экстрактор, ни в индексы.
# Тот же список продублирован в Node-воркере (tools/_shared.ts, isSecretFile):
# обе половины продукта индексируют один и тот же проект, и расхождение
# означало бы, что одна сторона файл бережёт, а другая — нет.
SECRET_FILE_PATTERNS = (
    ".env",
    ".pem",
    ".key",
    ".pfx",
    ".p12",
    "id_rsa",
    "id_ed25519",
    "credentials",
    "secrets",
)


def is_secret_file(rel_path: "str | Path") -> bool:
    """Секретный ли файл по имени: подстрока в имени, регистр не важен."""
    name = str(rel_path).replace("\\", "/").rsplit("/", 1)[-1].lower()
    return any(marker in name for marker in SECRET_FILE_PATTERNS)
