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
