"""Runtime configuration for an py-utils instance.

A single `Config` object pins the active project root and the storage
directory; everything else (graph, retriever, memory) takes a `Config`
or a `project_root` and derives its own paths from there.

Optional per-project overrides live in ``<project_root>/.py-utils.toml``:

    [ignore]
    dirs = ["my_generated_dir"]      # extra dirs to skip during graph_build
    [extensions]
    extra = ["*.foo", "*.bar"]       # extra source-file extensions to index
    [graph]
    max_file_bytes = 10485760        # override the default 5 MB skip threshold

Loaded once at Config init; subsystems read it via ``config.project``.
"""

from __future__ import annotations

import hashlib
import logging
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


logger = logging.getLogger(__name__)

DEFAULT_STORAGE = Path.home() / ".py-utils"
PROJECT_CONFIG_FILENAME = ".py-utils.toml"


@dataclass
class ProjectConfig:
    """Per-project overrides loaded from ``<project_root>/.py-utils.toml``.

    Empty when the file is absent or unparseable — every subsystem falls
    back to its own defaults so the file is purely additive.
    """

    extra_ignore_dirs: list[str] = field(default_factory=list)
    extra_extensions: list[str] = field(default_factory=list)
    max_file_bytes: Optional[int] = None
    raw: dict = field(default_factory=dict)

    @classmethod
    def load(cls, project_root: Path) -> "ProjectConfig":
        toml_path = project_root / PROJECT_CONFIG_FILENAME
        if not toml_path.is_file():
            return cls()
        try:
            data = tomllib.loads(toml_path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("Could not parse %s: %s", toml_path, exc)
            return cls()

        ignore = data.get("ignore") or {}
        extensions = data.get("extensions") or {}
        graph = data.get("graph") or {}

        return cls(
            extra_ignore_dirs=[str(d) for d in (ignore.get("dirs") or []) if d],
            extra_extensions=[str(e) for e in (extensions.get("extra") or []) if e],
            max_file_bytes=int(graph["max_file_bytes"]) if graph.get("max_file_bytes") else None,
            raw=data,
        )


@dataclass
class Config:
    project_root: Path
    storage_root: Path = field(default_factory=lambda: DEFAULT_STORAGE)
    project: ProjectConfig = field(default_factory=ProjectConfig)

    def __post_init__(self) -> None:
        self.project_root = Path(self.project_root).resolve()
        self.storage_root = Path(self.storage_root).resolve()
        self.storage_root.mkdir(parents=True, exist_ok=True)
        # Re-load the project config now that the path is resolved. Callers
        # who passed an explicit ``project=`` win — we only auto-load when
        # the default factory left it empty.
        if not self.project.raw and not self.project.extra_ignore_dirs and not self.project.extra_extensions:
            self.project = ProjectConfig.load(self.project_root)

    @property
    def project_slug(self) -> str:
        h = hashlib.md5(str(self.project_root).encode()).hexdigest()[:8]
        name = self.project_root.name or "_global"
        return f"{name}_{h}"

    @property
    def project_dir(self) -> Path:
        d = self.storage_root / "projects" / self.project_slug
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def graph_dir(self) -> Path:
        d = self.project_dir / "graph"
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def retriever_cache_dir(self) -> Path:
        d = self.project_dir / "retriever"
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def memory_dir(self) -> Path:
        d = self.project_dir / "memory"
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def models_dir(self) -> Path:
        d = self.storage_root / "models"
        d.mkdir(parents=True, exist_ok=True)
        return d
