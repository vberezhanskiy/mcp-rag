"""Runtime configuration for an mcp-rag instance.

A single `Config` object pins the active project root and the storage
directory; everything else (graph, retriever, memory) takes a `Config`
or a `project_root` and derives its own paths from there.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path


DEFAULT_STORAGE = Path.home() / ".mcp-rag"


@dataclass
class Config:
    project_root: Path
    storage_root: Path = field(default_factory=lambda: DEFAULT_STORAGE)

    def __post_init__(self) -> None:
        self.project_root = Path(self.project_root).resolve()
        self.storage_root = Path(self.storage_root).resolve()
        self.storage_root.mkdir(parents=True, exist_ok=True)

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
