"""Lazy-loaded sentence-transformers embedder, cached locally."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # 80MB, 384 dim

_embedder: Optional[SentenceTransformer] = None
_models_dir: Optional[Path] = None


def configure(models_dir: Path) -> None:
    """Set the on-disk cache for the embedder. Call once at startup."""
    global _models_dir
    _models_dir = Path(models_dir)
    _models_dir.mkdir(parents=True, exist_ok=True)


def get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is not None:
        return _embedder

    cache_dir = _models_dir or (Path.home() / ".mcp-rag" / "models")
    cache_dir.mkdir(parents=True, exist_ok=True)
    local_path = cache_dir / "all-MiniLM-L6-v2"

    if local_path.exists():
        logger.info("Loading embedder from local cache: %s", local_path)
        _embedder = SentenceTransformer(str(local_path))
    else:
        logger.info("Downloading embedder %s → %s", EMBED_MODEL, local_path)
        _embedder = SentenceTransformer(EMBED_MODEL, cache_folder=str(cache_dir))
        _embedder.save(str(local_path))
    return _embedder
