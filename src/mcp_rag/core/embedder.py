"""Lazy-loaded sentence-transformers embedder, cached locally."""

from __future__ import annotations

import logging
import os
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


def _detect_device() -> str:
    """Pick the best available torch device.

    Honors MCP_RAG_DEVICE override (cuda/mps/cpu) for forcing a specific
    backend (e.g. on a GPU-capable box where the user wants CPU to avoid
    VRAM pressure during a heavy build elsewhere).
    """
    forced = (os.getenv("MCP_RAG_DEVICE") or "").strip().lower()
    if forced in {"cuda", "mps", "cpu"}:
        return forced
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except Exception as e:
        logger.debug("device detection fell back to cpu: %s", e)
    return "cpu"


def get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is not None:
        return _embedder

    cache_dir = _models_dir or (Path.home() / ".mcp-rag" / "models")
    cache_dir.mkdir(parents=True, exist_ok=True)
    local_path = cache_dir / "all-MiniLM-L6-v2"
    device = _detect_device()

    if local_path.exists():
        logger.info("Loading embedder from local cache: %s (device=%s)", local_path, device)
        _embedder = SentenceTransformer(str(local_path), device=device)
    else:
        logger.info("Downloading embedder %s → %s (device=%s)", EMBED_MODEL, local_path, device)
        _embedder = SentenceTransformer(EMBED_MODEL, cache_folder=str(cache_dir), device=device)
        _embedder.save(str(local_path))
    return _embedder


def encode_batch_size() -> int:
    """Per-batch chunk count tuned for the active device.

    GPUs love bigger batches — the CPU-shaped 32/64 default leaves most of
    the kernel pipeline idle on a 16GB card.
    """
    device = _detect_device()
    if device == "cuda":
        return 256
    if device == "mps":
        return 128
    return 32
