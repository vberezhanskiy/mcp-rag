"""Lazy-loaded sentence-transformers embedder, cached locally."""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# Default: Qwen/Qwen3-Embedding-0.6B — released June 2025. 0.6B params,
# 1024-dim (flexible 32–1024 via Matryoshka), 32k context, 100+ languages,
# Apache 2.0. MTEB Multilingual 64.33 vs bge-m3 59.56 at the same size
# class. Encoding queries requires the registered "query" prompt; documents
# encode as-is. Override via MCP_RAG_EMBED_MODEL.
DEFAULT_EMBED_MODEL = "Qwen/Qwen3-Embedding-0.6B"

# GPUs love bigger batches; bge-m3 is a heavier model than MiniLM though, so
# we scale a bit more conservatively than for the previous tiny default.
_BATCH_SIZES_BY_PARAM_BUCKET = {
    # (param-bucket, device) → batch size
    ("small", "cuda"): 256,
    ("small", "mps"): 128,
    ("small", "cpu"): 32,
    ("base", "cuda"): 128,
    ("base", "mps"): 64,
    ("base", "cpu"): 16,
    ("large", "cuda"): 64,
    ("large", "mps"): 32,
    ("large", "cpu"): 8,
}

_embedder: Optional[SentenceTransformer] = None
_models_dir: Optional[Path] = None


def _embed_model_id() -> str:
    return (os.getenv("MCP_RAG_EMBED_MODEL") or DEFAULT_EMBED_MODEL).strip() or DEFAULT_EMBED_MODEL


def _model_dir_name(model_id: str) -> str:
    """Filesystem-safe folder name derived from the HF model id."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model_id)


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
    model_id = _embed_model_id()
    local_path = cache_dir / _model_dir_name(model_id)
    device = _detect_device()

    if local_path.exists():
        logger.info("Loading embedder %s from cache: %s (device=%s)", model_id, local_path, device)
        _embedder = SentenceTransformer(str(local_path), device=device)
    else:
        logger.info("Downloading embedder %s → %s (device=%s)", model_id, local_path, device)
        _embedder = SentenceTransformer(model_id, cache_folder=str(cache_dir), device=device)
        _embedder.save(str(local_path))
    return _embedder


def _param_bucket_for_active_model() -> str:
    """Rough param-count bucket so batch size shrinks for heavy models.

    Order matters: the small/base check has to win against the substring
    fallback for "large" — otherwise '0.6b' triggers '6b' and a 600M model
    gets the 8B batch budget (we'd ship under-batched on a 16 GB GPU).
    """
    model_id = _embed_model_id().lower()
    # Sub-1B variants first.
    if any(tag in model_id for tag in ("0.6b", "0.5b", "300m", "embeddinggemma")):
        return "base"
    if any(tag in model_id for tag in ("8b", "7b", "4b", " 6b", "-6b", "_6b", "large", "-l-", "-xl-")):
        return "large"
    if any(tag in model_id for tag in ("base", "bge-m3", "mpnet", "e5-base")):
        return "base"
    return "small"  # MiniLM, e5-small, distil*, etc.


def encode_batch_size() -> int:
    """Per-batch chunk count tuned for active device + model size."""
    device = _detect_device()
    bucket = _param_bucket_for_active_model()
    return _BATCH_SIZES_BY_PARAM_BUCKET.get((bucket, device), 32)


def _has_prompt(embedder: SentenceTransformer, name: str) -> bool:
    prompts = getattr(embedder, "prompts", None) or {}
    return name in prompts


def encode_query(texts, **kwargs) -> np.ndarray:
    """Encode user-side queries.

    Some models (Qwen3-Embedding, e5) lift quality 1-5% when queries are
    encoded with a registered "query" prompt while documents go through
    raw. We pass `prompt_name="query"` only when the model actually has
    that prompt registered — otherwise we encode as-is so the helper is
    safe to call with any model.
    """
    embedder = get_embedder()
    if _has_prompt(embedder, "query"):
        kwargs.setdefault("prompt_name", "query")
    return embedder.encode(texts, **kwargs)


def encode_documents(texts: Iterable[str], **kwargs) -> np.ndarray:
    """Encode passages/entities/chunks. Always raw, no prompt."""
    return get_embedder().encode(texts, **kwargs)
