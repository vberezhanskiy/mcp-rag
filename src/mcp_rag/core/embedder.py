"""Lazy-loaded sentence-transformers embedder, cached locally."""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
from sentence_transformers import CrossEncoder, SentenceTransformer

logger = logging.getLogger(__name__)

# Default: BAAI/bge-m3 — multilingual top-tier (100+ languages incl. RU/EN),
# 568M params (~1.2GB), 8k native context, 1024-dim, encoder-only (BERT-style),
# no query/passage prefix gymnastics, Apache 2.0. Verified to fit on a 16GB
# consumer GPU at the capped sequence length below.
#
# Heavier alternatives (opt-in via MCP_RAG_EMBED_MODEL env):
#   - Qwen/Qwen3-Embedding-0.6B — better MTEB Multilingual (64.33 vs 59.56)
#     and 32k context, but it's a decoder-only LLM with KV cache that
#     allocates aggressively on long inputs and OOMs on 16GB VRAM if you
#     don't shrink the batch even further than the defaults below.
#   - Qwen/Qwen3-Embedding-4B / 8B — more VRAM, more quality.
DEFAULT_EMBED_MODEL = "BAAI/bge-m3"

# Cross-encoder reranker. Bi-encoder retrieval (bm25 + dense) collects
# candidates fast, then this model scores each (query, candidate) pair
# directly — much higher quality on top-K, at ~50–200 ms per call.
# bge-reranker-base is multilingual, ~278 MB, pairs naturally with bge-m3.
DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-base"

# Batch sizes tuned so attention activations fit a 16 GB consumer GPU at the
# capped sequence length (see _MAX_SEQ_LEN). Bigger batches help only as long
# as `batch × seq_len² × heads × bytes` stays under VRAM. With Qwen3-0.6B
# (16 heads, 1024 hidden) and seq=2048, batch=32 in bf16 sits around ~2 GB
# for the attention buffer — comfortable headroom.
_BATCH_SIZES_BY_PARAM_BUCKET = {
    ("small", "cuda"): 256,
    ("small", "mps"): 128,
    ("small", "cpu"): 32,
    ("base", "cuda"): 32,
    ("base", "mps"): 16,
    ("base", "cpu"): 8,
    ("large", "cuda"): 8,
    ("large", "mps"): 4,
    ("large", "cpu"): 2,
}

# Hard cap on per-chunk token length. The retriever chunks at 60 lines (~1.5k
# tokens worst case) and the graph stores entity descriptions much shorter
# than that, so the model's native 8k–32k context is wasted compute and
# turns into OOM on consumer GPUs.
_MAX_SEQ_LEN = 2048

_embedder: Optional[SentenceTransformer] = None
_reranker: Optional[CrossEncoder] = None
_models_dir: Optional[Path] = None


def _embed_model_id() -> str:
    return (os.getenv("MCP_RAG_EMBED_MODEL") or DEFAULT_EMBED_MODEL).strip() or DEFAULT_EMBED_MODEL


def _reranker_model_id() -> str:
    return (os.getenv("MCP_RAG_RERANKER_MODEL") or DEFAULT_RERANKER_MODEL).strip() or DEFAULT_RERANKER_MODEL


def rerank_enabled() -> bool:
    """Cross-encoder rerank is on by default; flip MCP_RAG_RERANK=0 to skip."""
    return (os.getenv("MCP_RAG_RERANK") or "1").strip() not in {"0", "false", "no", "off"}


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

    # bf16 on CUDA halves activation memory; quality difference for
    # similarity scoring is below noise floor.
    model_kwargs = {}
    if device == "cuda":
        try:
            import torch
            model_kwargs["torch_dtype"] = torch.bfloat16
        except Exception:
            pass

    if local_path.exists():
        logger.info("Loading embedder %s from cache: %s (device=%s)", model_id, local_path, device)
        _embedder = SentenceTransformer(str(local_path), device=device, model_kwargs=model_kwargs)
    else:
        logger.info("Downloading embedder %s → %s (device=%s)", model_id, local_path, device)
        _embedder = SentenceTransformer(
            model_id, cache_folder=str(cache_dir), device=device, model_kwargs=model_kwargs,
        )
        _embedder.save(str(local_path))

    # Cap the sequence length to keep activation memory bounded; our chunks
    # never exceed this in practice.
    if hasattr(_embedder, "max_seq_length") and _embedder.max_seq_length > _MAX_SEQ_LEN:
        logger.info("Capping max_seq_length %d → %d", _embedder.max_seq_length, _MAX_SEQ_LEN)
        _embedder.max_seq_length = _MAX_SEQ_LEN
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


def get_reranker() -> CrossEncoder:
    """Lazy-load the cross-encoder reranker."""
    global _reranker
    if _reranker is not None:
        return _reranker
    cache_dir = _models_dir or (Path.home() / ".mcp-rag" / "models")
    cache_dir.mkdir(parents=True, exist_ok=True)
    model_id = _reranker_model_id()
    local_path = cache_dir / _model_dir_name(model_id)
    device = _detect_device()

    model_kwargs = {}
    if device == "cuda":
        try:
            import torch
            model_kwargs["torch_dtype"] = torch.bfloat16
        except Exception:
            pass

    if local_path.exists():
        logger.info("Loading reranker %s from cache: %s (device=%s)", model_id, local_path, device)
        _reranker = CrossEncoder(str(local_path), device=device, model_kwargs=model_kwargs)
    else:
        logger.info("Downloading reranker %s → %s (device=%s)", model_id, local_path, device)
        _reranker = CrossEncoder(model_id, cache_folder=str(cache_dir), device=device, model_kwargs=model_kwargs)
        _reranker.save_pretrained(str(local_path))
    if hasattr(_reranker, "max_length") and _reranker.max_length and _reranker.max_length > _MAX_SEQ_LEN:
        _reranker.max_length = _MAX_SEQ_LEN
    return _reranker


def rerank(query: str, candidates: Sequence[str], top_k: Optional[int] = None) -> list[tuple[int, float]]:
    """Cross-encode (query, candidate) pairs and return [(orig_index, score)] sorted desc.

    Caller passes the candidate texts in their original order; we return
    indices into that list so the caller can reorder its own metadata.
    Falls back gracefully (returns the input order, no scores) if the
    cross-encoder fails to load — pure quality knob, not a hard dep.
    """
    if not candidates:
        return []
    try:
        ce = get_reranker()
        pairs = [(query, c) for c in candidates]
        scores = ce.predict(pairs, show_progress_bar=False)
    except Exception as e:
        logger.warning("rerank failed (%s); falling back to input order", e)
        return [(i, 0.0) for i in range(len(candidates))]
    ranked = sorted(enumerate(scores), key=lambda it: float(it[1]), reverse=True)
    if top_k is not None:
        ranked = ranked[:top_k]
    return [(i, float(s)) for i, s in ranked]
