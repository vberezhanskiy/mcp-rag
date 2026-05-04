"""Long-term per-project memory with hybrid BM25 + dense search."""

from __future__ import annotations

import hashlib
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import faiss
import numpy as np
from diskcache import Cache
from rank_bm25 import BM25Okapi

from .embedder import _embed_model_id, encode_documents, encode_query

logger = logging.getLogger(__name__)


class Memory:
    def __init__(
        self,
        content: str,
        memory_type: str,
        tags: List[str],
        confidence: float = 1.0,
        metadata: Optional[Dict] = None,
    ) -> None:
        self.id = hashlib.md5(content.encode()).hexdigest()[:12]
        self.content = content
        self.memory_type = memory_type
        self.tags = tags
        self.confidence = confidence
        self.timestamp = datetime.now(timezone.utc).isoformat()
        self.metadata = metadata or {}

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "content": self.content,
            "type": self.memory_type,
            "tags": self.tags,
            "confidence": self.confidence,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }

    @staticmethod
    def from_dict(data: Dict) -> "Memory":
        mem = Memory(
            content=data["content"],
            memory_type=data["type"],
            tags=data["tags"],
            confidence=data.get("confidence", 1.0),
            metadata=data.get("metadata", {}),
        )
        mem.id = data.get("id", mem.id)
        mem.timestamp = data.get("timestamp", mem.timestamp)
        return mem


class MemorySystem:
    def __init__(self, memory_dir: Path) -> None:
        self.memory_dir = Path(memory_dir)
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.memories_file = self.memory_dir / "memories.json"
        self.cache = Cache(str(self.memory_dir / "embeddings_cache"))
        self.memories: List[Memory] = []
        self.bm25: Optional[BM25Okapi] = None
        self.faiss_index: Optional[faiss.Index] = None
        self._index_dirty = True
        self._load_memories()
        self._build_index()

    @staticmethod
    def _tokenize_text(text: str) -> List[str]:
        text = text or ""
        text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)
        text = text.replace("_", " ").replace("/", " ").replace("\\", " ").replace(".", " ").replace("-", " ")
        return [tok for tok in re.findall(r"[A-Za-zА-Яа-я0-9]+", text.lower()) if len(tok) > 1]

    def _load_memories(self) -> None:
        if self.memories_file.exists():
            try:
                with open(self.memories_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.memories = [Memory.from_dict(m) for m in data]
                logger.info("Loaded %d memories from %s", len(self.memories), self.memories_file)
            except Exception as e:
                logger.warning("Failed to load memories: %s", e)
                self.memories = []
        else:
            self.memories = []

    def _save_memories(self) -> None:
        try:
            with open(self.memories_file, "w", encoding="utf-8") as f:
                json.dump([m.to_dict() for m in self.memories], f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error("Failed to save memories: %s", e)

    def _get_embedding(self, text: str) -> np.ndarray:
        # Cache key includes the model id so a model swap doesn't return
        # vectors with a stale dimension.
        key = f"emb:{_embed_model_id()}:{hashlib.md5(text.encode()).hexdigest()}"
        cached = self.cache.get(key)
        if cached is not None:
            return np.frombuffer(cached, dtype="float32")
        vec = encode_documents([text], normalize_embeddings=True, show_progress_bar=False)[0]
        self.cache.set(key, vec.astype("float32").tobytes())
        return vec.astype("float32")

    def _build_index(self) -> None:
        if not self.memories:
            self.bm25 = None
            self.faiss_index = None
            self._index_dirty = False
            return

        corpus = [m.content for m in self.memories]
        self.bm25 = BM25Okapi([self._tokenize_text(doc) for doc in corpus])

        embeddings = np.stack([self._get_embedding(text) for text in corpus])
        dim = embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dim)
        self.faiss_index.add(embeddings.astype("float32"))
        self._index_dirty = False

    @staticmethod
    def _normalize_content(text: str) -> str:
        return " ".join((text or "").lower().strip().split())

    def _find_duplicate(self, memory: Memory) -> Optional[Memory]:
        normalized = self._normalize_content(memory.content)
        for existing in self.memories:
            if self._normalize_content(existing.content) == normalized:
                return existing
        return None

    def add_or_update_memory(self, memory: Memory, rebuild_index: bool = True) -> str:
        existing = self._find_duplicate(memory)
        if existing is not None:
            changed = False

            if memory.memory_type and existing.memory_type != memory.memory_type:
                existing.memory_type = memory.memory_type
                changed = True

            merged_tags = sorted(set(existing.tags + memory.tags))
            if merged_tags != existing.tags:
                existing.tags = merged_tags
                changed = True

            if memory.confidence > existing.confidence:
                existing.confidence = memory.confidence
                changed = True

            merged_metadata = dict(existing.metadata)
            for key, value in memory.metadata.items():
                if merged_metadata.get(key) != value:
                    merged_metadata[key] = value
                    changed = True
            existing.metadata = merged_metadata

            if changed:
                existing.timestamp = datetime.now(timezone.utc).isoformat()
                self._save_memories()
                if rebuild_index:
                    self._build_index()
                else:
                    self._index_dirty = True
                return "updated"
            return "duplicate"

        self.memories.append(memory)
        self._save_memories()
        if rebuild_index:
            self._build_index()
        else:
            self._index_dirty = True
        return "created"

    def add_memory(self, memory: Memory, rebuild_index: bool = True) -> str:
        return self.add_or_update_memory(memory, rebuild_index=rebuild_index)

    def add_memories_batch(self, memories: List[Memory]) -> dict:
        added = 0
        updated = 0
        for memory in memories:
            result = self.add_or_update_memory(memory, rebuild_index=False)
            if result == "created":
                added += 1
            elif result == "updated":
                updated += 1
        if added or updated:
            self._save_memories()
            self._build_index()
        return {"added": added, "updated": updated}

    def search(
        self,
        query: str,
        top_k: int = 5,
        memory_type: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> List[Tuple[Memory, float]]:
        if not self.memories:
            return []
        if self.bm25 is None or self.faiss_index is None or self._index_dirty:
            self._build_index()
        if self.bm25 is None or self.faiss_index is None:
            return []

        tokenized_query = self._tokenize_text(query)
        bm25_scores = self.bm25.get_scores(tokenized_query)
        q_emb = encode_query([query], normalize_embeddings=True)
        dense_scores, dense_indices = self.faiss_index.search(q_emb.astype("float32"), len(self.memories))

        combined: Dict[int, float] = {}
        max_bm25 = max(bm25_scores) if max(bm25_scores) > 0 else 1.0

        for idx, bm25_score in enumerate(bm25_scores):
            combined[idx] = 0.6 * (bm25_score / max_bm25)

        for idx, dense_score in zip(dense_indices[0], dense_scores[0]):
            if idx == -1:
                continue
            combined[idx] = combined.get(idx, 0.0) + 0.4 * float(dense_score)

        sorted_indices = sorted(
            combined.items(),
            key=lambda item: (
                -round(item[1], 8),
                self.memories[item[0]].memory_type,
                self._normalize_content(self.memories[item[0]].content),
                self.memories[item[0]].timestamp,
            ),
        )

        results: List[Tuple[Memory, float]] = []
        for idx, score in sorted_indices:
            memory = self.memories[idx]
            if memory_type and memory.memory_type != memory_type:
                continue
            if tags and not any(tag in memory.tags for tag in tags):
                continue
            results.append((memory, score))
            if len(results) >= top_k:
                break
        return results

    def get_all_memories(self, memory_type: Optional[str] = None) -> List[Memory]:
        memories = self.memories
        if memory_type:
            memories = [m for m in memories if m.memory_type == memory_type]
        return sorted(
            memories,
            key=lambda m: (m.memory_type, self._normalize_content(m.content), m.timestamp),
        )

    def delete_memory(self, memory_id: str) -> bool:
        before = len(self.memories)
        self.memories = [m for m in self.memories if m.id != memory_id]
        if len(self.memories) == before:
            return False
        self._save_memories()
        self._index_dirty = True
        self._build_index()
        return True

    def delete_memories_by_query(self, query: str) -> int:
        query_lower = (query or "").lower().strip()
        if not query_lower:
            return 0
        before = len(self.memories)
        self.memories = [m for m in self.memories if query_lower not in m.content.lower()]
        deleted = before - len(self.memories)
        if deleted:
            self._save_memories()
            self._index_dirty = True
            self._build_index()
        return deleted

    def update_memory(
        self,
        memory_id: str,
        content: Optional[str] = None,
        memory_type: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> bool:
        target = next((m for m in self.memories if m.id == memory_id), None)
        if target is None:
            return False
        changed = False
        if content is not None and content != target.content:
            target.content = content
            changed = True
        if memory_type is not None and memory_type != target.memory_type:
            target.memory_type = memory_type
            changed = True
        if tags is not None and list(tags) != list(target.tags or []):
            target.tags = list(tags)
            changed = True
        if not changed:
            return False
        target.timestamp = datetime.now(timezone.utc).isoformat()
        self._save_memories()
        self._index_dirty = True
        self._build_index()
        return True

    def clear_all(self) -> None:
        self.memories = []
        self._save_memories()
        self.bm25 = None
        self.faiss_index = None
        self._index_dirty = False
