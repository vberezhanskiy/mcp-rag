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


def _as_list(v) -> List[str]:
    if isinstance(v, list):
        return [str(x) for x in v]
    if isinstance(v, str) and v:
        return [v]
    return []


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

    # ── storage layout ──
    # Facts persist as `<memory_dir>/memories/<type>__<id>__<slug>.md` plus a
    # human-readable `MEMORY.md` index in the memory_dir root. JSON snapshot
    # is kept alongside as a backup / convenience for programmatic readers.
    # Layout matches Claude Code's per-project memory directory so a fact can
    # be read or hand-edited in any text editor.

    @property
    def md_dir(self) -> Path:
        d = self.memory_dir / "memories"
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def index_file(self) -> Path:
        return self.memory_dir / "MEMORY.md"

    _FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n(.*)\Z", re.DOTALL)

    def _load_memories(self) -> None:
        # Prefer the .md layout. If empty but the legacy JSON exists, read
        # JSON once and rewrite as .md on the next save (auto-migration).
        md_files = sorted((self.memory_dir / "memories").glob("*.md")) \
            if (self.memory_dir / "memories").exists() else []
        if md_files:
            self.memories = []
            for path in md_files:
                try:
                    self.memories.append(self._read_md(path))
                except Exception as e:
                    logger.warning("Failed to read %s: %s", path, e)
            logger.info("Loaded %d memories from %s", len(self.memories), self.md_dir)
            return

        if self.memories_file.exists():
            try:
                with open(self.memories_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.memories = [Memory.from_dict(m) for m in data]
                logger.info("Migrating %d memories from JSON → markdown layout", len(self.memories))
                self._save_memories()
                return
            except Exception as e:
                logger.warning("Failed to load memories.json: %s", e)
        self.memories = []

    def _save_memories(self) -> None:
        # Drop .md files for memory ids that no longer exist (delete/merge).
        live_ids = {m.id for m in self.memories}
        for f in self.md_dir.glob("*.md"):
            if self._extract_id_from_filename(f.name) not in live_ids:
                try:
                    f.unlink()
                except Exception:
                    pass

        for mem in self.memories:
            try:
                self._md_path_for(mem).write_text(self._render_md(mem), encoding="utf-8")
            except Exception as e:
                logger.warning("Failed to write memory %s: %s", mem.id, e)

        try:
            self.index_file.write_text(self._render_index(), encoding="utf-8")
        except Exception as e:
            logger.warning("Failed to write MEMORY.md: %s", e)

        try:
            with open(self.memories_file, "w", encoding="utf-8") as f:
                json.dump([m.to_dict() for m in self.memories], f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error("Failed to save memories.json snapshot: %s", e)

    # ── markdown helpers ──

    @classmethod
    def _read_md(cls, path: Path) -> Memory:
        text = path.read_text(encoding="utf-8")
        m = cls._FRONTMATTER_RE.match(text)
        if not m:
            return Memory(content=text.strip(), memory_type="general", tags=[])
        front = cls._parse_frontmatter(m.group(1))
        body = m.group(2).strip()
        content = body or front.get("content") or ""
        mem = Memory(
            content=content,
            memory_type=front.get("type") or front.get("memory_type") or "general",
            tags=_as_list(front.get("tags")),
            confidence=float(front.get("confidence", 1.0) or 1.0),
            metadata=front.get("metadata") or {},
        )
        if "id" in front:
            mem.id = str(front["id"])
        if "timestamp" in front:
            mem.timestamp = str(front["timestamp"])
        return mem

    @staticmethod
    def _parse_frontmatter(block: str) -> dict:
        # Lightweight YAML-ish parser. Enough for `key: value` and bracketed
        # list literals. We avoid pulling a full YAML dep just for this.
        out: dict = {}
        for line in block.splitlines():
            line = line.rstrip()
            if not line or line.startswith("#"):
                continue
            key, _, value = line.partition(":")
            if not _:
                continue
            key = key.strip()
            value = value.strip()
            if not value:
                out[key] = ""
                continue
            if value.startswith("[") and value.endswith("]"):
                inner = value[1:-1].strip()
                if not inner:
                    out[key] = []
                    continue
                items = [p.strip().strip("'\"") for p in inner.split(",")]
                out[key] = [p for p in items if p]
                continue
            out[key] = value.strip("'\"")
        return out

    def _md_path_for(self, mem: Memory) -> Path:
        return self.md_dir / f"{mem.memory_type}__{mem.id}__{self._slug(mem.content)}.md"

    @staticmethod
    def _extract_id_from_filename(name: str) -> str:
        parts = name.removesuffix(".md").split("__")
        return parts[1] if len(parts) >= 3 else ""

    @staticmethod
    def _slug(text: str, limit: int = 40) -> str:
        s = re.sub(r"[^A-Za-z0-9_]+", "-", (text or "").strip().lower()).strip("-")
        return (s[:limit] or "memory").rstrip("-")

    @staticmethod
    def _render_md(mem: Memory) -> str:
        tags = ", ".join(f"'{t}'" for t in mem.tags)
        meta_block = ""
        if mem.metadata:
            try:
                meta_block = "\nmetadata: " + json.dumps(mem.metadata, ensure_ascii=False)
            except Exception:
                meta_block = ""
        return (
            "---\n"
            f"id: {mem.id}\n"
            f"type: {mem.memory_type}\n"
            f"tags: [{tags}]\n"
            f"confidence: {mem.confidence}\n"
            f"timestamp: {mem.timestamp}{meta_block}\n"
            "---\n"
            f"{mem.content}\n"
        )

    def _render_index(self) -> str:
        if not self.memories:
            return "# Memory\n\n_No facts persisted yet._\n"
        groups: Dict[str, List[Memory]] = {}
        for mem in self.memories:
            groups.setdefault(mem.memory_type, []).append(mem)
        lines = [
            "# Memory",
            "",
            f"_{len(self.memories)} fact(s) across {len(groups)} type(s). "
            "Each entry below links to the corresponding `.md` file._",
            "",
        ]
        for memory_type in sorted(groups):
            entries = sorted(
                groups[memory_type],
                key=lambda m: (m.timestamp or "", m.id),
                reverse=True,
            )
            lines.append(f"## {memory_type} ({len(entries)})")
            for mem in entries:
                rel = self._md_path_for(mem).relative_to(self.memory_dir).as_posix()
                lines.append(f"- [{mem.id}]({rel}) — {self._one_line_hook(mem.content)}")
            lines.append("")
        return "\n".join(lines)

    @staticmethod
    def _one_line_hook(content: str, limit: int = 110) -> str:
        flat = " ".join((content or "").split())
        return flat if len(flat) <= limit else flat[: limit - 1] + "…"

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
