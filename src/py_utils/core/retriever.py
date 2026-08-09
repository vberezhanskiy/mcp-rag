"""Hybrid code retriever — BM25 (lexical) + dense embeddings (semantic).

Indexes the project once, caches the index on disk via diskcache, then
serves text queries that combine BM25 and FAISS scores.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import faiss
import numpy as np
from diskcache import Cache
from rank_bm25 import BM25Okapi
from tqdm import tqdm
import gitignore_parser

from .embedder import (
    _embed_model_id,
    encode_batch_size,
    encode_documents,
    encode_query,
    rerank,
    rerank_enabled,
)
from .formatter import format_code_result

logger = logging.getLogger(__name__)


class MultiLangCodeRetriever:
    def __init__(
        self,
        root_dir: str | Path,
        cache_dir: Path,
        chunk_size: int = 60,
        overlap: int = 15,
    ) -> None:
        self.root_dir = Path(root_dir).resolve()
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.cache = Cache(str(cache_dir))
        self.bm25: Optional[BM25Okapi] = None
        self.faiss_index: Optional[faiss.Index] = None
        self.chunks: List[str] = []
        self.file_paths: List[str] = []
        self.line_numbers: List[int] = []
        self._gitignore_parser = None
        gitignore_path = self.root_dir / ".gitignore"
        if gitignore_path.exists():
            try:
                self._gitignore_parser = gitignore_parser.parse_gitignore(gitignore_path)
            except Exception:
                pass
        self._load_or_build_index()

    @staticmethod
    def _tokenize_text(text: str) -> List[str]:
        text = text or ""
        text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)
        text = text.replace("_", " ").replace("/", " ").replace("\\", " ").replace(".", " ").replace("-", " ")
        tokens = [tok for tok in re.findall(r"[A-Za-zА-Яа-я0-9]+", text.lower()) if len(tok) > 1]
        return tokens or ([text.lower().strip()] if text.strip() else [])

    @staticmethod
    def _code_extensions() -> List[str]:
        return [
            "*.py", "*.pyi", "*.ipynb",
            "*.js", "*.jsx", "*.ts", "*.tsx", "*.vue",
            "*.java", "*.kt", "*.scala", "*.groovy",
            "*.go", "*.rs", "*.swift", "*.dart",
            "*.cs", "*.fs", "*.vb",
            "*.php", "*.rb", "*.pl", "*.pm",
            "*.cpp", "*.cc", "*.cxx", "*.h", "*.hpp", "*.c", "*.m", "*.mm",
            "*.sql", "*.sh", "*.bash", "*.yaml", "*.yml", "*.toml", "*.json", "*.xml",
            "*.html", "*.css", "*.scss", "*.less",
        ]

    @staticmethod
    def _ignore_dirs() -> set[str]:
        return {
            "node_modules", "venv", ".venv", "__pycache__", ".git", ".pytest_cache",
            "dist", "build", "logs", ".idea", ".vscode", "target", "bin", "obj",
            ".gradle", "vendor", "CMakeFiles", "Debug", "Release", ".angular",
        }

    def _should_ignore(self, path: Path) -> bool:
        if self._gitignore_parser is not None:
            try:
                if self._gitignore_parser(str(path)):
                    return True
            except Exception:
                pass
        ignore_exts = {".min.js", ".min.css", ".map", ".pyc", ".class", ".jar"}
        path_lower = str(path).lower()
        for ext in ignore_exts:
            if path_lower.endswith(ext):
                return True
        # Exact-match against path components — substring check would filter
        # files like Layout.tsx/Login.tsx because their parent dir contains
        # "out"/"log" as substring.
        parts = {p.lower() for p in path.parts}
        return bool(parts & self._ignore_dirs())

    def _should_ignore_dir(self, name: str) -> bool:
        return name in self._ignore_dirs()

    def _iter_indexable_files(self) -> List[Path]:
        files: list[Path] = []
        seen: set[str] = set()
        exts = {ext.lstrip("*").lower() for ext in self._code_extensions()}

        for dirpath, dirnames, filenames in os.walk(self.root_dir):
            dirnames[:] = [d for d in dirnames if not self._should_ignore_dir(d)]
            for fname in filenames:
                fname_lower = fname.lower()
                if not any(fname_lower.endswith(ext) for ext in exts):
                    continue
                fpath = os.path.join(dirpath, fname)
                if fpath in seen:
                    continue
                seen.add(fpath)
                path = Path(fpath)
                if not path.is_file() or self._should_ignore(path):
                    continue
                files.append(path)
        return sorted(files)

    def _build_index(self) -> None:
        import time
        t0 = time.time()
        logger.info("Building hybrid index for %s", self.root_dir)

        files = self._iter_indexable_files()
        logger.info("Found %d files in %.2fs", len(files), time.time() - t0)

        documents: list[str] = []
        file_paths: list[str] = []
        line_numbers: list[int] = []
        tokenized_corpus: list[list[str]] = []

        # Threshold below which a generic chunk is dropped as boilerplate. Files
        # whose entire body is shorter than this still get exactly one fallback
        # chunk so they don't disappear from the index entirely (config files,
        # short re-export shims, .storybook/withTheme.tsx-style snippets).
        MIN_CHUNK_CHARS = 60
        MIN_FALLBACK_CHARS = 10

        for path in tqdm(files, desc="Indexing"):
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    lines = f.readlines()
                rel = path.relative_to(self.root_dir).as_posix()
                added = 0
                for i in range(0, len(lines), self.chunk_size - self.overlap):
                    chunk_lines = lines[i:i + self.chunk_size]
                    chunk = "".join(chunk_lines).rstrip()
                    if len(chunk.strip()) < MIN_CHUNK_CHARS:
                        continue
                    documents.append(chunk)
                    file_paths.append(rel)
                    line_numbers.append(i + 1)
                    tokenized_corpus.append(self._tokenize_text(chunk))
                    added += 1
                if added == 0:
                    whole = "".join(lines).rstrip()
                    if len(whole.strip()) >= MIN_FALLBACK_CHARS:
                        documents.append(whole)
                        file_paths.append(rel)
                        line_numbers.append(1)
                        tokenized_corpus.append(self._tokenize_text(whole))
            except Exception as e:
                logger.warning("Skipped %s: %s", path, e)

        if not documents:
            logger.warning("No indexable code chunks found")
            return

        self.bm25 = BM25Okapi(tokenized_corpus)

        logger.info("Generating embeddings for %d chunks…", len(documents))
        embeddings = encode_documents(
            documents,
            normalize_embeddings=True,
            show_progress_bar=True,
            batch_size=encode_batch_size(),
        )
        dim = embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dim)
        self.faiss_index.add(embeddings.astype("float32"))

        self.chunks = documents
        self.file_paths = file_paths
        self.line_numbers = line_numbers

        cache_key = self._get_cache_key()
        self.cache[cache_key] = {
            "bm25": self.bm25,
            "faiss_index": faiss.serialize_index(self.faiss_index),
            "chunks": self.chunks,
            "file_paths": self.file_paths,
            "line_numbers": self.line_numbers,
        }
        logger.info("Index built and cached (%d chunks)", len(self.chunks))

    def _load_or_build_index(self) -> None:
        cache_key = self._get_cache_key()
        if cache_key in self.cache:
            logger.info("Loading retriever index from cache")
            data = self.cache[cache_key]
            self.bm25 = data["bm25"]
            self.faiss_index = faiss.deserialize_index(data["faiss_index"])
            self.chunks = data["chunks"]
            self.file_paths = data["file_paths"]
            self.line_numbers = data.get("line_numbers", [0] * len(self.chunks))
        else:
            self._evict_old_cache(cache_key)
            self._build_index()

    def _evict_old_cache(self, current_key: str) -> None:
        stale_keys = [k for k in self.cache if k != current_key]
        for k in stale_keys:
            del self.cache[k]
        if stale_keys:
            logger.info("Evicted %d stale cache entries", len(stale_keys))

    def _get_cache_key(self) -> str:
        h = hashlib.md5()
        exts = {ext.lstrip("*").lower() for ext in self._code_extensions()}
        for dirpath, dirnames, filenames in os.walk(self.root_dir):
            dirnames[:] = [d for d in dirnames if not self._should_ignore_dir(d)]
            for fname in filenames:
                fpath = os.path.join(dirpath, fname)
                if not any(fname.lower().endswith(ext) for ext in exts):
                    continue
                if self._should_ignore(Path(fpath)):
                    continue
                try:
                    stat = os.stat(fpath)
                    rel = os.path.relpath(fpath, self.root_dir)
                    h.update(f"{rel}:{stat.st_mtime_ns}:{stat.st_size}".encode())
                except OSError:
                    continue
        # Include model id and chunking params so changing the embedder
        # invalidates the cache (FAISS index would otherwise have a stale
        # dimension and fail the next encode/search).
        h.update(f"{self.chunk_size}:{self.overlap}:{_embed_model_id()}".encode())
        return h.hexdigest()

    def rebuild(self) -> None:
        """Drop the cache and rebuild from scratch."""
        self.cache.clear()
        self._build_index()

    @staticmethod
    def _query_literals(query: str) -> list[str]:
        """Extract identifier-shaped literals from a query.

        Catches kebab-case (epcp-flex), snake_case (use_user_store),
        dotted (data.api.url), and CamelCase (FlexProps) — the kinds
        of tokens BM25 splits but that the user almost certainly typed
        because the exact substring exists in code.
        """
        kebab_or_snake = re.findall(r"[A-Za-z][A-Za-z0-9]*(?:[-_.][A-Za-z0-9]+)+", query)
        camel = re.findall(r"\b[a-z]+(?:[A-Z][a-z0-9]+)+\b|\b[A-Z][a-z]+(?:[A-Z][a-z0-9]+)+\b", query)
        out: list[str] = []
        for tok in kebab_or_snake + camel:
            if len(tok) >= 4 and tok.lower() not in {t.lower() for t in out}:
                out.append(tok)
        return out

    def search(
        self,
        query: str,
        top_k_initial: int = 60,
        top_k_final: int = 8,
        max_chunk_preview: int = 1500,
    ) -> str:
        if not self.bm25 or not self.faiss_index:
            return "Code index is empty — no source files were found in the project."

        tokenized_query = self._tokenize_text(query)
        bm25_scores = self.bm25.get_scores(tokenized_query)
        bm25_top = np.argsort(bm25_scores)[::-1][:top_k_initial]

        q_emb = encode_query([query], normalize_embeddings=True)
        dense_scores, dense_indices = self.faiss_index.search(q_emb.astype("float32"), top_k_initial)

        max_bm25 = max(bm25_scores) if max(bm25_scores) > 0 else 1.0

        dense_score_map: Dict[int, float] = {}
        for idx, dscore in zip(dense_indices[0], dense_scores[0]):
            if idx != -1:
                dense_score_map[int(idx)] = float(dscore)

        # Identifier-shaped substrings (kebab/snake/dot/Camel) are nearly
        # always intentional — pre-collect chunks that contain them so
        # they're guaranteed to enter the rerank pool even if BM25's
        # tokenizer dismantled the literal. Track per-literal document
        # frequency so we can IDF-weight the bonus later: a rare token
        # like 'epcp-flex' is a far stronger signal than 'className'.
        literals = self._query_literals(query)
        literal_hits: set[int] = set()
        literal_idf: Dict[str, float] = {}
        lits_lower: list[str] = [l.lower() for l in literals]
        if literals:
            n_chunks = max(len(self.chunks), 1)
            per_lit_hits: Dict[str, set[int]] = {l: set() for l in lits_lower}
            for idx, chunk in enumerate(self.chunks):
                cl = chunk.lower()
                for l in lits_lower:
                    if l in cl:
                        per_lit_hits[l].add(idx)
                        literal_hits.add(idx)
            import math
            for l, hits in per_lit_hits.items():
                df = max(len(hits), 1)
                # log scaling — rare literal in 5/2000 chunks gets weight ~6;
                # common like 'className' in 1000/2000 gets ~0.7. We scale
                # this with 0.2 inside the blend so a single rare literal
                # contributes ~1.2 — enough to dominate when nothing else
                # in the corpus carries it.
                literal_idf[l] = math.log(n_chunks / df)

        candidates: Dict[int, Tuple[str, str, int, float]] = {}
        all_ids = set(int(i) for i in bm25_top) | set(dense_score_map.keys()) | literal_hits
        for idx in all_ids:
            bm25_norm = bm25_scores[idx] / max_bm25 if idx < len(bm25_scores) else 0.0
            dense_norm = dense_score_map.get(idx, 0.0)
            combined = 0.5 * bm25_norm + 0.5 * dense_norm
            candidates[idx] = (self.chunks[idx], self.file_paths[idx], self.line_numbers[idx], combined)

        # Wider rerank pool — top_k * 8 — keeps literal-hit chunks reachable
        # even when BM25 ranks them low because the unique substring was
        # tokenized away.
        rerank_pool = max(top_k_final * 8, 60)
        sorted_cands = sorted(candidates.items(), key=lambda x: x[1][3], reverse=True)[:rerank_pool]
        if not sorted_cands:
            return "Nothing relevant found."

        if rerank_enabled() and len(sorted_cands) > 1:
            chunks_for_rerank = [c[1][0] for c in sorted_cands]
            ranked = rerank(query, chunks_for_rerank, top_k=None)
            ce_by_pos = {pos: float(score) for pos, score in ranked}
            blended: list = []
            for pos, (chunk_idx, (chunk, path, line, hybrid_score)) in enumerate(sorted_cands):
                ce = ce_by_pos.get(pos, 0.0)
                # Blend cross-encoder + bm25/dense hybrid + IDF-weighted
                # literal bonus. CE leads on concept queries; the hybrid
                # term keeps exact-token matches alive; the literal term
                # ensures rare user-typed identifiers (epcp-flex,
                # useUserStore, data.api.url) outrank chunks that just
                # share a generic word like 'className'.
                bonus = 0.0
                if lits_lower:
                    cl = chunk.lower()
                    bonus = 0.2 * sum(literal_idf.get(l, 0.0) for l in lits_lower if l in cl)
                final = 0.65 * ce + 0.35 * float(hybrid_score) + bonus
                blended.append((chunk_idx, (chunk, path, line, final)))
            blended.sort(key=lambda x: x[1][3], reverse=True)
            sorted_cands = blended[:top_k_final]
        else:
            sorted_cands = sorted_cands[:top_k_final]

        output = []
        for _, (chunk, path, line_num, score) in sorted_cands:
            preview = chunk[:max_chunk_preview] + ("..." if len(chunk) > max_chunk_preview else "")
            chunk_lines = chunk.count("\n") + 1
            output.append(format_code_result(path, line_num, line_num + chunk_lines - 1, score, preview))

        return "\n\n".join(output) if output else "No relevant code found."
