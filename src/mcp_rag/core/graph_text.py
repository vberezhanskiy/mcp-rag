"""Text-level search for CodeGraph — FTS5 trigram regex search and
ast-grep structural matching.

Mixin: methods use ``self.db_path`` / ``self.project_root`` /
``self._max_file_bytes`` / ``self._get_files()`` from ``CodeGraph``.
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
from pathlib import Path
from typing import Optional


logger = logging.getLogger(__name__)


class GraphTextMixin:
    """Adds regex (FTS5) and structural (ast-grep) search to ``CodeGraph``."""

    _ASTGREP_LANG_FROM_EXT = {
        ".py": "python", ".js": "javascript", ".jsx": "javascript",
        ".ts": "typescript", ".tsx": "tsx", ".go": "go", ".rs": "rust",
        ".java": "java", ".c": "c", ".cpp": "cpp", ".cc": "cpp",
        ".h": "c", ".hpp": "cpp", ".rb": "ruby", ".php": "php",
        ".cs": "c_sharp", ".kt": "kotlin", ".swift": "swift", ".scala": "scala",
        ".lua": "lua", ".html": "html", ".css": "css", ".json": "json",
        ".yaml": "yaml", ".yml": "yaml",
    }

    # ── FTS5 trigram regex search ───────────────────────────────────────────

    def _ensure_text_index(self) -> int:
        """Lazily build the FTS5 trigram chunk index for regex search.

        A chunk is a 100-line slice of a project file; trigram tokenizer
        makes any-substring MATCH sub-second on millions of lines.
        Idempotent — re-runs are no-ops once the table has rows.
        """
        with sqlite3.connect(self.db_path) as con:
            try:
                count = con.execute("SELECT COUNT(*) FROM chunks_fts").fetchone()[0]
                if count > 0:
                    return count
            except sqlite3.OperationalError:
                con.execute(
                    "CREATE VIRTUAL TABLE chunks_fts USING fts5("
                    "content, file UNINDEXED, line_start UNINDEXED, "
                    "tokenize='trigram')"
                )

        chunks: list[tuple[str, str, int]] = []
        chunk_lines = 100
        for path in self._get_files():
            try:
                stat = path.stat()
                if stat.st_size > self._max_file_bytes:
                    continue
                text = path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            rel = path.relative_to(self.project_root).as_posix()
            lines = text.splitlines()
            for i in range(0, len(lines), chunk_lines):
                chunk = "\n".join(lines[i:i + chunk_lines])
                if chunk.strip():
                    chunks.append((chunk, rel, i + 1))

        if chunks:
            with sqlite3.connect(self.db_path) as con:
                con.executemany(
                    "INSERT INTO chunks_fts(content, file, line_start) VALUES (?,?,?)",
                    chunks,
                )
                con.commit()
        logger.info("FTS5 trigram chunk index built: %d chunks", len(chunks))
        return len(chunks)

    def search_regex(
        self,
        pattern: str,
        file_glob: Optional[str] = None,
        case_insensitive: bool = False,
        limit: int = 50,
    ) -> dict:
        """Regex search via FTS5 trigram pre-filter + Python re post-match.

        Required: at least one literal alphanumeric run of 3+ characters
        in the pattern (used as the trigram pre-filter). Patterns of
        only meta-characters are rejected to avoid full-scan degeneration.

        ``file_glob`` accepts either a glob pattern (``*.ts``, ``src/**/*.py``)
        or a plain substring (``components/Button``); wildcard chars are the
        signal that triggers fnmatch — anything else falls back to substring.
        """
        self._ensure_text_index()

        literal_runs = re.findall(r"[A-Za-z0-9_]{3,}", pattern)
        if not literal_runs:
            return {
                "matches": [],
                "warning": (
                    "Pattern has no literal alphanumeric run of 3+ chars — "
                    "FTS5 trigram pre-filter would degenerate to a full scan. "
                    "Add at least one literal substring (e.g. 'def\\s+helper')."
                ),
            }

        safe_runs = [run.replace('"', '""') for run in literal_runs[:6]]
        fts_expr = " ".join(f'"{run}"' for run in safe_runs)

        flags = re.IGNORECASE if case_insensitive else 0
        try:
            rx = re.compile(pattern, flags)
        except re.error as e:
            return {"matches": [], "warning": f"Invalid regex: {e}"}

        matches: list[dict] = []
        with sqlite3.connect(self.db_path) as con:
            try:
                cursor = con.execute(
                    "SELECT file, line_start, content FROM chunks_fts "
                    "WHERE chunks_fts MATCH ? LIMIT ?",
                    (fts_expr, limit * 20),
                )
            except sqlite3.OperationalError as e:
                return {"matches": [], "warning": f"FTS5 MATCH failed: {e}"}

            use_glob = bool(file_glob) and any(c in file_glob for c in "*?[")
            for file_path, line_start, content in cursor:
                if file_glob:
                    if use_glob:
                        import fnmatch
                        if not (fnmatch.fnmatchcase(file_path, file_glob)
                                or fnmatch.fnmatchcase(file_path.replace("\\", "/"), file_glob)):
                            continue
                    elif file_glob not in file_path:
                        continue
                for m in rx.finditer(content):
                    line_offset = content[:m.start()].count("\n")
                    line_no = (line_start or 0) + line_offset
                    line_lo = content.rfind("\n", 0, m.start()) + 1
                    line_hi = content.find("\n", m.end())
                    if line_hi == -1:
                        line_hi = len(content)
                    matches.append({
                        "file": file_path,
                        "line": line_no,
                        "match": m.group(0)[:200],
                        "context": content[line_lo:line_hi][:300],
                    })
                    if len(matches) >= limit:
                        break
                if len(matches) >= limit:
                    break

        return {"matches": matches, "warning": None}

    def text_index_clear(self) -> None:
        """Drop the FTS5 trigram index (forces a rebuild on next call)."""
        with sqlite3.connect(self.db_path) as con:
            try:
                con.execute("DROP TABLE IF EXISTS chunks_fts")
                con.execute("VACUUM")
            except sqlite3.OperationalError:
                pass

    # ── ast-grep structural search ──────────────────────────────────────────

    def structural_search(
        self,
        pattern: str,
        lang: Optional[str] = None,
        path_filter: Optional[str] = None,
        limit: int = 50,
    ) -> dict:
        """AST-precise structural search via the ast-grep CLI.

        Patterns use metavariables: ``fetch($URL)`` matches all `fetch()`
        calls regardless of argument shape. Requires ``ast-grep`` (or
        ``sg``) on PATH.
        """
        import shutil
        import subprocess

        binary = shutil.which("ast-grep") or shutil.which("sg")
        if not binary:
            return {
                "matches": [],
                "warning": (
                    "ast-grep not found on PATH. Install via "
                    "`cargo install ast-grep` or `brew install ast-grep` "
                    "(also called `sg`)."
                ),
            }

        cmd = [binary, "--pattern", pattern, "--json=stream"]
        if lang:
            cmd.extend(["--lang", lang])
        target = self.project_root
        if path_filter:
            target = self.project_root / path_filter
            if not target.exists():
                return {"matches": [], "warning": f"path_filter not found: {path_filter}"}
        cmd.append(str(target))

        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=60,
                encoding="utf-8", errors="replace",
            )
        except subprocess.TimeoutExpired:
            return {"matches": [], "warning": "ast-grep timed out (>60s)"}
        except Exception as e:
            return {"matches": [], "warning": f"ast-grep failed: {e}"}

        if proc.returncode not in (0, 1):
            return {
                "matches": [],
                "warning": f"ast-grep exited {proc.returncode}: {proc.stderr.strip()[:300]}",
            }

        matches: list[dict] = []
        for line in (proc.stdout or "").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            file_path = rec.get("file") or ""
            try:
                rel = Path(file_path).resolve().relative_to(self.project_root).as_posix()
            except (ValueError, OSError):
                rel = file_path
            r = rec.get("range") or {}
            start = (r.get("start") or {}).get("line", 0) + 1
            end = (r.get("end") or {}).get("line", 0) + 1
            matches.append({
                "file": rel,
                "line_start": start,
                "line_end": end,
                "code": (rec.get("text") or "")[:300],
            })
            if len(matches) >= limit:
                break

        return {"matches": matches, "warning": None, "binary": binary}
