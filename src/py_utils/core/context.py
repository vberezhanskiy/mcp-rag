"""Named retrieval-context bundles.

A context is a JSON snapshot saved at ``<project_dir>/contexts/<name>.json``
that pins a curated set of files / entities / notes for later reuse:

    {
      "name": "auth-refactor",
      "query": "rip out legacy session middleware",
      "files": ["src/auth/middleware.py", "src/auth/session.py"],
      "entities": ["LegacyAuthMiddleware", "SessionStore"],
      "notes": "Track JWT migration; legal needs token storage compliant.",
      "saved_at": "2026-05-06T20:11:00+00:00"
    }

Inspired by Copilot Spaces and Cline's Memory Bank — lets a multi-day task
keep its "working set" of relevant code as a single named handle that the
agent can pull back at any time.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


logger = logging.getLogger(__name__)


_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_\-]{0,63}$")


def _slug(name: str) -> str:
    """Reject anything that wouldn't be a safe filename — no traversal, no spaces."""
    if not _NAME_RE.match(name or ""):
        raise ValueError(
            f"Invalid context name {name!r}: use 1-64 chars of [A-Za-z0-9_-], "
            "starting with a letter or digit."
        )
    return name


class ContextStore:
    """File-backed CRUD over named context bundles.

    Storage is one JSON file per context under ``<project_dir>/contexts/``.
    Plain text on purpose so users can inspect / hand-edit / commit them.
    """

    def __init__(self, project_dir: Path) -> None:
        self.dir = Path(project_dir) / "contexts"
        self.dir.mkdir(parents=True, exist_ok=True)

    def _path(self, name: str) -> Path:
        return self.dir / f"{_slug(name)}.json"

    def save(
        self,
        name: str,
        query: Optional[str] = None,
        files: Optional[list[str]] = None,
        entities: Optional[list[str]] = None,
        notes: Optional[str] = None,
    ) -> dict:
        payload = {
            "name": _slug(name),
            "query": (query or "").strip(),
            "files": [f.replace("\\", "/") for f in (files or []) if f],
            "entities": [e for e in (entities or []) if e],
            "notes": (notes or "").strip(),
            "saved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        path = self._path(name)
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        payload["path"] = str(path)
        return payload

    def load(self, name: str) -> Optional[dict]:
        path = self._path(name)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning("Could not parse context %s: %s", name, e)
            return None
        data["path"] = str(path)
        return data

    def list(self) -> list[dict]:
        out: list[dict] = []
        for path in sorted(self.dir.glob("*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            out.append({
                "name": data.get("name") or path.stem,
                "saved_at": data.get("saved_at", ""),
                "files": len(data.get("files") or []),
                "entities": len(data.get("entities") or []),
                "query": (data.get("query") or "")[:120],
            })
        return out

    def delete(self, name: str) -> bool:
        path = self._path(name)
        if not path.exists():
            return False
        path.unlink()
        return True
