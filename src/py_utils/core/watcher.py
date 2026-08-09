"""File-system watcher that keeps the knowledge graph in sync with disk edits.

Architecture
------------
``watchdog`` runs an OS-level FS observer in its own thread. Each event is
forwarded to an asyncio ``Queue`` via ``loop.call_soon_threadsafe`` so the
async consumer can debounce bursts of writes (git checkout, formatter,
mass save) before touching the graph.

Per change we call ``CodeGraph.reindex_file(rebuild_faiss=False)`` —
re-encoding the FAISS index for every file save would be ~80s on
ui-kit-sized projects, so we mark FAISS dirty and let it rebuild on
demand from the next similarity query, OR when a consolidation
threshold (default: 25 dirty files) is crossed.

Tools that don't need FAISS (graph_explain, graph_find_usages, etc.)
see fresh data within seconds of a save.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Optional

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

from .graph import _CODE_EXTENSIONS, CodeGraph

logger = logging.getLogger(__name__)


class _Handler(FileSystemEventHandler):
    """Bridge from watchdog's thread to the asyncio loop."""

    def __init__(self, queue: asyncio.Queue, loop: asyncio.AbstractEventLoop):
        self.queue = queue
        self.loop = loop

    def _post(self, event_type: str, src: str, dst: Optional[str] = None) -> None:
        try:
            self.loop.call_soon_threadsafe(self.queue.put_nowait, (event_type, src, dst))
        except RuntimeError:
            # Loop already closed during shutdown — drop silently.
            pass

    def on_modified(self, event: FileSystemEvent) -> None:
        if not event.is_directory:
            self._post("modified", event.src_path)

    def on_created(self, event: FileSystemEvent) -> None:
        if not event.is_directory:
            self._post("created", event.src_path)

    def on_deleted(self, event: FileSystemEvent) -> None:
        if not event.is_directory:
            self._post("deleted", event.src_path)

    def on_moved(self, event: FileSystemEvent) -> None:
        if not event.is_directory:
            self._post("moved", event.src_path, event.dest_path)


class GraphWatcher:
    """Background reindexer driven by file-system events."""

    def __init__(
        self,
        graph: CodeGraph,
        debounce_seconds: float = 1.0,
        faiss_consolidation_threshold: int = 25,
    ) -> None:
        self.graph = graph
        self.debounce = debounce_seconds
        self.faiss_threshold = faiss_consolidation_threshold
        self._observer: Optional[Observer] = None
        self._task: Optional[asyncio.Task] = None
        self._queue: Optional[asyncio.Queue] = None
        self._suffixes = {ext.lstrip("*").lower() for ext in _CODE_EXTENSIONS}
        self._dirty_count = 0

    def start(self) -> None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            logger.warning("GraphWatcher.start outside running loop — skipping")
            return
        self._queue = asyncio.Queue()
        handler = _Handler(self._queue, loop)
        self._observer = Observer()
        self._observer.schedule(handler, str(self.graph.project_root), recursive=True)
        self._observer.start()
        self._task = loop.create_task(self._consume())
        logger.info("GraphWatcher started for %s", self.graph.project_root)

    async def _consume(self) -> None:
        assert self._queue is not None
        loop = asyncio.get_event_loop()
        while True:
            try:
                first = await self._queue.get()
            except asyncio.CancelledError:
                return
            # Latest event per path wins inside a debounce window.
            pending: dict[str, tuple[str, str, Optional[str]]] = {first[1]: first}
            deadline = loop.time() + self.debounce
            while True:
                timeout = deadline - loop.time()
                if timeout <= 0:
                    break
                try:
                    evt = await asyncio.wait_for(self._queue.get(), timeout=timeout)
                except asyncio.TimeoutError:
                    break
                pending[evt[1]] = evt
            await self._apply_batch(list(pending.values()))

    async def _apply_batch(self, events) -> None:
        for event_type, src, dst in events:
            try:
                src_path = Path(src)
                if event_type == "deleted":
                    rel = self._safe_rel(src_path)
                    if rel is not None:
                        self.graph._delete_file_data(rel)
                        self.graph._faiss_dirty = True
                        self._dirty_count += 1
                    continue
                if event_type == "moved":
                    old_rel = self._safe_rel(src_path)
                    if old_rel is not None:
                        self.graph._delete_file_data(old_rel)
                        self.graph._faiss_dirty = True
                    if dst:
                        dst_path = Path(dst)
                        if self._is_indexable(dst_path):
                            await self.graph.reindex_file(dst_path, rebuild_faiss=False)
                            self._dirty_count += 1
                    continue
                if not self._is_indexable(src_path):
                    continue
                await self.graph.reindex_file(src_path, rebuild_faiss=False)
                self._dirty_count += 1
            except Exception as e:
                logger.warning("watcher: %s on %s failed: %s", event_type, src, e)

        if self._dirty_count >= self.faiss_threshold:
            logger.info("GraphWatcher: consolidating FAISS after %d edits", self._dirty_count)
            try:
                self.graph._rebuild_faiss()
            except Exception as e:
                logger.warning("GraphWatcher: FAISS consolidation failed: %s", e)
            self._dirty_count = 0

    def _safe_rel(self, path: Path) -> Optional[str]:
        try:
            return path.relative_to(self.graph.project_root).as_posix()
        except ValueError:
            return None

    def _is_indexable(self, path: Path) -> bool:
        if not path.exists() or not path.is_file():
            return False
        if self.graph._should_ignore(path):
            return False
        name = path.name.lower()
        return any(name.endswith(s) for s in self._suffixes)

    def stop(self) -> None:
        if self._observer is not None:
            try:
                self._observer.stop()
                self._observer.join(timeout=2.0)
            except Exception:
                pass
            self._observer = None
        if self._task is not None:
            self._task.cancel()
            self._task = None
        logger.info("GraphWatcher stopped")
