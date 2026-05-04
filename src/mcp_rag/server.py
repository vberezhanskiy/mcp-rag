"""mcp-rag MCP server (stdio).

Exposes the code knowledge graph, hybrid code search, and project memory
as MCP tools. Pin a project once via ``--project`` (or ``MCP_RAG_PROJECT``
env), then call tools without repeating the path.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any, Optional

from mcp.server import Server
from mcp.server.lowlevel.helper_types import ReadResourceContents
from mcp.server.stdio import stdio_server
from mcp.types import Resource, ResourceTemplate, TextContent, Tool
from pydantic import AnyUrl

from .config import Config
from .core import embedder
from .core.formatter import format_memory_listing
from .core.graph import CodeGraph
from .core.memory import Memory, MemorySystem
from .core.retriever import MultiLangCodeRetriever
from .llm.extractor import LLMExtractor, NoOpExtractor, OpenAICompatExtractor

logger = logging.getLogger("mcp_rag")


class Services:
    """Lazily-built singletons bound to a single Config."""

    def __init__(self, config: Config, llm_extractor: LLMExtractor) -> None:
        self.config = config
        self.llm_extractor = llm_extractor
        self._graph: Optional[CodeGraph] = None
        self._retriever: Optional[MultiLangCodeRetriever] = None
        self._memory: Optional[MemorySystem] = None

    @property
    def graph(self) -> CodeGraph:
        if self._graph is None:
            self._graph = CodeGraph(
                project_root=self.config.project_root,
                graph_dir=self.config.graph_dir,
                llm_extractor=self.llm_extractor,
            )
        return self._graph

    @property
    def retriever(self) -> MultiLangCodeRetriever:
        if self._retriever is None:
            self._retriever = MultiLangCodeRetriever(
                root_dir=self.config.project_root,
                cache_dir=self.config.retriever_cache_dir,
            )
        return self._retriever

    @property
    def memory(self) -> MemorySystem:
        if self._memory is None:
            self._memory = MemorySystem(memory_dir=self.config.memory_dir)
        return self._memory


def _build_tools() -> list[Tool]:
    return [
        Tool(
            name="graph_build",
            description=(
                "Build or refresh the project's code knowledge graph. "
                "Indexes files via tree-sitter / regex (and an optional LLM fallback). "
                "By default indexes every stale file in one call. "
                "Run this before the first graph_* search."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "max_files": {
                        "type": "integer",
                        "description": "Optional cap on how many files to index in this call. Omit to index all stale files.",
                    },
                },
            },
        ),
        Tool(
            name="graph_index_file",
            description="Re-index a single file. Use after editing one file when a full rebuild is overkill.",
            inputSchema={
                "type": "object",
                "properties": {
                    "filepath": {"type": "string", "description": "Path relative to the project root (or absolute)"},
                },
                "required": ["filepath"],
            },
        ),
        Tool(
            name="graph_search",
            description=(
                "Search graph entities by name (classes, functions, components, …). "
                "Combines LIKE filtering with FAISS semantic re-ranking."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "entity_type": {"type": "string", "description": "Optional filter: class/function/method/import/…"},
                    "limit": {"type": "integer", "default": 15},
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="graph_find_usages",
            description="Find every place an entity is referenced (use before refactor).",
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "mode": {"type": "string", "enum": ["all", "callers"], "default": "all"},
                },
                "required": ["name"],
            },
        ),
        Tool(
            name="graph_get_file_deps",
            description="List a file's dependencies — what it imports, inherits from, or uses.",
            inputSchema={
                "type": "object",
                "properties": {"filepath": {"type": "string"}},
                "required": ["filepath"],
            },
        ),
        Tool(
            name="graph_file_structure",
            description="Show all classes/functions/etc. defined in a file.",
            inputSchema={
                "type": "object",
                "properties": {"filepath": {"type": "string"}},
                "required": ["filepath"],
            },
        ),
        Tool(
            name="graph_get_subgraph",
            description=(
                "BFS the graph around an entity up to the given depth. "
                "Each node's relations are capped (per_node_cap) — common "
                "names like Layout/Header otherwise return tens of thousands "
                "of edges. Prefer unique entity names for useful results."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "entity_name": {"type": "string"},
                    "depth": {"type": "integer", "default": 2, "minimum": 1, "maximum": 4},
                    "per_node_cap": {"type": "integer", "default": 50, "minimum": 5, "maximum": 500},
                },
                "required": ["entity_name"],
            },
        ),
        Tool(
            name="graph_stats",
            description="Show graph statistics: file/entity/relation counts and entity types breakdown.",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="graph_pending_files",
            description=(
                "List files where the graph and disk diverge: never-indexed, "
                "stale (mtime changed), and missing (in graph but deleted on disk). "
                "Useful to confirm coverage after graph_build."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "limit": {"type": "integer", "default": 200, "description": "Cap per category"},
                    "filter": {"type": "string", "description": "Optional substring filter applied to paths"},
                },
            },
        ),
        Tool(
            name="graph_explain",
            description=(
                "Compact one-shot view of a file: defined entities, "
                "dependencies (what it imports/uses), and who calls its "
                "exported functions/classes from elsewhere. Replaces calling "
                "graph_file_structure + graph_get_file_deps + graph_find_usages "
                "separately."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "filepath": {"type": "string"},
                    "top_callers": {"type": "integer", "default": 5, "description": "Max external callers to list per defined entity"},
                },
                "required": ["filepath"],
            },
        ),
        Tool(
            name="graph_dead_code",
            description=(
                "List entities (functions, classes, methods, components) that "
                "no relation in the graph points to — i.e. nothing in the indexed "
                "code calls, uses, instantiates or inherits them. Refactor aid. "
                "False positives: dynamic dispatch, public API entry points, "
                "framework callbacks invoked via reflection."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "entity_types": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Filter by types (default: function, method, class, component, interface)",
                    },
                    "limit": {"type": "integer", "default": 50, "minimum": 1, "maximum": 500},
                },
            },
        ),
        Tool(
            name="graph_clear",
            description="Wipe the knowledge graph for the current project.",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="search_code",
            description=(
                "Hybrid semantic+lexical search over project source. "
                "BM25 for exact tokens, dense embeddings for concepts. "
                "Best for 'where is the auth flow', 'how do we serialize X', etc."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "top_k": {"type": "integer", "default": 8, "minimum": 1, "maximum": 30},
                    "max_chunk_chars": {"type": "integer", "default": 1500},
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="memory_add",
            description="Persist a fact about this project (preferences, conventions, decisions).",
            inputSchema={
                "type": "object",
                "properties": {
                    "content": {"type": "string"},
                    "memory_type": {"type": "string", "default": "general"},
                    "tags": {"type": "array", "items": {"type": "string"}, "default": []},
                },
                "required": ["content"],
            },
        ),
        Tool(
            name="memory_search",
            description="Hybrid search over stored memories.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "top_k": {"type": "integer", "default": 5},
                    "memory_type": {"type": "string"},
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="memory_list",
            description="List all memories, optionally filtered by type.",
            inputSchema={
                "type": "object",
                "properties": {"memory_type": {"type": "string"}},
            },
        ),
        Tool(
            name="memory_delete",
            description="Delete a memory by id, or every memory whose content contains the given query.",
            inputSchema={
                "type": "object",
                "properties": {
                    "memory_id": {"type": "string"},
                    "query": {"type": "string"},
                },
            },
        ),
        Tool(
            name="memory_clear",
            description="Delete all memories for this project.",
            inputSchema={"type": "object", "properties": {}},
        ),
    ]


def _norm_path(p: str) -> str:
    """Normalize a project-relative path to POSIX-style separators."""
    return p.replace("\\", "/").lstrip("./")


async def _dispatch(services: Services, name: str, args: dict) -> str:
    g = services.graph
    if name == "graph_build":
        cap = args.get("max_files")
        result = await g.build(max_files=int(cap) if cap is not None else None)
        return json.dumps(result, indent=2)

    if name == "graph_index_file":
        path = Path(args["filepath"])
        if not path.is_absolute():
            path = services.config.project_root / path
        if not path.exists():
            return f"File not found: {args['filepath']}"
        await g.reindex_file(path)
        rel = path.relative_to(services.config.project_root).as_posix()
        entities = g.get_file_entities(rel)
        return f"Indexed {rel}: {len(entities)} entities."

    if name == "graph_search":
        results = g.search_entity(
            args["query"],
            entity_type=args.get("entity_type") or None,
            limit=int(args.get("limit", 15)),
        )
        if not results:
            return f"No entities found for {args['query']!r}."
        lines = [f"Found {len(results)} entities:"]
        for r in results:
            lines.extend(g.format_entity_result(r))
        return "\n".join(lines)

    if name == "graph_find_usages":
        target = args["name"]
        mode = args.get("mode", "all")
        if mode == "callers":
            callers = g.get_callers(target)
            if not callers:
                return f"No callers for {target!r}."
            lines = [f"{target!r} is called from {len(callers)} places:"]
            for c in callers:
                lines.append(f"  • {c['file']} — {c['caller']} → {target}")
            return "\n".join(lines)
        usages = g.find_usages(target)
        if not usages:
            return f"No usages of {target!r}."
        lines = [f"{len(usages)} usages of {target!r}:"]
        for u in usages:
            lines.append(f"  • {u['file']} — {u['from']} --{u['relation']}--> {u['to']}")
        return "\n".join(lines)

    if name == "graph_get_file_deps":
        rel = _norm_path(args["filepath"])
        deps = g.get_file_deps(rel)
        if not deps:
            return f"No dependencies for {rel!r}."
        lines = [f"Dependencies of {rel!r} ({len(deps)}):"]
        by_rel: dict[str, list[dict]] = {}
        for d in deps:
            by_rel.setdefault(d["relation"], []).append(d)
        for rel, items in by_rel.items():
            lines.append(f"  [{rel}]")
            for item in items:
                lines.append(f"    • {item['from']} → {item['to']}")
        return "\n".join(lines)

    if name == "graph_file_structure":
        rel = _norm_path(args["filepath"])
        entities = g.get_file_entities(rel)
        if not entities:
            return f"No entities in {rel!r}."
        lines = [f"Structure of {rel!r} ({len(entities)} entities):"]
        for e in entities:
            lines.extend(g.format_entity_result({"file": rel, **e}, include_snippet=False))
        return "\n".join(lines)

    if name == "graph_get_subgraph":
        depth = max(1, min(int(args.get("depth", 2)), 4))
        cap = max(5, min(int(args.get("per_node_cap", 50)), 500))
        sub = g.get_subgraph(args["entity_name"], depth=depth, per_node_cap=cap)
        if not sub["entities"]:
            return f"No subgraph for {args['entity_name']!r}."
        lines = [
            f"Subgraph around {args['entity_name']!r} (depth {depth}, per_node_cap {cap}):",
            f"  Entities: {len(sub['entities'])}, relations: {len(sub['relations'])}",
        ]
        if sub.get("truncated_nodes"):
            lines.append(
                f"  ⚠ partial result — {len(sub['truncated_nodes'])} hub-like nodes had "
                f"more relations than the cap. Common names (e.g. Layout/Header) "
                f"often share lexical IDs across many files; prefer a more specific "
                f"entity_name. Truncated:"
            )
            for tn in sub["truncated_nodes"][:5]:
                lines.append(f"    - {tn}")
            if len(sub["truncated_nodes"]) > 5:
                lines.append(f"    … and {len(sub['truncated_nodes']) - 5} more")
        lines.append("Entities:")
        for e in sub["entities"]:
            lines.extend(g.format_entity_result(e))
        lines.append("Relations:")
        for rel in sub["relations"][:50]:
            lines.append(f"  • {rel['from']} --{rel['relation']}--> {rel['to']}")
        if len(sub["relations"]) > 50:
            lines.append(f"  … and {len(sub['relations']) - 50} more")
        return "\n".join(lines)

    if name == "graph_stats":
        stats = g.get_stats()
        status = g.get_build_status()
        lines = [
            f"Knowledge graph — {services.config.project_root.name}:",
            f"  Files in graph: {stats['files']}",
            f"  Project files: {status['total_files']}",
            f"  Indexed project files: {status['indexed_project_files']}",
            f"  Stale: {status['stale_files']}, deleted: {status['deleted_files']}",
            f"  Entities: {stats['entities']}",
            f"  Relations: {stats['relations']}",
        ]
        if g.is_building:
            lines.append("  (build in progress)")
        if stats["by_type"]:
            lines.append("  By type:")
            for t, c in stats["by_type"].items():
                lines.append(f"    - {t}: {c}")
        if stats["entities"] == 0 and not g.is_building:
            lines.append("\nGraph is empty. Run graph_build first.")
        return "\n".join(lines)

    if name == "graph_pending_files":
        pending = g.get_pending_files()
        limit = max(1, int(args.get("limit", 200)))
        flt = (args.get("filter") or "").lower()

        def pick(items: list[str]) -> list[str]:
            picked = [p for p in items if flt in p.lower()] if flt else items
            return picked[:limit]

        unindexed = pick(pending["unindexed"])
        stale = pick(pending["stale"])
        missing = pick(pending["missing"])
        total = len(pending["unindexed"]) + len(pending["stale"]) + len(pending["missing"])
        if total == 0:
            return "Graph is fully in sync with disk."

        lines = [
            f"Pending vs disk — unindexed={len(pending['unindexed'])}, "
            f"stale={len(pending['stale'])}, missing={len(pending['missing'])}"
        ]
        if flt:
            lines[0] += f"  (filter={flt!r})"
        for label, items, full in (
            ("Unindexed (never built)", unindexed, pending["unindexed"]),
            ("Stale (mtime changed)", stale, pending["stale"]),
            ("Missing (gone from disk)", missing, pending["missing"]),
        ):
            if not items:
                continue
            lines.append(f"\n{label} ({len(items)} of {len(full)}):")
            for p in items:
                lines.append(f"  • {p}")
            if len(items) < len(full):
                lines.append(f"  … {len(full) - len(items)} more — raise limit or use filter")
        return "\n".join(lines)

    if name == "graph_explain":
        rel = _norm_path(args["filepath"])
        top_callers = max(1, min(int(args.get("top_callers", 5)), 50))
        info = g.explain_file(rel, top_callers=top_callers)
        if not info["entities"] and not info["deps"]:
            return f"Nothing in graph for {rel!r}. Did you run graph_build?"
        lines = [f"# {rel}"]
        if info["entities"]:
            lines.append(f"\n## Defined ({len(info['entities'])})")
            for e in info["entities"]:
                loc = f":{e['line_start']}" if e.get("line_start") else ""
                desc = f" — {e['description']}" if e.get("description") else ""
                lines.append(f"  • [{e['type']}] {e['name']}{loc}{desc}")
        if info["deps"]:
            by_rel: dict[str, list[dict]] = {}
            for d in info["deps"]:
                by_rel.setdefault(d["relation"], []).append(d)
            lines.append(f"\n## Depends on ({len(info['deps'])})")
            for r, items in by_rel.items():
                targets = sorted({i["to"] for i in items})
                lines.append(f"  [{r}] {', '.join(targets)}")
        if info["used_by"]:
            lines.append(f"\n## Used by ({len(info['used_by'])} entities have external callers)")
            for ub in info["used_by"]:
                lines.append(f"  • {ub['type']} {ub['name']} — {ub['total']} caller(s)")
                for c in ub["callers"]:
                    lines.append(f"      ← {c['file']} :: {c['from']}  [{c['relation']}]")
        else:
            lines.append("\n## Used by\n  (no external callers found — internal-only or potential dead code)")
        return "\n".join(lines)

    if name == "graph_dead_code":
        types = args.get("entity_types") or None
        limit = max(1, min(int(args.get("limit", 50)), 500))
        results = g.find_dead_code(entity_types=types, limit=limit)
        if not results:
            return "No dead code found (every defined entity is referenced somewhere)."
        lines = [
            f"{len(results)} possibly-dead entities "
            f"(no callers/usages/instantiations in the graph):",
        ]
        if not types:
            lines[0] += " — types: function, method, class, component, interface"
        for r in results:
            loc = f":{r['line_start']}" if r.get("line_start") else ""
            lines.append(f"  • [{r['type']}] {r['name']}  ({r['file']}{loc})")
        lines.append(
            "\nNote: dynamic dispatch, public API entry points, and framework "
            "callbacks (e.g. event handlers wired by string name) won't show "
            "incoming relations and may be false positives."
        )
        return "\n".join(lines)

    if name == "graph_clear":
        g.clear()
        return f"Knowledge graph cleared for {services.config.project_root.name!r}."

    if name == "search_code":
        top_k = max(1, min(int(args.get("top_k", 8)), 30))
        max_chunk = max(100, min(int(args.get("max_chunk_chars", 1500)), 5000))
        return services.retriever.search(
            args["query"],
            top_k_initial=max(top_k * 4, 30),
            top_k_final=top_k,
            max_chunk_preview=max_chunk,
        )

    if name == "memory_add":
        mem = Memory(
            content=args["content"],
            memory_type=args.get("memory_type", "general"),
            tags=list(args.get("tags") or []),
        )
        result = services.memory.add_or_update_memory(mem)
        return f"Memory {result}: id={mem.id} type={mem.memory_type}"

    if name == "memory_search":
        results = services.memory.search(
            args["query"],
            top_k=int(args.get("top_k", 5)),
            memory_type=args.get("memory_type") or None,
        )
        if not results:
            return "No matching memories."
        lines = [f"{len(results)} matching memories:"]
        for mem, score in results:
            tags = " ".join(mem.tags)
            lines.append(f"  [{mem.memory_type}] score={score:.2f} {mem.content}" + (f" {tags}" if tags else ""))
        return "\n".join(lines)

    if name == "memory_list":
        memories = services.memory.get_all_memories(memory_type=args.get("memory_type") or None)
        return format_memory_listing(memories, title="Memories")

    if name == "memory_delete":
        if args.get("memory_id"):
            ok = services.memory.delete_memory(args["memory_id"])
            return "Deleted." if ok else "Not found."
        if args.get("query"):
            n = services.memory.delete_memories_by_query(args["query"])
            return f"Deleted {n} memories matching {args['query']!r}."
        return "Provide either memory_id or query."

    if name == "memory_clear":
        services.memory.clear_all()
        return "All memories cleared."

    return f"Unknown tool: {name}"


# ─── Resources ─────────────────────────────────────────────────────────────
#
# Same backend as the tool dispatcher, but exposed via the MCP resource
# protocol so Claude Code (and any other MCP client) can let users attach
# rich context with one @-pick instead of three tool round-trips.


def _render_overview(services: Services) -> str:
    g = services.graph
    cfg = services.config
    stats = g.get_stats()
    status = g.get_build_status()
    lines = [
        f"# Project overview — {cfg.project_root.name}",
        "",
        f"Root: `{cfg.project_root}`",
        f"Indexed files: {status['indexed_project_files']} / {status['total_files']}",
        f"Entities: {stats['entities']:,} | relations: {stats['relations']:,}",
    ]
    if stats["entities"] == 0:
        lines.append("\n⚠ Graph is empty — run `graph_build` first.")
        return "\n".join(lines)
    if stats["by_type"]:
        lines.append("\n## Entity types")
        for t, c in stats["by_type"].items():
            lines.append(f"- {t}: {c:,}")
    # Most-referenced entities — cheap "important things to know about" digest.
    import sqlite3
    with sqlite3.connect(g.db_path) as con:
        rows = con.execute(
            "SELECT to_name, COUNT(*) AS c FROM relations "
            "WHERE relation IN ('calls', 'instantiates', 'uses', 'imports') "
            "GROUP BY to_name ORDER BY c DESC LIMIT 15"
        ).fetchall()
    if rows:
        lines.append("\n## Most referenced symbols")
        for name, c in rows:
            lines.append(f"- {name} — {c} refs")
    return "\n".join(lines)


def _render_file(services: Services, path: str) -> str:
    rel = _norm_path(path)
    info = services.graph.explain_file(rel, top_callers=5)
    if not info["entities"] and not info["deps"]:
        return f"# {rel}\n\nNot in graph. Did you run `graph_build`?"
    # Filter to "real" definitions — regex sweep records every `name(` form as
    # a "function" call target with description "Referenced call target", which
    # buries actual class/function declarations from tree-sitter (those have
    # "Extracted from <node>" descriptions or originate from structured parsers).
    primary_types = {"class", "function", "method", "component", "interface", "enum", "type", "module"}
    primary = [
        e for e in info["entities"]
        if e["type"] in primary_types
        and len(e.get("name") or "") > 1
        and "Referenced" not in (e.get("description") or "")
    ]
    lines = [f"# {rel}"]
    if primary:
        lines.append(f"\n## Defined ({len(primary)})")
        for e in primary[:80]:
            loc = f":{e['line_start']}" if e.get("line_start") else ""
            desc = f" — {e['description']}" if e.get("description") else ""
            lines.append(f"- [{e['type']}] **{e['name']}**{loc}{desc}")
        if len(primary) > 80:
            lines.append(f"- … {len(primary) - 80} more")
    if info["deps"]:
        by_rel: dict[str, list[dict]] = {}
        for d in info["deps"]:
            by_rel.setdefault(d["relation"], []).append(d)
        lines.append(f"\n## Dependencies ({len(info['deps'])})")
        for r, items in by_rel.items():
            targets = sorted({i["to"] for i in items})
            lines.append(f"- **{r}** → {', '.join(targets[:30])}"
                         + (f"  (…{len(targets)-30} more)" if len(targets) > 30 else ""))
    primary_names = {e["name"] for e in primary}
    used_by = [ub for ub in info["used_by"] if ub["name"] in primary_names]
    if used_by:
        lines.append(f"\n## External callers")
        for ub in used_by:
            lines.append(f"- {ub['type']} **{ub['name']}** — {ub['total']} caller(s)")
            for c in ub["callers"]:
                lines.append(f"  - `{c['file']}` :: {c['from']}  *[{c['relation']}]*")
    return "\n".join(lines)


def _render_search(services: Services, query: str) -> str:
    return f"# Semantic search: {query!r}\n\n" + services.retriever.search(
        query, top_k_initial=40, top_k_final=8, max_chunk_preview=1200
    )


def _render_explain(services: Services, entity: str) -> str:
    g = services.graph
    results = g.search_entity(entity, limit=5)
    if not results:
        return f"# {entity}\n\nNot found in graph."
    lines = [f"# {entity}", ""]
    for r in results:
        lines.extend(g.format_entity_result(r))
        callers = g.get_callers(r["name"])
        if callers:
            lines.append(f"  callers ({len(callers)}):")
            for c in callers[:10]:
                lines.append(f"    ← {c['file']} :: {c['caller']}")
    return "\n".join(lines)


def _resource_kind_and_param(uri: AnyUrl) -> tuple[str, str]:
    """Parse rag://kind/param → (kind, decoded param)."""
    from urllib.parse import unquote
    raw = str(uri)
    if not raw.startswith("rag://"):
        return ("", "")
    rest = raw[len("rag://"):]
    if "/" in rest:
        kind, _, param = rest.partition("/")
        return (kind, unquote(param))
    return (rest, "")


def build_server(services: Services) -> Server:
    server = Server("mcp-rag")
    tools = _build_tools()

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        return tools

    @server.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
        try:
            text = await _dispatch(services, name, arguments or {})
        except Exception as e:
            logger.exception("Tool %s failed", name)
            text = f"Error in {name}: {e}"
        return [TextContent(type="text", text=text)]

    @server.list_resources()
    async def list_resources() -> list[Resource]:
        return [
            Resource(
                uri=AnyUrl("rag://overview"),
                name="Project overview",
                description="Top-level digest of the project: file count, entity types, most-referenced symbols. Pick this first when entering an unfamiliar codebase.",
                mimeType="text/markdown",
            ),
        ]

    @server.list_resource_templates()
    async def list_resource_templates() -> list[ResourceTemplate]:
        return [
            ResourceTemplate(
                uriTemplate="rag://file/{path}",
                name="File context",
                description="Defined entities + dependencies + external callers for a project-relative path. Bundles graph_file_structure + graph_get_file_deps + graph_find_usages.",
                mimeType="text/markdown",
            ),
            ResourceTemplate(
                uriTemplate="rag://search/{query}",
                name="Semantic search",
                description="Hybrid BM25 + dense-embedding search across project source. Pick when you don't know exact names — ranks by meaning.",
                mimeType="text/markdown",
            ),
            ResourceTemplate(
                uriTemplate="rag://explain/{entity}",
                name="Entity card",
                description="Locations of an entity (class/function/component) plus its callers. Useful for refactor scoping.",
                mimeType="text/markdown",
            ),
        ]

    @server.read_resource()
    async def read_resource(uri: AnyUrl):
        kind, param = _resource_kind_and_param(uri)
        try:
            if kind == "overview":
                content = _render_overview(services)
            elif kind == "file" and param:
                content = _render_file(services, param)
            elif kind == "search" and param:
                content = _render_search(services, param)
            elif kind == "explain" and param:
                content = _render_explain(services, param)
            else:
                content = f"Unknown resource: {uri}"
        except Exception as e:
            logger.exception("read_resource %s failed", uri)
            content = f"Error rendering {uri}: {e}"
        return [ReadResourceContents(content=content, mime_type="text/markdown")]

    return server


def _resolve_llm_extractor() -> LLMExtractor:
    if os.getenv("MCP_RAG_LLM_BASE_URL") and os.getenv("MCP_RAG_LLM_API_KEY"):
        try:
            return OpenAICompatExtractor()
        except Exception as e:
            logger.warning("LLM extractor disabled: %s", e)
    return NoOpExtractor()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="mcp-rag", description="MCP server: code graph + RAG search")
    parser.add_argument(
        "--project",
        default=os.getenv("MCP_RAG_PROJECT") or os.getcwd(),
        help="Project root (defaults to MCP_RAG_PROJECT or cwd)",
    )
    parser.add_argument(
        "--storage",
        default=os.getenv("MCP_RAG_STORAGE"),
        help="Storage root for SQLite/FAISS/cache (default ~/.mcp-rag)",
    )
    parser.add_argument(
        "--log-level",
        default=os.getenv("MCP_RAG_LOG", "INFO"),
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=args.log_level, format="%(levelname)s %(name)s: %(message)s")

    storage = Path(args.storage) if args.storage else None
    config = Config(
        project_root=Path(args.project),
        storage_root=storage or (Path.home() / ".mcp-rag"),
    )
    embedder.configure(config.models_dir)

    services = Services(config=config, llm_extractor=_resolve_llm_extractor())
    server = build_server(services)

    async def _run() -> None:
        async with stdio_server() as (read, write):
            await server.run(read, write, server.create_initialization_options())

    logger.info("mcp-rag serving project=%s storage=%s", config.project_root, config.storage_root)
    asyncio.run(_run())


if __name__ == "__main__":
    main()
