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
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

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
                "Run this before the first graph_* search."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "max_files": {"type": "integer", "default": 200, "description": "Max files indexed in this call"},
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
            description="BFS the graph around an entity up to the given depth.",
            inputSchema={
                "type": "object",
                "properties": {
                    "entity_name": {"type": "string"},
                    "depth": {"type": "integer", "default": 2, "minimum": 1, "maximum": 4},
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


async def _dispatch(services: Services, name: str, args: dict) -> str:
    g = services.graph
    if name == "graph_build":
        result = await g.build(max_files=int(args.get("max_files", 200)))
        return json.dumps(result, indent=2)

    if name == "graph_index_file":
        path = Path(args["filepath"])
        if not path.is_absolute():
            path = services.config.project_root / path
        if not path.exists():
            return f"File not found: {args['filepath']}"
        await g.reindex_file(path)
        rel = str(path.relative_to(services.config.project_root))
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
        deps = g.get_file_deps(args["filepath"])
        if not deps:
            return f"No dependencies for {args['filepath']!r}."
        lines = [f"Dependencies of {args['filepath']!r} ({len(deps)}):"]
        by_rel: dict[str, list[dict]] = {}
        for d in deps:
            by_rel.setdefault(d["relation"], []).append(d)
        for rel, items in by_rel.items():
            lines.append(f"  [{rel}]")
            for item in items:
                lines.append(f"    • {item['from']} → {item['to']}")
        return "\n".join(lines)

    if name == "graph_file_structure":
        entities = g.get_file_entities(args["filepath"])
        if not entities:
            return f"No entities in {args['filepath']!r}."
        lines = [f"Structure of {args['filepath']!r} ({len(entities)} entities):"]
        for e in entities:
            lines.extend(g.format_entity_result({"file": args["filepath"], **e}, include_snippet=False))
        return "\n".join(lines)

    if name == "graph_get_subgraph":
        depth = max(1, min(int(args.get("depth", 2)), 4))
        sub = g.get_subgraph(args["entity_name"], depth=depth)
        if not sub["entities"]:
            return f"No subgraph for {args['entity_name']!r}."
        lines = [
            f"Subgraph around {args['entity_name']!r} (depth {depth}):",
            f"  Entities: {len(sub['entities'])}, relations: {len(sub['relations'])}",
            "Entities:",
        ]
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
