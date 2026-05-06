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
from .core.context import ContextStore
from .core.formatter import format_memory_listing
from .core.graph import CodeGraph
from .core.memory import Memory, MemorySystem
from .core.metrics import default_metrics
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
        self._contexts: Optional[ContextStore] = None
        # Background graph_build task — kept alive on the Services object so
        # asyncio doesn't GC it while it runs.
        self._build_task: Optional[asyncio.Task] = None

    @property
    def graph(self) -> CodeGraph:
        if self._graph is None:
            self._graph = CodeGraph(
                project_root=self.config.project_root,
                graph_dir=self.config.graph_dir,
                llm_extractor=self.llm_extractor,
                project_config=self.config.project,
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

    @property
    def contexts(self) -> ContextStore:
        if not hasattr(self, "_contexts") or self._contexts is None:
            self._contexts = ContextStore(self.config.project_dir)
        return self._contexts


def _memory_disabled() -> bool:
    """``MCP_RAG_NO_MEMORY=1`` hides the memory_* tools.

    Useful when the host already has its own persistent memory (e.g.
    Claude Code's ``~/.claude/memory/``) and the duplicate surface in
    the tool list is just noise.
    """
    return (os.getenv("MCP_RAG_NO_MEMORY") or "").strip() in {"1", "true", "yes", "on"}


def _build_tools() -> list[Tool]:
    tools = [
        Tool(
            name="graph_build",
            description=(
                "Build or refresh the per-project code knowledge graph. "
                "Walks source via tree-sitter / regex extractors and stores "
                "entities + relations in SQLite + FAISS. By default indexes "
                "every stale file in one call. "
                "Auto-runs on the first data-needing graph_* tool when the "
                "graph is empty, so you usually don't call this directly — "
                "only after large branch switches, after changing extractor "
                "settings, or when graph_pending_files reports stale files."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "max_files": {
                        "type": "integer",
                        "description": "Optional cap on how many files to index in this call. Omit to index all stale files.",
                    },
                    "background": {
                        "type": "boolean",
                        "default": False,
                        "description": "If true, run the build as a background asyncio task and return immediately. Use on big projects where the synchronous call would hit Claude Code's ~30s tool-call timeout. Poll graph_stats / graph_pending_files for progress; graph_stats shows '(build in progress)' while it's running.",
                    },
                },
            },
        ),
        Tool(
            name="graph_index_file",
            description=(
                "Re-index ONE file after edits. Useful when you've just "
                "changed a single file and want its entities/relations "
                "refreshed without sweeping the whole project. For a full "
                "project refresh use graph_build (it's already incremental "
                "via mtime check)."
            ),
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
                "Find entities by NAME (substring/multi-token match on the "
                "entities table) with optional type filter. Returns a typed "
                "list of declarations — classes, functions, methods, "
                "components, etc. — that contain the query in their name. "
                "Niche: 'list all classes with Button in the name', "
                "'every use* hook', 'all *Provider components'.\n\n"
                "NOT for concept search — descriptive queries like "
                "'extract antd styles' return nothing because this matches "
                "identifier text, not meaning. Use search_code for that.\n"
                "NOT for refactor scope — for an exact name's call sites "
                "use graph_find_usages."
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
            description=(
                "Every place an EXACT-named entity is referenced — calls, "
                "JSX <Component> usages, instantiations, inheritance, "
                "uses-as-property. Mode 'callers' narrows to calls only. "
                "Run before rename/delete/refactor.\n\n"
                "Needs the literal entity name. For partial-name discovery "
                "use graph_search; for concept search use search_code."
            ),
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
            description=(
                "Outgoing edges of a file: what it imports, inherits from, "
                "or uses. Grouped by relation type.\n\n"
                "graph_explain returns this PLUS the file's own declarations "
                "PLUS external callers in one call. Prefer graph_explain "
                "unless you specifically want only the deps section."
            ),
            inputSchema={
                "type": "object",
                "properties": {"filepath": {"type": "string"}},
                "required": ["filepath"],
            },
        ),
        Tool(
            name="graph_file_structure",
            description=(
                "Bare list of declarations in a file (classes, functions, "
                "methods, components, interfaces, types, …) with line "
                "numbers.\n\n"
                "graph_explain wraps this with deps and external callers. "
                "Prefer graph_explain unless you need a stripped-down "
                "list for further programmatic processing."
            ),
            inputSchema={
                "type": "object",
                "properties": {"filepath": {"type": "string"}},
                "required": ["filepath"],
            },
        ),
        Tool(
            name="graph_get_subgraph",
            description=(
                "BFS the relation graph around an entity, up to the given "
                "depth. Returns reached entities + edges between them. "
                "Each node's relations are capped (per_node_cap, default 50) "
                "and overflowed nodes are listed under truncated_nodes so "
                "the caller knows the result is partial.\n\n"
                "Reliable on uniquely-named entities only. Common names "
                "(Layout, Header, props, value) recur as `to_name` across "
                "hundreds of files and produce mostly truncated_nodes — "
                "use graph_find_usages for those instead."
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
            description=(
                "Counts (files, entities, relations) plus a per-type "
                "breakdown. Quick freshness sanity check at session "
                "start; pair with graph_pending_files to see *which* "
                "files diverge."
            ),
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="graph_pending_files",
            description=(
                "Three lists of files where the graph and disk diverge:\n"
                "  unindexed — exist on disk, never indexed\n"
                "  stale     — indexed but mtime has changed\n"
                "  missing   — in the graph, deleted from disk\n"
                "Use after graph_build to confirm coverage, or before a "
                "session to decide whether to rebuild. `filter` does a "
                "substring match on path, `limit` caps each category."
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
                "Compact dossier on a file in one call:\n"
                "  • Declarations defined in it (filtered to real "
                "    class/function/component rows, regex noise dropped)\n"
                "  • Dependencies grouped by relation (imports, uses, "
                "    inherits)\n"
                "  • External callers per defined entity\n\n"
                "Default starting point when reading or about to edit a "
                "file — replaces graph_file_structure + graph_get_file_deps "
                "+ graph_find_usages in one round trip."
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
            name="graph_find_similar",
            description=(
                "Entities semantically nearest to a given anchor by FAISS "
                "vector similarity. The embed text combines name + "
                "outgoing relations digest + code snippet, so structural "
                "fingerprints cluster (e.g. all antd-component wrappers "
                "that `instantiate Ant*; call cn` end up close).\n\n"
                "Run before writing a new helper or component to find an "
                "existing one that does the same thing. Anchor must be "
                "the name of an indexed entity. For text concept search "
                "(no anchor) use search_code; for partial-name discovery "
                "use graph_search."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "entity_name": {"type": "string"},
                    "limit": {"type": "integer", "default": 10, "minimum": 1, "maximum": 50},
                    "min_score": {"type": "number", "default": 0.4, "minimum": 0.0, "maximum": 1.0},
                    "entity_types": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional type filter (e.g. ['function','component']).",
                    },
                },
                "required": ["entity_name"],
            },
        ),
        Tool(
            name="graph_dead_code",
            description=(
                "Definitions (function/method/class/component/interface) "
                "that no relation in the graph points at — i.e. nothing in "
                "the indexed code calls, uses, instantiates, or inherits "
                "them. Refactor candidates.\n\n"
                "Pass exclude_paths globs (e.g. ['demoapp/*', "
                "'**/*.stories.*']) to hide scaffolding/sample code where "
                "no-usages is expected. Known false positives in any "
                "codebase: dynamic dispatch, public API entry points, "
                "framework callbacks wired by string name (event handlers, "
                "DI tokens, route handlers)."
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
                    "exclude_paths": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional fnmatch globs to skip (e.g. ['demoapp/*', '**/*.stories.*']) — useful for hiding scaffolding where no-usages is expected.",
                    },
                },
            },
        ),
        Tool(
            name="graph_visualize",
            description=(
                "Render the project's dependency graph as a self-contained "
                "HTML file with three drill-down levels:\n"
                "  • Modules — top-level path-prefix nodes, cross-module "
                "    edges weighted by relation count\n"
                "  • Files — double-click a module to see its files plus "
                "    file→file edges (dashed = link to outside the module)\n"
                "  • Entities — double-click a file to see its declared "
                "    classes/functions/components and their relations\n\n"
                "Self-contained: vis-network from CDN, all data inlined as "
                "JSON. Returns the path to the generated HTML — open it "
                "in a browser."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "output_path": {
                        "type": "string",
                        "description": "Where to write the HTML (default: ~/.mcp-rag/projects/<slug>/graph.html so the project working tree stays clean).",
                    },
                    "module_depth": {
                        "type": "integer",
                        "default": 2,
                        "minimum": 1,
                        "maximum": 4,
                        "description": "How many leading path segments form a module id (e.g. 2 → 'storybook/src/components')",
                    },
                },
            },
        ),
        Tool(
            name="graph_clear",
            description=(
                "Wipe the knowledge graph for this project. Destructive — "
                "the next data-needing graph_* call will trigger a full "
                "rebuild via auto-build. Use only when changing extractor "
                "settings, switching embedding models with different "
                "dimensions, or troubleshooting suspected corruption. "
                "graph_build is incremental, so this is rarely needed."
            ),
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="metrics",
            description=(
                "In-process metrics snapshot — counters (tool calls, "
                "errors), latency histograms (mean/p50/p95/max in ms), "
                "and gauges (FAISS size, cache hit rate). Useful for "
                "tuning batch sizes / device choice without standing up "
                "Prometheus.\n\n"
                "Pass `reset=true` to zero everything after the snapshot."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "reset": {"type": "boolean", "default": False},
                },
            },
        ),
        Tool(
            name="search_regex",
            description=(
                "Sub-second regex search across project file content "
                "via SQLite FTS5 trigram pre-filter + Python `re` "
                "post-match.\n\n"
                "Pre-filter requires at least one literal alphanumeric "
                "run of 3+ chars in the pattern (e.g. `def\\s+test_\\w+`, "
                "`epcp-flex`, `TODO|FIXME`). Pure meta-character patterns "
                "are rejected to avoid full-scan degeneration.\n\n"
                "Complements search_code (semantic) and "
                "graph_structural_search (AST). Use when you need raw "
                "byte-level pattern matching — short literals where "
                "dense embedders smudge, or whole-line patterns no "
                "tokenizer would catch."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Python regex pattern."},
                    "file_glob": {
                        "type": "string",
                        "description": "Optional substring filter on result file paths.",
                    },
                    "case_insensitive": {"type": "boolean", "default": False},
                    "limit": {"type": "integer", "default": 50, "minimum": 1, "maximum": 500},
                },
                "required": ["pattern"],
            },
        ),
        Tool(
            name="graph_filter_by_trait",
            description=(
                "Filter entities by detected traits — language-agnostic "
                "markers extracted at index time from the entity head:\n"
                "  • async / generator / abstract / static / deprecated\n"
                "  • exported / default-export (TS/JS surface)\n"
                "  • test (entity in a test file by path heuristic)\n\n"
                "Combine with entity_types and path_filter to narrow. "
                "Examples:\n"
                "  ['async','exported']            — exported async APIs\n"
                "  ['deprecated']                  — refactor backlog\n"
                "  ['abstract'] type=class         — base classes only\n"
                "  ['test'] negate via filter on   — write tests for missing surface\n"
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "traits": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of trait tokens to require (or any of, see match).",
                    },
                    "match": {
                        "type": "string",
                        "enum": ["all", "any"],
                        "default": "all",
                        "description": "all = AND across traits, any = OR.",
                    },
                    "entity_types": {"type": "array", "items": {"type": "string"}},
                    "path_filter": {"type": "string", "description": "Substring to match in file paths."},
                    "limit": {"type": "integer", "default": 50, "minimum": 1, "maximum": 500},
                },
                "required": ["traits"],
            },
        ),
        Tool(
            name="graph_structural_search",
            description=(
                "AST-precise structural search via ast-grep. Patterns "
                "use metavariables — ``fetch($URL)`` matches every fetch "
                "call regardless of argument shape; ``$X.then($CB)`` "
                "finds every promise-then chain.\n\n"
                "Complements text search (BM25/dense miss structural "
                "shape) and entity search (graph_search/find_usages need "
                "exact names). Requires ast-grep (or `sg`) on PATH; "
                "install via `cargo install ast-grep` or `brew install "
                "ast-grep`.\n\n"
                "Use case: refactor candidates ('every direct fetch "
                "call'), pattern-based audits ('every empty catch "
                "block'), structural codemod scoping."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "ast-grep pattern with $VARS metavariables.",
                    },
                    "lang": {
                        "type": "string",
                        "description": "Language hint (python, javascript, typescript, tsx, go, rust, java, c, cpp, ruby, php, c_sharp, kotlin, swift, html, css, json, yaml). Auto-skipped when omitted — ast-grep will try several.",
                    },
                    "path_filter": {
                        "type": "string",
                        "description": "Optional sub-path under the project root to restrict the search (e.g. 'src/components').",
                    },
                    "limit": {"type": "integer", "default": 50, "minimum": 1, "maximum": 500},
                },
                "required": ["pattern"],
            },
        ),
        Tool(
            name="graph_repo_map",
            description=(
                "Token-budgeted skeleton of the most-important code in "
                "the project, ranked by personalized PageRank over the "
                "relation graph (calls/uses/imports/instantiates/"
                "inherits). Optional focus_files / focus_entities bias "
                "the ranking toward areas you're working on (10× and "
                "50× respectively).\n\n"
                "Use case: load the most relevant N tokens of project "
                "context into an LLM at session start. Beats blind "
                "file listing because the ranking reflects actual call "
                "topology — central abstractions surface first."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "token_budget": {"type": "integer", "default": 8000, "minimum": 500, "maximum": 60000},
                    "focus_files": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Substrings of file paths to bias toward (e.g. ['src/auth', 'routes']).",
                    },
                    "focus_entities": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Exact entity names to bias toward (50× weight on personalization vector).",
                    },
                },
            },
        ),
        Tool(
            name="graph_test_coverage",
            description=(
                "Map production entities to the test files that exercise "
                "them. Reverse-traverses the graph: any function/method/"
                "class/component referenced (calls/uses/instantiates) from "
                "a test file is marked as covered. Test files detected by "
                "path/filename heuristics: test_*.py, *.test.tsx, *.spec.*, "
                "**/__tests__/**, *_test.go, *Test.java, etc.\n\n"
                "Modes:\n"
                "  • 'summary'   — counts + by-type covered/uncovered ratios\n"
                "  • 'uncovered' — list production defs with no test refs\n"
                "  • 'entity'    — list tests that reference one entity\n\n"
                "Limitations: misses dynamic dispatch and indirection through "
                "string-based test fixtures or DI containers."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "mode": {
                        "type": "string",
                        "enum": ["summary", "uncovered", "entity"],
                        "default": "summary",
                    },
                    "entity_name": {
                        "type": "string",
                        "description": "Required when mode='entity'.",
                    },
                    "test_globs": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional override for test-file detection (default covers Python/JS/TS/Go/JVM/Rust).",
                    },
                    "target_path_filter": {
                        "type": "string",
                        "description": "Optional substring filter on production file paths (e.g. 'src/auth').",
                    },
                    "limit": {"type": "integer", "default": 50, "minimum": 1, "maximum": 500},
                },
            },
        ),
        Tool(
            name="graph_find_clones",
            description=(
                "Detect clusters of near-duplicate definitions across the "
                "codebase. Pairs are flagged when FAISS cosine similarity "
                "is high (semantic match) AND outgoing-relation Jaccard "
                "overlap is high (structural match: same calls/uses "
                "downstream). Pairs are merged into clusters via "
                "union-find; only clusters with >=2 members are returned.\n\n"
                "Use case: 'consolidate these N implementations'. Catches "
                "copy-pasted helpers, parallel auth flows, repeated "
                "validation/normalization functions that text search misses."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "min_score": {
                        "type": "number",
                        "default": 0.85,
                        "minimum": 0.0,
                        "maximum": 1.0,
                        "description": "FAISS cosine threshold. Lower = looser semantic match, more clusters.",
                    },
                    "min_shape_overlap": {
                        "type": "number",
                        "default": 0.3,
                        "minimum": 0.0,
                        "maximum": 1.0,
                        "description": "Jaccard threshold on outgoing (relation, target) sets. Filters out semantically-similar but structurally-different pairs.",
                    },
                    "top_k_per_entity": {
                        "type": "integer",
                        "default": 5,
                        "minimum": 1,
                        "maximum": 20,
                    },
                    "entity_types": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional restriction (default: function, method, class, component, interface).",
                    },
                    "limit": {"type": "integer", "default": 50, "minimum": 1, "maximum": 200},
                },
            },
        ),
        Tool(
            name="search_code",
            description=(
                "Hybrid lexical + semantic search OVER CODE TEXT (not the "
                "entity table). Stack: BM25 → dense embeddings (bge-m3) → "
                "cross-encoder rerank → IDF-weighted bonus for kebab/snake/"
                "dotted/Camel literals the user typed verbatim. Mixed "
                "RU/EN queries supported.\n\n"
                "Default tool for concept questions ('how does theming "
                "work', 'where is the auth flow'), pattern hunting, and "
                "any query where you don't know the exact entity name. "
                "Returns code chunks with file/lines/score.\n\n"
                "For an exact entity name's references → graph_find_usages. "
                "For partial-name discovery in the entity registry → "
                "graph_search. For an indexed entity's nearest neighbors → "
                "graph_find_similar."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "top_k": {"type": "integer", "default": 8, "minimum": 1, "maximum": 30},
                    "max_chunk_chars": {"type": "integer", "default": 1500},
                    "hyde": {
                        "type": "boolean",
                        "default": False,
                        "description": "When true and an LLM is configured (MCP_RAG_LLM_*), draft a hypothetical code snippet from the query first and append it before retrieval. Bridges the question-vs-code embedding gap for natural-language queries.",
                    },
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="context_save",
            description=(
                "Save a named retrieval-context bundle for this project: "
                "the curated set of files / entities / notes you're "
                "actively working with. Persists to "
                "<project_dir>/contexts/<name>.json so the same handle "
                "can be reloaded across sessions or shared via git.\n\n"
                "Use case: 'auth-refactor' — pin the auth module files, "
                "the entities being renamed, and a note about the JWT "
                "migration plan; pull the bundle back at the start of "
                "the next session via context_load."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Bundle name. 1-64 chars of [A-Za-z0-9_-], starting with letter/digit.",
                    },
                    "query": {"type": "string", "description": "Optional query/topic that scoped this bundle."},
                    "files": {"type": "array", "items": {"type": "string"}, "description": "Project-relative file paths."},
                    "entities": {"type": "array", "items": {"type": "string"}, "description": "Entity names central to this context."},
                    "notes": {"type": "string", "description": "Free-form notes (decisions, TODOs, gotchas)."},
                },
                "required": ["name"],
            },
        ),
        Tool(
            name="context_load",
            description=(
                "Load a saved context bundle by name. Returns query + "
                "files + entities + notes so a fresh agent session can "
                "rehydrate the prior working set in one call."
            ),
            inputSchema={
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            },
        ),
        Tool(
            name="context_list",
            description="List every saved context bundle for this project (name, saved_at, file/entity counts, query preview).",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="context_delete",
            description="Delete a saved context bundle by name. Returns whether anything was removed.",
            inputSchema={
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            },
        ),
        Tool(
            name="memory_save",
            description=(
                "Persist a fact about this project (decision, convention, "
                "DTO shape, operational note). Stored per project root, "
                "indexed for hybrid search. Returns the memory id.\n\n"
                "If you already track project notes in another system "
                "(Claude Code's local memory, Obsidian, etc.), prefer that "
                "and skip this — mcp-rag memory is most useful when "
                "multiple tools/IDEs share the same MCP server."
            ),
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
            description=(
                "Hybrid BM25 + dense search across stored project "
                "memories. Optional filter by memory_type. Top score "
                "with a wide gap to the runner-up usually means a clean "
                "hit; flat distribution means no real match."
            ),
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
            description=(
                "List all stored memories for this project, optionally "
                "filtered by memory_type. Useful for review and curation."
            ),
            inputSchema={
                "type": "object",
                "properties": {"memory_type": {"type": "string"}},
            },
        ),
        Tool(
            name="memory_delete",
            description=(
                "Remove a single memory by its id, or — if `query` is "
                "given instead — every memory whose content contains that "
                "substring (case-insensitive). Pass exactly one of the "
                "two."
            ),
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
            description=(
                "Wipe all memories for this project. Destructive and "
                "non-reversible — use only when starting over."
            ),
            inputSchema={"type": "object", "properties": {}},
        ),
    ]
    if _memory_disabled():
        tools = [t for t in tools if not t.name.startswith("memory_")]
    return tools


def _norm_path(p: str) -> str:
    """Normalize a project-relative path to POSIX-style separators."""
    return p.replace("\\", "/").lstrip("./")


# Tools that read graph data and benefit from an auto-build on the first call
# of a fresh session. Inspection / lifecycle / non-graph tools are excluded so
# they don't block on a 100s+ index build the user didn't ask for.
_GRAPH_TOOLS_NEEDING_DATA = {
    "graph_search",
    "graph_find_usages",
    "graph_get_file_deps",
    "graph_file_structure",
    "graph_get_subgraph",
    "graph_explain",
    "graph_dead_code",
    "graph_find_similar",
    "graph_find_clones",
    "graph_test_coverage",
    "graph_repo_map",
    "graph_filter_by_trait",
    "graph_visualize",
}


_AUTO_BUILD_SYNC_THRESHOLD = 300


async def _ensure_graph_built(services: Services) -> Optional[str]:
    """If the graph is empty, build it (sync for small projects, async for
    big ones) and return a status banner. None when data is already there.
    """
    g = services.graph
    if g.get_stats()["entities"] > 0:
        return None

    # If a background build is already running (kicked off at server start
    # or by a prior call), don't start another — just tell the caller to
    # wait. Partial state may still serve some queries.
    bt = services._build_task
    if bt is not None and not bt.done():
        status = g.get_build_status()
        return (
            f"ℹ Background build still running — "
            f"{status['indexed_project_files']} of {status['total_files']} indexed so far. "
            f"Retry in a moment for fresh data.\n\n"
        )

    status = g.get_build_status()
    total = status["total_files"]
    # Large projects don't fit Claude Code's ~30s tool-call window — do
    # this in the background and let the caller poll graph_stats.
    if total > _AUTO_BUILD_SYNC_THRESHOLD:
        logger.info("Auto-build (background) for %d files", total)
        services._build_task = asyncio.create_task(g.build())
        return (
            f"ℹ Graph empty for {total} files — kicked off a background build. "
            f"Retry your request in ~1 min, or watch progress with graph_stats / "
            f"graph_pending_files.\n\n"
        )

    # Small project — synchronous is fine, user gets results in one round-trip.
    logger.info("Auto-build (sync) for %d files", total)
    try:
        result = await g.build()
    except Exception as e:
        logger.exception("auto-build failed")
        return f"⚠ auto-build failed: {e}\n"
    return (
        f"ℹ Graph was empty — auto-built: "
        f"{result['indexed']} files, {result['entities']} entities, "
        f"{result['relations']} relations.\n\n"
    )


async def _dispatch(services: Services, name: str, args: dict) -> str:
    if name in _GRAPH_TOOLS_NEEDING_DATA:
        banner = await _ensure_graph_built(services) or ""
    else:
        banner = ""
    result = await _dispatch_inner(services, name, args)
    return banner + result if banner else result


async def _dispatch_inner(services: Services, name: str, args: dict) -> str:
    g = services.graph
    if name == "graph_build":
        cap = args.get("max_files")
        cap = int(cap) if cap is not None else None
        if bool(args.get("background", False)):
            existing = services._build_task
            if existing is not None and not existing.done():
                return (
                    "Background build already in progress. "
                    "Check graph_stats — it will show '(build in progress)' "
                    "until done."
                )
            async def _bg_build():
                try:
                    result = await g.build(max_files=cap)
                    logger.info("Background graph_build done: %s", result)
                except Exception:
                    logger.exception("Background graph_build failed")
            services._build_task = asyncio.create_task(_bg_build())
            status = g.get_build_status()
            return (
                f"Background build started — {status['stale_files']} stale, "
                f"{status['deleted_files']} deleted out of {status['total_files']} project files. "
                f"Poll graph_stats / graph_pending_files; graph_stats shows '(build in progress)' "
                f"while it's running."
            )
        result = await g.build(max_files=cap)
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
        q = args["query"]
        results = g.search_entity(
            q,
            entity_type=args.get("entity_type") or None,
            limit=int(args.get("limit", 15)),
        )
        if not results:
            return (
                f"No entities found by name for {q!r}.\n\n"
                f"graph_search matches identifiers (class/function/component "
                f"names). For concept search across code, run search_code "
                f"with the same query."
            )
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
        if stats.get("embedding_cache_rows"):
            cache_line = f"  Embedding cache: {stats['embedding_cache_rows']:,} entries"
            if "last_cache_hit_rate" in stats:
                cache_line += f"  (last rebuild hit rate: {stats['last_cache_hit_rate']*100:.1f}%)"
            lines.append(cache_line)
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

    if name == "graph_find_similar":
        anchor = args["entity_name"]
        limit = max(1, min(int(args.get("limit", 10)), 50))
        min_score = max(0.0, min(float(args.get("min_score", 0.4)), 1.0))
        types = args.get("entity_types") or None
        out = g.find_similar_entities(
            entity_name=anchor,
            limit=limit,
            min_score=min_score,
            entity_types=types,
        )
        if out.get("warning"):
            return out["warning"]
        results = out["results"]
        if not results:
            return (
                f"No similar entities found for {anchor!r} above score {min_score}. "
                "Try lowering min_score or check the name with graph_search first."
            )
        lines = [f"{len(results)} entities semantically near {anchor!r}:"]
        for r in results:
            loc = f":{r['line_start']}" if r.get("line_start") else ""
            desc = f" — {r['description']}" if r.get("description") else ""
            lines.append(f"  • [{r['type']}] {r['name']}  score={r['score']:.3f}  ({r['file']}{loc}){desc}")
        return "\n".join(lines)

    if name == "graph_dead_code":
        types = args.get("entity_types") or None
        limit = max(1, min(int(args.get("limit", 50)), 500))
        exclude_paths = args.get("exclude_paths") or None
        results = g.find_dead_code(entity_types=types, limit=limit, exclude_paths=exclude_paths)
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

    if name == "metrics":
        # Refresh a couple of useful gauges right before snapshot so the
        # picture is current at read time even when nothing has called
        # the underlying tool recently.
        try:
            faiss = g.faiss_index
            default_metrics.set_gauge("faiss.entities", float(faiss.ntotal if faiss else 0))
        except Exception:  # pragma: no cover
            pass
        try:
            stats = g.get_stats()
            default_metrics.set_gauge("graph.entities", float(stats.get("entities", 0)))
            default_metrics.set_gauge("graph.relations", float(stats.get("relations", 0)))
            default_metrics.set_gauge("graph.embedding_cache_rows", float(stats.get("embedding_cache_rows", 0)))
        except Exception:  # pragma: no cover
            pass
        snap = default_metrics.snapshot()
        if bool(args.get("reset")):
            default_metrics.reset()
        lines = ["# mcp-rag metrics"]
        if snap["counters"]:
            lines.append("\n## Counters")
            for k, v in sorted(snap["counters"].items()):
                lines.append(f"  {k}: {v}")
        if snap["gauges"]:
            lines.append("\n## Gauges")
            for k, v in sorted(snap["gauges"].items()):
                lines.append(f"  {k}: {v:g}")
        if snap["histograms"]:
            lines.append("\n## Latency (ms)")
            lines.append(f"  {'metric':<40s}  count   mean    p50    p95    max")
            for k in sorted(snap["histograms"]):
                h = snap["histograms"][k]
                lines.append(
                    f"  {k:<40s}  {h['count']:>5d}  {h['mean']:>5.1f}  {h['p50']:>5.1f}  {h['p95']:>5.1f}  {h['max']:>5.1f}"
                )
        if len(lines) == 1:
            lines.append("\nNo metrics recorded yet.")
        if bool(args.get("reset")):
            lines.append("\n_Metrics reset after snapshot._")
        return "\n".join(lines)

    if name == "search_regex":
        out = g.search_regex(
            pattern=str(args.get("pattern") or "").strip(),
            file_glob=args.get("file_glob") or None,
            case_insensitive=bool(args.get("case_insensitive")),
            limit=int(args.get("limit", 50)),
        )
        if out.get("warning"):
            return out["warning"]
        matches = out["matches"]
        if not matches:
            return f"No regex matches for pattern: {args.get('pattern')!r}"
        lines = [f"{len(matches)} regex match(es):"]
        for m in matches:
            lines.append(f"  • {m['file']}:{m['line']}")
            lines.append(f"      {m['context'].rstrip()[:200]}")
        return "\n".join(lines)

    if name == "graph_filter_by_trait":
        traits = args.get("traits") or []
        if not isinstance(traits, list) or not traits:
            return "❌ traits must be a non-empty list."
        rows = g.find_by_trait(
            traits=[str(t) for t in traits],
            entity_types=args.get("entity_types") or None,
            path_filter=args.get("path_filter") or None,
            limit=int(args.get("limit", 50)),
            match=str(args.get("match") or "all"),
        )
        if not rows:
            return f"No entities with traits {traits} (match={args.get('match', 'all')})."
        lines = [f"{len(rows)} entities matching traits {traits}:"]
        for r in rows:
            loc = f":{r['line_start']}" if r.get("line_start") else ""
            tags = " ".join(f"#{t}" for t in r.get("traits") or [])
            lines.append(f"  • [{r['type']}] {r['name']}  ({r['file']}{loc})  {tags}")
        return "\n".join(lines)

    if name == "graph_structural_search":
        out = g.structural_search(
            pattern=str(args.get("pattern") or "").strip(),
            lang=args.get("lang") or None,
            path_filter=args.get("path_filter") or None,
            limit=int(args.get("limit", 50)),
        )
        if out.get("warning"):
            return out["warning"]
        matches = out["matches"]
        if not matches:
            return f"No matches for pattern: {args.get('pattern')!r}"
        lines = [f"{len(matches)} structural match(es) for pattern {args.get('pattern')!r}:"]
        for m in matches:
            lines.append(f"  • {m['file']}:{m['line_start']}-{m['line_end']}")
            lines.append(f"      {m['code'][:200]}")
        return "\n".join(lines)

    if name == "graph_repo_map":
        out = g.build_repo_map(
            token_budget=int(args.get("token_budget", 8000)),
            focus_files=args.get("focus_files") or None,
            focus_entities=args.get("focus_entities") or None,
        )
        if out.get("warning"):
            return out["warning"]
        return out["markdown"] or "Repo map empty — graph has no rankable entities yet."

    if name == "graph_test_coverage":
        mode = (args.get("mode") or "summary").strip()
        out = g.find_test_coverage(
            mode=mode,
            entity_name=args.get("entity_name"),
            test_globs=args.get("test_globs") or None,
            target_path_filter=args.get("target_path_filter") or None,
            limit=int(args.get("limit", 50)),
        )
        if "error" in out:
            return f"❌ {out['error']}"
        if mode == "summary":
            lines = [
                f"Test coverage — {services.config.project_root.name}",
                f"  Test files: {out['test_files']}",
                f"  Production files: {out['production_files']}",
                f"  Production defs: {out['production_entities']}",
                f"  Covered: {out['covered']}  ({out['coverage_pct']:.1f}%)",
                f"  Uncovered: {out['uncovered']}",
            ]
            if out["by_type"]:
                lines.append("  By type:")
                for t, c in sorted(out["by_type"].items()):
                    total_t = c["covered"] + c["uncovered"]
                    pct = (c["covered"] / total_t * 100) if total_t else 0.0
                    lines.append(f"    - {t}: {c['covered']}/{total_t}  ({pct:.0f}%)")
            return "\n".join(lines)
        if mode == "uncovered":
            items = out["uncovered"]
            if not items:
                return f"All production entities are covered (across {out['test_files']} test file(s))."
            lines = [
                f"{out['uncovered_count']} uncovered production entities "
                f"(across {out['test_files']} test file(s)):"
            ]
            for e in items:
                loc = f":{e['line_start']}" if e.get("line_start") else ""
                lines.append(f"  • [{e['type']}] {e['name']}  ({e['file']}{loc})")
            return "\n".join(lines)
        # mode == "entity"
        hits = out.get("tests") or []
        if not hits:
            return f"No tests reference {out['entity']!r}."
        lines = [f"{out['test_count']} test reference(s) to {out['entity']!r}:"]
        for h in hits:
            lines.append(f"  • {h['file']} :: {h['from']}")
        return "\n".join(lines)

    if name == "graph_find_clones":
        out = g.find_clones(
            min_score=float(args.get("min_score", 0.85)),
            min_shape_overlap=float(args.get("min_shape_overlap", 0.3)),
            top_k_per_entity=int(args.get("top_k_per_entity", 5)),
            entity_types=args.get("entity_types") or None,
            limit=int(args.get("limit", 50)),
        )
        if out.get("warning"):
            return out["warning"]
        clusters = out["clusters"]
        if not clusters:
            return (
                "No clone clusters above the thresholds. "
                "Try lowering min_score or min_shape_overlap."
            )
        lines = [f"{len(clusters)} clone cluster(s) found:"]
        for i, cluster in enumerate(clusters, 1):
            lines.append(
                f"\n## Cluster {i} — {len(cluster['members'])} members "
                f"(avg score={cluster['avg_score']:.2f}, "
                f"shape overlap={cluster['avg_shape_overlap']:.2f})"
            )
            for m in cluster["members"]:
                loc = f":{m['line_start']}" if m.get("line_start") else ""
                lines.append(f"  • [{m['type']}] {m['name']}  ({m['file']}{loc})")
        lines.append(
            "\nNote: false positives possible for legitimate parallel "
            "implementations (test fixtures vs prod, polyfills, "
            "interface implementations)."
        )
        return "\n".join(lines)

    if name == "graph_visualize":
        # Default to the per-project storage dir (alongside graph.db,
        # retriever cache, etc.) so the project working tree stays clean.
        out = args.get("output_path") or str(services.config.project_dir / "graph.html")
        depth = max(1, min(int(args.get("module_depth", 2)), 4))
        info = g.visualize(output_path=Path(out), module_depth=depth)
        return (
            f"Graph visualization written to: {info['output_path']}\n"
            f"Modules: {info['modules']}, files: {info['files']}, "
            f"relations: {info['relations']}\n"
            f"Open the file in a browser. Double-click any node to drill in, "
            f"breadcrumb buttons up top to navigate back."
        )

    if name == "graph_clear":
        g.clear()
        return f"Knowledge graph cleared for {services.config.project_root.name!r}."

    if name == "search_code":
        top_k = max(1, min(int(args.get("top_k", 8)), 30))
        max_chunk = max(100, min(int(args.get("max_chunk_chars", 1500)), 5000))
        query = args["query"]
        hyde = bool(args.get("hyde"))
        if hyde and isinstance(services.llm_extractor, OpenAICompatExtractor):
            hyde_prompt = (
                "You are helping search a codebase. Given the user's question, "
                "write a SHORT hypothetical code snippet (function signature + "
                "3-7 lines of body) that, if it existed, would best answer the "
                "question. Return ONLY the code, no commentary, no explanation, "
                "no markdown fences.\n\n"
                f"Question: {query}"
            )
            try:
                hypo = await services.llm_extractor.complete(hyde_prompt, max_tokens=300)
            except Exception as e:
                logger.warning("HyDE expansion failed: %s", e)
                hypo = ""
            if hypo:
                query = f"{query}\n\n{hypo}"
        return services.retriever.search(
            query,
            top_k_initial=max(top_k * 4, 30),
            top_k_final=top_k,
            max_chunk_preview=max_chunk,
        )

    if name == "context_save":
        try:
            payload = services.contexts.save(
                name=str(args.get("name") or "").strip(),
                query=args.get("query"),
                files=args.get("files") or [],
                entities=args.get("entities") or [],
                notes=args.get("notes"),
            )
        except ValueError as e:
            return f"❌ {e}"
        return (
            f"✅ Saved context {payload['name']!r}: "
            f"{len(payload['files'])} file(s), {len(payload['entities'])} entity(ies)\n"
            f"   path: {payload['path']}"
        )

    if name == "context_load":
        data = services.contexts.load(str(args.get("name") or "").strip())
        if not data:
            return f"No context named {args.get('name')!r}."
        lines = [f"# Context: {data['name']}", f"_saved {data.get('saved_at', '?')}_"]
        if data.get("query"):
            lines.append(f"\n**Query:** {data['query']}")
        if data.get("files"):
            lines.append(f"\n## Files ({len(data['files'])})")
            for f in data["files"]:
                lines.append(f"- `{f}`")
        if data.get("entities"):
            lines.append(f"\n## Entities ({len(data['entities'])})")
            for e in data["entities"]:
                lines.append(f"- {e}")
        if data.get("notes"):
            lines.append(f"\n## Notes\n{data['notes']}")
        return "\n".join(lines)

    if name == "context_list":
        bundles = services.contexts.list()
        if not bundles:
            return "No saved contexts. Use context_save to create one."
        lines = [f"{len(bundles)} saved context(s):"]
        for b in bundles:
            lines.append(
                f"  • {b['name']} — {b['files']} file(s), {b['entities']} entity(ies)"
                f"  ({b['saved_at']})"
            )
            if b.get("query"):
                lines.append(f"      query: {b['query']}")
        return "\n".join(lines)

    if name == "context_delete":
        ok = services.contexts.delete(str(args.get("name") or "").strip())
        return f"✅ Deleted context {args.get('name')!r}." if ok else f"No context named {args.get('name')!r}."

    if name.startswith("memory_") and _memory_disabled():
        return (
            "memory_* tools are disabled on this server "
            "(MCP_RAG_NO_MEMORY=1). Unset the env var to re-enable."
        )

    if name == "memory_save":
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
        default_metrics.inc(f"tool.{name}.calls")
        try:
            with default_metrics.timer(f"tool.{name}"):
                text = await _dispatch(services, name, arguments or {})
        except Exception as e:
            default_metrics.inc(f"tool.{name}.errors")
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
    parser.add_argument(
        "--no-watch",
        action="store_true",
        default=(os.getenv("MCP_RAG_NO_WATCH") or "").strip() in {"1", "true", "yes"},
        help="Disable the filesystem watcher (auto-reindex on edits).",
    )
    subparsers = parser.add_subparsers(dest="command", required=False)
    subparsers.add_parser(
        "repl",
        help="Interactive REPL (no MCP) — explore the graph and search from a terminal.",
    )
    return parser.parse_args()


# ── Interactive REPL ────────────────────────────────────────────────────────
# Maps friendly REPL words to MCP tool names + a function that turns the rest
# of the line into a kwargs dict. Reuses _dispatch() so REPL output is exactly
# what an MCP client would see.

def _repl_kwargs_query(rest: str) -> dict:
    return {"query": rest.strip()} if rest.strip() else {}


def _repl_kwargs_name(rest: str) -> dict:
    return {"name": rest.strip()} if rest.strip() else {}


def _repl_kwargs_entity(rest: str) -> dict:
    return {"entity_name": rest.strip()} if rest.strip() else {}


def _repl_kwargs_filepath(rest: str) -> dict:
    return {"filepath": rest.strip()} if rest.strip() else {}


def _repl_kwargs_pattern(rest: str) -> dict:
    return {"pattern": rest.strip()} if rest.strip() else {}


def _repl_kwargs_empty(rest: str) -> dict:  # noqa: ARG001
    return {}


_REPL_COMMANDS: dict[str, tuple[str, callable, str]] = {
    "search":   ("search_code",          _repl_kwargs_query,    "search <query>            hybrid BM25+dense+rerank"),
    "find":     ("graph_search",         _repl_kwargs_query,    "find <name>               entity-name lookup"),
    "usages":   ("graph_find_usages",    _repl_kwargs_name,     "usages <name>             callers/usages of an entity"),
    "explain":  ("graph_explain",        _repl_kwargs_filepath, "explain <file>            file dossier"),
    "similar":  ("graph_find_similar",   _repl_kwargs_entity,   "similar <name>            FAISS-nearest entities"),
    "clones":   ("graph_find_clones",    _repl_kwargs_empty,    "clones                    detect clone clusters"),
    "coverage": ("graph_test_coverage",  _repl_kwargs_empty,    "coverage                  test-coverage summary"),
    "repomap":  ("graph_repo_map",       _repl_kwargs_empty,    "repomap                   PageRank-ranked project skeleton"),
    "struct":   ("graph_structural_search", _repl_kwargs_pattern, "struct <pattern>          ast-grep structural search"),
    "regex":    ("search_regex",         _repl_kwargs_pattern,  "regex <pattern>           FTS5-trigram + Python re search"),
    "contexts": ("context_list",         _repl_kwargs_empty,    "contexts                  list saved retrieval bundles"),
    "metrics":  ("metrics",              _repl_kwargs_empty,    "metrics                   in-process counters/latency/gauges"),
    "dead":     ("graph_dead_code",      _repl_kwargs_empty,    "dead                      possibly-dead defs"),
    "viz":      ("graph_visualize",      _repl_kwargs_empty,    "viz                       write graph.html, open in browser"),
    "build":    ("graph_build",          _repl_kwargs_empty,    "build                     index/refresh stale files"),
    "stats":    ("graph_stats",          _repl_kwargs_empty,    "stats                     graph counts"),
    "pending":  ("graph_pending_files",  _repl_kwargs_empty,    "pending                   files diverging from disk"),
}


def _repl_help() -> str:
    lines = ["mcp-rag REPL — commands:"]
    for word in sorted(_REPL_COMMANDS):
        _, _, doc = _REPL_COMMANDS[word]
        lines.append("  " + doc)
    lines.append("  help, ?                    this message")
    lines.append("  quit, exit, q              leave the REPL")
    return "\n".join(lines)


def _repl_run(services: Services) -> int:
    """Blocking interactive REPL — dispatches to the same async tools the MCP
    server exposes. Returns the process exit code (0 on clean exit)."""
    print(f"mcp-rag REPL — project: {services.config.project_root}")
    print("Type `help` for commands, `quit` to leave.\n")
    loop = asyncio.new_event_loop()
    try:
        while True:
            try:
                line = input("rag> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()  # newline after ^C / ^D
                return 0
            if not line:
                continue
            if line in {"quit", "exit", "q"}:
                return 0
            if line in {"help", "?"}:
                print(_repl_help())
                continue
            cmd, _, rest = line.partition(" ")
            entry = _REPL_COMMANDS.get(cmd.lower())
            if not entry:
                print(f"Unknown command: {cmd!r}. Type `help`.")
                continue
            tool_name, kwargs_fn, _doc = entry
            kwargs = kwargs_fn(rest)
            try:
                text = loop.run_until_complete(_dispatch(services, tool_name, kwargs))
            except Exception as e:
                text = f"Error in {tool_name}: {e}"
            print(text)
            print()  # trailing blank line so consecutive outputs don't merge
    finally:
        loop.close()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    storage = Path(args.storage) if args.storage else None
    config = Config(
        project_root=Path(args.project),
        storage_root=storage or (Path.home() / ".mcp-rag"),
    )
    embedder.configure(config.models_dir)

    # Mirror logs to a per-project file — Claude Code doesn't persist
    # the MCP subprocess's stderr, so without this the user has no way
    # to see what happened during background builds, watcher activity,
    # etc. The path is printed at INFO level so it's discoverable.
    log_file = os.getenv("MCP_RAG_LOG_FILE") or str(config.project_dir / "server.log")
    try:
        from logging.handlers import RotatingFileHandler
        fh = RotatingFileHandler(log_file, maxBytes=5 * 1024 * 1024, backupCount=2, encoding="utf-8")
        fh.setLevel(args.log_level)
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        logging.getLogger().addHandler(fh)
        logger.info("Log file: %s", log_file)
    except Exception as e:
        logger.warning("Could not attach file log handler at %s: %s", log_file, e)

    services = Services(config=config, llm_extractor=_resolve_llm_extractor())

    if getattr(args, "command", None) == "repl":
        # Quiet down the noisy startup logs in interactive mode — the user
        # doesn't want eager-build progress drowning out their prompt.
        logging.getLogger().setLevel(max(logging.WARNING, getattr(logging, args.log_level, logging.INFO)))
        raise SystemExit(_repl_run(services))

    server = build_server(services)

    async def _run() -> None:
        watcher: Optional["GraphWatcher"] = None
        if not args.no_watch:
            from .core.watcher import GraphWatcher
            try:
                watcher = GraphWatcher(services.graph)
                watcher.start()
            except Exception as e:
                logger.warning("Failed to start file watcher: %s — proceeding without auto-reindex", e)
                watcher = None

        # Eager background build at boot: if the graph is empty for a
        # non-trivial project, start indexing right away so the first
        # tool call doesn't pay the full cost. Skipped for tiny projects
        # (sync auto-build at first call is faster than the round trip).
        try:
            if services.graph.get_stats()["entities"] == 0:
                status = services.graph.get_build_status()
                if status["total_files"] > 50:
                    logger.info("Starting eager background build for %d files", status["total_files"])
                    services._build_task = asyncio.create_task(services.graph.build())
        except Exception as e:
            logger.warning("eager build start failed: %s", e)

        try:
            async with stdio_server() as (read, write):
                await server.run(read, write, server.create_initialization_options())
        finally:
            if watcher is not None:
                watcher.stop()
            if services._build_task is not None and not services._build_task.done():
                services._build_task.cancel()

    logger.info("mcp-rag serving project=%s storage=%s watch=%s",
                config.project_root, config.storage_root, not args.no_watch)
    asyncio.run(_run())


if __name__ == "__main__":
    main()
