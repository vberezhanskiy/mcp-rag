**English** · [Русский](README.ru.md)

# mcp-rag

A self-hosted MCP server that turns any source tree into a **code knowledge graph** plus **hybrid semantic + lexical search** + an **interactive HTML visualization**. Designed for Claude Code, but speaks plain MCP so any MCP-aware client (Cursor, Continue, custom agents) can plug in.

The goal: give an AI agent the same kind of cross-file structural awareness an IDE has — "who calls this function", "what defines this component", "what's similar to this helper" — without re-reading the project on every prompt.

---

## Why this exists

Out of the box, Claude Code (and most coding agents) read files on demand via `Read`/`Grep`/`Glob`. That covers ~90% of work, but breaks down on:

- **Concept questions** — "where does the auth flow live?" — Grep needs exact tokens; Read costs context.
- **Refactor scope** — "if I rename `Button`, what breaks?" — needs structural usage tracking, not text search.
- **Code discovery** — "is there already a helper that does X?" — needs semantic matching, not name patterns.
- **Multi-language projects** — Angular + NestJS + Python + FastAPI — each has its own conventions, but cross-cutting questions ("which frontend calls this endpoint?") need a unified index.

mcp-rag pre-computes a **per-project graph** (tree-sitter for 10+ languages, regex fallback for the rest) and a **chunk index** (BM25 + dense embeddings via `bge-m3`, cross-encoder rerank). The result: an MCP tool surface where one call replaces 5–10 file reads.

---

## Quick start

### Install

```bash
git clone <your-fork>
cd mcp-rag
python -m venv .venv
.venv/Scripts/activate           # or `source .venv/bin/activate`
pip install -e ".[llm,dev]"
```

### Plug into Claude Code

```bash
claude mcp add rag --scope user -- /abs/path/to/.venv/bin/mcp-rag
```

Or edit `~/.claude.json` directly:

```json
{
  "mcpServers": {
    "rag": {
      "type": "stdio",
      "command": "/abs/path/to/.venv/bin/mcp-rag"
    }
  }
}
```

The server picks up the project from Claude Code's `cwd` at startup. Restart Claude Code from the project root and you're set. First `graph_*` call will auto-build the graph.

### First flow

```
graph_build               # or just call any graph_* tool — it auto-builds when empty
graph_explain src/Foo.tsx # one-call file dossier
search_code "auth flow"   # concept search across code text
graph_find_usages User    # exact-name refactor scope
graph_visualize           # writes HTML, open in browser
```

---

## Configuration

Everything is environment-driven. Set in your MCP config under `"env"` or in your shell.

### Project & storage

| Variable | Default | What it does |
|---|---|---|
| `MCP_RAG_PROJECT` | cwd | Project root to index. Overridden by `--project` CLI arg. |
| `MCP_RAG_STORAGE` | `~/.mcp-rag` | Where graph / faiss / cache / models live. |

### Embedder

| Variable | Default | What it does |
|---|---|---|
| `MCP_RAG_EMBED_MODEL` | `BAAI/bge-m3` | Sentence-transformers model id. Anything compatible works. |
| `MCP_RAG_RERANKER_MODEL` | `BAAI/bge-reranker-base` | Cross-encoder for `search_code` rerank. |
| `MCP_RAG_RERANK` | `1` | Set `0` to disable rerank (faster but noisier). |
| `MCP_RAG_DEVICE` | auto | `cuda`/`mps`/`cpu`. Auto-detects; force-override here. |

### LLM extractor (optional)

Tree-sitter + regex cover the common case. Set these if you want a fallback for unusual languages:

| Variable | Required | What it does |
|---|---|---|
| `MCP_RAG_LLM_BASE_URL` | yes | OpenAI-compatible Chat Completions endpoint. |
| `MCP_RAG_LLM_API_KEY` | yes | Bearer token. |
| `MCP_RAG_LLM_MODEL` | no, default `deepseek-chat` | Model id. |

### Behavior toggles

| Variable | Default | What it does |
|---|---|---|
| `MCP_RAG_NO_WATCH` | unset | Set `1` to disable the file-system watcher (no auto-reindex on saves). |
| `MCP_RAG_NO_MEMORY` | unset | Set `1` to hide `memory_*` tools — useful if your host already has its own memory store (Claude Code's `~/.claude/memory/`, etc.). |
| `MCP_RAG_LOG` | `INFO` | `DEBUG`/`INFO`/`WARNING`/`ERROR`. |

### Example: full Claude Code config

```json
{
  "mcpServers": {
    "rag": {
      "type": "stdio",
      "command": "D:\\Projects\\mcp-rag\\.venv\\Scripts\\mcp-rag.exe",
      "env": {
        "MCP_RAG_NO_MEMORY": "1",
        "MCP_RAG_RERANK": "1",
        "MCP_RAG_DEVICE": "cuda"
      }
    }
  }
}
```

### Example: with the LLM fallback

Claude Code passes `env` values to the subprocess literally — there's no
shell-style `${VAR}` expansion — so the API key has to be a real string,
not a reference. Either drop it inline (and keep the file out of git)
or wrap the launcher in a shell script that loads from `.env`:

```json
{
  "mcpServers": {
    "rag": {
      "type": "stdio",
      "command": "/usr/local/bin/mcp-rag",
      "env": {
        "MCP_RAG_LLM_BASE_URL": "https://api.deepseek.com/v1",
        "MCP_RAG_LLM_API_KEY": "sk-replace-with-your-key",
        "MCP_RAG_LLM_MODEL": "deepseek-chat"
      }
    }
  }
}
```

Wrapper-script alternative on Windows (`start-mcp-rag.bat`):

```bat
@echo off
set MCP_RAG_LLM_BASE_URL=https://api.deepseek.com/v1
set MCP_RAG_LLM_API_KEY=sk-...
set MCP_RAG_LLM_MODEL=deepseek-chat
"D:\Projects\mcp-rag\.venv\Scripts\mcp-rag.exe" %*
```

```json
{
  "mcpServers": {
    "rag": {
      "type": "stdio",
      "command": "cmd.exe",
      "args": ["/c", "D:\\path\\start-mcp-rag.bat"]
    }
  }
}
```

---

## Tool reference

### Daily-use (90% of value)

| Tool | What it does |
|---|---|
| `search_code` | Hybrid BM25 + dense + cross-encoder + IDF-weighted literal bonus over project source. RU/EN. Concept queries, pattern hunting, when you don't know exact names. |
| `graph_explain` | One-call file dossier: declared entities + dependency map + external callers. Replaces three other tools. |
| `graph_find_usages` | Every place an exact-named entity is referenced (calls, JSX usages, instantiations, inheritance). Use before rename/refactor. |
| `graph_pending_files` | Files where graph and disk diverge (unindexed / stale / missing). Sanity check after edits. |
| `graph_stats` | File / entity / relation counts plus per-type breakdown. |

### Situational

| Tool | What it does |
|---|---|
| `graph_find_similar` | Semantically nearest entities to an anchor — dedup detection. The embed text combines `name + outgoing relations + snippet`, so structural fingerprints cluster (all antd-component wrappers end up close). |
| `graph_dead_code` | Functions/classes/components no relation points at. Pass `exclude_paths` globs to skip scaffolding. |
| `graph_get_subgraph` | BFS the relation graph around an entity. Capped per node — common names like `Layout`/`Header` produce mostly truncated nodes; reliable on uniquely-named entities. |
| `graph_visualize` | Renders an interactive HTML page with three drill-down levels (modules → files → entities). Self-contained, vis-network from CDN. |
| `graph_build` / `graph_clear` / `graph_index_file` | Lifecycle. `graph_build` is auto-called on first data-needing tool when the graph is empty. |

### Niche

| Tool | What it does |
|---|---|
| `graph_search` | Find entities by name (substring match in the entity table) with type filter. Niche: "list all classes with `Button` in name". For concept search use `search_code`; for exact-name refactor use `graph_find_usages`. |
| `graph_get_file_deps` / `graph_file_structure` | Subsets of `graph_explain`. Prefer `graph_explain` unless you only need one section. |

### Memory (toggle with `MCP_RAG_NO_MEMORY=1`)

| Tool | What it does |
|---|---|
| `memory_add` / `memory_search` / `memory_list` / `memory_delete` / `memory_clear` | Per-project memory store, indexed for hybrid search. Useful for hosts without their own memory layer. |

---

## Resources (`@`-attachable in Claude Code)

| URI | What it returns |
|---|---|
| `rag://overview` | Project digest: top-level structure, entity-type breakdown, most-referenced symbols. |
| `rag://file/{path}` | Compact file dossier (same shape as `graph_explain`). |
| `rag://search/{query}` | Bundled `search_code` result block. |
| `rag://explain/{entity}` | Entity card with location, snippet, callers. |

---

## Architecture

```
project source
      │
      ▼
┌─ extractors ────────────────────────────────────────┐
│  tree-sitter (10+ langs)                            │
│  regex (HTML, CSS, configs, JSX usages)             │
│  optional LLM fallback (OpenAI-compat)              │
└──────────────────┬──────────────────────────────────┘
                   ▼
   ┌────────────────────────────────┐
   │ entities + relations → SQLite  │
   │ chunks → diskcache             │
   └────┬──────────────┬────────────┘
        ▼              ▼
   ┌─────────┐   ┌────────────────┐
   │ graph_* │   │ search_code    │
   │ tools   │   │  bm25 +        │
   │         │   │  bge-m3 dense +│
   │ + FAISS │   │  bge-reranker +│
   │  index  │   │  literal bonus │
   └─────────┘   └────────────────┘
```

A file-system watcher (watchdog) keeps the graph fresh between runs of `graph_build`. FAISS rebuild is deferred — re-encoding ~40k entities on every save would be too expensive — so similarity tools (`graph_find_similar`) trigger a rebuild on demand if the dirty flag is set.

---

## Hardware & performance

Tested on **RTX 5060 Ti 16 GB**, Windows 11, Python 3.13, on a project of 1931 files / 42k entities / 50k relations.

- Graph build with bge-m3 + bf16: **~140s** end-to-end.
- FAISS rebuild only: **~80s** for 42k entities.
- `search_code` query (post-warmup): **~100–200 ms** including cross-encoder rerank.
- File-watcher reindex: **~50–250 ms** per file (debounce 1s).

For CPU-only setups, set `MCP_RAG_DEVICE=cpu` — works, just slower. For Apple Silicon, `MCP_RAG_DEVICE=mps` (auto-detected when available).

### Model size vs VRAM

| Model | Params | Approx VRAM (bf16) | Languages | Notes |
|---|---|---|---|---|
| `sentence-transformers/all-MiniLM-L6-v2` | 23M | ~80 MB | EN only | The lightweight escape hatch — fits anywhere, runs on CPU at meaningful speed, but no RU/multilingual support. Pick this if your project is English-only and you want minimal footprint. |
| `BAAI/bge-m3` (default) | 568M | ~1.2 GB | 100+ | Encoder-only, safe on 16GB GPUs, 8k context. Good RU/EN balance, no prefix gymnastics. |
| `Qwen/Qwen3-Embedding-0.6B` | 600M | unstable on 16GB | 100+ | Decoder-only LLM with KV cache; better MTEB Multilingual on paper but OOMs in practice on consumer GPUs once a real corpus is fed in. Documented as opt-in, not a recommendation. |

### Preset: lightweight (EN-only, CPU-friendly)

Pair MiniLM bi-encoder for retrieval with the MS MARCO cross-encoder
for rerank — both ~80 MB, both run with meaningful throughput on CPU,
both English-focused. Drop-in config:

```json
"env": {
  "MCP_RAG_EMBED_MODEL":    "sentence-transformers/all-MiniLM-L6-v2",
  "MCP_RAG_RERANKER_MODEL": "cross-encoder/ms-marco-MiniLM-L-6-v2"
}
```

Pick this for English-only repos when you don't have a GPU or want
the smallest possible footprint. Multilingual support is gone, but
search quality on English code is still strong because the
cross-encoder rerank lifts the top-K out of MiniLM's noisier output.

---

## Storage layout

```
~/.mcp-rag/
├── models/
│   └── BAAI_bge-m3/                           # downloaded once
└── projects/
    └── <project_name>_<short-hash>/
        ├── graph/
        │   └── graph.db                       # SQLite: entities, relations, file_meta
        ├── retriever/
        │   └── cache.db                       # diskcache: bm25 + faiss + chunks
        ├── memory/
        │   ├── memories.json
        │   └── embeddings_cache/
        └── graph.html                         # graph_visualize output
```

Project root is hashed so two checkouts of the same repo at different paths get separate stores.

---

## Languages

Tree-sitter (structural extraction): Python, JS/TS/TSX/JSX, Vue, Svelte, Astro, Go, Rust, Java, C#, C/C++, PHP, Ruby.

Regex extractors: HTML, CSS/SCSS/LESS/Sass, JSON, YAML, TOML, Jinja templates, Godot/GDScript.

Anything else routes to the LLM fallback if configured, otherwise gets a file-level entity only.

---

## CLI

```
mcp-rag --help
  --project PATH        project root (default: cwd)
  --storage PATH        storage root (default: ~/.mcp-rag)
  --log-level LEVEL     DEBUG | INFO | WARNING | ERROR
  --no-watch            disable filesystem watcher
```

The CLI flags are the same names as the env vars without the `MCP_RAG_` prefix.

---

## Development

```bash
pip install -e ".[llm,dev]"          # ruff + pytest extras
ruff check src/
pytest                                 # tests are still TBD; see TODO.md
```

Smoke check the MCP handshake without going through Claude Code:

```bash
echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"smoke","version":"0.1"}}}' | mcp-rag --project /path/to/repo
```

---

## License

MIT.
