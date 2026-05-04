# mcp-rag

MCP server that exposes a **code knowledge graph** + **hybrid semantic/lexical search** for any project, intended to be plugged into Claude Code (or any MCP-aware client).

## What it gives you

- `graph_*` — entities (classes, functions, methods, imports, components, …) and relations (calls, imports, inherits, uses, defines) extracted from the project, stored in SQLite, with a FAISS-backed semantic re-ranker.
- `search_code` — BM25 + dense-embedding hybrid search across all source files.
- `memory_*` — long-term per-project memory (facts, preferences, project info) with hybrid search.

Tree-sitter handles 10+ languages structurally; everything else falls back to lightweight regex extractors. An optional LLM extractor can be plugged in for unsupported languages.

## Install

```bash
pip install mcp-rag
```

## Use with Claude Code

Add to your MCP config:

```json
{
  "mcpServers": {
    "rag": {
      "command": "mcp-rag",
      "args": ["--project", "/path/to/your/project"]
    }
  }
}
```

Then in Claude Code: `graph_build` → `search_code "authentication flow"` → `graph_find_usages "User"` → etc.

## Storage

Per-project SQLite + FAISS files live under `~/.mcp-rag/<project_name>_<hash>/`.

## License

MIT
