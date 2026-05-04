# mcp-rag — backlog

Ranked by value × effort. Pick from the top.

## Picker UX
- **Top-N files as concrete resources** — `resources/list` returns the 30 most "important" project files (top defined-entity count or top external-caller count) so they show up in Claude Code's `@`-picker without needing path autocomplete inside templates. Option B from the resources discussion.

## Config & DX
- **`.mcp-rag.toml` in project root** — extra ignore dirs (`generated/`, `vendor/`), `max_file_size_mb`, optional LLM block. Auto-discovered from `cwd`. Currently only env vars (`MCP_RAG_DEVICE`, `MCP_RAG_LLM_*`).
- **Project context tool / resource** — `project_overview` already exists as a resource; promote/extend with detected stack (`package.json`, `pyproject.toml`, etc.) and a 1-paragraph human-readable summary so a single fetch primes a fresh session.

## Live mode
- **File-watcher → auto reindex** — `watchdog`-driven background task. Saves the user from manual `graph_index_file` after edits. Needs debounce + concurrent-build safety.

## Quality
- **pytest suite** — smoke build, path normalization, ignore-filter (with the substring vs exact-match regression), JSX extraction. Add GitHub Actions to run it on Python 3.11–3.13.

## Visualisation
- **Mermaid output** for `graph_explain` / `graph_get_subgraph` — Claude Code renders Mermaid; a `mermaid graph LR` block per file beats a flat list for understanding deps at a glance.

## Refactor aids
- **Tests-coverage map** — reverse call-graph from test files → prod entities. Answers "if I change X, which tests cover it?" Useful in typed projects with explicit imports.

## Skipped on purpose
- ~~`graph_export` (DOT/JSON)~~ — Claude Code has no UI for it.
- ~~web retriever / summary system / rules system~~ — agent-specific features from codeAgent, overkill for an MCP tool.
- ~~`memory_*` tools~~ — user has their own `~/.claude/memory/` system; cross-tool sharing is the only remaining justification.
