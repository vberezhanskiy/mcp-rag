"""Text formatters for tool output. Stateless helpers."""

from __future__ import annotations

from typing import Iterable, Optional


def format_section(title: str, lines: list[str], *, emoji: str = "") -> str:
    prefix = f"{emoji} " if emoji else ""
    header = f"{prefix}{title}:"
    body = [line for line in lines if line]
    return "\n".join([header, *body]) if body else header


def format_code_result(path: str, line_start: int, line_end: int, score: float, preview: str) -> str:
    return (
        f"source=code\n"
        f"file={path}\n"
        f"lines={line_start}-{line_end}\n"
        f"score={score:.4f}\n"
        f"```text\n{preview}\n```"
    )


def format_scored_memory(memory, score: float) -> str:
    return f"  [{memory.memory_type}] score={score:.2f} {memory.content[:120]}"


def format_memory_listing(memories: Iterable, *, title: str = "Memory List", emoji: str = "", limit: Optional[int] = None) -> str:
    items = list(memories)
    if limit is not None:
        items = items[:limit]
    if not items:
        return "Memory is empty."

    lines = [f"Total memories: {len(items)}."]
    by_type: dict[str, list] = {}
    for memory in items:
        by_type.setdefault(memory.memory_type, []).append(memory)

    for memory_type in sorted(by_type):
        typed_items = by_type[memory_type]
        lines.append(f"  [{memory_type}] ({len(typed_items)}):")
        for memory in sorted(typed_items, key=lambda item: item.timestamp, reverse=True):
            tags = " ".join(memory.tags)
            lines.append(f"    - {memory.content[:100]}" + (f" {tags}" if tags else ""))

    return format_section(title, lines, emoji=emoji)


def format_graph_relations(label: str, items: list[str], *, indent: str = "    ") -> list[str]:
    if not items:
        return []
    return [f"{indent}{label}: {', '.join(items)}"]
