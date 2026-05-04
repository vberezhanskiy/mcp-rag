"""LLM-based fallback extractor for languages tree-sitter doesn't cover.

Default is `NoOpExtractor` — returns nothing, so the graph relies on
tree-sitter + regex parsers only. Plug in `OpenAICompatExtractor` to
enable LLM extraction for arbitrary file types.
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Protocol


logger = logging.getLogger(__name__)


_EXTRACT_PROMPT = """\
Analyze the following code file and extract the knowledge graph.
Return a JSON object with this exact structure:
{{
  "entities": [
    {{"name": "ClassName", "type": "class", "description": "brief description"}},
    {{"name": "function_name", "type": "function", "description": "brief description"}},
    {{"name": "module_name", "type": "import", "description": ""}}
  ],
  "relations": [
    {{"from": "ClassName", "relation": "defines", "to": "method_name"}},
    {{"from": "function_name", "relation": "calls", "to": "other_function"}},
    {{"from": "function_name", "relation": "imports", "to": "module_name"}},
    {{"from": "class_name", "relation": "inherits", "to": "parent_class"}}
  ]
}}

Relation types: defines, calls, imports, inherits, uses, instantiates
Only extract top-level meaningful entities. Skip private helpers (_name).
Return ONLY the JSON, no explanation.

File: {filepath}
```
{code}
```"""


class LLMExtractor(Protocol):
    async def extract(self, rel_path: str, code: str) -> dict:
        """Return {"entities": [...], "relations": [...]}."""
        ...


class NoOpExtractor:
    """Default — returns empty extraction. tree-sitter + regex carry the load."""

    async def extract(self, rel_path: str, code: str) -> dict:  # noqa: ARG002
        return {"entities": [], "relations": []}


class OpenAICompatExtractor:
    """Calls any OpenAI-compatible Chat Completions endpoint.

    Configure via env or constructor args:
      MCP_RAG_LLM_BASE_URL — e.g. https://api.deepseek.com/v1
      MCP_RAG_LLM_API_KEY  — bearer token
      MCP_RAG_LLM_MODEL    — model id (defaults to "deepseek-chat")
    """

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        timeout: float = 120.0,
    ) -> None:
        self.base_url = (base_url or os.getenv("MCP_RAG_LLM_BASE_URL", "")).rstrip("/")
        self.api_key = api_key or os.getenv("MCP_RAG_LLM_API_KEY", "")
        self.model = model or os.getenv("MCP_RAG_LLM_MODEL", "deepseek-chat")
        self.timeout = timeout
        if not self.base_url or not self.api_key:
            raise ValueError(
                "OpenAICompatExtractor requires base_url and api_key "
                "(or MCP_RAG_LLM_BASE_URL / MCP_RAG_LLM_API_KEY env)."
            )

    async def extract(self, rel_path: str, code: str) -> dict:
        try:
            import httpx
        except ImportError as e:
            raise RuntimeError("httpx not installed — install with `pip install mcp-rag[llm]`") from e

        truncated = code[:8000] + ("\n... [truncated]" if len(code) > 8000 else "")
        prompt = _EXTRACT_PROMPT.format(filepath=rel_path, code=truncated)

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self.model,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0,
                        "max_tokens": 2048,
                        "stream": False,
                    },
                )
            if resp.status_code != 200:
                logger.error("LLM extract HTTP %s for %s", resp.status_code, rel_path)
                return {"entities": [], "relations": []}
            text = resp.json()["choices"][0]["message"]["content"].strip()
            text = _strip_code_fence(text)
            return json.loads(text)
        except Exception as e:
            logger.warning("LLM extract failed for %s: %s", rel_path, e)
            return {"entities": [], "relations": []}


def _strip_code_fence(text: str) -> str:
    if "```" not in text:
        return text
    m = re.search(r"```(?:json|json5)?\s*(.+?)```", text, re.DOTALL)
    return m.group(1).strip() if m else text
