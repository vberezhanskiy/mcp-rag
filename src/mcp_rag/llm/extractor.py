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

    Subclass and override `_endpoint_url()` / `_request_headers()` to plug in
    proxies that need a different URL shape or extra auth headers (e.g. when
    routing through a SaaS bearer + provider-selection header).
    """

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        timeout: float = 120.0,
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        self.base_url = (base_url or os.getenv("MCP_RAG_LLM_BASE_URL", "")).rstrip("/")
        self.api_key = api_key or os.getenv("MCP_RAG_LLM_API_KEY", "")
        self.model = model or os.getenv("MCP_RAG_LLM_MODEL", "deepseek-chat")
        self.timeout = timeout
        self.extra_headers = dict(extra_headers or {})

    def _endpoint_url(self) -> str:
        if not self.base_url:
            raise ValueError(
                "base_url is required (or MCP_RAG_LLM_BASE_URL env)"
            )
        return f"{self.base_url}/chat/completions"

    def _request_headers(self) -> dict[str, str]:
        if not self.api_key:
            raise ValueError(
                "api_key is required (or MCP_RAG_LLM_API_KEY env)"
            )
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        headers.update(self.extra_headers)
        return headers

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
                    self._endpoint_url(),
                    headers=self._request_headers(),
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
            return _parse_json_lenient(text)
        except Exception as e:
            logger.warning("LLM extract failed for %s: %s", rel_path, e)
            return {"entities": [], "relations": []}


def _strip_code_fence(text: str) -> str:
    if "```" not in text:
        return text
    m = re.search(r"```(?:json|json5)?\s*(.+?)```", text, re.DOTALL)
    return m.group(1).strip() if m else text


def _parse_json_lenient(text: str) -> dict:
    """Parse JSON, falling back to json5 (trailing commas, comments) if available."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        try:
            import json5  # type: ignore[import-not-found]
        except ImportError:
            return {"entities": [], "relations": []}
        try:
            return json5.loads(text)
        except Exception:
            return {"entities": [], "relations": []}
