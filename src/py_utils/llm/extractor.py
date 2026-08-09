"""LLM-based entity/relation extractor for source files.

Default is `NoOpExtractor` — returns nothing. Plug in
`OpenAICompatExtractor` (or any subclass) to enable real extraction: every
source file becomes one chat-completion call that returns a JSON graph.
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
    """Default — returns empty extraction. Plug in a real extractor to build graphs."""

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
        # Diagnostic string from the most recent extract() call. Empty
        # string when extraction succeeded normally; a one-line summary
        # of the failure mode otherwise (HTTP status, empty response,
        # JSON parse error, etc.). Read by graph_index_file / graph_build
        # wrappers so users can see *why* entities = 0 without digging
        # into worker stderr (which Electron silently swallows).
        self.last_diagnostic: str = ""

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
        self.last_diagnostic = ""
        # Input/output caps are env-tunable. The 8 KB default that used to
        # ship here silently dropped 80 % of any 40 KB router/service file,
        # so the LLM saw a fragment and returned either empty results or
        # JSON broken at the truncation seam. Modern long-context models
        # (DeepSeek V4-Flash 1M, Claude Sonnet 200K, GPT-4.1 1M, etc.)
        # handle 100 KB inputs without issues — that covers ~95 % of real
        # source files. ``max_tokens`` on the response side is generous
        # enough for even God-files (1000+ entities × ~25 tokens of JSON
        # each fits in 32 K); the provider only bills actual completion
        # tokens, so a high cap costs nothing when responses are short.
        # Override via env on smaller models where 100 KB blows the context.
        max_chars = _env_int("MCP_RAG_LLM_EXTRACT_MAX_CHARS", 100_000)
        max_tokens = _env_int("MCP_RAG_LLM_EXTRACT_MAX_TOKENS", 32_000)
        if len(code) > max_chars:
            truncated = code[:max_chars] + "\n... [truncated]"
            logger.info(
                "LLM extract: truncated %s from %d to %d chars (set MCP_RAG_LLM_EXTRACT_MAX_CHARS to raise)",
                rel_path, len(code), max_chars,
            )
        else:
            truncated = code
        prompt = _EXTRACT_PROMPT.format(filepath=rel_path, code=truncated)
        try:
            text = await self.complete(prompt, max_tokens=max_tokens, temperature=0)
        except Exception as e:
            self.last_diagnostic = f"complete() raised {type(e).__name__}: {e}"
            logger.warning("LLM extract failed for %s: %s", rel_path, e)
            return {"entities": [], "relations": []}
        if not text:
            self.last_diagnostic = (
                "complete() returned empty string — check worker logs for HTTP status. "
                "Common causes: 401 (bad bearer / _DP_USER_TOKEN not set), "
                "5xx (provider down), connection refused (backend not running)."
            )
            logger.warning("LLM extract for %s returned empty response", rel_path)
            return {"entities": [], "relations": []}
        result = _parse_json_lenient(_strip_code_fence(text), context=rel_path)
        # Empty result with a non-empty response usually means the model
        # produced prose / a code block / a different schema — log a preview
        # so users can adjust the prompt or model.
        if not result.get("entities") and not result.get("relations"):
            self.last_diagnostic = (
                f"LLM returned {len(text)} chars but parsed to empty graph. "
                f"Raw response head: {text[:200]!r}"
            )
            logger.warning(
                "LLM extract for %s parsed to empty graph; raw response preview: %r",
                rel_path, text[:300],
            )
        return result

    async def complete(self, prompt: str, max_tokens: int = 400, temperature: float = 0.0) -> str:
        """Generic chat-completion. Returns the assistant's text or '' on failure.

        Used by ``extract()`` and by host features that want a one-shot LLM
        call without its own client (e.g. HyDE query expansion in search_code).
        """
        try:
            import httpx
        except ImportError as e:
            raise RuntimeError("httpx not installed — install with `pip install py-utils[llm]`") from e

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(
                    self._endpoint_url(),
                    headers=self._request_headers(),
                    json={
                        "model": self.model,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                        "stream": False,
                    },
                )
            if resp.status_code != 200:
                logger.error("LLM complete HTTP %s: %s", resp.status_code, resp.text[:300])
                return ""
            return resp.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            logger.warning("LLM complete failed: %s", e)
            return ""


def _strip_code_fence(text: str) -> str:
    if "```" not in text:
        return text
    m = re.search(r"```(?:json|json5)?\s*(.+?)```", text, re.DOTALL)
    return m.group(1).strip() if m else text


def _parse_json_lenient(text: str, context: str = "") -> dict:
    """Parse JSON, falling back to json5 (trailing commas, comments) if available.

    On failure, logs a warning with the parser error and a preview of the
    offending text so callers can tell "model returned no graph" from
    "model returned a broken graph the parser dropped".
    """
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        first_err = e
    try:
        import json5  # type: ignore[import-not-found]
    except ImportError:
        logger.warning(
            "LLM extract JSON parse failed for %s (no json5 fallback installed): %s; preview: %r",
            context or "<unknown>", first_err, text[:300],
        )
        return {"entities": [], "relations": []}
    try:
        return json5.loads(text)
    except Exception as e:
        logger.warning(
            "LLM extract JSON parse failed for %s (json + json5 both rejected): json=%s; json5=%s; preview: %r",
            context or "<unknown>", first_err, e, text[:300],
        )
        return {"entities": [], "relations": []}


def _env_int(name: str, default: int) -> int:
    v = os.getenv(name, "").strip()
    if not v:
        return default
    try:
        n = int(v)
        return n if n > 0 else default
    except ValueError:
        logger.warning("Invalid %s=%r, falling back to %d", name, v, default)
        return default
