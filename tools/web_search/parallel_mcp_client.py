"""Anonymous Parallel Search MCP context for the research worker."""

import asyncio
import hashlib
import json
import math
import os
from urllib.parse import urlsplit

from main import __version__

MCP_URL = "https://search.parallel.ai/mcp"
USER_AGENT = (
    f"Tesla-Agentic-System/{__version__} "
    "(https://github.com/longaxoloti/Agentic-System)"
)
MAX_CONTEXT_CHARS = 60_000


def research_search_provider() -> str:
    """Keep existing browser behavior unless the user explicitly selects a backend."""
    provider = os.getenv("RESEARCH_SEARCH_PROVIDER", "browser").strip().lower() or "browser"
    if provider not in {"browser", "parallel", "none"}:
        raise ValueError("RESEARCH_SEARCH_PROVIDER must be browser, parallel or none")
    return provider


def _results(response) -> list[dict]:
    if response.isError:
        raise RuntimeError("Parallel MCP tool returned an error")
    payload = response.structuredContent
    if payload is None:
        text = "\n".join(
            block.text for block in response.content if block.type == "text"
        )
        payload = json.loads(text)
    if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
        raise ValueError("Parallel MCP returned an invalid results payload")
    return [item for item in payload["results"] if isinstance(item, dict)]


def _valid_url(value) -> bool:
    if not isinstance(value, str) or any(c.isspace() for c in value):
        return False
    try:
        parsed = urlsplit(value)
        return (
            parsed.scheme in {"http", "https"}
            and bool(parsed.hostname)
            and not parsed.username
            and not parsed.password
        )
    except ValueError:
        return False


def _context(items: list[dict], *, label: str) -> str:
    parts = []
    for item in items:
        excerpts = item.get("excerpts", [])
        if not isinstance(excerpts, list):
            excerpts = []
        content = "\n".join(s for s in excerpts if isinstance(s, str))[:12_000]
        parts.append(
            f"=== PARALLEL {label} ===\nURL: {item['url']}\n"
            f"TITLE_HINT: {str(item.get('title') or '')[:500]}\nCONTENT:\n{content}"
        )
    return "\n\n".join(parts)


async def collect_parallel_context(
    query: str, *, session_id: str, max_sources: int = 5
) -> tuple[str, list[str]]:
    """Search, then fetch selected sources within one bounded MCP session.

    No credentials are read. The explicit selection is never replaced by another
    provider on failure. Cancellation propagates to the caller.
    """
    if not query.strip() or max_sources <= 0:
        return "", []
    timeout = float(os.getenv("PARALLEL_MCP_TIMEOUT_SECONDS", "60"))
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("PARALLEL_MCP_TIMEOUT_SECONDS must be finite and positive")

    import httpx
    from mcp import ClientSession
    from mcp.client.streamable_http import streamable_http_client

    # Avoid exposing local chat identifiers, while keeping related calls stable.
    conversation = hashlib.sha256(session_id.encode()).hexdigest()
    async with asyncio.timeout(timeout):
        async with httpx.AsyncClient(
            headers={"User-Agent": USER_AGENT}, timeout=timeout
        ) as http_client:
            async with streamable_http_client(MCP_URL, http_client=http_client) as (read, write, _):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    tools = {tool.name for tool in (await session.list_tools()).tools}
                    if not {"web_search", "web_fetch"} <= tools:
                        raise RuntimeError("Parallel MCP search/fetch tools are unavailable")
                    search = await session.call_tool(
                        "web_search",
                        {"objective": query, "search_queries": [query], "session_id": conversation},
                    )
                    selected = []
                    urls = []
                    for item in _results(search):
                        url = item.get("url")
                        if _valid_url(url) and url not in urls:
                            selected.append(item)
                            urls.append(url)
                        if len(urls) >= min(max_sources, 20):
                            break
                    if not urls:
                        return "No web results found by Parallel Search.", []
                    fetched = await session.call_tool(
                        "web_fetch",
                        {"urls": urls, "search_queries": [query], "session_id": conversation},
                    )
                    pages = [item for item in _results(fetched) if item.get("url") in urls]
                    context = _context(selected, label="SEARCH RESULT")
                    context += "\n\n" + _context(pages, label="FETCH RESULT")
                    return context[:MAX_CONTEXT_CHARS], urls
