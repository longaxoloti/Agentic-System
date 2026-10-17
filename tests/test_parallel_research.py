"""Research dispatcher tests with real MCP framing and a deterministic HTTP peer.

Like the other research tests, these use the installation's main/config.py.
Model synthesis is scripted; no LLM, browser, database or remote service is used.
"""

import asyncio
import hashlib
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
import mcp.client.streamable_http  # Load SDK types before replacing the HTTP client.
from langchain_core.messages import HumanMessage

from tools.web_search import parallel_mcp_client as parallel


@pytest.fixture
def node(monkeypatch):
    from graph.nodes import research

    monkeypatch.setattr(research, "unload_model", AsyncMock())
    monkeypatch.setattr(research, "load_context", lambda _: {})
    monkeypatch.setattr(research, "CAMOUFOX_ENABLED", True)
    monkeypatch.setattr(research, "RESEARCH_MAX_DISCOVERED_SOURCES", 2)
    monkeypatch.setattr(research, "get_workspace_priming_context", lambda **_: "Cite evidence.")
    llm = SimpleNamespace(ainvoke=AsyncMock(return_value=SimpleNamespace(content="Scripted synthesis")))
    monkeypatch.setattr(research, "get_llm", lambda **_: llm)
    monkeypatch.setattr(research, "_persist_research_sources", lambda _: 0)
    monkeypatch.setenv("RESEARCH_SEARCH_PROVIDER", "parallel")
    monkeypatch.setenv("PARALLEL_MCP_TIMEOUT_SECONDS", "5")
    # Fail immediately if the selected path accidentally starts a browser.
    monkeypatch.setattr(research, "ChromeCDPClient", lambda: pytest.fail("unexpected browser"))
    monkeypatch.setattr(research, "CamoFoxMCPClient", lambda **_: pytest.fail("unexpected browser"))
    return research, llm


def state(**overrides):
    return {
        "messages": [HumanMessage(content="Research Python release notes")],
        "session_id": "test-research-conversation",
        "search_query": "Python latest release notes",
        "topic": "Python", "tasks": ["Summarize releases"],
        "persist_research_to_db": False,
        **overrides,
    }


@pytest.fixture
def peer(monkeypatch):
    requests = []
    clients = []
    options = {"error": None, "empty": False, "text": False, "hang": False, "missing_tool": False}
    entered = asyncio.Event()
    cancelled = asyncio.Event()
    client_type = httpx.AsyncClient

    async def handle(request):
        assert str(request.url) == parallel.MCP_URL
        assert request.headers["user-agent"] == parallel.USER_AGENT
        assert "authorization" not in request.headers
        message = json.loads(request.content)
        requests.append(message)
        method = message["method"]
        if "id" not in message:
            return httpx.Response(202)
        if method == "initialize":
            result = {"protocolVersion": "2025-03-26", "capabilities": {"tools": {}},
                      "serverInfo": {"name": "fixture", "version": "1"}}
        elif method == "tools/list":
            result = {"tools": [{"name": name, "inputSchema": {"type": "object"}}
                                for name in (["web_search"] if options["missing_tool"]
                                             else ["web_search", "web_fetch"])]}
        else:
            assert method == "tools/call"
            tool = message["params"]["name"]
            if options["hang"]:
                entered.set()
                try:
                    await asyncio.Event().wait()
                finally:
                    cancelled.set()
            if options["error"] == "http" or (options["error"] == "fetch" and tool == "web_fetch"):
                return httpx.Response(403, text="Forbidden")
            payload = {"results": []}
            if not options["empty"]:
                payload["results"] = [
                    {"url": "https://python.org/releases/", "title": "Python releases",
                     "excerpts": ["Release details" if tool == "web_search" else "Fetched release evidence"]},
                    {"url": "javascript:alert(1)", "excerpts": ["Invalid"]},
                    {"url": "https://user:password@host/a", "excerpts": ["Invalid"]},
                    {"url": "https://python.org/releases/", "excerpts": ["Duplicate"]},
                    {"url": "https://docs.python.org/3/whatsnew/", "excerpts": ["What's new"]},
                ]
                if tool == "web_search":
                    payload["results"] += [
                        payload["results"][0],
                        {"url": "javascript:alert(1)", "excerpts": ["Invalid"]},
                    ]
            result = {"content": [{"type": "text", "text": json.dumps(payload)}]}
            if not options["text"]:
                result["structuredContent"] = payload
            if options["error"] == "tool":
                result = {"isError": True, "content": [{"type": "text", "text": "Unavailable"}]}
            if options["error"] == "malformed":
                result = {"content": [{"type": "text", "text": "not JSON"}]}
        return httpx.Response(200, json={"jsonrpc": "2.0", "id": message["id"], "result": result})

    def factory(**kwargs):
        client = client_type(transport=httpx.MockTransport(handle), **kwargs)
        clients.append(client)
        return client

    monkeypatch.setattr(httpx, "AsyncClient", factory)
    # Ambient credentials must not silently turn the anonymous path into a paid one.
    monkeypatch.setenv("PARALLEL_API_KEY", "unused-sentinel")
    return SimpleNamespace(requests=requests, clients=clients, options=options,
                           entered=entered, cancelled=cancelled)


@pytest.mark.asyncio
@pytest.mark.parametrize("text", [False, True])
async def test_dispatch_search_fetch_and_synthesis(node, peer, text):
    research, llm = node
    peer.options["text"] = text
    result = await research.research_node(state())
    calls = [m["params"] for m in peer.requests if m["method"] == "tools/call"]
    assert [c["name"] for c in calls] == ["web_search", "web_fetch"]
    assert calls[0]["arguments"]["objective"] == "Python latest release notes"
    assert calls[0]["arguments"]["search_queries"] == ["Python latest release notes"]
    expected_id = hashlib.sha256(b"test-research-conversation").hexdigest()
    assert all(c["arguments"]["session_id"] == expected_id for c in calls)
    assert calls[1]["arguments"]["urls"] == [
        "https://python.org/releases/", "https://docs.python.org/3/whatsnew/"
    ]
    prompt = llm.ainvoke.call_args.args[0][1].content
    assert "Release details" in prompt and "Fetched release evidence" in prompt
    task = result["task_results"][0]
    assert task["result"] == "Scripted synthesis"
    assert task["sources"] == calls[1]["arguments"]["urls"]
    assert all(client.is_closed for client in peer.clients)


@pytest.mark.asyncio
@pytest.mark.parametrize("setting", ["disabled", "none"])
async def test_disabled_search_never_contacts_mcp(node, peer, monkeypatch, setting):
    research, llm = node
    if setting == "disabled":
        monkeypatch.setattr(research, "CAMOUFOX_ENABLED", False)
    else:
        monkeypatch.setenv("RESEARCH_SEARCH_PROVIDER", "none")
    result = await research.research_node(state())
    assert not peer.requests
    assert not result["task_results"][0]["sources"]
    assert "Research unavailable" in llm.ainvoke.call_args.args[0][1].content


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", [None, "", "browser"])
async def test_default_and_saved_browser_choice(node, peer, monkeypatch, provider):
    research, _ = node
    if provider is None:
        monkeypatch.delenv("RESEARCH_SEARCH_PROVIDER")
    else:
        monkeypatch.setenv("RESEARCH_SEARCH_PROVIDER", provider)
    browser = SimpleNamespace(ping=AsyncMock(return_value=False), close=AsyncMock())
    monkeypatch.setattr(research, "CHROME_CDP_ENABLED", False)
    monkeypatch.setattr(research, "CamoFoxMCPClient", lambda **_: browser)
    result = await research.research_node(state())
    browser.ping.assert_awaited_once()
    browser.close.assert_awaited_once()
    assert not peer.requests
    assert not result["task_results"][0]["sources"]


@pytest.mark.asyncio
async def test_daily_sources_preserve_precedence(node, peer, monkeypatch):
    research, llm = node
    collect = AsyncMock(return_value=("Fixed source evidence", ["https://example.com/news"]))
    monkeypatch.setattr(research, "_collect_context_from_fixed_urls", collect)
    await research.research_node(state(report_mode="daily_report", fixed_source_urls=["https://example.com/news"]))
    assert not peer.requests
    collect.assert_awaited_once_with(["https://example.com/news"])
    assert "Fixed source evidence" in llm.ainvoke.call_args.args[0][1].content


@pytest.mark.asyncio
@pytest.mark.parametrize("error", ["tool", "http", "malformed", "fetch"])
async def test_errors_reach_synthesis_without_fallback(node, peer, error):
    research, llm = node
    peer.options["error"] = error
    result = await research.research_node(state())
    assert not result["task_results"][0]["sources"]
    assert "No web evidence was collected" in llm.ainvoke.call_args.args[0][1].content
    assert all(client.is_closed for client in peer.clients)


@pytest.mark.asyncio
async def test_missing_tools_fail_before_search(node, peer):
    research, llm = node
    peer.options["missing_tool"] = True
    result = await research.research_node(state())
    assert not result["task_results"][0]["sources"]
    assert "Research unavailable" in llm.ainvoke.call_args.args[0][1].content
    assert not any(m["method"] == "tools/call" for m in peer.requests)


@pytest.mark.asyncio
@pytest.mark.parametrize("timeout", ["0", "-1", "nan", "inf", "invalid"])
async def test_invalid_timeout_never_opens_transport(node, peer, monkeypatch, timeout):
    research, llm = node
    monkeypatch.setenv("PARALLEL_MCP_TIMEOUT_SECONDS", timeout)
    result = await research.research_node(state())
    assert not peer.requests
    assert not result["task_results"][0]["sources"]
    assert "Research unavailable" in llm.ainvoke.call_args.args[0][1].content


@pytest.mark.asyncio
async def test_timeout_closes_transport(node, peer, monkeypatch):
    research, llm = node
    peer.options["hang"] = True
    monkeypatch.setenv("PARALLEL_MCP_TIMEOUT_SECONDS", "0.1")
    result = await research.research_node(state())
    assert not result["task_results"][0]["sources"]
    assert "Research unavailable" in llm.ainvoke.call_args.args[0][1].content
    assert peer.cancelled.is_set()
    assert all(client.is_closed for client in peer.clients)


@pytest.mark.asyncio
async def test_cancellation_propagates_and_closes_transport(node, peer):
    research, llm = node
    peer.options["hang"] = True
    task = asyncio.create_task(research.research_node(state()))
    await asyncio.wait_for(peer.entered.wait(), 2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    llm.ainvoke.assert_not_awaited()
    assert peer.cancelled.is_set()
    assert all(client.is_closed for client in peer.clients)


@pytest.mark.asyncio
async def test_empty_results_skip_fetch(node, peer):
    research, llm = node
    peer.options["empty"] = True
    result = await research.research_node(state())
    assert not result["task_results"][0]["sources"]
    assert "No web results" in llm.ainvoke.call_args.args[0][1].content
    assert len([m for m in peer.requests if m["method"] == "tools/call"]) == 1


@pytest.mark.parametrize("url", ["file:///tmp/test", "https://user:password@host/a", "https://[invalid", "https://a b.com/", None])
def test_reject_invalid_source_urls(url):
    assert not parallel._valid_url(url)


def test_invalid_provider_is_not_silently_replaced(monkeypatch):
    monkeypatch.setenv("RESEARCH_SEARCH_PROVIDER", "typo")
    with pytest.raises(ValueError, match="RESEARCH_SEARCH_PROVIDER"):
        parallel.research_search_provider()
