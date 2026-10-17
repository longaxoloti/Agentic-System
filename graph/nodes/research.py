import logging
import asyncio
import random
from collections import deque
from datetime import datetime
from urllib.parse import urlparse, urljoin
import re
import os
from pathlib import Path
from typing import Optional
from tools.web_search.crawl4ai_client import crawl_url_to_markdown
from langchain_core.messages import HumanMessage, SystemMessage
from graph.state import AgentState
from graph.llm_router import get_llm
from main.config import (
    CAMOUFOX_ENABLED,
    CHROME_CDP_ENABLED,
    CAMOFOX_BEHAVIOR_MAX_DELAY_SECONDS,
    CAMOFOX_BEHAVIOR_MIN_DELAY_SECONDS,
    CAMOFOX_CHALLENGE_MAX_RETRIES,
    CAMOFOX_REUSE_TAB,
    GOOGLE_MIN_INTERVAL_SECONDS,
    GOOGLE_COOLDOWN_ON_CHALLENGE_SECONDS,
    GOOGLE_SEARCH_CACHE_TTL_SECONDS,
    GOOGLE_SEARCH_LOCK_TIMEOUT_SECONDS,
    RESEARCH_MAX_SEARCH_QUERIES,
    RESEARCH_MAX_DISCOVERED_SOURCES,
    RESEARCH_SOURCE_ALLOWLIST,
    DAILY_REPORT_FIXED_SOURCE_URLS,
    OLLAMA_ORCHESTRATOR_MODEL,
    OLLAMA_RESEARCH_MODEL,
)
from tools.web_search.camofox_mcp_client import CamoFoxMCPClient
from tools.web_search.chrome_cdp_client import ChromeCDPClient
from tools.web_search.google_guard import GoogleGuard, GoogleGuardConfig
from tools.web_search.research_result_cache import ResearchResultCache, CacheConfig
from tools.web_search.file_lock import file_lock
from tools.web_search.parallel_mcp_client import collect_parallel_context, research_search_provider
from tools.agent_org.workspace_priming import get_workspace_priming_context
from tools.agent_org.ollama_manager import unload_model, load_context

logger = logging.getLogger(__name__)
_knowledge_service = None


def _get_knowledge_service():
    global _knowledge_service
    if _knowledge_service is None:
        from storage.knowledge_service import KnowledgeService
        _knowledge_service = KnowledgeService()
    return _knowledge_service


def _persist_research_sources(urls: list[str]) -> int:
    """Persist discovered research URLs into unified knowledge."""
    if not urls:
        return 0

    try:
        knowledge_service = _get_knowledge_service()
    except Exception as exc:
        logger.debug("Knowledge service unavailable; skip URL persistence: %s", exc)
        return 0

    try:
        knowledge_service.save(
            chat_id="research",
            content="\n".join(urls),
            category="research_sources",
            title="Research sources",
            metadata={"urls": urls},
        )
        return len(urls)
    except Exception as exc:
        logger.debug("Failed to persist research URLs to unified knowledge: %s", exc)
        return 0


def _extract_crawled_articles(context: str) -> list[dict]:
    if not context:
        return []

    pattern = re.compile(
        r"=== CRAWL4AI ARTICLE ===\n"
        r"URL: (?P<url>.+?)\n"
        r"DOMAIN: (?P<domain>.*?)\n"
        r"TITLE_HINT: (?P<title>.*?)\n"
        r"CONTENT:\n(?P<content>.*?)(?=\n=== |\Z)",
        re.DOTALL,
    )
    items: list[dict] = []
    for match in pattern.finditer(context):
        url = (match.group("url") or "").strip()
        content = (match.group("content") or "").strip()
        if not url or not content:
            continue
        items.append(
            {
                "source_url": url,
                "domain": (match.group("domain") or "").strip(),
                "title_hint": (match.group("title") or "").strip(),
                "content": content,
            }
        )
    return items


def _persist_crawled_articles(
    *,
    chat_id: str,
    topic: str,
    query: str,
    articles: list[dict],
) -> dict:
    if not articles:
        return {"saved": 0, "deduplicated": 0, "failed": 0}

    topic_tokens = _build_topic_signal_tokens(topic, query)
    min_db_topic_score = float(os.getenv("RESEARCH_DB_TOPIC_MIN_SCORE", "0.30"))

    filtered_articles: list[dict] = []
    skipped_off_topic = 0
    for item in articles:
        source_url = (item.get("source_url") or "").strip()
        title_hint = (item.get("title_hint") or "").strip()
        content = (item.get("content") or "")

        score = _score_crawled_article_topic_relevance(
            url=source_url,
            title=title_hint,
            content=content,
            topic_tokens=topic_tokens,
        )
        if topic_tokens and score < min_db_topic_score:
            skipped_off_topic += 1
            logger.info(
                "Research DB filter | skip_off_topic url=%s score=%.3f min=%.3f",
                source_url,
                score,
                min_db_topic_score,
            )
            continue
        filtered_articles.append(item)

    if skipped_off_topic:
        logger.info(
            "Research DB filter | topic=%s query=%s kept=%s skipped_off_topic=%s",
            topic,
            query,
            len(filtered_articles),
            skipped_off_topic,
        )

    if not filtered_articles:
        return {"saved": 0, "deduplicated": 0, "failed": 0}

    try:
        knowledge_service = _get_knowledge_service()
    except Exception as exc:
        logger.debug("Knowledge service unavailable; skip article persistence: %s", exc)
        return {"saved": 0, "deduplicated": 0, "failed": len(filtered_articles)}

    counters = {"saved": 0, "deduplicated": 0, "failed": 0}
    for item in filtered_articles:
        source_url = item.get("source_url", "")
        title = item.get("title_hint", "") or source_url
        content = item.get("content", "")
        metadata = {
            "source_url": source_url,
            "domain": item.get("domain", ""),
            "topic": topic,
            "query": query,
            "ingested_at": datetime.utcnow().isoformat(),
            "ingestion_pipeline": "research_node_crawl4ai",
        }
        try:
            result = knowledge_service.save_deduplicated(
                chat_id=chat_id,
                content=content,
                category="web_news",
                title=title[:200],
                metadata=metadata,
                source_url=source_url,
            )
            if result.get("deduplicated"):
                counters["deduplicated"] += 1
            else:
                counters["saved"] += 1
        except Exception as exc:
            logger.debug("Failed to persist crawled article %s: %s", source_url, exc)
            counters["failed"] += 1
    return counters

# DEBUG LOGGING
DEBUG_RESEARCH = os.getenv("DEBUG_RESEARCH", "").lower() in {"1", "true", "yes"}
DEBUG_LOG_FILE = Path("data/logs/research_debug.log") if DEBUG_RESEARCH else None

def _debug_log(step: str, key: str, value, prefix: str = ""):
    """Log debug info to both logger and file if DEBUG_RESEARCH is enabled."""
    if not DEBUG_RESEARCH:
        return
    msg = f"[RESEARCH_DEBUG] {step}::{key} {prefix}\n"
    if isinstance(value, (dict, list)):
        import json
        msg += json.dumps(value, ensure_ascii=False, indent=2)[:2500] + "\n"
    else:
        msg += str(value)[:2500] + "\n"
    logger.info(msg.strip())
    if DEBUG_LOG_FILE:
        try:
            DEBUG_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(DEBUG_LOG_FILE, "a", encoding="utf-8") as f:
                f.write(msg)
        except Exception as e:
            logger.warning("Failed to write debug log: %s", e)

# END DEBUG


def _is_crawlable_seed_url(url: str) -> bool:
    lowered = (url or "").strip().lower()
    if not lowered:
        return False
    if not (lowered.startswith("http://") or lowered.startswith("https://")):
        return False
    if lowered.endswith((".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg", ".css", ".js")):
        return False
    return True


def _is_news_hub_url(url: str) -> bool:
    """Detect if URL is a news hub/category/dashboard page (contains multiple articles)."""
    lowered = (url or "").strip().lower()
    # Pattern: URLs with /hub/, /category/, /news/, /topic/, /tag/, /section/ typically contain multiple articles
    hub_patterns = ["/hub/", "/category/", "/news/", "/topic/", "/tag/", "/section/"]
    return any(pattern in lowered for pattern in hub_patterns)


def _is_listing_seed_url(url: str) -> bool:
    """Detect generic listing pages (homepages/category pages) that aggregate many article links."""
    parsed = urlparse((url or "").strip())
    if parsed.scheme not in {"http", "https"}:
        return False
    domain = _normalize_domain(url)
    path = (parsed.path or "/").strip() or "/"
    if path == "/":
        return domain in {
            "vnexpress.net",
            "cafef.vn",
            "techcrunch.com",
            "apnews.com",
            "bbc.com",
            "theguardian.com",
            "nytimes.com",
        }
    return _is_news_hub_url(url)


def _extract_article_links_from_markdown(markdown: str, base_url: str) -> list[dict[str, str]]:
    """Extract article links from hub page markdown."""
    if not markdown:
        return []
    
    # Pattern to match markdown links: [text](url "optional title")
    link_pattern = re.compile(r'\[([^\]]+)\]\(([^)\s]+)(?:\s+"[^"]*")?\)')
    base_domain = _normalize_domain(base_url)
    
    article_links: list[dict[str, str]] = []
    seen = set()
    
    for match in link_pattern.finditer(markdown):
        link_text = match.group(1).strip()
        link_url = match.group(2).strip()
        if " " in link_url:
            link_url = link_url.split(" ", 1)[0].strip()
        link_url = link_url.strip("<>'\"")
        
        if not link_url:
            continue
        
        # Convert relative URLs to absolute
        if not link_url.startswith(('http://', 'https://')):
            link_url = urljoin(base_url, link_url)

        if link_url in seen or link_url.rstrip("/") == base_url.rstrip("/"):
            continue

        # Keep extraction focused on the same publisher domain.
        if _normalize_domain(link_url) != base_domain:
            continue
        
        # Skip common non-article URLs
        skip_patterns = [
            "/rss", ".rss", "xml", "/feed", "/sitemap",
            "/contact", "/about", "/privacy", "/terms",
            "/archive", "/search", "/login", "/subscribe",
            "/video", ".mp4", ".mp3", "/gallery"
        ]
        if any(pattern in link_url.lower() for pattern in skip_patterns):
            continue
        
        # Skip if it's just domain root or hub page itself
        if not link_text or len(link_text) < 3:
            continue
        
        # Prefer links that look like news articles (longer text, not just single word)
        if len(link_text) > 5:
            seen.add(link_url)
            article_links.append({"url": link_url, "title": link_text})
    
    return article_links


def _tokenize_for_match(text: str) -> set[str]:
    cleaned = re.sub(r"[^\w\sÀ-ỹ]", " ", (text or "").lower(), flags=re.UNICODE)
    tokens = {tok for tok in cleaned.split() if len(tok) >= 3}
    stop = {
        "the", "and", "for", "with", "that", "this", "from", "into", "after", "over", "under",
        "cua", "cho", "voi", "nhung", "trong", "sau", "truoc", "mot", "nhieu", "theo", "lai",
    }
    return {tok for tok in tokens if tok not in stop}


def _best_article_url_for_headline(headline: str, articles: list[dict], fallback_source: str) -> str:
    headline_tokens = _tokenize_for_match(headline)
    if not headline_tokens:
        return fallback_source

    best_url = fallback_source
    best_score = 0.0
    fallback_domain = _normalize_domain(fallback_source)
    domain_candidate = ""

    for article in articles:
        article_url = (article.get("source_url") or "").strip()
        if not article_url:
            continue
        if not domain_candidate and _normalize_domain(article_url) == fallback_domain:
            domain_candidate = article_url

        title_hint = (article.get("title_hint") or "").strip()
        title_tokens = _tokenize_for_match(title_hint)
        content_preview = (article.get("content") or "")[:1200]
        content_tokens = _tokenize_for_match(content_preview)

        title_overlap = len(headline_tokens & title_tokens)
        content_overlap = len(headline_tokens & content_tokens)
        score = (3.0 * title_overlap) + (1.0 * content_overlap)

        if score > best_score:
            best_score = score
            best_url = article_url

    if best_score > 0:
        return best_url
    if domain_candidate:
        return domain_candidate
    return fallback_source


def _rewrite_hub_sources_with_article_urls(
    result_text: str,
    *,
    articles: list[dict],
    hub_urls: list[str],
) -> str:
    if not result_text:
        return result_text

    normalized_hubs = {u.rstrip("/") for u in hub_urls if u}
    if not normalized_hubs:
        return result_text

    lines = result_text.splitlines()
    if not lines:
        return result_text

    def _extract_headline(before_idx: int) -> str:
        for idx in range(before_idx, -1, -1):
            candidate = lines[idx].strip()
            if not candidate:
                continue
            if candidate.startswith("**"):
                return candidate.strip("*").strip()
        return ""

    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped.lower().startswith("- source:"):
            continue

        current_source = stripped.split(":", 1)[1].strip()
        if not current_source:
            continue

        normalized_current = current_source.rstrip("/")
        if normalized_current not in normalized_hubs:
            continue

        headline = _extract_headline(i - 1)
        replacement = _best_article_url_for_headline(headline, articles, current_source)
        if replacement and replacement != current_source:
            indent = line[: len(line) - len(line.lstrip())]
            lines[i] = f"{indent}- Source: {replacement}"

    return "\n".join(lines)


async def _collect_context_from_fixed_urls(urls: list[str]) -> tuple[str, list[str]]:
    context_parts: list[str] = []
    crawled_urls: list[str] = []
    seen_seed_urls: set[str] = set()
    seen_article_urls: set[str] = set()

    # Stop by total evidence size instead of fixed "N articles per source".
    context_char_budget = int(os.getenv("DAILY_FIXED_SOURCE_CONTEXT_CHAR_BUDGET", "180000"))
    running_context_chars = 0

    per_source_queues: dict[str, deque[dict[str, str]]] = {}
    source_order: list[str] = []

    for raw_url in urls:
        source_url = (raw_url or "").strip()
        if not source_url or source_url in seen_seed_urls:
            continue
        seen_seed_urls.add(source_url)
        if not _is_crawlable_seed_url(source_url):
            logger.warning("Skip non-crawlable fixed source URL: %s", source_url)
            continue

        source_domain = _normalize_domain(source_url)
        seed_markdown = await crawl_url_to_markdown(source_url)
        if not seed_markdown:
            logger.info("Fixed-source crawl returned empty content: %s", source_url)
            continue

        extracted_links = _extract_article_links_from_markdown(seed_markdown, source_url)
        is_listing = _is_listing_seed_url(source_url) or len(extracted_links) >= 8

        queue_items: list[dict[str, str]] = []
        if is_listing and extracted_links:
            logger.info(
                "Detected listing source, extracted %d article links: %s",
                len(extracted_links),
                source_url,
            )
            for item in extracted_links:
                article_url = (item.get("url") or "").strip()
                if not article_url or article_url in seen_article_urls:
                    continue
                seen_article_urls.add(article_url)
                queue_items.append(
                    {
                        "url": article_url,
                        "title": (item.get("title") or "").strip(),
                        "domain": _normalize_domain(article_url),
                    }
                )
        else:
            # Treat as single-article source when no listing signal is found.
            queue_items.append(
                {
                    "url": source_url,
                    "title": "fixed_source",
                    "domain": source_domain,
                    "prefetched_content": seed_markdown,
                }
            )

        if not queue_items:
            # Fallback to seed markdown when link extraction is empty.
            queue_items.append(
                {
                    "url": source_url,
                    "title": "fixed_source",
                    "domain": source_domain,
                    "prefetched_content": seed_markdown,
                }
            )

        per_source_queues[source_url] = deque(queue_items)
        source_order.append(source_url)

    if not source_order:
        return "", []

    # Balanced crawl: one article per source each round until all queues are exhausted
    # or the global context budget is reached.
    progress_made = True
    while progress_made:
        progress_made = False
        for source_url in source_order:
            queue = per_source_queues.get(source_url)
            if not queue:
                continue

            while queue:
                candidate = queue.popleft()
                article_url = (candidate.get("url") or "").strip()
                if not article_url or article_url in crawled_urls:
                    continue

                article_markdown = (candidate.get("prefetched_content") or "").strip()
                if not article_markdown:
                    article_markdown = await crawl_url_to_markdown(article_url)

                if not article_markdown or len(article_markdown.strip()) < 100:
                    logger.debug("Article crawl returned insufficient content: %s", article_url)
                    continue

                estimated_next_size = running_context_chars + len(article_markdown)
                if estimated_next_size > context_char_budget and context_parts:
                    logger.info(
                        "Stop fixed-source crawl due to context budget (%s chars).",
                        context_char_budget,
                    )
                    return "\n\n".join(context_parts).strip(), crawled_urls

                crawled_urls.append(article_url)
                context_parts.append(
                    f"=== CRAWL4AI ARTICLE ===\n"
                    f"URL: {article_url}\n"
                    f"DOMAIN: {(candidate.get('domain') or _normalize_domain(article_url)).strip()}\n"
                    f"TITLE_HINT: {(candidate.get('title') or '').strip()}\n"
                    f"CONTENT:\n{article_markdown}"
                )
                running_context_chars = estimated_next_size
                progress_made = True
                # Move to next source after one successful article for fairness.
                break

    return "\n\n".join(context_parts).strip(), crawled_urls

async def research_node(state: AgentState) -> dict:
    logger.info("--- RESEARCH NODE ---")
    await unload_model(OLLAMA_ORCHESTRATOR_MODEL)

    session_id = state.get("session_id", "default")
    ctx = load_context(session_id)
    user_text = ctx.get("user_message", "")
    topic = ctx.get("topic", state.get("topic", ""))
    search_query = ctx.get("search_query", state.get("search_query", ""))
    tasks = ctx.get("tasks", state.get("tasks", []))
    step_index = ctx.get("step_index", 0)

    if not user_text:
        user_message = next(
            (m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), None
        )
        user_text = user_message.content if user_message else ""

    research_query = (search_query or "").strip() or (topic or "").strip() or user_text.strip()
    persist_enabled = bool(state.get("persist_research_to_db", True))
    report_mode = (
        str(state.get("report_mode") or ctx.get("report_mode") or "").strip().lower()
    )
    fixed_source_urls = (
        state.get("fixed_source_urls")
        or ctx.get("fixed_source_urls")
        or DAILY_REPORT_FIXED_SOURCE_URLS
    )
    if not isinstance(fixed_source_urls, list):
        fixed_source_urls = DAILY_REPORT_FIXED_SOURCE_URLS
    daily_fixed_sources_mode = report_mode in {"daily_fixed_sources", "daily_report"} or session_id.startswith("tg_daily")
    logger.info("Research query (topic-first): %r", research_query)
    _debug_log("START", "user_text", user_text)
    _debug_log("START", "topic", topic)
    _debug_log("START", "search_query", search_query)
    _debug_log("START", "research_query", research_query)
    _debug_log("START", "report_mode", report_mode)
    _debug_log("START", "fixed_source_urls", fixed_source_urls)
    _debug_log("START", "daily_fixed_sources_mode", daily_fixed_sources_mode)
    if daily_fixed_sources_mode:
        logger.info(
            "Research running in daily fixed-source mode (urls=%s)",
            len(fixed_source_urls),
        )
        collected_context, discovered_sources = await _collect_context_from_fixed_urls(
            fixed_source_urls
        )
        if not collected_context:
            collected_context = (
                "Research unavailable: fixed_source_urls is empty or could not be crawled. "
                "Configure one or more reliable news URLs for daily report mode."
            )
    elif not CAMOUFOX_ENABLED:
        collected_context = (
            "Research unavailable: CAMOUFOX_ENABLED=false disables interactive web research."
        )
        discovered_sources = []
    elif research_search_provider() == "none":
        collected_context = "Research unavailable: RESEARCH_SEARCH_PROVIDER=none."
        discovered_sources = []
    elif research_search_provider() == "parallel":
        try:
            collected_context, discovered_sources = await collect_parallel_context(
                research_query,
                session_id=session_id,
                max_sources=RESEARCH_MAX_DISCOVERED_SOURCES,
            )
        except Exception as exc:
            logger.warning("Parallel research failed: %s", type(exc).__name__)
            collected_context = (
                "Research unavailable: Parallel search/fetch failed. "
                "No web evidence was collected; do not present an answer as web-verified."
            )
            discovered_sources = []
    else:
        logger.info("Research crawl mode is active.")
        camoufox_user_id = str(state.get("chat_id") or session_id or "agent")
        camoufox_session_key = f"research_{session_id}"
        
        browser = None
        if CHROME_CDP_ENABLED:
            logger.info("Attempting to connect to Chrome CDP...")
            cdp_client = ChromeCDPClient()
            if await cdp_client.ping():
                browser = cdp_client
                logger.info("Successfully connected to Chrome CDP. Using as primary browser.")
            else:
                logger.warning("Chrome CDP connection failed. Ensure Chrome is running with --remote-debugging-port. Falling back to Camoufox MCP.")
                await cdp_client.close()
                
        if browser is None:
            browser = CamoFoxMCPClient(
                user_id=camoufox_user_id,
                session_key=camoufox_session_key,
            )
        discovered_sources: list[str] = []
        try:
            is_up = await browser.ping()
            if not is_up:
                collected_context = (
                    "Research unavailable: Camoufox MCP server/browser is down. "
                    "No fallback sources are allowed."
                )
            else:
                closed = await browser.close_all_tabs()
                if closed:
                    logger.info("Closed %s stale Camoufox tab(s) before new crawl.", closed)

                await asyncio.sleep(3)

                crawl_result = await perform_camoufox_direct_crawl(
                    user_text=user_text,
                    topic=research_query,
                    browser=browser,
                    search_query_override=search_query,
                )
                collected_context = crawl_result.get("context", "")
                discovered_sources = crawl_result.get("sources", [])
                _debug_log("CRAWL_RESULT", "discovered_sources_count", len(discovered_sources))
                _debug_log("CRAWL_RESULT", "discovered_sources_list", discovered_sources)
                _debug_log("CRAWL_RESULT", "context_total_chars", len(collected_context), prefix="total char length")
                _debug_log("CRAWL_RESULT", "context_sample", collected_context[:1500], prefix="first 1500 chars")
                logger.info(
                    "Research evidence | bot_detected=%s discovered_urls=%s crawled_urls=%s",
                    crawl_result.get("bot_detected_count", 0),
                    len(crawl_result.get("discovered_urls", [])),
                    len(discovered_sources),
                )
        finally:
            await browser.close()

    chat_id_for_persistence = str(state.get("chat_id") or session_id or "research")
    crawled_articles = _extract_crawled_articles(collected_context)
    if persist_enabled:
        article_persistence = _persist_crawled_articles(
            chat_id=chat_id_for_persistence,
            topic=topic or research_query,
            query=research_query,
            articles=crawled_articles,
        )
    else:
        article_persistence = {"saved": 0, "deduplicated": 0, "failed": 0}

    if crawled_articles and persist_enabled:
        logger.info(
            "Research article persistence | extracted=%s saved=%s deduplicated=%s failed=%s",
            len(crawled_articles),
            article_persistence.get("saved", 0),
            article_persistence.get("deduplicated", 0),
            article_persistence.get("failed", 0),
        )

    # =========================== Synthesize ===================================
    task_description = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(tasks))
    synthesis_user_prompt = (
        f"Research query used: {research_query}\n"
        f"Topic: {topic}\n"
        f"Tasks assigned to you:\n{task_description}\n\n"
        f"Original user query: {user_text}\n\n"
        "Content filtering guidance:\n"
        "- Remove any promotional, sponsored, or irrelevant content.\n"
        "- Keep only factual and educational information.\n\n"
        "Source citation guidance:\n"
        "- For each item, cite a direct article URL, not a hub/category homepage.\n"
        "- Never use fixed dashboard URLs as item-level source when article URLs are available.\n\n"
        f"Gathered context:\n{collected_context}"
    )
    _debug_log("SYNTHESIS_PREP", "model_name", OLLAMA_RESEARCH_MODEL)
    _debug_log("SYNTHESIS_PREP", "user_prompt_char_count", len(synthesis_user_prompt), prefix="chars")
    _debug_log("SYNTHESIS_PREP", "user_prompt_sample", synthesis_user_prompt[:2000], prefix="first 2000 chars")

    llm = get_llm(task_type="research", temperature=0.3)
    system_prompt = get_workspace_priming_context(model_role="researcher") or (
        "You are a Research Agent. Synthesize the provided context into a clear, well-structured answer. "
        "Always cite your sources with URLs. Filter to keep only relevant information. "
        "Never refuse to engage — if context is thin, reason from what you have and flag any uncertainty."
    )

    logger.info("Synthesizing research results with %s...", OLLAMA_RESEARCH_MODEL)
    try:
        response = await llm.ainvoke(
            [SystemMessage(content=system_prompt), HumanMessage(content=synthesis_user_prompt)]
        )
        result_text = response.content.strip()
        if daily_fixed_sources_mode and crawled_articles:
            result_text = _rewrite_hub_sources_with_article_urls(
                result_text,
                articles=crawled_articles,
                hub_urls=fixed_source_urls,
            )
        _debug_log("MODEL_OUTPUT", "response_length", len(result_text), prefix="chars")
        _debug_log("MODEL_OUTPUT", "full_response", result_text)
    except Exception as e:
        logger.error(f"Error during LLM synthesis: {e}", exc_info=True)
        result_text = "An error occurred while synthesising the research data."
        _debug_log("MODEL_OUTPUT", "error_occurred", str(e))

    # =========================== Extract Sources ===================================
    if daily_fixed_sources_mode or CAMOUFOX_ENABLED:
        sources = list(dict.fromkeys(discovered_sources))[:10]
    else:
        sources = re.findall(r'https?://[^\s"<>]+', collected_context)
        sources = list(dict.fromkeys(sources))[:10]

    persisted_sources = _persist_research_sources(sources) if persist_enabled else 0
    if persisted_sources:
        logger.info("Persisted %d research source URL(s) into bookmarks.", persisted_sources)

    _debug_log("FINAL_RESULT", "final_sources_count", len(sources))
    _debug_log("FINAL_RESULT", "final_sources_list", sources)

    task_result = {
        "step_index": step_index,
        "model": OLLAMA_RESEARCH_MODEL,
        "result": result_text,
        "sources": sources,
        "persistence": {
            "articles_extracted": len(crawled_articles),
            "articles_saved": article_persistence.get("saved", 0),
            "articles_deduplicated": article_persistence.get("deduplicated", 0),
            "articles_failed": article_persistence.get("failed", 0),
            "source_urls_saved": persisted_sources,
        },
    }

    return {
        "task_results": [task_result],
        "active_model": OLLAMA_RESEARCH_MODEL,
    }

def _normalize_domain(url: str) -> str:
    return urlparse(url).netloc.replace("www.", "").lower()

def _is_allowlisted_domain(domain: str) -> bool:
    return any(domain == allowed or domain.endswith(f".{allowed}") for allowed in RESEARCH_SOURCE_ALLOWLIST)

def _is_probable_article_url(url: str) -> bool:
    lowered = url.lower()
    blocked_fragments = [
        "google.com/search",
        "google.com/sorry",
        "news.google.com/search",
        "youtube.com",
        "accounts.google.com",
        "/follow?",
        "?iid=",
        "/live-updates",
        "/tag/",
        "/topics/",
        "/video/",
        "/watch",
        "/privacy",
        "/terms",
        "/settings",
    ]
    if any(fragment in lowered for fragment in blocked_fragments):
        return False
    if lowered.endswith((".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg", ".css", ".js")):
        return False
    if not (lowered.startswith("http://") or lowered.startswith("https://")):
        return False

    parsed = urlparse(url)
    path = (parsed.path or "").strip()
    if path in {"", "/"}:
        return False

    return True

def _compact_search_query(query: str) -> str:
    raw = (query or "").strip()
    if not raw:
        return ""

    cleaned = re.sub(r"[^\w\sÀ-ỹ]", " ", raw, flags=re.UNICODE)
    tokens = [token for token in re.split(r"\s+", cleaned) if token]
    if not tokens:
        return raw[:120]

    return " ".join(tokens[:10])[:120]

def _extract_refs_from_snapshot(snapshot_text: str) -> list[tuple[str, str]]:
    pattern = re.compile(r"\[link\s+(e\d+)\]\s*([^\n\r]+)", flags=re.IGNORECASE)
    seen_refs: set[str] = set()
    refs: list[tuple[str, str]] = []
    for match in pattern.finditer(snapshot_text or ""):
        ref_id = match.group(1).strip()
        label = match.group(2).strip()
        if ref_id in seen_refs:
            continue
        if len(label) < 6:
            continue
        seen_refs.add(ref_id)
        refs.append((ref_id, label[:180]))
    return refs


def _build_ref_candidates(snapshot_text: str, refs_details: Optional[list[dict]] = None) -> list[dict]:
    candidates: list[dict] = []
    seen: set[str] = set()

    for item in refs_details or []:
        ref_id = str((item or {}).get("refId") or "").strip()
        if not ref_id or ref_id in seen:
            continue
        title = str((item or {}).get("title") or "").strip()
        if len(title) < 6:
            continue
        candidates.append(
            {
                "ref_id": ref_id,
                "title": title[:180],
                "url": str((item or {}).get("url") or "").strip(),
                "domain": str((item or {}).get("domain") or "").strip(),
                "path": str((item or {}).get("path") or "").strip(),
            }
        )
        seen.add(ref_id)

    if candidates:
        return candidates

    # Backward-compatible fallback when browser client doesn't provide refsDetails.
    for ref_id, label in _extract_refs_from_snapshot(snapshot_text):
        if ref_id in seen:
            continue
        candidates.append(
            {
                "ref_id": ref_id,
                "title": label,
                "url": "",
                "domain": "",
                "path": "",
            }
        )
        seen.add(ref_id)
    return candidates

def _extract_urls_from_text(text: str) -> list[str]:
    urls = re.findall(r"https?://[^\s\"'<>\)\]]+", text or "")
    deduped: list[str] = []
    seen: set[str] = set()
    for raw_url in urls:
        normalized = raw_url.rstrip(".,;)\"]'")
        if normalized in seen:
            continue
        if not _is_probable_article_url(normalized):
            continue
        seen.add(normalized)
        deduped.append(normalized)
    return deduped


def _build_topic_signal_tokens(*texts: str) -> set[str]:
    joined = " ".join((text or "").strip().lower() for text in texts if text)
    cleaned = re.sub(r"[^\w\sÀ-ỹ]", " ", joined, flags=re.UNICODE)
    tokens = [token for token in re.split(r"\s+", cleaned) if token]
    stopwords = {
        "the", "and", "for", "with", "that", "this", "from", "into", "latest", "today", "news",
        "thong", "tin", "moi", "nhat", "ve", "va", "cua", "la", "cac", "cho", "mot", "nhung",
        "toi", "ban", "giup", "hay", "find", "about", "update", "current", "breaking",
    }
    return {token for token in tokens if len(token) >= 3 and token not in stopwords}


def _topic_overlap_score(text: str, topic_tokens: set[str]) -> float:
    if not text or not topic_tokens:
        return 0.0
    cleaned = re.sub(r"[^\w\sÀ-ỹ]", " ", (text or "").lower(), flags=re.UNICODE)
    tokens = {token for token in re.split(r"\s+", cleaned) if len(token) >= 3}
    if not tokens:
        return 0.0
    overlap = len(tokens & topic_tokens)
    if overlap == 0:
        return 0.0
    return overlap / max(1, min(len(topic_tokens), 8))


def _score_candidate_topic_relevance(candidate: dict, topic_tokens: set[str]) -> float:
    url = (candidate.get("url") or "").strip()
    title = (candidate.get("title") or "").strip()
    snapshot = (candidate.get("snapshot") or "").strip()[:1200]

    # Strongest signal: title and URL path; snapshot is weaker/noisy.
    title_score = _topic_overlap_score(title, topic_tokens)
    url_score = _topic_overlap_score(url, topic_tokens)
    snapshot_score = _topic_overlap_score(snapshot, topic_tokens)

    return (2.0 * title_score) + (1.5 * url_score) + (0.6 * snapshot_score)


def _score_crawled_article_topic_relevance(
    *,
    url: str,
    title: str,
    content: str,
    topic_tokens: set[str],
) -> float:
    if not topic_tokens:
        return 1.0

    candidate = {
        "url": url,
        "title": title,
        "snapshot": (content or "")[:2200],
    }
    base_score = _score_candidate_topic_relevance(candidate, topic_tokens)

    # Add a light content-only overlap bonus so genuine topical articles win,
    # while generic category pages are demoted.
    content_bonus = 0.8 * _topic_overlap_score((content or "")[:2200], topic_tokens)
    return base_score + content_bonus


def _rank_candidates_by_topic(candidates: list[dict], topic_tokens: set[str]) -> list[dict]:
    if not candidates:
        return []

    scored: list[dict] = []
    for item in candidates:
        candidate = dict(item)
        candidate["topic_score"] = _score_candidate_topic_relevance(candidate, topic_tokens)
        scored.append(candidate)

    # Prefer allowlisted sources first when topic score ties.
    scored.sort(
        key=lambda row: (
            float(row.get("topic_score") or 0.0),
            1.0 if row.get("allowlisted") else 0.0,
        ),
        reverse=True,
    )
    return scored


def _filter_candidates_by_topic(
    candidates: list[dict],
    topic_tokens: set[str],
    *,
    min_score: float,
) -> list[dict]:
    ranked = _rank_candidates_by_topic(candidates, topic_tokens)
    filtered = [row for row in ranked if float(row.get("topic_score") or 0.0) >= min_score]

    # Safety fallback: if strict filter removed all URLs, keep top few to avoid empty crawl.
    if filtered:
        return filtered
    return ranked[: max(1, min(4, len(ranked)))]


def _select_refs_by_topic_signal(
    refs: list[dict],
    topic_tokens: set[str],
    *,
    max_refs: int,
) -> list[str]:
    if not refs:
        return []
    if not topic_tokens:
        return [str(ref.get("ref_id") or "") for ref in refs[:max_refs] if ref.get("ref_id")]

    scored = []
    for ref in refs:
        ref_id = str(ref.get("ref_id") or "").strip()
        if not ref_id:
            continue
        title = str(ref.get("title") or "")
        path = str(ref.get("path") or "")
        url = str(ref.get("url") or "")
        score = (2.0 * _topic_overlap_score(title, topic_tokens)) + (
            1.5 * _topic_overlap_score(path, topic_tokens)
        ) + (1.0 * _topic_overlap_score(url, topic_tokens))
        scored.append((ref_id, score))

    scored.sort(key=lambda item: item[1], reverse=True)
    selected: list[str] = []
    for ref_id, _ in scored:
        if ref_id not in selected:
            selected.append(ref_id)
        if len(selected) >= max_refs:
            break
    return selected

async def _generate_search_queries(topic: str, user_text: str) -> list[str]:
    llm = get_llm(task_type="research", temperature=0.2)
    prompt = (
        "You are a research assistant. Please generate web search queries to find reliable, current sources.\n"
        f"Topic: {topic}\n"
        f"User context: {user_text}\n\n"
        "Requirements:\n"
        "- Return up to 2 queries.\n"
        "- Each query should be natural, like a real user, without machine-like strings such as: site, google, .com, http.\n"
        "- Do not number, do not use bullet points, do not explain further.\n"
        "- Focus on the latest information from today or the last 24 hours.\n"
        "- Prefer queries with terms like today, latest, current, breaking, updated."
    )
    try:
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        candidates = [line.strip("-• \t") for line in (response.content or "").splitlines() if line.strip()]
        cleaned: list[str] = []
        for candidate in candidates:
            compact = _compact_search_query(candidate)
            if compact and compact not in cleaned:
                cleaned.append(compact)
            if len(cleaned) >= RESEARCH_MAX_SEARCH_QUERIES:
                break
        if cleaned:
            return cleaned
    except Exception as error:
        logger.warning("Search query generation failed: %s", error)

    fallback = _compact_search_query(topic or user_text)
    if fallback:
        return [fallback, f"{fallback} today"][:RESEARCH_MAX_SEARCH_QUERIES]
    return ["latest global news today", "breaking news today"][:RESEARCH_MAX_SEARCH_QUERIES]


async def _select_refs_with_llm(
    snapshot_text: str,
    query: str,
    refs_details: Optional[list[dict]] = None,
    topic_tokens: Optional[set[str]] = None,
    max_refs: int = 4,
) -> list[str]:
    refs = _build_ref_candidates(snapshot_text, refs_details)
    if not refs:
        return []

    topic_tokens = topic_tokens or set()
    preselected_refs = _select_refs_by_topic_signal(refs, topic_tokens, max_refs=max_refs)
    if preselected_refs:
        refs_for_llm = [entry for entry in refs if str(entry.get("ref_id") or "") in set(preselected_refs)]
    else:
        refs_for_llm = refs[:max_refs]

    refs_prompt = "\n".join(
        (
            f"{ref.get('ref_id')}: "
            f"title={str(ref.get('title') or '')[:140]} | "
            f"domain={str(ref.get('domain') or '')[:80]} | "
            f"path={str(ref.get('path') or '')[:140]} | "
            f"url={str(ref.get('url') or '')[:220]}"
        )
        for ref in refs_for_llm[:40]
    )
    llm = get_llm(task_type="research", temperature=0.0)
    prompt = (
        "Select the refs corresponding to the most reliable and relevant news articles.\n"
        f"Search query: {query}\n\n"
        "Prioritize refs that directly match the requested topic and avoid generic category pages.\n"
        "Prefer links where title and URL path both contain topic signals.\n"
        "Avoid URLs that look like broad hubs/categories unless they explicitly match the topic.\n\n"
        "List of refs:\n"
        f"{refs_prompt}\n\n"
        "Return only the refs, separated by spaces (e.g., e4 e8 e12)."
    )
    try:
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        selected = re.findall(r"e\d+", response.content or "")
        uniq_selected: list[str] = []
        valid_ref_ids = {str(ref.get("ref_id") or "") for ref in refs_for_llm}
        for ref_id in selected:
            if ref_id in valid_ref_ids and ref_id not in uniq_selected:
                uniq_selected.append(ref_id)
            if len(uniq_selected) >= max_refs:
                break
        if uniq_selected:
            return uniq_selected
    except Exception as error:
        logger.warning("LLM ref selection failed: %s", error)

    fallback_refs = [str(ref.get("ref_id") or "") for ref in refs[:max_refs] if ref.get("ref_id")]
    return preselected_refs[:max_refs] or fallback_refs


async def _should_scroll_for_more_results(
    snapshot_text: str,
    query: str,
    refs_count: int,
    scroll_round: int,
) -> bool:
    if refs_count >= 8 and len((snapshot_text or "").strip()) >= 1200:
        return False

    llm = get_llm(task_type="research", temperature=0.0)
    prompt = (
        "Decide whether one shallow Google results scroll is needed to discover additional reliable sources.\n"
        f"Search query: {query}\n"
        f"Current refs count: {refs_count}\n"
        f"Current snapshot length: {len((snapshot_text or '').strip())}\n\n"
        f"Shallow scroll rounds already done: {scroll_round}\n\n"
        "Rules:\n"
        "- Answer YES only when current visible results are insufficient for source diversity/reliability.\n"
        "- If repeated scrolling is no longer adding meaningful new candidates, answer NO.\n"
        "- Prefer NO when there are already enough credible options.\n"
        "- We only allow a shallow scroll, not deep scrolling.\n\n"
        "Return exactly one token: YES or NO."
    )
    try:
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        answer = (response.content or "").strip().upper()
        return answer.startswith("YES")
    except Exception as error:
        logger.warning("LLM scroll decision failed: %s", error)
        return refs_count < 4 or len((snapshot_text or "").strip()) < 700


async def _rerank_non_allowlisted_sources(topic: str, candidates: list[dict]) -> list[dict]:
    if len(candidates) <= 1:
        return candidates

    llm = get_llm(task_type="research", temperature=0.0)
    prompt_items = "\n".join(
        f"- {index + 1}. {item['url']} | title: {item.get('title', '')[:120]}"
        for index, item in enumerate(candidates[:20])
    )
    prompt = (
        "Rank the URLs by source credibility and relevance to the topic.\n"
        f"Topic: {topic}\n\n"
        "URLs:\n"
        f"{prompt_items}\n\n"
        "Return only the order of numbers, separated by spaces (e.g., 2 1 3)."
    )

    try:
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        order = [int(number) - 1 for number in re.findall(r"\d+", response.content or "")]
        ordered: list[dict] = []
        used_indexes: set[int] = set()
        for index in order:
            if 0 <= index < len(candidates) and index not in used_indexes:
                ordered.append(candidates[index])
                used_indexes.add(index)
        for index, candidate in enumerate(candidates):
            if index not in used_indexes:
                ordered.append(candidate)
        return ordered
    except Exception as error:
        logger.warning("LLM source reranking failed: %s", error)
        return candidates


async def _select_sources_hybrid(topic: str, candidates: list[dict]) -> list[dict]:
    allowlisted = [candidate for candidate in candidates if candidate.get("allowlisted")]
    non_allowlisted = [candidate for candidate in candidates if not candidate.get("allowlisted")]
    ranked_non_allowlisted = await _rerank_non_allowlisted_sources(topic, non_allowlisted)
    combined = allowlisted + ranked_non_allowlisted
    return combined[:RESEARCH_MAX_DISCOVERED_SOURCES]


def _is_low_signal_snapshot(snapshot: str, refs_count: int) -> bool:
    text = (snapshot or "").strip()
    if not text:
        return True
    if len(text) < 120:
        return True
    if refs_count == 0 and len(text) < 500:
        return True
    return False


def _looks_like_bot_challenge(url: str, snapshot_text: str) -> bool:
    text = f"{url}\n{snapshot_text}".lower()
    markers = [
        "google.com/sorry",
        "unusual traffic",
        "verify you are human",
        "i am not a robot",
        "captcha",
        "cloudflare",
        "attention required",
    ]
    return any(marker in text for marker in markers)


async def _behavior_pause(multiplier: float = 1.0) -> None:
    base = random.uniform(CAMOFOX_BEHAVIOR_MIN_DELAY_SECONDS, CAMOFOX_BEHAVIOR_MAX_DELAY_SECONDS)
    await asyncio.sleep(max(0.2, base * max(0.2, multiplier)))


async def perform_camoufox_direct_crawl(
    user_text: str,
    topic: str,
    browser,
    search_query_override: str = "",
) -> dict:
    if (search_query_override or "").strip() and (search_query_override or "").strip().upper() != "NONE":
        search_queries = [_compact_search_query(search_query_override)]
    else:
        search_queries = (await _generate_search_queries(topic=topic, user_text=user_text))[:RESEARCH_MAX_SEARCH_QUERIES]
    logger.info("Research evidence | generated_queries=%s max_used=%s", search_queries, RESEARCH_MAX_SEARCH_QUERIES)
    _debug_log("CAMOUFOX_START", "search_queries_generated", search_queries)

    topic_tokens = _build_topic_signal_tokens(topic, user_text, search_query_override)
    _debug_log("CAMOUFOX_START", "topic_signal_tokens", sorted(topic_tokens))

    google_guard = GoogleGuard(
        config=GoogleGuardConfig(
            min_interval_seconds=GOOGLE_MIN_INTERVAL_SECONDS,
            cooldown_on_challenge_seconds=GOOGLE_COOLDOWN_ON_CHALLENGE_SECONDS,
        )
    )
    cache = ResearchResultCache(config=CacheConfig(ttl_seconds=GOOGLE_SEARCH_CACHE_TTL_SECONDS))

    context_parts: list[str] = []
    discovered_candidates: list[dict] = []
    seen_urls: set[str] = set()
    bot_detected_count = 0
    reusable_tab_id: str | None = None

    for search_query_idx, search_query in enumerate(search_queries):
        if not await browser.ping():
            logger.warning("Camoufox became unavailable during crawl loop.")
            break

        # Cache hit: skip Google entirely.
        cached = cache.get(search_query)
        if cached:
            cached_sources = cached.get("selected_sources") or []
            if isinstance(cached_sources, list) and cached_sources:
                logger.info("Research evidence | google_cache_hit query=%s sources=%d", search_query, len(cached_sources))
                _debug_log("GOOGLE_CACHE", "hit_query", search_query)
                for item in cached_sources:
                    url = (item or {}).get("url")
                    if not isinstance(url, str) or not url:
                        continue
                    if url in seen_urls:
                        continue
                    seen_urls.add(url)
                    discovered_candidates.append(
                        {
                            "url": url,
                            "title": (item or {}).get("title", ""),
                            "domain": (item or {}).get("domain", _normalize_domain(url)),
                            "allowlisted": bool((item or {}).get("allowlisted")),
                        }
                    )
                continue

        try:
            with file_lock(
                "data/state/google_search.lock",
                timeout_seconds=GOOGLE_SEARCH_LOCK_TIMEOUT_SECONDS,
            ):
                ok, reason = google_guard.can_hit_google()
                if not ok:
                    logger.warning("Research evidence | google_blocked reason=%s query=%s", reason, search_query)
                    context_parts.append(
                        "=== GOOGLE SEARCH SKIPPED ===\n"
                        f"QUERY: {search_query}\n"
                        f"REASON: {reason}\n"
                        "NOTE: Using cache/other sources only to reduce bot-detection risk.\n"
                    )
                    continue

                query_handled = False
                for attempt in range(CAMOFOX_CHALLENGE_MAX_RETRIES + 1):
                    tab_id = reusable_tab_id if CAMOFOX_REUSE_TAB and reusable_tab_id else None
                    if not tab_id:
                        tab_id = await browser.create_tab("https://www.google.com")
                        if not tab_id:
                            logger.warning("Camoufox create_tab failed for query: %s", search_query)
                            break
                        if CAMOFOX_REUSE_TAB:
                            reusable_tab_id = tab_id

                    google_guard.mark_google_hit()
                    if not await browser.search_google(tab_id, search_query):
                        logger.warning("Camoufox search_google failed for query: %s", search_query)
                        if CAMOFOX_REUSE_TAB and reusable_tab_id:
                            await browser.close_tab(reusable_tab_id)
                            reusable_tab_id = None
                        await _behavior_pause(multiplier=1.0 + attempt)
                        continue

                    logger.info(
                        "Research evidence | behavior_pause query=%d/%d attempt=%d/%d",
                        search_query_idx + 1,
                        len(search_queries),
                        attempt + 1,
                        CAMOFOX_CHALLENGE_MAX_RETRIES + 1,
                    )
                    await _behavior_pause(multiplier=1.2 + 0.5 * attempt)

                    first_page = await browser.get_snapshot_page(tab_id)
                    if not first_page:
                        logger.warning("No snapshot data from Camoufox for query: %s", search_query)
                        await _behavior_pause(multiplier=1.0 + attempt)
                        continue

                    page_url = str(first_page.get("url") or "")
                    snapshot_text = str(first_page.get("snapshot") or "")
                    refs_count = int(first_page.get("refsCount") or 0)

                    scroll_attempts = 0
                    while True:
                        if scroll_attempts >= 8:
                            logger.info(
                                "Research evidence | shallow_scroll_guard_stop query=%s attempts=%s",
                                search_query,
                                scroll_attempts,
                            )
                            break

                        should_scroll = await _should_scroll_for_more_results(
                            snapshot_text=snapshot_text,
                            query=search_query,
                            refs_count=refs_count,
                            scroll_round=scroll_attempts,
                        )
                        if not should_scroll:
                            break
                        if not hasattr(browser, "scroll_for_more_results"):
                            break

                        before_urls = set(_extract_urls_from_text(snapshot_text))
                        scrolled = await browser.scroll_for_more_results(
                            tab_id,
                            steps=1,
                            pixels_per_step=280,
                        )
                        if not scrolled:
                            break

                        await _behavior_pause(multiplier=0.6)
                        after_scroll_page = await browser.get_snapshot_page(tab_id)
                        if not after_scroll_page:
                            break

                        next_snapshot = str(after_scroll_page.get("snapshot") or "")
                        after_urls = set(_extract_urls_from_text(next_snapshot))
                        newly_discovered = len(after_urls - before_urls)

                        page_url = str(after_scroll_page.get("url") or page_url)
                        snapshot_text = next_snapshot
                        refs_count = int(after_scroll_page.get("refsCount") or refs_count)
                        first_page = after_scroll_page
                        scroll_attempts += 1

                        logger.info(
                            "Research evidence | shallow_scroll query=%s attempt=%s new_urls=%s refs=%s",
                            search_query,
                            scroll_attempts,
                            newly_discovered,
                            refs_count,
                        )

                        if newly_discovered == 0:
                            break

                    if _looks_like_bot_challenge(page_url, snapshot_text):
                        bot_detected_count += 1
                        google_guard.mark_challenge(reason=f"serp_challenge:{page_url}")
                        logger.info(
                            "Research evidence | bot_detected query=%s url=%s attempt=%d/%d cooldown=%ss",
                            search_query,
                            page_url,
                            attempt + 1,
                            CAMOFOX_CHALLENGE_MAX_RETRIES + 1,
                            int(GOOGLE_COOLDOWN_ON_CHALLENGE_SECONDS),
                        )
                        if CAMOFOX_REUSE_TAB and reusable_tab_id:
                            await browser.close_tab(reusable_tab_id)
                            reusable_tab_id = None
                        await _behavior_pause(multiplier=2.0 + attempt)
                        continue

                    if _is_low_signal_snapshot(snapshot_text, refs_count):
                        logger.info(
                            "Low-signal snapshot detected (refs=%s, len=%s), retry query=%s url=%s attempt=%d/%d",
                            refs_count,
                            len(snapshot_text),
                            search_query,
                            page_url,
                            attempt + 1,
                            CAMOFOX_CHALLENGE_MAX_RETRIES + 1,
                        )
                        await _behavior_pause(multiplier=1.5 + 0.5 * attempt)
                        continue

                    context_parts.append(
                        f"=== CAMOUFOX SEARCH PAGE ===\n"
                        f"QUERY: {search_query}\n"
                        f"URL: {page_url}\n"
                        f"SNAPSHOT:\n{snapshot_text[:7000]}"
                    )

                    for extracted_url in _extract_urls_from_text(snapshot_text):
                        if extracted_url in seen_urls:
                            continue
                        seen_urls.add(extracted_url)
                        domain = _normalize_domain(extracted_url)
                        is_allowlisted = _is_allowlisted_domain(domain)
                        _debug_log("CAMOUFOX_URL_EXTRACTED", "url", extracted_url, prefix=f"allowlisted={is_allowlisted}")
                        discovered_candidates.append(
                            {
                                "url": extracted_url,
                                "title": "",
                                "domain": domain,
                                "allowlisted": is_allowlisted,
                            }
                        )

                    selected_refs = await _select_refs_with_llm(
                        snapshot_text=snapshot_text,
                        query=search_query,
                        refs_details=first_page.get("refsDetails") if isinstance(first_page, dict) else None,
                        topic_tokens=topic_tokens,
                        max_refs=4,
                    )
                    logger.info(
                        "Research evidence | query=%s refs_selected=%s refs_count=%s",
                        search_query,
                        selected_refs,
                        refs_count,
                    )

                    for ref_id in selected_refs:
                        await _behavior_pause(multiplier=0.8)
                        if not await browser.click(tab_id, ref_id):
                            continue
                        await _behavior_pause(multiplier=1.0)
                        landed_page = await browser.get_snapshot_page(tab_id)
                        if not landed_page:
                            await _behavior_pause(multiplier=0.7)
                            continue

                        landed_url = str(landed_page.get("url") or "")
                        if not landed_url or not _is_probable_article_url(landed_url):
                            continue

                        if _looks_like_bot_challenge(landed_url, str(landed_page.get("snapshot") or "")):
                            bot_detected_count += 1
                            google_guard.mark_challenge(reason=f"click_challenge:{landed_url}")
                            logger.info("Research evidence | bot_detected clicked_ref=%s url=%s", ref_id, landed_url)
                            continue

                        if landed_url in seen_urls:
                            continue

                        seen_urls.add(landed_url)
                        domain = _normalize_domain(landed_url)
                        ref_title = next(
                            (label for candidate_ref, label in _extract_refs_from_snapshot(snapshot_text) if candidate_ref == ref_id),
                            "",
                        )
                        candidate = {
                            "url": landed_url,
                            "title": ref_title,
                            "domain": domain,
                            "allowlisted": _is_allowlisted_domain(domain),
                            "snapshot": str(landed_page.get("snapshot") or "")[:1200],
                        }
                        topic_score = _score_candidate_topic_relevance(candidate, topic_tokens)
                        candidate["topic_score"] = topic_score
                        if topic_tokens and topic_score < 0.18:
                            logger.info(
                                "Research evidence | skip_low_topic_signal ref=%s url=%s score=%.3f",
                                ref_id,
                                landed_url,
                                topic_score,
                            )
                            continue
                        discovered_candidates.append(candidate)

                    collected_segments = [snapshot_text]
                    next_offset = first_page.get("nextOffset")
                    has_more = bool(first_page.get("hasMore"))

                    for _ in range(2):
                        if not has_more or next_offset is None:
                            break
                        extra_page = await browser.get_snapshot_page(tab_id, offset=int(next_offset))
                        if not extra_page:
                            break
                        collected_segments.append(str(extra_page.get("snapshot") or ""))
                        has_more = bool(extra_page.get("hasMore"))
                        next_offset = extra_page.get("nextOffset")

                    merged_snapshot = "\n\n".join(segment for segment in collected_segments if segment).strip()
                    if merged_snapshot:
                        context_parts.append(
                            f"=== CAMOUFOX SOURCE ===\n"
                            f"URL: {page_url}\n"
                            f"SNAPSHOT:\n{merged_snapshot[:6000]}"
                        )
                    query_handled = True
                    break

        except TimeoutError as error:
            logger.warning("Research evidence | google_lock_timeout query=%s error=%s", search_query, error)
            context_parts.append(
                "=== GOOGLE SEARCH SKIPPED ===\n"
                f"QUERY: {search_query}\n"
                "REASON: google_search_worker_busy (lock timeout)\n"
            )
            continue


        if not query_handled:
            logger.info("Research evidence | query_exhausted_after_retries query=%s", search_query)

    if reusable_tab_id:
        await browser.close_tab(reusable_tab_id)

    logger.info("Research evidence | discovered_url_candidates=%s", len(discovered_candidates))
    _debug_log("CAMOUFOX_URL_DISCOVERY", "all_discovered_candidates_count", len(discovered_candidates))
    _debug_log("CAMOUFOX_URL_DISCOVERY", "all_discovered_candidates", [c["url"] for c in discovered_candidates])

    if not discovered_candidates:
        return {
            "context": (
                "No crawlable sources discovered from Camoufox search flow. "
                "Treat context as insufficient and avoid unsupported claims."
            ),
            "sources": [],
            "discovered_urls": [],
            "bot_detected_count": bot_detected_count,
        }

    topic_filtered_candidates = _filter_candidates_by_topic(
        discovered_candidates,
        topic_tokens,
        min_score=0.14,
    )
    logger.info(
        "Research evidence | topic_filtered_candidates=%s of discovered=%s",
        len(topic_filtered_candidates),
        len(discovered_candidates),
    )
    _debug_log(
        "CAMOUFOX_URL_SELECTION",
        "topic_filtered_candidates",
        [
            {
                "url": c.get("url"),
                "topic_score": c.get("topic_score", 0.0),
                "title": (c.get("title") or "")[:120],
            }
            for c in topic_filtered_candidates[:20]
        ],
    )

    selected_sources = await _select_sources_hybrid(topic=topic, candidates=topic_filtered_candidates)
    selected_urls = [source["url"] for source in selected_sources]
    logger.info("Research evidence | selected_urls=%s", selected_urls)
    _debug_log("CAMOUFOX_URL_SELECTION", "selected_sources_count", len(selected_sources))
    _debug_log("CAMOUFOX_URL_SELECTION", "selected_urls", selected_urls)
    for i, source in enumerate(selected_sources):
        _debug_log(
            "CAMOUFOX_URL_SELECTION",
            f"selected_source_{i}",
            {"url": source["url"], "domain": source.get("domain"), "allowlisted": source.get("allowlisted")},
        )

    if selected_sources:
        try:
            cache.set(
                search_queries[0] if search_queries else topic,
                selected_sources=selected_sources,
                meta={"topic": topic, "bot_detected_count": bot_detected_count},
            )
        except Exception as error:
            logger.warning("Failed to write google search cache: %s", error)

    crawled_urls: list[str] = []
    total_crawled_chars = 0
    min_post_crawl_topic_score = float(os.getenv("RESEARCH_POST_CRAWL_TOPIC_MIN_SCORE", "0.28"))
    for source in selected_sources:
        source_url = source["url"]
        source_domain = source.get("domain", "")
        source_title = source.get("title", "")
        logger.info("Crawl4AI → crawling article: %s", source_url)
        _debug_log("CRAWL4AI_FETCH", "source_url", source_url)
        _debug_log("CRAWL4AI_FETCH", "source_domain", source_domain)
        _debug_log("CRAWL4AI_FETCH", "source_title", source_title)
        cached_html = None
        if hasattr(browser, "get_page_html_by_url"):
            try:
                cached_html = await browser.get_page_html_by_url(source_url)
                if cached_html:
                    logger.info("Research evidence | using cached browser HTML for %s", source_url)
            except Exception as error:
                logger.debug("Cached HTML lookup failed for %s: %s", source_url, error)

        article_markdown = await crawl_url_to_markdown(
            source_url,
            html=cached_html,
            base_url=source_url if cached_html else None,
            cdp_url=getattr(browser, "cdp_url", None),
        )
        if not article_markdown:
            logger.info("Research evidence | crawl_empty url=%s", source_url)
            _debug_log("CRAWL4AI_RESULT", "crawl_failed_empty", source_url)
            continue

        post_crawl_topic_score = _score_crawled_article_topic_relevance(
            url=source_url,
            title=source_title,
            content=article_markdown,
            topic_tokens=topic_tokens,
        )
        if topic_tokens and post_crawl_topic_score < min_post_crawl_topic_score:
            logger.info(
                "Research evidence | skip_post_crawl_low_topic url=%s score=%.3f min=%.3f",
                source_url,
                post_crawl_topic_score,
                min_post_crawl_topic_score,
            )
            _debug_log(
                "CRAWL4AI_RESULT",
                "post_crawl_topic_rejected",
                {
                    "url": source_url,
                    "title": source_title,
                    "score": post_crawl_topic_score,
                    "min": min_post_crawl_topic_score,
                },
            )
            continue

        crawled_urls.append(source_url)
        total_crawled_chars += len(article_markdown)
        _debug_log("CRAWL4AI_RESULT", "crawl_successful_url", source_url)
        _debug_log("CRAWL4AI_RESULT", "crawl_content_length", len(article_markdown), prefix="chars")
        _debug_log("CRAWL4AI_RESULT", "crawl_content_sample", article_markdown[:1200], prefix="first 1200 chars")
        context_parts.append(
            f"=== CRAWL4AI ARTICLE ===\n"
            f"URL: {source_url}\n"
            f"DOMAIN: {source_domain}\n"
            f"TITLE_HINT: {source_title}\n"
            f"CONTENT:\n{article_markdown}"
        )

    logger.info(
        "Research evidence | crawled_urls=%s total_crawled_chars=%s",
        len(crawled_urls),
        total_crawled_chars,
    )
    _debug_log("CRAWL4AI_SUMMARY", "total_crawled_urls", len(crawled_urls))
    _debug_log("CRAWL4AI_SUMMARY", "total_crawled_chars", total_crawled_chars)
    _debug_log("CRAWL4AI_SUMMARY", "crawled_urls", crawled_urls)

    if not context_parts:
        context_text = (
            "No crawlable content retrieved from Camoufox + Crawl4AI pipeline. "
            "Treat context as insufficient and avoid unsupported claims."
        )
    else:
        context_text = "\n\n".join(context_parts)

    return {
        "context": context_text,
        "sources": crawled_urls,
        "discovered_urls": selected_urls,
        "bot_detected_count": bot_detected_count,
    }
