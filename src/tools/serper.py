# pip install httpx trafilatura crawl4ai
# crawl4ai-setup  (one-time, installs Playwright browsers)

# httpx>=0.27.0
# trafilatura>=1.9.0
# crawl4ai>=0.8.0

import asyncio
import os
from dataclasses import dataclass
from typing import Optional

import httpx
import trafilatura

# ── Config ────────────────────────────────────────────────────────────────────

SERPER_API_KEY       = os.getenv("SERPER_API_KEY", "")
SERPER_URL           = "https://google.serper.dev/search"
HTTPX_CONCURRENCY    = 20
CRAWL4AI_CONCURRENCY = 5
MIN_CONTENT_LENGTH   = 300
HTTPX_TIMEOUT        = 10
CRAWL4AI_TIMEOUT     = 20

SKIP_PATTERNS = [
    "youtube.com", "youtu.be",
    "twitter.com", "x.com",
    "instagram.com", "tiktok.com",
    "facebook.com", "linkedin.com",
    "pinterest.com", ".pdf",
]

REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}

# ── Models ────────────────────────────────────────────────────────────────────

@dataclass
class SearchResult:
    title:    str
    url:      str
    snippet:  str
    position: int

@dataclass
class FetchedResult:
    title:    str
    url:      str
    content:  str
    source:   str          # "scraped" | "crawl4ai" | "snippet"
    position: int
    success:  bool
    error:    Optional[str] = None

# ── Serper ────────────────────────────────────────────────────────────────────

async def serper_search(
    query: str,
    max_results: int,
    client: httpx.AsyncClient,
) -> list[SearchResult]:
    try:
        resp = await client.post(
            SERPER_URL,
            json={"q": query, "num": max_results},
            headers={"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"},
            timeout=10,
        )
        resp.raise_for_status()
        return [
            SearchResult(
                title    = item.get("title", ""),
                url      = item.get("link", ""),
                snippet  = item.get("snippet", ""),
                position = item.get("position", 0),
            )
            for item in resp.json().get("organic", [])[:max_results]
        ]
    except Exception:
        return []

# ── URL Filter ────────────────────────────────────────────────────────────────

def should_skip(url: str) -> bool:
    url_lower = url.lower()
    return any(p in url_lower for p in SKIP_PATTERNS)

# ── httpx + Trafilatura ───────────────────────────────────────────────────────

async def fetch_with_httpx(url: str, client: httpx.AsyncClient) -> Optional[str]:
    try:
        resp = await client.get(
            url,
            headers=REQUEST_HEADERS,
            timeout=HTTPX_TIMEOUT,
            follow_redirects=True,
        )
        if resp.status_code != 200:
            return None
        text = trafilatura.extract(
            resp.text,
            include_comments=False,
            include_tables=True,
            no_fallback=False,
        )
        return text if text and len(text) >= MIN_CONTENT_LENGTH else None
    except Exception:
        return None

# ── Crawl4AI Fallback ─────────────────────────────────────────────────────────

async def fetch_with_crawl4ai(url: str) -> Optional[str]:
    try:
        from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig
        browser_cfg = BrowserConfig(headless=True, verbose=False)
        run_cfg     = CrawlerRunConfig(page_timeout=CRAWL4AI_TIMEOUT * 1000)
        async with AsyncWebCrawler(config=browser_cfg) as crawler:
            result = await crawler.arun(url=url, config=run_cfg)
        if result.success and result.markdown:
            text = result.markdown.strip()
            return text if len(text) >= MIN_CONTENT_LENGTH else None
        return None
    except Exception:
        return None

# ── Per-URL Fetch Pipeline ────────────────────────────────────────────────────

async def fetch_url(
    result: SearchResult,
    client: httpx.AsyncClient,
    crawl4ai_sem: asyncio.Semaphore,
) -> FetchedResult:
    if should_skip(result.url):
        return FetchedResult(
            title    = result.title,
            url      = result.url,
            content  = result.snippet,
            source   = "snippet",
            position = result.position,
            success  = False,
            error    = "skipped: non-text URL",
        )

    text = await fetch_with_httpx(result.url, client)
    if text:
        return FetchedResult(
            title    = result.title,
            url      = result.url,
            content  = text,
            source   = "scraped",
            position = result.position,
            success  = True,
        )

    async with crawl4ai_sem:
        text = await fetch_with_crawl4ai(result.url)
    if text:
        return FetchedResult(
            title    = result.title,
            url      = result.url,
            content  = text,
            source   = "crawl4ai",
            position = result.position,
            success  = True,
        )

    return FetchedResult(
        title    = result.title,
        url      = result.url,
        content  = result.snippet,
        source   = "snippet",
        position = result.position,
        success  = False,
        error    = "all fetch methods failed",
    )

# ── Parallel Fetch ────────────────────────────────────────────────────────────

async def fetch_all(search_results: list[SearchResult]) -> list[FetchedResult]:
    httpx_sem    = asyncio.Semaphore(HTTPX_CONCURRENCY)
    crawl4ai_sem = asyncio.Semaphore(CRAWL4AI_CONCURRENCY)

    async with httpx.AsyncClient() as client:
        async def bounded_fetch(result: SearchResult) -> FetchedResult:
            async with httpx_sem:
                return await fetch_url(result, client, crawl4ai_sem)

        return await asyncio.gather(*[bounded_fetch(r) for r in search_results])

# ── Main Pipeline ─────────────────────────────────────────────────────────────

async def run_pipeline(query: str, max_results: int = 5) -> list[FetchedResult]:
    async with httpx.AsyncClient() as client:
        search_results = await serper_search(query, max_results, client)

    if not search_results:
        return []

    return await fetch_all(search_results)
