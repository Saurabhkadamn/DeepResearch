"""
Deep Research - Web Search Tool
Tavily API for web search.
"""

import asyncio
import httpx
from ..config import TAVILY_API_KEY, MAX_SEARCH_RESULTS


async def tavily_search(query: str, max_results: int = MAX_SEARCH_RESULTS) -> list[dict]:
    """Search the web using Tavily API."""
    if not TAVILY_API_KEY:
        print("[WARN] TAVILY_API_KEY not set")
        return []

    payload = {
        "api_key": TAVILY_API_KEY,
        "query": query,
        "max_results": max_results,
        "include_answer": False,
        "include_raw_content": False,
        "search_depth": "advanced",
    }

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post("https://api.tavily.com/search", json=payload)
            response.raise_for_status()
            data = response.json()
            return [
                {
                    "url": r.get("url", ""),
                    "title": r.get("title", ""),
                    "content": r.get("content", ""),
                    "score": r.get("score", 0.0),
                }
                for r in data.get("results", [])
            ]
    except Exception as e:
        print(f"[ERROR] Tavily search failed for '{query}': {e}")
        return []


async def search_multiple(queries: list[str]) -> dict[str, list[dict]]:
    """Run multiple searches in parallel. Returns {query: results}."""

    async def _search(q):
        results = await tavily_search(q)
        return q, results

    tasks = [_search(q) for q in queries]
    raw = await asyncio.gather(*tasks, return_exceptions=True)

    results = {}
    for item in raw:
        if isinstance(item, Exception):
            print(f"[ERROR] Search failed: {item}")
            continue
        query, res = item
        results[query] = res
    return results
