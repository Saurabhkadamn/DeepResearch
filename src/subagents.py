"""
Deep Research - Sub-Agent Spawning
Each research section gets its own isolated LLM call with scoped context.
Pushes granular structured progress events for real-time UI streaming.
"""

import asyncio
from .llm import call_llm, parse_llm_json
from .tools.search import tavily_search, search_multiple
from .tools.documents import search_docs, get_doc
from .prompts import RESEARCH_AGENT_PROMPT


async def run_research_subagent(
    section: dict,
    doc_ids: list[str],
    model: str = "",
    event_callback=None,
) -> dict:
    """
    Run a single research sub-agent for one section.
    Pushes granular events via event_callback for real-time UI.
    """
    section_id = section.get("id", "unknown")
    title = section.get("title", "Untitled")
    description = section.get("description", "")
    search_queries = section.get("search_queries", [])
    relevant_docs = section.get("relevant_docs", [])

    async def emit(event_type, **data):
        if event_callback:
            await event_callback({"event": event_type, "section_id": section_id, "section_title": title, **data})

    await emit("section_start", queries=search_queries)

    # 1. Web search — one query at a time for granular streaming
    search_results = {}
    for query in search_queries:
        await emit("search_start", query=query)
        results = await tavily_search(query)
        search_results[query] = results

        # Emit each found URL
        for r in results:
            url = r.get("url", "")
            domain = url.split("//")[-1].split("/")[0] if url else "unknown"
            await emit("search_result", query=query, url=url, domain=domain, title=r.get("title", ""))

        await emit("search_done", query=query, count=len(results))

    # 2. Document search
    doc_extracts = []
    for doc_id in relevant_docs:
        doc = get_doc(doc_id)
        if doc:
            doc_extracts.append({
                "doc_id": doc["id"],
                "filename": doc["filename"],
                "content": doc["content"],
            })

    keyword_results = search_docs(title)
    for kr in keyword_results:
        if kr["doc_id"] not in [d["doc_id"] for d in doc_extracts]:
            doc_extracts.append(kr)

    if doc_extracts:
        await emit("docs_found", count=len(doc_extracts),
                    filenames=[d.get("filename", d.get("doc_id", "?")) for d in doc_extracts])

    # 3. Format for LLM
    search_str = _format_search_results(search_results)
    doc_str = _format_doc_extracts(doc_extracts)

    # 4. LLM analysis
    await emit("analyzing")

    prompt = RESEARCH_AGENT_PROMPT.format(
        section_title=title,
        section_description=description,
        search_results=search_str or "No search results found.",
        doc_extracts=doc_str or "No relevant documents.",
    )

    response = await call_llm(prompt, model=model)
    findings = parse_llm_json(response)

    # 5. Build sources
    sources = []
    seen_urls = set()
    for query_results in search_results.values():
        for r in query_results:
            url = r.get("url", "")
            if url and url not in seen_urls:
                seen_urls.add(url)
                sources.append({"url": url, "title": r.get("title", ""), "snippet": r.get("content", "")[:150]})

    total_results = sum(len(v) for v in search_results.values())
    confidence = findings.get("confidence", 0.5) if findings else 0.5
    content = findings.get("content", "") if findings else ""
    content_size = len(content.encode("utf-8"))

    await emit("section_done", search_count=total_results, source_count=len(sources),
               confidence=confidence, content_size=content_size,
               gaps=findings.get("gaps", []) if findings else [])

    return {
        "section_id": section_id,
        "section_title": title,
        "content": content,
        "key_points": findings.get("key_points", []) if findings else [],
        "sources": sources,
        "confidence": confidence,
        "gaps": findings.get("gaps", []) if findings else [],
        "search_count": total_results,
    }


async def run_all_subagents(
    sections: list[dict],
    doc_ids: list[str],
    model: str = "",
    progress_callback=None,
    event_callback=None,
) -> list[dict]:
    """
    Run research sub-agents for ALL sections in parallel.
    event_callback receives structured events for rich UI.
    progress_callback receives text strings for simple progress log.
    """

    # Emit initial section list
    if event_callback:
        section_list = [{"id": s.get("id", ""), "title": s.get("title", ""), "queries": s.get("search_queries", [])} for s in sections]
        await event_callback({"event": "research_start", "sections": section_list, "total": len(sections)})

    completed = 0

    async def _run_one(section):
        nonlocal completed
        section_id = section.get("id", "")
        title = section.get("title", "")

        # Also push simple progress text
        if progress_callback:
            await progress_callback(f"🔍 Researching: {title}")

        result = await run_research_subagent(section, doc_ids, model, event_callback=event_callback)

        completed += 1
        if progress_callback:
            await progress_callback(f"✅ {title}: {result.get('search_count', 0)} results, {result.get('confidence', 0):.0%} confidence")

        if event_callback:
            await event_callback({"event": "progress_update", "completed": completed, "total": len(sections)})

        return result

    tasks = [_run_one(s) for s in sections]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    findings = []
    for r in results:
        if isinstance(r, Exception):
            print(f"[ERROR] Sub-agent failed: {r}")
            continue
        findings.append(r)

    if event_callback:
        await event_callback({"event": "research_round_done", "findings_count": len(findings)})

    return findings


def _format_search_results(results: dict) -> str:
    if not results:
        return ""
    lines = []
    for query, items in results.items():
        lines.append(f"\n### Search: \"{query}\"")
        for item in items:
            lines.append(f"**[{item.get('title', '')}]({item.get('url', '')})**")
            lines.append(f"{item.get('content', '')[:300]}\n")
    return "\n".join(lines)


def _format_doc_extracts(extracts: list) -> str:
    if not extracts:
        return ""
    lines = []
    for doc in extracts:
        filename = doc.get("filename", doc.get("doc_id", "unknown"))
        content = doc.get("content", doc.get("excerpt", ""))[:400]
        lines.append(f"**📄 {filename}:**\n{content}\n")
    return "\n".join(lines)