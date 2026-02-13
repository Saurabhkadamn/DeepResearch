"""
Deep Research - Document Store
Pre-loaded test documents. Replace with real upload handling later.
"""

# Pre-loaded test documents (simulates user uploads)
PRELOADED_DOCS = {
    "doc_001": {
        "id": "doc_001",
        "filename": "product_roadmap.pdf",
        "content": (
            "AI Platform Roadmap 2025:\n"
            "Q1 - Launch deep research feature with multi-model support.\n"
            "Q2 - Multi-agent workflows and custom agent builder.\n"
            "Q3 - Custom model fine-tuning pipeline for enterprise clients.\n"
            "Q4 - Enterprise analytics dashboard with usage insights.\n"
            "Key focus areas: education sector, K-12 districts, higher education.\n"
            "LMS integration targets: Canvas, Blackboard, Moodle.\n"
            "Governance: SOC2 compliance, GDPR, FERPA for education data."
        ),
    },
    "doc_002": {
        "id": "doc_002",
        "filename": "competitor_analysis.docx",
        "content": (
            "Competitor Analysis - AI in Education:\n"
            "1. Cognii - AI-powered tutoring, NLP-based assessment. Funding: $10M.\n"
            "2. Century Tech - Personalized learning paths, UK-based. 1M+ students.\n"
            "3. Squirrel AI - Adaptive learning, China-based. $150M funding.\n"
            "4. Carnegie Learning - Math-focused AI tutoring. US K-12 market.\n"
            "5. Khanmigo (Khan Academy) - GPT-4 powered tutor. Free tier available.\n"
            "Our differentiators: Multi-model support (Claude, GPT, Gemini), "
            "enterprise governance, LMS connectors, no-code agent builder."
        ),
    },
    "doc_003": {
        "id": "doc_003",
        "filename": "meeting_notes_jan2025.md",
        "content": (
            "# Team Meeting Notes - January 2025\n\n"
            "## Key Decisions:\n"
            "- Deep research feature is top priority for Q1\n"
            "- Will use LangGraph for agent orchestration\n"
            "- OpenRouter for multi-model access during development\n"
            "- Tavily for web search API\n"
            "- Target: MVP demo by end of February\n\n"
            "## Open Questions:\n"
            "- How to handle document context in research queries?\n"
            "- What's the right UX for plan confirmation?\n"
            "- Should we support real-time collaboration on research?"
        ),
    },
}


def get_all_docs() -> list[dict]:
    """Get all pre-loaded documents."""
    return list(PRELOADED_DOCS.values())


def get_doc(doc_id: str) -> dict | None:
    """Get a specific document by ID."""
    return PRELOADED_DOCS.get(doc_id)


def get_doc_summaries() -> str:
    """Get formatted document summaries for LLM context."""
    if not PRELOADED_DOCS:
        return "No documents available."
    lines = []
    for doc in PRELOADED_DOCS.values():
        preview = doc["content"][:150].replace("\n", " ")
        lines.append(f"- [{doc['id']}] {doc['filename']}: {preview}...")
    return "\n".join(lines)


def search_docs(query: str) -> list[dict]:
    """Simple keyword search across documents."""
    query_lower = query.lower()
    results = []
    for doc in PRELOADED_DOCS.values():
        if query_lower in doc["content"].lower():
            results.append({
                "doc_id": doc["id"],
                "filename": doc["filename"],
                "excerpt": doc["content"][:300],
            })
    return results
