"""
Deep Research - Document Store
Pre-loaded test documents. Replace with real upload handling later.
"""

# Pre-loaded test documents (simulates user uploads)
PRELOADED_DOCS = {
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
