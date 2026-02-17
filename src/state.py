"""
Deep Research - LangGraph State Schema
"""

from __future__ import annotations


class ResearchState(dict):
    """
    State that flows through the LangGraph.
    Using dict subclass for LangGraph compatibility.
    """
    pass


# Default state factory
def create_initial_state(
    user_query: str,
    chat_id: str = "",
    user_id: str = "",
    model: str = "",
    doc_ids: list[str] | None = None,
) -> dict:
    return {
        # Identity
        "research_id": "",
        "chat_id": chat_id,
        "user_id": user_id,

        # Input
        "user_query": user_query,
        "model": model,
        "doc_ids": doc_ids or [],

        # Context (populated by collect_context)
        "chat_history_context": "",
        "doc_summaries": "",

        # Query Analysis
        "research_mode": "",  # "simple" | "deep"
        "needs_clarification": False,
        "clarification_message": "",  # LLM's natural language message to user
        "clarification_conversation": [],  # [{role, content}, ...] full conversation log

        # Planning (todo tool)
        "todos": [],

        # Plan
        "plan": None,  # {"summary": str, "sections": [...], "estimated_time": int}
        "plan_approved": False,
        "plan_edits": {},

        # Virtual filesystem (context management)
        "vfs": {},  # serialized VirtualFileSystem

        # Research
        "findings": [],  # [{"section_id", "content", "sources", "confidence", "gaps"}]
        "all_sources": [],
        "research_loop_count": 0,
        "has_gaps": False,

        # Output
        "report": "",
        "status": "analyzing",
        "progress": [],  # list of progress messages
        "error": "",
    }
