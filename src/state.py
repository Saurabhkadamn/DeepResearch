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

        # Context Brief (produced by context_brief node)
        # Shape: {
        #   "conversation_summary": str,
        #   "user_background": str,
        #   "relevant_docs": str,
        # }
        "context_brief": {},

        # Extended Thinking (produced by extended_thinking node)
        # Free-form reasoning text from the reasoning model.
        # NOT JSON — the planner reads this text directly.
        "thinking": "",        # full reasoning text
        "thinking_trace": "",  # chain-of-thought (internal monologue from reasoning model)

        # Query Analysis
        "research_mode": "",           # "simple" | "deep"
        "needs_clarification": False,
        "clarification_message": "",
        "clarification_conversation": [],  # [{role, content}, ...]

        # Planning
        "todos": [],
        "plan": None,
        "plan_approved": False,
        "plan_edits": {},

        # Virtual filesystem (context management)
        "vfs": {},

        # Research
        "findings": [],
        "all_sources": [],
        "research_loop_count": 0,
        "has_gaps": False,

        # Output
        "report": "",
        "status": "analyzing",
        "progress": [],
        "error": "",
    }