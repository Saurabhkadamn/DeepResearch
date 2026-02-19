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

        # ── NEW: Context Brief ──────────────────────────────────────────
        # Structured compression of all raw context, produced by context_brief node.
        # Shape: {
        #   "user_intent": str,
        #   "key_entities": [str],
        #   "relevant_history": str,
        #   "relevant_docs": str,
        #   "implicit_constraints": [str],
        #   "what_user_already_knows": str
        # }
        "context_brief": {},

        # ── NEW: Extended Thinking ──────────────────────────────────────
        # Produced by extended_thinking node before analysis.
        # problem_type: "reasoning" | "research" | "corpus"
        #   - reasoning: answer lives in model knowledge, light search only
        #   - research:  answer needs heavy web research (current data)
        #   - corpus:    answer lives in uploaded docs, analyze those
        "problem_type": "",
        "hypothesis": "",       # model's best guess before searching
        "done_criteria": "",    # what "complete" looks like for THIS query
        "thinking_summary": "", # compressed output passed to downstream nodes

        # Query Analysis
        "research_mode": "",  # "simple" | "deep"
        "needs_clarification": False,
        "clarification_message": "",
        "clarification_conversation": [],  # [{role, content}, ...]

        # Planning (todo tool)
        "todos": [],

        # Plan
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