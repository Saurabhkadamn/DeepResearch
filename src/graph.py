"""
Deep Research - LangGraph Graph Definition
Wires all nodes into the state machine with HITL interrupts.

Full flow:
  collect_context
      → analyze_query              classify simple/deep, check if clarification needed
      → [HITL] wait_for_clarification  (loops back to analyze_query)
      → context_brief              compress clarified context into structured brief
      → extended_thinking          reasoning model forms hypothesis + classifies problem
      → generate_plan              uses hypothesis + key_questions from thinking
      → [HITL] wait_for_confirmation
      → execute_research  ←──────────────────────────────┐
      → synthesize_and_check  (gap check vs done_criteria)┘ if gaps
      → write_report
      → END

WHY this order:
  context_brief and extended_thinking run AFTER clarification so they
  have the fully-understood, clarified intent — not the raw ambiguous query.
  The brief should capture what the user actually wants, not what they first typed.
"""

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from .nodes import (
    collect_context,
    context_brief_node,
    extended_thinking_node,
    analyze_query,
    generate_plan,
    execute_research,
    synthesize_and_check,
    write_report,
    simple_answer,
    clarification_passthrough,
    confirmation_passthrough,
    route_after_analysis,
    route_after_clarification,
    route_after_plan,
    route_after_synthesis,
)


def build_research_graph():
    graph = StateGraph(dict)

    # ── Nodes ──────────────────────────────────────────────────────────
    graph.add_node("collect_context",        collect_context)
    graph.add_node("analyze_query",          analyze_query)
    graph.add_node("wait_for_clarification", clarification_passthrough)
    graph.add_node("context_brief",          context_brief_node)
    graph.add_node("extended_thinking",      extended_thinking_node)
    graph.add_node("generate_plan",          generate_plan)
    graph.add_node("execute_research",       execute_research)
    graph.add_node("synthesize_and_check",   synthesize_and_check)
    graph.add_node("write_report",           write_report)
    graph.add_node("simple_answer",          simple_answer)
    graph.add_node("wait_for_confirmation",  confirmation_passthrough)

    # ── Entry ──────────────────────────────────────────────────────────
    graph.set_entry_point("collect_context")
    graph.add_edge("collect_context", "analyze_query")

    # ── Analysis routing ───────────────────────────────────────────────
    # simple queries skip briefing+thinking entirely (no need to reason about trivial asks)
    # deep queries: clarify first (if needed), THEN brief + think + plan
    graph.add_conditional_edges(
        "analyze_query",
        route_after_analysis,
        {
            "simple_answer":          "simple_answer",
            "wait_for_clarification": "wait_for_clarification",
            "context_brief":          "context_brief",   # deep, no clarification needed
            "error":                  END,
        },
    )

    graph.add_edge("simple_answer", END)

    # After clarification → re-run analyze_query with enriched context
    graph.add_conditional_edges(
        "wait_for_clarification",
        route_after_clarification,
        {
            "analyze_query": "analyze_query",
        },
    )

    # ── Brief → Think → Plan pipeline (runs with clarified intent) ─────
    graph.add_edge("context_brief",     "extended_thinking")
    graph.add_edge("extended_thinking", "generate_plan")

    # ── Plan → confirmation ────────────────────────────────────────────
    graph.add_conditional_edges(
        "generate_plan",
        route_after_plan,
        {
            "wait_for_confirmation": "wait_for_confirmation",
            "error":                 END,
        },
    )

    graph.add_edge("wait_for_confirmation", "execute_research")

    # ── Research loop ──────────────────────────────────────────────────
    graph.add_edge("execute_research", "synthesize_and_check")

    graph.add_conditional_edges(
        "synthesize_and_check",
        route_after_synthesis,
        {
            "execute_research": "execute_research",
            "write_report":     "write_report",
        },
    )

    graph.add_edge("write_report", END)

    # ── Compile with HITL interrupts ───────────────────────────────────
    checkpointer = MemorySaver()
    compiled = graph.compile(
        checkpointer=checkpointer,
        interrupt_before=["wait_for_clarification", "wait_for_confirmation"],
    )

    return compiled


# Singleton
_graph = None

def get_graph():
    global _graph
    if _graph is None:
        _graph = build_research_graph()
    return _graph