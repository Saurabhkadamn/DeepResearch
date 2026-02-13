"""
Deep Research - LangGraph Graph Definition
Wires all nodes into the state machine with HITL interrupts.
"""

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from .nodes import (
    collect_context,
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
    """
    Build the LangGraph state machine.

    Flow:
    collect_context → analyze_query ─┬→ simple_answer → END
                                     ├→ wait_for_clarification → generate_plan
                                     └→ generate_plan → wait_for_confirmation
                                            → execute_research ←──┐
                                            → synthesize_and_check ┘ (if gaps)
                                            → write_report → END
    """
    graph = StateGraph(dict)

    # Nodes
    graph.add_node("collect_context", collect_context)
    graph.add_node("analyze_query", analyze_query)
    graph.add_node("generate_plan", generate_plan)
    graph.add_node("execute_research", execute_research)
    graph.add_node("synthesize_and_check", synthesize_and_check)
    graph.add_node("write_report", write_report)
    graph.add_node("simple_answer", simple_answer)
    graph.add_node("wait_for_clarification", clarification_passthrough)
    graph.add_node("wait_for_confirmation", confirmation_passthrough)

    # Entry
    graph.set_entry_point("collect_context")

    # Edges
    graph.add_edge("collect_context", "analyze_query")

    graph.add_conditional_edges(
        "analyze_query",
        route_after_analysis,
        {
            "simple_answer": "simple_answer",
            "wait_for_clarification": "wait_for_clarification",
            "generate_plan": "generate_plan",
            "error": END,
        },
    )

    graph.add_edge("simple_answer", END)

    # After clarification → go straight to plan (don't re-analyze)
    graph.add_conditional_edges(
        "wait_for_clarification",
        route_after_clarification,
        {
            "generate_plan": "generate_plan",
        },
    )

    graph.add_conditional_edges(
        "generate_plan",
        route_after_plan,
        {
            "wait_for_confirmation": "wait_for_confirmation",
            "error": END,
        },
    )

    graph.add_edge("wait_for_confirmation", "execute_research")
    graph.add_edge("execute_research", "synthesize_and_check")

    graph.add_conditional_edges(
        "synthesize_and_check",
        route_after_synthesis,
        {
            "execute_research": "execute_research",
            "write_report": "write_report",
        },
    )

    graph.add_edge("write_report", END)

    # Compile with checkpointer and HITL interrupts
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
