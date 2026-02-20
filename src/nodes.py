"""
Deep Research - LangGraph Nodes
Each function is a node in the state machine.

Node execution order:
  collect_context → analyze_query → [clarify?] → context_brief
  → extended_thinking → generate_plan → execute_research
  → synthesize_and_check → write_report

WHY context_brief and extended_thinking come AFTER clarification:
  We brief and think on the CLARIFIED intent, not the raw ambiguous query.
  There is no point forming a hypothesis before we know what the user wants.
"""

import json
import uuid
import asyncio
from .llm import call_llm, call_llm_thinking, parse_llm_json
from .config import MAX_RESEARCH_LOOPS, CHAT_HISTORY_LIMIT, LLM_MODEL_QUALITY

# Real-time progress callback registry
# Maps research_id -> async callback function
_progress_callbacks = {}

def register_progress_callback(research_id: str, callback):
    _progress_callbacks[research_id] = callback

def unregister_progress_callback(research_id: str):
    _progress_callbacks.pop(research_id, None)

from .prompts import (
    CONTEXT_BRIEF_PROMPT,
    EXTENDED_THINKING_PROMPT,
    QUERY_ANALYZER_PROMPT,
    PLAN_GENERATOR_PROMPT,
    SYNTHESIS_PROMPT,
    REPORT_WRITER_PROMPT,
)
from .tools.todo import format_todos_for_prompt
from .tools.filesystem import VirtualFileSystem
from .tools.documents import get_doc_summaries, search_docs
from .tools.search import tavily_search
from .subagents import run_all_subagents
from .history import get_messages_for_context


def _progress(state: dict, msg: str) -> dict:
    """Add a progress message to state AND push via real-time callback."""
    state["progress"] = state.get("progress", []) + [msg]
    rid = state.get("research_id", "")
    cb = _progress_callbacks.get(rid)
    if cb:
        try:
            asyncio.ensure_future(cb(msg))
        except RuntimeError:
            pass
    return state


# ============================================================
# NODE 1: COLLECT CONTEXT
# ============================================================
async def collect_context(state: dict) -> dict:
    """Gather chat history and document summaries into state."""
    state["status"] = "collecting"
    _progress(state, "📥 Collecting context...")

    chat_id = state.get("chat_id", "")
    if chat_id:
        messages = get_messages_for_context(chat_id, CHAT_HISTORY_LIMIT)
        if messages:
            history_str = "\n".join(
                [f"[{m['role']}]: {m['content'][:200]}" for m in messages]
            )
            state["chat_history_context"] = history_str
            state["_history_count"] = len(messages)
            _progress(state, f"💬 Loaded {len(messages)} chat messages")

    state["doc_summaries"] = get_doc_summaries()
    state["vfs"] = {}

    # Use full UUID — 8-char truncation has collision risk
    if not state.get("research_id"):
        state["research_id"] = str(uuid.uuid4())

    _progress(state, "✅ Context ready")
    return state


# ============================================================
# NODE 2: ANALYZE QUERY
# Runs immediately after collect_context, BEFORE briefing/thinking.
# ============================================================
async def analyze_query(state: dict) -> dict:
    """
    Classify query as simple/deep, check if clarification needed.
    Runs BEFORE context_brief and extended_thinking.
    """
    state["status"] = "analyzing"
    _progress(state, "🔍 Analyzing query...")

    clarification_history = ""
    conversation_log = state.get("clarification_conversation", [])
    if conversation_log:
        clarification_history = "\n".join(
            [f"[{turn['role']}]: {turn['content']}" for turn in conversation_log]
        )

    prompt = QUERY_ANALYZER_PROMPT.format(
        query=state["user_query"],
        clarification_conversation=clarification_history or "No clarification conversation yet",
        doc_summaries=state.get("doc_summaries", "No documents"),
        chat_history=state.get("chat_history_context", "No history"),
    )

    response = await call_llm(prompt, model=state.get("model", ""))
    analysis = parse_llm_json(response)

    if not analysis:
        state["error"] = "Failed to analyze query"
        state["status"] = "error"
        return state

    state["research_mode"]         = analysis.get("mode", "deep")
    state["needs_clarification"]   = analysis.get("needs_clarification", False)
    state["clarification_message"] = analysis.get("clarification_message", "")

    reasoning = analysis.get("reasoning", "")
    if reasoning:
        _progress(state, f"💭 {reasoning}")

    if state["research_mode"] == "simple":
        _progress(state, "📝 Simple query — generating direct answer")
    else:
        _progress(state, "🔬 Deep research required")

    if state["needs_clarification"]:
        state["status"] = "clarifying"
        _progress(state, "❓ Need more details from user")
    else:
        state["status"] = "briefing"

    return state


# ============================================================
# NODE 3: CONTEXT BRIEF
# Runs AFTER clarification, BEFORE extended_thinking.
# ============================================================
async def context_brief_node(state: dict) -> dict:
    """
    Summarize conversation history into clean background context.
    Single job: what has been discussed. No interpretation, no intent extraction.
    """
    state["status"] = "briefing"
    _progress(state, "📋 Building context brief...")

    history_count = state.get("_history_count", 0)

    clarification_str = ""
    conversation_log = state.get("clarification_conversation", [])
    if conversation_log:
        clarification_str = "\n\nClarification conversation:\n" + "\n".join(
            [f"[{t['role']}]: {t['content']}" for t in conversation_log]
        )

    prompt = CONTEXT_BRIEF_PROMPT.format(
        query=state["user_query"],
        history_count=history_count,
        chat_history=state.get("chat_history_context", "No prior conversation.") + clarification_str,
        doc_summaries=state.get("doc_summaries", "No documents available."),
    )

    response = await call_llm(prompt, model=state.get("model", ""))
    brief = parse_llm_json(response)

    if not brief:
        brief = {
            "conversation_summary": "No prior conversation",
            "user_background": "Unknown",
            "relevant_docs": "None",
        }
        _progress(state, "⚠️ Context brief failed, using defaults")
    else:
        summary = brief.get("conversation_summary", "")
        _progress(state, f"📋 Context: {summary[:80]}")

    state["context_brief"] = brief
    return state


# ============================================================
# NODE 4: EXTENDED THINKING
# Runs AFTER context_brief, BEFORE generate_plan.
# Free-form reasoning — output is text, not JSON.
# ============================================================
async def extended_thinking_node(state: dict) -> dict:
    """
    Use a reasoning model to think deeply about the problem before any research.
    Output is free-form text — no JSON parsing.
    The planner reads this thinking and turns it into a research plan.
    """
    state["status"] = "thinking"
    _progress(state, "🧠 Thinking about the problem...")

    brief = state.get("context_brief", {})

    prompt = EXTENDED_THINKING_PROMPT.format(
        query=state["user_query"],
        conversation_summary=brief.get("conversation_summary", "No prior conversation"),
        relevant_docs=brief.get("relevant_docs", "None"),
    )

    try:
        # Always use the dedicated reasoning model — never the fast model
        thinking_trace, final_answer = await call_llm_thinking(prompt)

        # Store raw text — thinking is free-form, not JSON
        state["thinking"]       = final_answer
        state["thinking_trace"] = thinking_trace

        preview = final_answer[:120] if final_answer else ""
        _progress(state, f"🧠 Thinking complete: {preview}{'...' if len(final_answer) > 120 else ''}")

    except Exception as e:
        print(f"[THINKING] Reasoning model call failed: {e}")
        state["thinking"]       = ""
        state["thinking_trace"] = ""
        _progress(state, f"⚠️ Thinking step failed ({str(e)[:60]}), continuing...")

    # Update status so api.py fires thinking_complete event
    state["status"] = "planning"
    return state


# ============================================================
# NODE 5: GENERATE PLAN
# Reads researcher's thinking and turns it into a concrete plan.
# ============================================================
async def generate_plan(state: dict) -> dict:
    """Generate structured research plan driven by extended thinking output."""
    state["status"] = "planning"
    _progress(state, "📋 Creating research plan...")

    brief = state.get("context_brief", {})

    prompt = PLAN_GENERATOR_PROMPT.format(
        query=state["user_query"],
        conversation_summary=brief.get("conversation_summary", "No prior conversation"),
        thinking=state.get("thinking", ""),
        relevant_docs=state.get("doc_summaries", "No documents"),
    )

    response = await call_llm(prompt, model=state.get("model", ""))
    plan = parse_llm_json(response)

    if not plan or "sections" not in plan:
        state["error"] = "Failed to generate plan"
        state["status"] = "error"
        return state

    state["plan"] = plan

    todos = [{"id": "plan", "task": "Generate research plan", "status": "done"}]
    for section in plan["sections"]:
        todos.append({
            "id": section["id"],
            "task": f"Research: {section['title']}",
            "status": "pending",
        })
    todos.append({"id": "synthesis", "task": "Synthesize findings", "status": "pending"})
    todos.append({"id": "report",    "task": "Write final report",  "status": "pending"})
    state["todos"] = todos

    section_count = len(plan["sections"])
    _progress(state, f"📋 Plan: {section_count} sections ready")

    state["status"] = "awaiting_confirmation"
    return state


# ============================================================
# NODE 6: EXECUTE RESEARCH (parallel sub-agents)
# ============================================================
async def execute_research(state: dict) -> dict:
    """Run parallel sub-agents for each section. Write to virtual filesystem."""
    state["status"] = "researching"
    _progress(state, "⚡ Starting research...")

    plan = state.get("plan", {})
    sections = plan.get("sections", [])

    if not sections:
        state["error"] = "No sections to research"
        state["status"] = "error"
        return state

    for todo in state.get("todos", []):
        if todo["status"] == "pending" and todo["id"].startswith("sec"):
            todo["status"] = "in_progress"

    rid = state.get("research_id", "")

    async def progress_cb(msg):
        _progress(state, msg)

    async def event_cb(event):
        cb = _progress_callbacks.get(rid)
        if cb:
            await cb(event)

    # Pass thinking to sub-agents so they know the broader research goal
    thinking_summary = state.get("thinking", "")[:500]

    findings = await run_all_subagents(
        sections=sections,
        doc_ids=state.get("doc_ids", []),
        model=state.get("model", ""),
        thinking_summary=thinking_summary,
        progress_callback=progress_cb,
        event_callback=event_cb,
    )

    vfs = VirtualFileSystem.from_dict(state.get("vfs", {}))
    all_sources = []

    for finding in findings:
        section_id = finding["section_id"]
        content    = finding.get("content", "")
        vfs.write(f"/findings/{section_id}", content)
        meta = {
            "key_points": finding.get("key_points", []),
            "confidence": finding.get("confidence", 0.5),
            "gaps":       finding.get("gaps", []),
        }
        vfs.write(f"/findings/{section_id}_meta", str(meta))

        content_size = len(content.encode("utf-8")) if content else 0
        await event_cb({
            "event": "vfs_write",
            "section_id": section_id,
            "path": f"/findings/{section_id}",
            "size": content_size,
        })

        all_sources.extend(finding.get("sources", []))

        for todo in state.get("todos", []):
            if todo["id"] == section_id:
                todo["status"] = "done"

    state["findings"]            = findings
    state["all_sources"]         = all_sources
    state["vfs"]                 = vfs.to_dict()
    state["research_loop_count"] = state.get("research_loop_count", 0) + 1

    _progress(state, f"📊 Research round {state['research_loop_count']}: {len(findings)} sections done")
    return state


# ============================================================
# NODE 7: SYNTHESIZE & GAP CHECK
# ============================================================
async def synthesize_and_check(state: dict) -> dict:
    """Check findings completeness against the researcher's thinking."""
    state["status"] = "synthesizing"
    _progress(state, "🔄 Checking research completeness...")

    for todo in state.get("todos", []):
        if todo["id"] == "synthesis":
            todo["status"] = "in_progress"

    vfs = VirtualFileSystem.from_dict(state.get("vfs", {}))
    findings_files = vfs.read_all_findings()

    findings_summary = ""
    for path, content in findings_files.items():
        if not path.endswith("_meta"):
            section_id   = path.split("/")[-1]
            meta_content = vfs.read(f"{path}_meta")
            findings_summary += f"\n### {section_id}\n{content}\nMeta: {meta_content}\n"

    plan = state.get("plan", {})
    prompt = SYNTHESIS_PROMPT.format(
        query=state["user_query"],
        thinking=state.get("thinking", "Produce a comprehensive, well-sourced report"),
        plan_summary=plan.get("summary", ""),
        findings_summary=findings_summary or "No findings yet",
        max_loops=MAX_RESEARCH_LOOPS,
        current_loop=state.get("research_loop_count", 1),
    )

    response = await call_llm(prompt, model=state.get("model", ""))
    synthesis = parse_llm_json(response)

    if not synthesis:
        state["has_gaps"] = False
        _progress(state, "⚠️ Synthesis check skipped, proceeding to report")
        return state

    completeness = synthesis.get("completeness_score", 0.8)
    is_complete  = synthesis.get("is_complete", True)
    gaps         = synthesis.get("gaps_to_fill", [])

    _progress(state, f"📈 Completeness: {completeness:.0%}")

    if not is_complete and gaps and state.get("research_loop_count", 1) < MAX_RESEARCH_LOOPS:
        state["has_gaps"] = True
        _progress(state, f"🔄 {len(gaps)} gap(s) found, researching more...")

        for gap in gaps:
            section_id  = gap.get("section_id", "")
            new_queries = gap.get("search_queries", [])
            if new_queries and state.get("plan"):
                for section in state["plan"]["sections"]:
                    if section["id"] == section_id:
                        section["search_queries"] = new_queries
                        break
                for todo in state.get("todos", []):
                    if todo["id"] == section_id:
                        todo["status"] = "pending"
    else:
        state["has_gaps"] = False
        for todo in state.get("todos", []):
            if todo["id"] == "synthesis":
                todo["status"] = "done"

        if state.get("research_loop_count", 1) >= MAX_RESEARCH_LOOPS:
            _progress(state, f"⏹️ Max research rounds ({MAX_RESEARCH_LOOPS}) reached")
        else:
            _progress(state, "✅ Research complete, writing report...")

    return state


# ============================================================
# NODE 8: WRITE REPORT
# ============================================================
async def write_report(state: dict) -> dict:
    """Write final report from virtual filesystem findings. Uses quality model."""
    state["status"] = "writing"
    _progress(state, "✍️ Writing final report (this may take 30-60 seconds)...")
    print(f"[REPORT] Starting report generation for {state.get('research_id', '?')}")

    for todo in state.get("todos", []):
        if todo["id"] == "report":
            todo["status"] = "in_progress"

    vfs = VirtualFileSystem.from_dict(state.get("vfs", {}))
    findings_files = vfs.read_all_findings()

    all_findings = ""
    for path, content in findings_files.items():
        if not path.endswith("_meta"):
            section_id    = path.split("/")[-1]
            all_findings += f"\n### Section: {section_id}\n{content}\n"

    sources = state.get("all_sources", [])
    seen = set()
    sources_str = ""
    for i, s in enumerate(sources, 1):
        url = s.get("url", "")
        if url and url not in seen:
            seen.add(url)
            sources_str += f"{i}. [{s.get('title', 'Source')}]({url})\n"

    plan = state.get("plan", {})
    prompt = REPORT_WRITER_PROMPT.format(
        query=state["user_query"],
        thinking=state.get("thinking", ""),
        plan_summary=plan.get("summary", ""),
        all_findings=all_findings or "No findings available",
        all_sources=sources_str or "No sources",
    )

    report = await call_llm(
        prompt,
        model=LLM_MODEL_QUALITY,
        system_prompt="You are an expert research report writer. Write in Markdown.",
        temperature=0.4,
        max_tokens=8192,
    )

    state["report"] = report
    state["status"] = "completed"

    print(f"[REPORT] Done! {len(report)} chars")
    for todo in state.get("todos", []):
        if todo["id"] == "report":
            todo["status"] = "done"

    _progress(state, "📊 Report complete!")
    return state


# ============================================================
# NODE 9: SIMPLE ANSWER
# ============================================================
async def simple_answer(state: dict) -> dict:
    """Direct answer for simple queries. Quick web search + LLM."""
    state["status"] = "writing"
    _progress(state, "📝 Generating answer...")

    results = await tavily_search(state["user_query"], max_results=3)
    search_context = "\n".join(
        [f"- {r['title']}: {r['content'][:200]}" for r in results]
    )

    prompt = f"""Answer this query concisely and accurately.

Query: {state["user_query"]}

Web Search Results:
{search_context or "None"}

Available Documents:
{state.get("doc_summaries", "None")}

Chat History:
{state.get("chat_history_context", "None")}

Provide a clear, well-formatted answer in Markdown."""

    report = await call_llm(
        prompt,
        model=state.get("model", ""),
        system_prompt="You are a helpful AI assistant. Give clear, concise answers.",
        temperature=0.3,
    )

    state["report"] = report
    state["status"] = "completed"
    _progress(state, "✅ Done")
    return state


# ============================================================
# HITL PASSTHROUGH NODES
# ============================================================
async def clarification_passthrough(state: dict) -> dict:
    """Passthrough after user provides clarification answer."""
    state["needs_clarification"] = False
    state["status"] = "analyzing"
    _progress(state, "💬 Got your response, re-analyzing...")
    return state


async def confirmation_passthrough(state: dict) -> dict:
    """Passthrough after user approves/edits plan."""
    state["plan_approved"] = True
    state["status"] = "researching"

    edits = state.get("plan_edits", {})
    if edits and state.get("plan"):
        added = edits.get("add_sections", [])
        for title in added:
            state["plan"]["sections"].append({
                "id": f"sec_added_{len(state['plan']['sections'])+1}",
                "title": title,
                "description": f"User-requested: {title}",
                "search_queries": [title],
                "relevant_docs": [],
            })
            state["todos"].insert(-2, {
                "id": f"sec_added_{len(state['plan']['sections'])}",
                "task": f"Research: {title}",
                "status": "pending",
            })

    _progress(state, "✅ Plan approved! Starting research...")
    return state


# ============================================================
# ROUTING FUNCTIONS
# ============================================================
def route_after_analysis(state: dict) -> str:
    if state.get("status") == "error":
        return "error"
    if state.get("research_mode") == "simple":
        return "simple_answer"
    if state.get("needs_clarification"):
        return "wait_for_clarification"
    return "context_brief"


def route_after_clarification(state: dict) -> str:
    return "analyze_query"


def route_after_plan(state: dict) -> str:
    if state.get("status") == "error":
        return "error"
    return "wait_for_confirmation"


def route_after_synthesis(state: dict) -> str:
    if state.get("has_gaps"):
        return "execute_research"
    return "write_report"