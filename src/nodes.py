"""
Deep Research - LangGraph Nodes
Each function is a node in the state machine.
Uses todo planning + virtual filesystem for context management.
"""

import json
import uuid
import asyncio
from .llm import call_llm, parse_llm_json

# Real-time progress callback registry
# Maps research_id -> async callback function
_progress_callbacks = {}

def register_progress_callback(research_id: str, callback):
    _progress_callbacks[research_id] = callback

def unregister_progress_callback(research_id: str):
    _progress_callbacks.pop(research_id, None)
from .prompts import (
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
from .config import MAX_RESEARCH_LOOPS, CHAT_HISTORY_LIMIT, LLM_MODEL_QUALITY


def _progress(state: dict, msg: str) -> dict:
    """Add a progress message to state AND push via real-time callback."""
    state["progress"] = state.get("progress", []) + [msg]
    # Push to WebSocket immediately if callback registered
    rid = state.get("research_id", "")
    cb = _progress_callbacks.get(rid)
    if cb:
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.ensure_future(cb(msg))
            else:
                loop.run_until_complete(cb(msg))
        except Exception as e:
            print(f"[PROGRESS] Failed to push: {e}")
    return state


# ============================================================
# NODE 1: COLLECT CONTEXT
# ============================================================
async def collect_context(state: dict) -> dict:
    """Gather chat history and document summaries into state."""
    _progress(state, "📥 Collecting context...")

    # Chat history
    chat_id = state.get("chat_id", "")
    if chat_id:
        messages = get_messages_for_context(chat_id, CHAT_HISTORY_LIMIT)
        if messages:
            history_str = "\n".join(
                [f"[{m['role']}]: {m['content'][:200]}" for m in messages]
            )
            state["chat_history_context"] = history_str
            _progress(state, f"💬 Loaded {len(messages)} chat messages")

    # Document summaries
    state["doc_summaries"] = get_doc_summaries()

    # Init virtual filesystem
    state["vfs"] = {}

    # Init research ID
    state["research_id"] = str(uuid.uuid4())[:8]

    _progress(state, "✅ Context ready")
    return state


# ============================================================
# NODE 2: ANALYZE QUERY
# ============================================================
async def analyze_query(state: dict) -> dict:
    """Classify query as simple/deep, check if clarification needed."""
    state["status"] = "analyzing"
    _progress(state, "🧠 Analyzing query...")

    prompt = QUERY_ANALYZER_PROMPT.format(
        query=state["user_query"],
        chat_history=state.get("chat_history_context", "No history"),
        doc_summaries=state.get("doc_summaries", "No documents"),
    )

    response = await call_llm(prompt, model=state.get("model", ""))
    analysis = parse_llm_json(response)

    if not analysis:
        state["error"] = "Failed to analyze query"
        state["status"] = "error"
        return state

    state["research_mode"] = analysis.get("mode", "deep")
    state["needs_clarification"] = analysis.get("needs_clarification", False)
    state["clarification_questions"] = analysis.get("clarification_questions", [])

    reasoning = analysis.get("reasoning", "")
    _progress(state, f"🔍 {reasoning}")

    if state["research_mode"] == "simple":
        _progress(state, "📝 Simple query — generating direct answer")
    else:
        _progress(state, "🔬 Deep research required")

    if state["needs_clarification"]:
        state["status"] = "clarifying"
        _progress(state, f"❓ Need {len(state['clarification_questions'])} clarification(s)")
    else:
        state["status"] = "planning"

    return state


# ============================================================
# NODE 3: GENERATE PLAN
# ============================================================
async def generate_plan(state: dict) -> dict:
    """Generate structured research plan. Updates todo list."""
    state["status"] = "planning"
    _progress(state, "📋 Creating research plan...")

    clarification_str = ""
    if state.get("clarification_answers"):
        clarification_str = json.dumps(state["clarification_answers"])

    prompt = PLAN_GENERATOR_PROMPT.format(
        query=state["user_query"],
        clarification_answers=clarification_str or "None",
        doc_summaries=state.get("doc_summaries", "No documents"),
        chat_history=state.get("chat_history_context", "No history"),
    )

    response = await call_llm(prompt, model=state.get("model", ""))
    plan = parse_llm_json(response)

    if not plan or "sections" not in plan:
        state["error"] = "Failed to generate plan"
        state["status"] = "error"
        return state

    state["plan"] = plan

    # Create todo list from plan (deepagents pattern)
    todos = [{"id": "plan", "task": "Generate research plan", "status": "done"}]
    for section in plan["sections"]:
        todos.append({
            "id": section["id"],
            "task": f"Research: {section['title']}",
            "status": "pending",
        })
    todos.append({"id": "synthesis", "task": "Synthesize findings", "status": "pending"})
    todos.append({"id": "report", "task": "Write final report", "status": "pending"})
    state["todos"] = todos

    section_count = len(plan["sections"])
    est_time = plan.get("estimated_time_seconds", 90)
    _progress(state, f"📋 Plan: {section_count} sections, ~{est_time}s estimated")

    state["status"] = "awaiting_confirmation"
    return state


# ============================================================
# NODE 4: EXECUTE RESEARCH (parallel sub-agents)
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

    # Update todos to in_progress
    for todo in state.get("todos", []):
        if todo["status"] == "pending" and todo["id"].startswith("sec"):
            todo["status"] = "in_progress"

    # Build callbacks for real-time streaming
    rid = state.get("research_id", "")

    async def progress_cb(msg):
        _progress(state, msg)

    async def event_cb(event):
        """Push structured events directly to WebSocket for rich UI."""
        # Look up callback EACH TIME (not captured once)
        cb = _progress_callbacks.get(rid)
        if cb:
            await cb(event)
        else:
            etype = event.get("event", "?") if isinstance(event, dict) else "?"
            print(f"[EVENT_CB] {etype} for {rid}, rt_cb=NO")

    # Run all sub-agents in parallel with both callbacks
    findings = await run_all_subagents(
        sections=sections,
        doc_ids=state.get("doc_ids", []),
        model=state.get("model", ""),
        progress_callback=progress_cb,
        event_callback=event_cb,
    )

    # Write findings to virtual filesystem
    vfs = VirtualFileSystem.from_dict(state.get("vfs", {}))
    all_sources = []

    for finding in findings:
        section_id = finding["section_id"]
        content = finding.get("content", "")
        vfs.write(f"/findings/{section_id}", content)
        meta = {
            "key_points": finding.get("key_points", []),
            "confidence": finding.get("confidence", 0.5),
            "gaps": finding.get("gaps", []),
        }
        vfs.write(f"/findings/{section_id}_meta", str(meta))

        # Emit VFS write event
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

    state["findings"] = findings
    state["all_sources"] = all_sources
    state["vfs"] = vfs.to_dict()
    state["research_loop_count"] = state.get("research_loop_count", 0) + 1

    _progress(state, f"📊 Research round {state['research_loop_count']}: {len(findings)} sections done")

    return state


# ============================================================
# NODE 5: SYNTHESIZE & GAP CHECK
# ============================================================
async def synthesize_and_check(state: dict) -> dict:
    """Check findings for gaps. Loop back if needed."""
    state["status"] = "synthesizing"
    _progress(state, "🔄 Checking research completeness...")

    # Update synthesis todo
    for todo in state.get("todos", []):
        if todo["id"] == "synthesis":
            todo["status"] = "in_progress"

    # Build findings summary from virtual filesystem
    vfs = VirtualFileSystem.from_dict(state.get("vfs", {}))
    findings_files = vfs.read_all_findings()

    findings_summary = ""
    for path, content in findings_files.items():
        if not path.endswith("_meta"):
            section_id = path.split("/")[-1]
            meta_content = vfs.read(f"{path}_meta")
            findings_summary += f"\n### {section_id}\n{content}\nMeta: {meta_content}\n"

    plan = state.get("plan", {})
    prompt = SYNTHESIS_PROMPT.format(
        query=state["user_query"],
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
    is_complete = synthesis.get("is_complete", True)
    gaps = synthesis.get("gaps_to_fill", [])

    _progress(state, f"📈 Completeness: {completeness:.0%}")

    if not is_complete and gaps and state.get("research_loop_count", 1) < MAX_RESEARCH_LOOPS:
        state["has_gaps"] = True
        _progress(state, f"🔄 {len(gaps)} gap(s) found, researching more...")

        # Update plan sections with gap-filling queries
        for gap in gaps:
            section_id = gap.get("section_id", "")
            new_queries = gap.get("search_queries", [])
            if new_queries and state.get("plan"):
                for section in state["plan"]["sections"]:
                    if section["id"] == section_id:
                        section["search_queries"] = new_queries
                        break
                # Reset that section's todo
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
# NODE 6: WRITE REPORT (uses quality model)
# ============================================================
async def write_report(state: dict) -> dict:
    """Write final report from virtual filesystem findings. Uses quality model."""
    state["status"] = "writing"
    _progress(state, "✍️ Writing final report (this may take 30-60 seconds)...")
    print(f"[REPORT] Starting report generation for {state.get('research_id', '?')}")

    # Update todo
    for todo in state.get("todos", []):
        if todo["id"] == "report":
            todo["status"] = "in_progress"

    # Read all findings from virtual filesystem
    vfs = VirtualFileSystem.from_dict(state.get("vfs", {}))
    findings_files = vfs.read_all_findings()

    all_findings = ""
    for path, content in findings_files.items():
        if not path.endswith("_meta"):
            section_id = path.split("/")[-1]
            all_findings += f"\n### Section: {section_id}\n{content}\n"

    # Format sources
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
        plan_summary=plan.get("summary", ""),
        all_findings=all_findings or "No findings available",
        all_sources=sources_str or "No sources",
    )

    # Use QUALITY model for the final report
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
# NODE 7: SIMPLE ANSWER (for non-deep queries)
# ============================================================
async def simple_answer(state: dict) -> dict:
    """Direct answer for simple queries. Quick web search + LLM."""
    state["status"] = "writing"
    _progress(state, "📝 Generating answer...")

    # Quick search
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
    """Passthrough after user provides clarification answers."""
    state["needs_clarification"] = False
    state["status"] = "planning"
    _progress(state, "💬 Clarification received, continuing...")
    return state


async def confirmation_passthrough(state: dict) -> dict:
    """Passthrough after user approves/edits plan."""
    state["plan_approved"] = True
    state["status"] = "researching"

    # Apply plan edits if any
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
    return "generate_plan"


def route_after_clarification(state: dict) -> str:
    """After clarification, go straight to plan — don't re-analyze."""
    return "generate_plan"


def route_after_plan(state: dict) -> str:
    if state.get("status") == "error":
        return "error"
    return "wait_for_confirmation"


def route_after_synthesis(state: dict) -> str:
    if state.get("has_gaps"):
        return "execute_research"
    return "write_report"