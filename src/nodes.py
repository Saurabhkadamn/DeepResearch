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
            # asyncio.create_task() is safe inside any running async context.
            # get_event_loop() is deprecated in Python 3.10+ and run_until_complete()
            # deadlocks when called from within a running event loop.
            asyncio.ensure_future(cb(msg))
        except RuntimeError:
            pass  # no running loop — progress drop is acceptable
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
    # DO NOT overwrite research_id — api.py already set it and registered
    # the progress callback under that id. Overwriting breaks all progress streaming.
    if not state.get("research_id"):
        state["research_id"] = str(uuid.uuid4())[:8]

    _progress(state, "✅ Context ready")
    return state


# ============================================================
# NODE 3: CONTEXT BRIEF
# Runs AFTER clarification, BEFORE extended_thinking.
# Now has the fully-clarified intent to work with.
# ============================================================
async def context_brief_node(state: dict) -> dict:
    """
    Compress context into a structured brief using the CLARIFIED query.

    Runs after analyze_query (and any clarification), so we have:
    - The user's confirmed/clarified intent
    - Full clarification conversation
    - Chat history and doc summaries
    """
    state["status"] = "briefing"
    _progress(state, "📋 Building context brief...")

    history_count = state.get("_history_count", 0)

    # Include clarification conversation in history if it happened
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
            "user_intent": state["user_query"],
            "key_entities": [],
            "relevant_history": "None",
            "relevant_docs": "None",
            "implicit_constraints": [],
            "what_user_already_knows": "Unknown",
            "missing_context": "None",
        }
        _progress(state, "⚠️ Context brief failed, using defaults")
    else:
        intent = brief.get("user_intent", "")
        _progress(state, f"🎯 Intent: {intent}")

    state["context_brief"] = brief
    return state


# ============================================================
# NODE 4: EXTENDED THINKING
# Runs AFTER context_brief, BEFORE generate_plan.
# Has the clarified intent and structured brief to reason with.
# ============================================================
async def extended_thinking_node(state: dict) -> dict:
    """
    Use a reasoning model to think about the problem before searching.

    What this does that no other node does:
    - Classifies the PROBLEM TYPE (reasoning vs research vs corpus)
    - Forms a HYPOTHESIS before looking at any data
    - Defines DONE CRITERIA specific to this query
    - Identifies KEY QUESTIONS the research must answer

    This shapes everything downstream:
    - generate_plan uses key_questions as section seeds
    - synthesize_and_check uses done_criteria as the quality bar
    - write_report uses hypothesis to frame the narrative
    """
    state["status"] = "thinking"
    _progress(state, "🧠 Thinking about the problem...")

    brief = state.get("context_brief", {})

    prompt = EXTENDED_THINKING_PROMPT.format(
        query=state["user_query"],
        user_intent=brief.get("user_intent", state["user_query"]),
        key_entities=", ".join(brief.get("key_entities", [])) or "None identified",
        relevant_history=brief.get("relevant_history", "None"),
        relevant_docs=brief.get("relevant_docs", "None"),
        implicit_constraints=", ".join(brief.get("implicit_constraints", [])) or "None",
        what_user_already_knows=brief.get("what_user_already_knows", "Unknown"),
        missing_context=brief.get("missing_context", "None"),
    )

    try:
        # Always use the dedicated reasoning model — NEVER the user's fast model override.
        # A non-reasoning model passed here would silently produce poor thinking output.
        thinking_trace, final_answer = await call_llm_thinking(prompt)

        thinking_output = parse_llm_json(final_answer)

        if not thinking_output:
            thinking_output = {}
            if thinking_trace:
                thinking_output["thinking_summary"] = thinking_trace[:500]
            _progress(state, "⚠️ Thinking output parse failed, using fallback")

    except Exception as e:
        print(f"[THINKING] Reasoning model call failed: {e}")
        thinking_output = {}
        _progress(state, f"⚠️ Thinking step failed ({str(e)[:60]}), continuing...")

    # Write to state — with safe defaults if thinking failed
    state["problem_type"]     = thinking_output.get("problem_type", "research")
    state["hypothesis"]       = thinking_output.get("hypothesis", "")
    state["done_criteria"]    = thinking_output.get("done_criteria", "")
    state["thinking_summary"] = thinking_output.get("thinking_summary", "")

    state["_key_questions"]        = thinking_output.get("key_questions_to_answer", [])
    state["_recommended_approach"] = thinking_output.get("recommended_approach", "")

    problem_type  = state["problem_type"]
    hypothesis    = state["hypothesis"] if isinstance(state["hypothesis"], str) else str(state["hypothesis"])
    done_criteria = state["done_criteria"] if isinstance(state["done_criteria"], str) else str(state["done_criteria"])

    type_emoji = {"reasoning": "💭", "research": "🌐", "corpus": "📄", "hybrid": "🔀"}.get(problem_type, "🔬")
    _progress(state, f"{type_emoji} Problem type: {problem_type}")

    if hypothesis:
        _progress(state, f"💡 Hypothesis: {hypothesis[:120]}{'...' if len(hypothesis) > 120 else ''}")
    if done_criteria:
        _progress(state, f"🎯 Done when: {done_criteria[:120]}{'...' if len(done_criteria) > 120 else ''}")

    # ✅ Update status so api.py knows thinking is done and can fire thinking_complete event
    state["status"] = "planning"
    return state


# ============================================================
# NODE 2: ANALYZE QUERY
# Runs immediately after collect_context, BEFORE briefing/thinking.
# Only has access to raw query + chat history + doc summaries.
# ============================================================
async def analyze_query(state: dict) -> dict:
    """
    Classify query as simple/deep, check if clarification needed.

    Runs BEFORE context_brief and extended_thinking — no thinking fields
    exist yet. Only reads: raw query, chat history, doc summaries,
    and any clarification conversation accumulated so far.
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
        state["status"] = "briefing"   # next stop: context_brief

    return state


# ============================================================
# NODE 5: GENERATE PLAN
# Now uses hypothesis, done_criteria, key_questions from thinking.
# ============================================================
async def generate_plan(state: dict) -> dict:
    """Generate structured research plan. Updates todo list."""
    state["status"] = "planning"
    _progress(state, "📋 Creating research plan...")

    brief = state.get("context_brief", {})

    clarification_str = ""
    conversation_log = state.get("clarification_conversation", [])
    if conversation_log:
        clarification_str = "\n".join(
            [f"[{turn['role']}]: {turn['content']}" for turn in conversation_log]
        )

    # Format key_questions for prompt
    key_questions = state.get("_key_questions", [])
    key_questions_str = "\n".join([f"- {q}" for q in key_questions]) if key_questions else "Not specified"

    prompt = PLAN_GENERATOR_PROMPT.format(
        # From thinking step
        problem_type=state.get("problem_type", "research"),
        hypothesis=state.get("hypothesis", ""),
        done_criteria=state.get("done_criteria", ""),
        key_questions=key_questions_str,
        recommended_approach=state.get("_recommended_approach", ""),
        # Standard inputs
        query=state["user_query"],
        context_brief=json.dumps(brief, indent=2) if brief else "Not available",
        clarification_conversation=clarification_str or "None",
        doc_summaries=state.get("doc_summaries", "No documents"),
        # Legacy
        chat_history=state.get("chat_history_context", "No history"),
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
    est_time = plan.get("estimated_time_seconds", 90)
    _progress(state, f"📋 Plan: {section_count} sections, ~{est_time}s estimated")

    state["status"] = "awaiting_confirmation"
    return state


# ============================================================
# NODE 6: EXECUTE RESEARCH (parallel sub-agents)
# Unchanged — sub-agents don't need the thinking metadata.
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

    findings = await run_all_subagents(
        sections=sections,
        doc_ids=state.get("doc_ids", []),
        model=state.get("model", ""),
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
# Now checks against done_criteria from thinking step.
# ============================================================
async def synthesize_and_check(state: dict) -> dict:
    """Check findings for gaps against done_criteria. Loop back if needed."""
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
            section_id  = path.split("/")[-1]
            meta_content = vfs.read(f"{path}_meta")
            findings_summary += f"\n### {section_id}\n{content}\nMeta: {meta_content}\n"

    plan = state.get("plan", {})
    prompt = SYNTHESIS_PROMPT.format(
        query=state["user_query"],
        done_criteria=state.get("done_criteria", "Produce a comprehensive, well-sourced report"),
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
    coverage     = synthesis.get("done_criteria_coverage", "")

    _progress(state, f"📈 Completeness: {completeness:.0%}")
    if coverage:
        coverage_str = coverage if isinstance(coverage, str) else json.dumps(coverage)
        _progress(state, f"📋 Coverage: {coverage_str[:120]}")

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
# NODE 8: WRITE REPORT (uses quality model)
# Now passes hypothesis + done_criteria to the report writer.
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
        hypothesis=state.get("hypothesis", "Not specified"),
        done_criteria=state.get("done_criteria", "Comprehensive, well-sourced report"),
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
# NODE 9: SIMPLE ANSWER (for non-deep queries)
# ============================================================
async def simple_answer(state: dict) -> dict:
    """Direct answer for simple queries. Quick web search + LLM."""
    state["status"] = "writing"
    _progress(state, "📝 Generating answer...")

    results = await tavily_search(state["user_query"], max_results=3)
    search_context = "\n".join(
        [f"- {r['title']}: {r['content'][:200]}" for r in results]
    )

    brief = state.get("context_brief", {})
    hypothesis = state.get("hypothesis", "")

    prompt = f"""Answer this query concisely and accurately.

Query: {state["user_query"]}
User Intent: {brief.get("user_intent", state["user_query"])}
{f"Starting hypothesis: {hypothesis}" if hypothesis else ""}

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
    """Passthrough after user provides clarification answer via chat."""
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
    # Deep query, no clarification needed → go brief + think
    return "context_brief"


def route_after_clarification(state: dict) -> str:
    """
    After user provides clarification, re-run analyze_query.
    analyze_query will set needs_clarification=False once satisfied,
    which routes to context_brief and proceeds.
    """
    return "analyze_query"


def route_after_plan(state: dict) -> str:
    if state.get("status") == "error":
        return "error"
    return "wait_for_confirmation"


def route_after_synthesis(state: dict) -> str:
    if state.get("has_gaps"):
        return "execute_research"
    return "write_report"