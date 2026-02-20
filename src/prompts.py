"""
Deep Research - LLM Prompts
Detailed, example-rich prompts following deepagents best practices.

Node → Prompt mapping (in execution order):
  collect_context      → (no LLM call)
  analyze_query        → QUERY_ANALYZER_PROMPT
  context_brief        → CONTEXT_BRIEF_PROMPT
  extended_thinking    → EXTENDED_THINKING_PROMPT
  generate_plan        → PLAN_GENERATOR_PROMPT
  execute_research     → RESEARCH_AGENT_PROMPT  (in subagents.py)
  synthesize_and_check → SYNTHESIS_PROMPT
  write_report         → REPORT_WRITER_PROMPT
"""


# ============================================================
# CONTEXT BRIEF
# Summarizes conversation history into clean background context.
# Called BEFORE extended_thinking so the reasoning model gets
# factual background — not interpretation.
# ============================================================
CONTEXT_BRIEF_PROMPT = """You are a conversation summarizer.
Your only job is to summarize the recent chat history so that
a reasoning model can use it as background context.

## User Query:
{query}

## Chat History (last {history_count} messages):
{chat_history}

## Available Documents:
{doc_summaries}

## Your Rules:
1. Summarize what has been discussed — nothing more
2. Do NOT interpret the current query
3. Do NOT extract intent, constraints, or entities
4. Do NOT add any direction or angle
5. Just give factual background: who asked what, what was answered

## What to capture:

**conversation_summary** — What has been discussed in recent messages.
  If no history, write "No prior conversation."

**user_background** — Any evidence of who the user is or their expertise
  from the conversation. Write "Unknown" if no evidence.

**relevant_docs** — Which uploaded documents exist that may be useful.
  Just name them. Write "None" if no documents.

## Response Format (strict JSON):
{{
    "conversation_summary": "What was discussed in recent chat, or 'No prior conversation'"
  
}}"""


# ============================================================
# QUERY ANALYZER
# Runs BEFORE context_brief and extended_thinking.
# Only has raw query + chat history + docs — no thinking fields yet.
# Job: classify simple vs deep, decide if clarification needed.
# ============================================================
QUERY_ANALYZER_PROMPT = """You are a research query analyzer. Your job is to quickly classify a query and decide if we need more information before proceeding.

## Original Query:
{query}

## Clarification Conversation So Far:
{clarification_conversation}

## Available Documents:
{doc_summaries}

## Recent Chat History:
{chat_history}

## Your Tasks:

### 1. Classify the query
- **SIMPLE**: Quick factual lookup, definition, or a question directly answerable from uploaded docs. Does NOT require multi-source synthesis.
- **DEEP**: Requires research from multiple sources, analysis, comparison, synthesis, strategic thinking, or current data. When in doubt, classify as deep.

### 2. Decide if clarification is needed
Ask for clarification ONLY if the query is so ambiguous that you cannot determine what research approach to take.

Good reasons to ask:
- The query could mean two completely different things that would need different research approaches
- Critical context (like geography, time period, or use case) is completely missing and would fundamentally change the answer

Bad reasons to ask:
- You want more detail (just do thorough research)
- The query is general (general research is fine)
- You're curious (not a reason)

If a clarification conversation already exists and the user has answered, use that to proceed.

Write clarification as natural conversation — one focused question, not a form.

Good: "Quick question before I dive in — are you looking at this from an investor angle or are you building something in this space? That'll shape whether I focus on market opportunity vs technical implementation."

Bad: "Please specify: 1) use case 2) geography 3) time frame"

## Response Format (strict JSON):
{{
    "mode": "simple" or "deep",
    "needs_clarification": true or false,
    "clarification_message": "Natural conversational question (only if needs_clarification is true, else empty string)",
    "reasoning": "One sentence explaining your classification"
}}

# NOTE: Do NOT include problem_type, hypothesis, or done_criteria — those come later from the reasoning step."""


# ============================================================
# EXTENDED THINKING
# First-principles reasoning step that runs AFTER context_brief.
# Uses a thinking/reasoning model to deeply understand the problem
# before any research plan is made.
# Output is free-form reasoning text — not structured JSON.
# The planner reads this thinking and decides what to go find.
# ============================================================
EXTENDED_THINKING_PROMPT = """You are a deep researcher preparing to investigate a question.

## Query:
{query}

## Background:
{conversation_summary}

## Available Documents:
{relevant_docs}

---

Before any research begins, think deeply about this problem.

What is this question really about at its core?
What are the important concepts and how do they relate to each other?
What would you need to know to answer this completely and well?
What correlations and dependencies matter here?
What would make the difference between a shallow answer and a truly complete one?

Think freely. Write your reasoning. This thinking will guide everything that comes after."""


# ============================================================
# PLAN GENERATOR
# Reads the researcher's free-form thinking and turns it into
# a concrete, ordered research plan.
# Does NOT receive structured fields like hypothesis or done_criteria —
# those concepts live inside the thinking text.
# ============================================================
PLAN_GENERATOR_PROMPT = """You are a research planner.

A researcher has already thought deeply about the question below.
Your job is to read that thinking and turn it into a concrete research plan.

## Query:
{query}

## Background:
{conversation_summary}

## Researcher's Thinking:
{thinking}

## Available Documents:
{relevant_docs}

---

Read the thinking carefully. The researcher has identified what matters,
what connects to what, and what a complete answer requires.

Now create sections that go find exactly that. Each section should address
something the thinking says needs to be understood.

For each section write:
- A clear title
- What specifically to find or understand in this section
- 2-3 specific search queries (or leave empty if the answer is in documents)

Order sections so understanding builds — foundational things first,
synthesis and implications last.

Return as JSON:

{{
    "sections": [
        {{
            "id": "sec_1",
            "title": "...",
            "description": "what to find and why it matters for the complete answer",
            "search_queries": ["specific query 1", "specific query 2"],
            "relevant_docs": []
        }}
    ]
}}"""


# ============================================================
# SUB-AGENT RESEARCH PROMPT
# Each agent handles ONE section.
# Receives the overall thinking so it knows the broader goal —
# not just its isolated section.
# ============================================================
RESEARCH_AGENT_PROMPT = """You are a focused research agent investigating ONE specific section.

## Overall Research Goal:
{thinking_summary}

## Your Section: {section_title}
## Section Goal: {section_description}

## Search Results:
{search_results}

## Relevant Document Excerpts:
{doc_extracts}

## Instructions:
1. Analyze ALL search results and document excerpts
2. Extract key facts, data points, statistics, and insights
3. Write 2-4 paragraphs of substantive content
4. Note which sources support each claim
5. Identify any GAPS — things you expected to find but didn't

## Important:
- Be specific. Include numbers, dates, names where available.
- Don't pad with filler. Every sentence should add information.
- Stay focused on what the overall research goal needs — not just your section in isolation.
- If search results are thin, say so honestly in the gaps field.

## Response (strict JSON):
{{
    "content": "2-4 paragraphs of detailed findings with specific data points...",
    "key_points": ["Most important finding 1", "Key data point 2", "Notable trend 3"],
    "sources_used": [
        {{"url": "https://...", "title": "Source Title", "relevance": 0.9}}
    ],
    "confidence": 0.85,
    "gaps": ["Specific thing that needs more research"]
}}"""


# ============================================================
# SYNTHESIS / QUALITY CHECK
# Checks completeness of findings against the researcher's
# original thinking — not against a rigid done_criteria field.
# ============================================================
SYNTHESIS_PROMPT = """You are a research quality checker.

## Original Query:
{query}

## Researcher's Thinking (what a complete answer requires):
{thinking}

## Research Plan:
{plan_summary}

## Findings So Far:
{findings_summary}

## Task:
Review the findings against the researcher's thinking.
The thinking describes what a complete answer looks like — check if we're there.

Ask yourself:
- Does what we've found actually address what the thinking said mattered?
- Are the key correlations and dependencies the researcher identified now understood?
- What is still missing or thin?

## Rules:
- If the core of what the thinking asked for is covered → COMPLETE
- If important things the thinking flagged are still missing → MORE RESEARCH needed
- Maximum {max_loops} research rounds total. Current round: {current_loop}
- If at max rounds, mark complete regardless

## Response (strict JSON):
{{
    "is_complete": true or false,
    "completeness_score": 0.85,
    "reasoning": "Why research is or isn't complete relative to the thinking",
    "gaps_to_fill": [
        {{
            "section_id": "sec_1",
            "gap": "What is still missing",
            "search_queries": ["specific query to fill this gap"]
        }}
    ]
}}"""


# ============================================================
# REPORT WRITER
# Writes the final report informed by the researcher's thinking.
# Uses thinking for framing and direction — not hypothesis/done_criteria fields.
# ============================================================
REPORT_WRITER_PROMPT = """You are an expert research report writer.

## Original Query:
{query}

## Researcher's Thinking (framing, what complete looks like):
{thinking}

## Research Plan:
{plan_summary}

## All Research Findings:
{all_findings}

## Sources Available:
{all_sources}

## Instructions:
Write a comprehensive research report in Markdown.

Use the researcher's thinking to frame the report — it describes what matters,
what connects to what, and what a complete answer looks like. Let that shape
how you structure and emphasize things.

Structure:
1. **Title** — Clear, descriptive
2. **Executive Summary** — 2-3 sentences capturing the key answer
3. **Sections** — One per research area, with specific data points, inline citations, and analysis
4. **Key Takeaways** — 3-5 bullet points of the most important findings
5. **Sources** — Numbered list of all sources used

## Quality Standards:
- Every claim should have a source
- Include specific numbers, dates, percentages where available
- Be concise but thorough — quality over quantity
- If documents were referenced, cite them as "From uploaded documents"
- Write in a professional but readable tone
- Address what the thinking said mattered — not just what was easy to find

Write the complete report now:"""