# CLAUDE.md — DeepResearch Agent

This file provides guidance for AI assistants working in this codebase. Read it fully before making changes.

---

## Project Overview

**DeepResearch** is an AI-powered multi-stage research agent. It accepts a user query, optionally asks clarifying questions, plans a research strategy with human-in-the-loop (HITL) approval, spawns parallel sub-agents to search the web, synthesizes findings, detects gaps, and produces a comprehensive research report.

Key capabilities:
- Extended reasoning before searching (hypothesis formation)
- Human-in-the-loop confirmation at two stages: clarification and plan approval
- Parallel sub-agent execution per research section
- Real-time progress streaming via WebSocket
- Virtual in-memory filesystem to prevent context overflow
- Dual LLM strategy: fast model for planning, quality model for report writing

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.8+ |
| State Machine | LangGraph 0.2.60 |
| API Server | FastAPI + Uvicorn |
| Real-time | WebSockets |
| LLM Gateway | OpenRouter (via httpx) |
| Web Search | Tavily API |
| Validation | Pydantic v2 |
| Frontend | Single-page HTML (static/index.html) |

---

## Repository Structure

```
DeepResearch/
├── main.py                  # Entry point — calls src.api.run()
├── requirements.txt         # Python dependencies
├── README.md                # Project documentation
├── CLAUDE.md                # This file
├── data/                    # Runtime data (chat history JSON, gitignored)
├── static/
│   └── index.html           # Full SPA frontend (dark-theme, WebSocket-driven)
└── src/
    ├── __init__.py
    ├── api.py               # FastAPI app — REST + WebSocket endpoints
    ├── config.py            # Configuration from environment variables
    ├── llm.py               # OpenRouter LLM client
    ├── graph.py             # LangGraph state machine definition
    ├── state.py             # ResearchState schema (TypedDict, 30+ fields)
    ├── nodes.py             # LangGraph node implementations
    ├── subagents.py         # Parallel sub-agent orchestration
    ├── prompts.py           # All LLM prompts
    ├── history.py           # JSON file-based chat history manager
    └── tools/
        ├── __init__.py
        ├── todo.py          # Planning todo list tool
        ├── filesystem.py    # Virtual in-memory filesystem
        ├── search.py        # Tavily web search wrapper
        └── documents.py     # Document store (preloaded docs placeholder)
```

---

## Running the Project

### Prerequisites

Create a `.env` file in the project root:

```env
OPENROUTER_API_KEY=your_key_here
TAVILY_API_KEY=your_key_here

# Optional model overrides
LLM_MODEL_FAST=deepseek/deepseek-chat-v3-0324
LLM_MODEL_QUALITY=deepseek/deepseek-chat-v3-0324
LLM_MODEL_REASONING=deepseek/deepseek-r1

# Optional server/behavior settings
HOST=0.0.0.0
PORT=8000
MAX_SEARCH_RESULTS=5
MAX_RESEARCH_LOOPS=3
CHAT_HISTORY_LIMIT=10
```

### Start the Server

```bash
pip install -r requirements.txt
python main.py
```

Server starts at `http://localhost:8000`. Open the browser to use the web UI.

---

## Configuration (`src/config.py`)

All configuration is loaded from environment variables with sensible defaults. Key variables:

| Variable | Default | Purpose |
|---|---|---|
| `OPENROUTER_API_KEY` | required | LLM API access |
| `TAVILY_API_KEY` | required | Web search |
| `LLM_MODEL_FAST` | `deepseek/deepseek-chat-v3-0324` | Planning, analysis, sub-agents |
| `LLM_MODEL_QUALITY` | `deepseek/deepseek-chat-v3-0324` | Final report writing |
| `LLM_MODEL_REASONING` | `deepseek/deepseek-r1` | Extended thinking node |
| `MAX_SEARCH_RESULTS` | `5` | Tavily results per query |
| `MAX_RESEARCH_LOOPS` | `3` | Max synthesis → research cycles |
| `CHAT_HISTORY_LIMIT` | `10` | Messages of history included in context |

---

## Research Pipeline

The pipeline is a LangGraph state machine. Stages execute in order:

```
collect_context
      │
analyze_query ──► [simple query] ──► FastAPI returns direct LLM response
      │
      ▼ [deep research]
wait_for_clarification (HITL interrupt — optional)
      │
context_brief_node
      │
extended_thinking_node  (reasoning model: hypothesis + done criteria)
      │
generate_plan
      │
wait_for_confirmation (HITL interrupt — user approves/edits/cancels)
      │
execute_research  (parallel sub-agents per section)
      │
synthesize_and_check ──► [has gaps + loops remaining] ──► execute_research
      │
write_report
      │
END
```

### Stage Descriptions

| Stage | Node | What it does |
|---|---|---|
| `collecting` | `collect_context` | Load chat history and documents into state |
| `analyzing` | `analyze_query` | Classify query as simple or deep; check if clarification is needed |
| `clarifying` | `wait_for_clarification` | HITL interrupt; waits for user to submit clarification via API |
| `briefing` | `context_brief_node` | Compress chat/doc context into a structured brief |
| `thinking` | `extended_thinking_node` | Use reasoning model to classify problem type, form hypothesis, define done criteria |
| `planning` | `generate_plan` | Generate research plan with sections and search queries |
| `awaiting_confirmation` | `wait_for_confirmation` | HITL interrupt; waits for user to approve, edit, or cancel the plan |
| `researching` | `execute_research` | Spawn parallel sub-agents — one per research section |
| `synthesizing` | `synthesize_and_check` | Validate findings against done criteria; flag gaps |
| `writing` | `write_report` | Generate final markdown report using quality model |
| `completed` | END | Research finished |

---

## Key Source Files

### `src/state.py` — ResearchState

The single shared TypedDict passed through all LangGraph nodes. Important fields:

```python
# Identity
research_id: str          # UUID for this research session
chat_id: str              # Links to chat history

# Input
user_query: str           # The original question
model: str                # LLM model override from request

# Research control
research_mode: str        # "simple" or "deep"
needs_clarification: bool
plan_approved: bool
has_gaps: bool
research_loop_count: int  # Tracks synthesis→research iterations

# Content
todos: list               # Research plan items
plan: str                 # Human-readable plan text
findings: dict            # {section_title: content} written by sub-agents
all_sources: list         # Accumulated search result URLs
report: str               # Final report markdown

# Progress
status: str               # Current pipeline stage name
progress: list            # Event log for WebSocket streaming
error: str                # Set on failure
```

When adding new pipeline stages, add corresponding status values and update `state.py`.

### `src/nodes.py` — Node Implementations

Each function matches a LangGraph node. Conventions:
- Accept `state: ResearchState` and return a dict of state updates.
- Use `progress_callback` parameter (passed from `api.py`) to emit real-time events.
- Call `call_llm()` or `call_llm_thinking()` from `llm.py` — never use the OpenRouter API directly.
- Parse LLM JSON output with `parse_llm_json()` from `llm.py`.
- Log errors to `state["error"]` and set `state["status"] = "error"`.

### `src/llm.py` — LLM Client

| Function | Model used | Purpose |
|---|---|---|
| `call_llm(prompt, system)` | Fast model | Standard completions for planning/analysis |
| `call_llm_thinking(prompt, system)` | Reasoning model | Extended thinking with chain-of-thought |
| `call_llm_streaming(prompt, system)` | Fast model | Streaming responses (simple chat mode) |
| `call_llm_simple(messages)` | Fast model | Raw message list completion |
| `parse_llm_json(text)` | — | Extract JSON from LLM response (handles markdown fences) |

### `src/prompts.py` — Prompts

All prompts are module-level string constants. Prompts include:
- Detailed instruction blocks
- Example input/output pairs
- JSON schema specifications for structured outputs

When modifying prompts:
- Keep the example section intact — it strongly conditions output format.
- Never remove JSON schema documentation from a prompt.
- Test prompt changes end-to-end; LLM output format changes can silently break `parse_llm_json`.

### `src/subagents.py` — Sub-Agent Orchestration

`run_all_subagents(todos, research_id, progress_callback)` runs sub-agents in parallel using `asyncio.gather`. Each sub-agent:
1. Receives a single research section (todo item) as its scope.
2. Decides which queries to search.
3. Calls `tavily_search()` for each query.
4. Writes findings to the virtual filesystem keyed by section title.
5. Emits granular events: `section_start`, `search_start`, `search_result`, `search_done`, `analyzing`, `synthesis_done`.

### `src/api.py` — API Layer

The API manages two types of sessions:
- **Chat sessions** (`chat_id`): Simple conversation history via `history.py`.
- **Research sessions** (`research_id`): Active LangGraph graph instances stored in `research_sessions` dict.

HITL interrupts work by:
1. LangGraph suspends at `wait_for_clarification` or `wait_for_confirmation`.
2. The API stores the suspended graph instance in `research_sessions`.
3. The client sends a follow-up HTTP request (`/clarify` or `/confirm-plan`).
4. The API resumes the graph by calling `graph.ainvoke(resume_input, config)`.

### `src/tools/filesystem.py` — Virtual Filesystem

Used to store sub-agent findings without polluting the main LLM context:
```python
fs = VirtualFilesystem()
fs.write("section_title.md", content)   # Sub-agent writes findings
fs.read("section_title.md")             # Synthesis reads them back
fs.read_all_findings()                  # Returns all files as a dict
```

### `src/history.py` — Chat History

Persists to `data/chat_history.json` as a JSON dict keyed by `chat_id`. The `data/` directory must exist for history to work. On first run, create it with `mkdir -p data`.

---

## API Endpoints

### Deep Research Flow

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/api/v1/deep-research/start` | Start a research session; returns `research_id` |
| `POST` | `/api/v1/deep-research/{id}/clarify` | Submit clarification answer |
| `POST` | `/api/v1/deep-research/{id}/confirm-plan` | Approve / edit / cancel the plan |
| `GET` | `/api/v1/deep-research/{id}/status` | Polling fallback for progress |
| `GET` | `/api/v1/deep-research/{id}/report` | Retrieve the final report |
| `WS` | `/ws/deep-research/{id}` | WebSocket for real-time events |

### Chat Mode

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/api/v1/chat` | Simple chat completion with history |

### Chat Management

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/v1/chats` | List all chat sessions |
| `POST` | `/api/v1/chats` | Create a new chat |
| `GET` | `/api/v1/chats/{id}/messages` | Get messages for a chat |
| `DELETE` | `/api/v1/chats/{id}` | Delete a chat |
| `GET` | `/api/v1/documents` | List available preloaded documents |

### WebSocket Event Types

Events pushed over the WebSocket during research:

| Event type | When emitted |
|---|---|
| `status` | Pipeline stage transitions |
| `progress` | General progress messages |
| `section_start` | Sub-agent begins a section |
| `search_start` | Sub-agent starts a search query |
| `search_result` | Individual search result returned |
| `search_done` | All searches for a section complete |
| `docs_found` | Documents loaded from store |
| `analyzing` | Sub-agent analyzing gathered results |
| `synthesis_done` | Synthesis node completed |
| `clarification_needed` | Graph suspended for clarification |
| `plan_ready` | Graph suspended for plan approval |
| `report` | Final report text (chunked or complete) |
| `error` | An error occurred |
| `done` | Research complete |

---

## Development Conventions

### Adding a New LangGraph Node

1. Implement the node function in `src/nodes.py`:
   ```python
   async def my_new_node(state: ResearchState, progress_callback=None) -> dict:
       # ... logic ...
       return {"status": "my_stage", "some_field": value}
   ```
2. Register it in `src/graph.py` with `graph.add_node("my_new_node", my_new_node)`.
3. Add the appropriate edge or conditional edge.
4. Add the new status string to `src/state.py` comments/documentation.
5. Update the frontend stage tracker in `static/index.html` if it should appear in the UI.

### Adding a New API Endpoint

1. Add the route in `src/api.py`.
2. If it interacts with an active research session, access `research_sessions[research_id]`.
3. If it modifies graph state, resume the graph with the new input and await the result.

### Modifying Prompts

- All prompts live in `src/prompts.py`.
- Sub-agent prompts are in `src/subagents.py` (inline within functions).
- Always validate JSON output format after changing a prompt — `parse_llm_json` is tolerant but the downstream dict access is not.

### Extending the Virtual Filesystem

The `VirtualFilesystem` in `src/tools/filesystem.py` is in-memory only. For persistence across server restarts, replace the internal `self.files` dict with a database or file-backed store.

### Model Selection

When adding support for a new OpenRouter model:
1. Add it to the reasoning model list in `src/llm.py` `call_llm_thinking()` if it supports chain-of-thought.
2. Set it via the appropriate `LLM_MODEL_*` environment variable.
3. No code changes required for fast/quality model swaps — they are fully environment-driven.

---

## Testing

There are currently no automated tests. When adding tests:
- Use `pytest` and `pytest-asyncio` for async node functions.
- Mock `call_llm` and `tavily_search` to avoid live API calls in unit tests.
- Place tests in a `tests/` directory at the project root.
- Test node functions by passing a mock `ResearchState` dict and asserting on the returned state updates.

---

## Common Pitfalls

| Pitfall | Remedy |
|---|---|
| LLM returns JSON wrapped in markdown fences | Always use `parse_llm_json()` — it strips fences |
| HITL interrupt not resuming | Ensure `research_sessions[id]` still holds the graph instance (not GC'd) |
| Sub-agent findings missing | Check that `VirtualFilesystem` instance is shared across sub-agent calls |
| Chat history file missing | Create `data/` directory before first run: `mkdir -p data` |
| WebSocket event not reaching client | Verify `progress_callback` is wired through from `api.py` to the node |
| Research loops infinitely | `MAX_RESEARCH_LOOPS` env var caps synthesis→research iterations |
| OpenRouter rate limits | Add exponential backoff in `llm.py` `call_llm()` for 429 responses |

---

## Data Flow Summary

```
User query
  │
  ▼
api.py ──► graph.ainvoke() ──► collect_context
                                    │
                               analyze_query
                                    │
                    ┌───────────────┴───────────────┐
                    │ simple                        │ deep
                    ▼                               ▼
             direct LLM response          [optional clarification]
             returned via HTTP                      │
                                          context_brief_node
                                                    │
                                          extended_thinking_node
                                          (problem type, hypothesis,
                                           done criteria)
                                                    │
                                          generate_plan ──► HITL approve
                                                    │
                                          execute_research
                                          ┌─────────────────────┐
                                          │ sub-agent 1 (section 1) │
                                          │ sub-agent 2 (section 2) │  ← parallel
                                          │ sub-agent N (section N) │
                                          └─────────────────────┘
                                                    │
                                          synthesize_and_check
                                                    │
                                          ┌─────────┴──────────┐
                                          │ gaps found          │ complete
                                          ▼                     ▼
                                    execute_research       write_report
                                    (up to MAX loops)           │
                                                          WebSocket: report
                                                          HTTP: GET /report
```

---

## File Modification Checklist

Before modifying a file, consider these dependencies:

| File | Consumers | Impact of change |
|---|---|---|
| `state.py` | All nodes, api.py | Field renames break all nodes |
| `prompts.py` | nodes.py, subagents.py | Format changes break JSON parsing |
| `graph.py` | api.py | Edge changes alter pipeline flow |
| `llm.py` | nodes.py, subagents.py | Signature changes break callers |
| `config.py` | All modules | Env var name changes break deployments |
| `api.py` | static/index.html | Endpoint/event changes break the UI |
| `static/index.html` | Users | UI-only, safe to change independently |
