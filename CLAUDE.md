# CLAUDE.md — Deep Research Agent

Comprehensive guide for AI assistants working on this codebase.

## Project Overview

A deep research agent built on raw **LangGraph** + **FastAPI**. Given a user query it:
1. Classifies the query (simple vs. deep)
2. Optionally asks a single clarifying question (HITL)
3. Runs a reasoning model to think about the problem
4. Generates a research plan and asks the user to approve it (HITL)
5. Spawns parallel sub-agents (one per plan section) that search the web via Tavily
6. Checks completeness and loops if needed (gap detection)
7. Writes a final Markdown report using the quality model

Both a REST API and a WebSocket interface are provided. The frontend is a single `static/index.html` page.

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env   # or create .env manually (see Environment Variables below)

# Run
python main.py
# → http://localhost:8000
```

---

## Environment Variables

Create a `.env` file in the project root:

```env
# LLM gateway (required)
OPENROUTER_API_KEY=sk-or-...
LLM_MODEL_FAST=deepseek/deepseek-chat-v3-0324       # cheap, high-volume calls
LLM_MODEL_QUALITY=deepseek/deepseek-chat-v3-0324    # final report writing
LLM_MODEL_REASONING=deepseek/deepseek-r1             # thinking/reasoning step

# Reasoning budget (tokens for thinking model)
REASONING_BUDGET_TOKENS=8000

# Web search (required for deep research)
TAVILY_API_KEY=tvly-...

# Research tuning
MAX_SEARCH_RESULTS=5
MAX_RESEARCH_LOOPS=3
CHAT_HISTORY_LIMIT=10

# Storage
DATA_DIR=data

# Server
HOST=0.0.0.0
PORT=8000
```

All settings are centralised in `src/config.py` and loaded via `python-dotenv`.

---

## Project Structure

```
DeepResearch/
├── main.py                      # Entry point — calls src/api.run()
├── requirements.txt
├── data/
│   └── chat_history.json        # JSON-file chat persistence (auto-created)
├── src/
│   ├── __init__.py
│   ├── config.py                # All env vars & constants
│   ├── llm.py                   # OpenRouter HTTP client (4 call variants)
│   ├── history.py               # JSON chat history CRUD
│   ├── state.py                 # LangGraph state schema + create_initial_state()
│   ├── prompts.py               # All LLM prompts (one per node)
│   ├── nodes.py                 # LangGraph node functions + routing functions
│   ├── subagents.py             # Parallel sub-agent spawning
│   ├── graph.py                 # LangGraph graph definition + singleton
│   ├── api.py                   # FastAPI app + WebSocket handler
│   └── tools/
│       ├── __init__.py
│       ├── todo.py              # Todo planning helpers (deepagents pattern)
│       ├── filesystem.py        # VirtualFileSystem (in-memory dict)
│       ├── search.py            # Tavily web search wrapper
│       └── documents.py        # Pre-loaded document store (currently empty)
└── static/
    └── index.html               # Single-page test UI
```

---

## Architecture

### LangGraph State Machine

The graph is compiled once at startup (`graph.py:get_graph()`) and stored as a singleton.
State flows through nodes as a plain `dict` (LangGraph `StateGraph(dict)`).
A `MemorySaver` checkpointer enables HITL interrupts.

**Full node execution order:**

```
collect_context
  → analyze_query
      ├─(simple)──────────────→ simple_answer → END
      ├─(needs clarification)─→ wait_for_clarification  [HITL interrupt]
      │                              ↑ loops back to analyze_query after user reply
      └─(deep, no clarify)────→ context_brief
                                    → extended_thinking
                                    → generate_plan
                                    → wait_for_confirmation  [HITL interrupt]
                                    → execute_research  ←──────────┐
                                    → synthesize_and_check         │
                                        ├─(gaps found)─────────────┘
                                        └─(complete)──→ write_report → END
```

**HITL interrupts** are declared in `graph.py:build_research_graph()`:
```python
interrupt_before=["wait_for_clarification", "wait_for_confirmation"]
```
The graph pauses, the API returns, and the frontend sends user input back via
`POST /api/v1/deep-research/{id}/clarify` or `/confirm-plan` (or via WebSocket).
`api.py:_resume_graph()` calls `graph.update_state()` then `graph.astream(None, ...)`.

### State Schema (`src/state.py`)

All state lives in a flat dict created by `create_initial_state()`. Key fields:

| Field | Type | Set by |
|-------|------|--------|
| `research_id` | str (UUID) | `collect_context` |
| `user_query` | str | API caller |
| `research_mode` | `"simple"` \| `"deep"` | `analyze_query` |
| `needs_clarification` | bool | `analyze_query` |
| `clarification_conversation` | list[dict] | HITL flow |
| `context_brief` | dict | `context_brief_node` |
| `thinking` | str (free text) | `extended_thinking_node` |
| `thinking_trace` | str (raw CoT) | `extended_thinking_node` |
| `plan` | dict | `generate_plan` |
| `todos` | list[dict] | `generate_plan` |
| `vfs` | dict | `execute_research` (VirtualFileSystem) |
| `findings` | list[dict] | `execute_research` |
| `all_sources` | list[dict] | `execute_research` |
| `has_gaps` | bool | `synthesize_and_check` |
| `report` | str (Markdown) | `write_report` / `simple_answer` |
| `status` | str | every node |
| `progress` | list[str] | `_progress()` helper |

### Dual + Reasoning Model Strategy

Three model tiers configured in `.env`:

| Tier | Env var | Used for |
|------|---------|----------|
| **FAST** | `LLM_MODEL_FAST` | `analyze_query`, `context_brief`, `generate_plan`, `synthesize_and_check`, all sub-agents |
| **QUALITY** | `LLM_MODEL_QUALITY` | `write_report` only (1 call, higher quality) |
| **REASONING** | `LLM_MODEL_REASONING` | `extended_thinking` only (`call_llm_thinking`) |

### LLM Client (`src/llm.py`)

Four async functions, all calling OpenRouter:

| Function | Use |
|----------|-----|
| `call_llm(prompt, model, ...)` | Standard JSON-returning call |
| `call_llm_thinking(prompt, model, ...)` | Reasoning model, returns `(thinking_trace, final_answer)` |
| `call_llm_streaming(prompt, model, ...)` | SSE streaming, yields text chunks |
| `call_llm_simple(messages, model, ...)` | Multi-turn chat (normal chat mode) |

`parse_llm_json(response)` strips markdown fences and parses JSON — use it whenever
nodes expect structured output.

**Timeouts:** standard=120s, reasoning=180s, search=30s.

### Virtual Filesystem (`src/tools/filesystem.py`)

An in-memory `dict`-backed filesystem (`VirtualFileSystem`). Sub-agents write findings to `/findings/{section_id}` and metadata to `/findings/{section_id}_meta`. The report writer reads everything under `/findings/` via `vfs.read_all_findings()`. This prevents context overflow on long research tasks.

State serialises it as `state["vfs"] = vfs.to_dict()` and deserialises with `VirtualFileSystem.from_dict(state["vfs"])`.

### Sub-Agents (`src/subagents.py`)

`run_all_subagents()` runs all plan sections in **parallel** via `asyncio.gather`.
Each `run_research_subagent()`:
1. Runs Tavily searches **sequentially** per query (for granular streaming events)
2. Searches pre-loaded documents
3. Calls the fast LLM to produce a JSON findings object
4. Returns `{section_id, content, key_points, sources, confidence, gaps}`

Events are pushed to the WebSocket via `event_callback` for real-time UI updates.

### WebSocket Protocol (`src/api.py`)

The WebSocket at `/ws/deep-research/{id}` streams typed JSON events:

| Event type | Direction | Description |
|------------|-----------|-------------|
| `progress` | server→client | Free-text status update |
| `pipeline_stage` | server→client | Named pipeline stage started |
| `thinking_complete` | server→client | Reasoning model finished, includes thinking text |
| `clarification_needed` | server→client | Pause for user input |
| `plan_ready` | server→client | Plan generated, pause for confirmation |
| `research_event` | server→client | Granular sub-agent events (search results, etc.) |
| `vfs_write` | server→client | Virtual filesystem write event |
| `report_complete` | server→client | Final report ready |
| `heartbeat` | server→client | Keep-alive (every 300s) |
| `error` | server→client | Error event |
| `clarify` | client→server | User clarification reply |
| `confirm_plan` | client→server | Plan approval/edit/cancel |

---

## API Endpoints

### Health
- `GET /health` — Returns `{"status": "ok", "version": "0.3.0"}`
- `GET /` — Serves `static/index.html`

### Normal Chat
- `POST /api/v1/chat` — Body: `{message, chat_id?, model?}`

### Deep Research
- `POST /api/v1/deep-research/start` — Body: `{query, chat_id?, model?, doc_ids?[]}`
- `POST /api/v1/deep-research/{id}/clarify` — Body: `{message}`
- `POST /api/v1/deep-research/{id}/confirm-plan` — Body: `{action: "approve"|"edit"|"cancel", edits?}`
- `GET /api/v1/deep-research/{id}/status` — Polling fallback
- `GET /api/v1/deep-research/{id}/report` — Final report + sources
- `WS /ws/deep-research/{id}` — Real-time streaming

### Chat Management
- `GET /api/v1/chats` — List conversations
- `POST /api/v1/chats` — Create conversation
- `GET /api/v1/chats/{id}/messages` — Get messages
- `DELETE /api/v1/chats/{id}` — Delete conversation

### Documents
- `GET /api/v1/documents` — List pre-loaded documents

---

## Key Design Conventions

### Prompt Conventions (`src/prompts.py`)

- Every LLM-calling node has exactly **one prompt constant** in `prompts.py`
- Prompts use `.format(**kwargs)` with named placeholders
- Nodes that return structured data require `**strict JSON**` in the prompt
- The `extended_thinking` node is the only one that does **not** parse JSON — it stores free-form text
- The node-to-prompt mapping is documented at the top of `prompts.py`

### Node Conventions (`src/nodes.py`)

- Every node is an `async def` that takes `state: dict` and returns `state: dict`
- Routing functions are sync `def route_after_X(state: dict) -> str`
- Use the `_progress(state, msg)` helper to emit status messages — it both appends to `state["progress"]` and pushes via registered callback
- Always set `state["status"]` at the start of the node to the appropriate stage name
- On error: set `state["error"]` and `state["status"] = "error"`

### Status Values (Pipeline Stages)

```
collecting → analyzing → clarifying → briefing → thinking → planning
→ awaiting_confirmation → researching → synthesizing → writing → completed
```

Error state: `"error"`. Cancelled state: `"cancelled"`.

### LLM Response Parsing

Always use `parse_llm_json(response)` — it handles markdown code fences (`\`\`\`json`).
On parse failure it returns `{}` and logs the error. Callers should guard:
```python
result = parse_llm_json(response)
if not result:
    state["error"] = "Failed to parse ..."
    return state
```

### Research Sessions

`api.py` maintains two in-memory dicts:
- `research_sessions: dict[str, dict]` — session state, config, thread_id
- `ws_connections: dict[str, WebSocket]` — active WebSocket connections

Sessions are **not persisted** across server restarts. Chat history is persisted in `data/chat_history.json`.

### Graph Resume Pattern

After HITL interrupts, resuming uses:
```python
graph.update_state(config, current_state, as_node="wait_for_clarification")
async for event in graph.astream(None, config=config, stream_mode="values"):
    ...
```
The first event replays the checkpoint and must be skipped if `status` matches the
resuming-from status. See `_resume_graph()` in `api.py`.

### Progress Callbacks

Real-time progress during graph execution uses a callback registry in `nodes.py`:
```python
register_progress_callback(research_id, async_callback_fn)
unregister_progress_callback(research_id)
```
The callback receives either a string (progress message) or a dict (structured event).
Always call `unregister_progress_callback` in a `finally` block.

---

## Adding New Features

### Adding a New Node

1. Write the async node function in `src/nodes.py`
2. Add the prompt to `src/prompts.py` (update the node→prompt mapping comment)
3. Register the node in `graph.py:build_research_graph()` with `graph.add_node()`
4. Add edges/conditional edges in the graph
5. Add a new `status` value if needed (update `api.py` progress_map)
6. Update WebSocket event handling in `_run_graph()` / `_resume_graph()` if it's a HITL node

### Adding a New Tool

Create a module in `src/tools/`, import it in the relevant node file.
Tools that make external HTTP calls should use `httpx.AsyncClient` with an appropriate timeout.

### Changing Prompts

Edit the constant in `src/prompts.py`. The prompt contract (JSON shape) must match
what the calling node's `parse_llm_json` result handler expects.

### Changing Models

Update `.env` variables. To use a new reasoning model, check `llm.py:call_llm_thinking`
— it handles OpenRouter's unified `reasoning` parameter and falls back through multiple
response formats (reasoning_details, `<think>` tags, content blocks).

### Adding Document Upload

The `src/tools/documents.py` `PRELOADED_DOCS` dict is currently empty.
Implement real upload handling by populating this dict (or replacing with a database)
and exposing a `POST /api/v1/documents` endpoint.

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `fastapi` | 0.115.6 | Web framework + WebSocket |
| `uvicorn[standard]` | 0.34.0 | ASGI server |
| `pydantic` | 2.10.3 | Request/response validation |
| `langgraph` | 0.2.60 | State machine + HITL |
| `langchain-core` | 0.3.28 | LangGraph dependency |
| `httpx` | 0.28.1 | Async HTTP (OpenRouter + Tavily) |
| `python-dotenv` | 1.0.1 | `.env` loading |
| `websockets` | 14.1 | WebSocket transport |

---

## Testing

There are currently **no automated tests**. Manual testing is done via:
- `http://localhost:8000` — the HTML test UI
- Direct API calls (curl / Postman) against REST endpoints
- WebSocket connections to `/ws/deep-research/{id}`

When adding tests, use `pytest` + `httpx.AsyncClient` with FastAPI's `TestClient`
or `AsyncClient` for integration tests.

---

## Common Pitfalls

- **Double-start guard**: `_run_graph` sets `session["graph_started"] = True` to prevent duplicate execution when both a WebSocket connection and an HTTP resume race. Always check this flag.
- **Checkpoint replay skip**: After `_resume_graph`, the first event replays the previous checkpoint state. Skip it if `status` matches `resuming_from_status`.
- **UUID truncation**: `research_id` uses full UUIDs. `chat_id` uses 8-char truncated UUIDs (collision-acceptable for chat IDs). Do not truncate research IDs.
- **Thinking is not JSON**: `extended_thinking` stores free-form text in `state["thinking"]`. Do not call `parse_llm_json` on it.
- **VFS serialisation**: Always call `vfs.to_dict()` before storing in state and `VirtualFileSystem.from_dict(state["vfs"])` when reading back.
- **Model fallthrough**: If `state["model"]` is empty, `call_llm` falls back to `LLM_MODEL_FAST`. The quality model override in `write_report` is explicit — it always passes `LLM_MODEL_QUALITY`.
