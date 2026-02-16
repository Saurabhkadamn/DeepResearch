# 🔬 Deep Research Agent

A deep research agent built on **raw LangGraph** with patterns borrowed from LangChain's DeepAgents — planning tools, sub-agent spawning, virtual filesystem for context management.

## Architecture

```
USER QUERY
    │
    ▼
collect_context → analyze_query ─┬→ simple_answer → END
                                 ├→ clarification (HITL) → analyze_query
                                 └→ generate_plan → confirm plan (HITL)
                                        → execute_research (parallel sub-agents) ←┐
                                        → synthesize_and_check (gap detection) ───┘
                                        → write_report (quality model) → END
```

### Key Patterns (from DeepAgents)

1. **Todo Planning Tool** — Forces the LLM to plan before acting, tracks progress
2. **Sub-Agent Spawning** — Each research section gets an isolated sub-agent with scoped context
3. **Virtual Filesystem** — Findings written to in-memory files, prevents context overflow
4. **Dual Model Strategy** — Fast model for analysis/planning, quality model for final report
5. **Detailed System Prompts** — Example-rich prompts with clear JSON response formats

### What's Different (keeping it lean)

- No middleware system (saves tokens)
- No real filesystem (in-memory dict)
- No Anthropic-specific optimizations (model-agnostic via OpenRouter)
- HTML test page instead of Streamlit

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure environment
cp .env.example .env
# Edit .env with your API keys

# Deep Research - Environment Variables

# OpenRouter (LLM)
OPENROUTER_API_KEY=
LLM_MODEL_FAST=
LLM_MODEL_QUALITY=

# Tavily (Web Search)
TAVILY_API_KEY =

# Research Settings
MAX_SEARCH_RESULTS=5
MAX_RESEARCH_LOOPS=1

# Server
HOST=0.0.0.0
PORT=8000



# 3. Run
python main.py
```

Open `http://localhost:8000` for the test UI.

## API Endpoints

### Normal Chat
- `POST /api/v1/chat` — Simple chat with history

### Deep Research
- `POST /api/v1/deep-research/start` — Start research session
- `POST /api/v1/deep-research/{id}/clarify` — Submit clarification answers
- `POST /api/v1/deep-research/{id}/confirm-plan` — Approve/edit/cancel plan
- `GET /api/v1/deep-research/{id}/status` — Polling fallback
- `GET /api/v1/deep-research/{id}/report` — Get final report

### WebSocket
- `WS /ws/deep-research/{id}` — Real-time progress, plan review, report streaming

### Chat Management
- `GET /api/v1/chats` — List all conversations
- `POST /api/v1/chats` — Create new conversation
- `GET /api/v1/chats/{id}/messages` — Get chat messages
- `DELETE /api/v1/chats/{id}` — Delete conversation

## Project Structure

```
deep-research/
├── main.py                  # Entry point
├── requirements.txt
├── .env.example
├── data/
│   └── chat_history.json    # Persistent chat storage
├── src/
│   ├── config.py            # Settings, dual model config
│   ├── llm.py               # OpenRouter client (fast + quality)
│   ├── history.py           # JSON chat history (multi-conversation)
│   ├── state.py             # LangGraph state schema
│   ├── prompts.py           # All LLM prompts (detailed, with examples)
│   ├── nodes.py             # LangGraph node functions
│   ├── subagents.py         # Parallel research sub-agents
│   ├── graph.py             # LangGraph state machine + HITL
│   ├── api.py               # FastAPI + WebSocket
│   └── tools/
│       ├── todo.py          # Planning tool (deepagents pattern)
│       ├── filesystem.py    # Virtual in-memory filesystem
│       ├── search.py        # Tavily web search
│       └── documents.py     # Pre-loaded test documents
└── static/
    └── index.html           # HTML test page
```

## Tech Stack

- **LangGraph** — State machine with HITL interrupts
- **FastAPI** — REST + WebSocket API
- **Tavily** — Web search
- **OpenRouter** — Multi-model LLM gateway
- **httpx** — Async HTTP client
