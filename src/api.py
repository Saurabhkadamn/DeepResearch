"""
Deep Research - FastAPI API + WebSocket
Normal chat mode + Deep research mode in one service.
"""

import json
import asyncio
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .config import HOST, PORT
from .state import create_initial_state
from .graph import get_graph
from .llm import call_llm_simple
from .nodes import register_progress_callback, unregister_progress_callback
from .history import (
    create_chat, add_message, get_messages, get_messages_for_context,
    list_chats, delete_chat,
)
from .tools.documents import get_all_docs


# ============================================================
# SESSION STORE (in-memory, replace with Redis in production)
# ============================================================
research_sessions: dict[str, dict] = {}
ws_connections: dict[str, WebSocket] = {}


# ============================================================
# MODELS
# ============================================================
class ChatRequest(BaseModel):
    message: str
    chat_id: str = ""
    model: str = ""

class ChatResponse(BaseModel):
    reply: str
    chat_id: str

class StartResearchRequest(BaseModel):
    query: str
    chat_id: str = ""
    model: str = ""
    doc_ids: list[str] = Field(default_factory=list)

class StartResearchResponse(BaseModel):
    research_id: str
    chat_id: str
    status: str
    ws_url: str

class PlanActionRequest(BaseModel):
    action: str  # "approve" | "edit" | "cancel"
    edits: dict = Field(default_factory=dict)

class ClarifyRequest(BaseModel):
    message: str  # Free-text chat message from user


# ============================================================
# APP
# ============================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 Deep Research service starting...")
    get_graph()
    print("✅ LangGraph ready")
    yield
    print("👋 Shutting down...")

app = FastAPI(title="Deep Research API", version="0.2.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static HTML test page
app.mount("/static", StaticFiles(directory="static"), name="static")


# ============================================================
# BASIC ENDPOINTS
# ============================================================
@app.get("/")
async def root():
    return FileResponse("static/index.html")

@app.get("/health")
async def health():
    return {"status": "ok", "service": "deep-research"}

@app.get("/api/v1/chats")
async def api_list_chats():
    return {"chats": list_chats()}

@app.post("/api/v1/chats")
async def api_create_chat():
    chat_id = create_chat()
    return {"chat_id": chat_id}

@app.delete("/api/v1/chats/{chat_id}")
async def api_delete_chat(chat_id: str):
    delete_chat(chat_id)
    return {"deleted": True}

@app.get("/api/v1/chats/{chat_id}/messages")
async def api_get_messages(chat_id: str):
    return {"messages": get_messages(chat_id)}

@app.get("/api/v1/documents")
async def api_list_documents():
    return {"documents": get_all_docs()}


# ============================================================
# NORMAL CHAT (simple proxy to OpenRouter)
# ============================================================
@app.post("/api/v1/chat", response_model=ChatResponse)
async def normal_chat(request: ChatRequest):
    """Normal chat mode — simple LLM call with history context."""
    chat_id = request.chat_id
    if not chat_id:
        chat_id = create_chat(request.message[:50])

    add_message(chat_id, "user", request.message, "chat")

    history = get_messages_for_context(chat_id, limit=10)
    messages = [{"role": "system", "content": "You are a helpful AI assistant."}]
    messages.extend(history)

    reply = await call_llm_simple(messages, model=request.model)
    add_message(chat_id, "assistant", reply, "chat")

    return ChatResponse(reply=reply, chat_id=chat_id)


# ============================================================
# DEEP RESEARCH - START
# ============================================================
@app.post("/api/v1/deep-research/start", response_model=StartResearchResponse)
async def start_research(request: StartResearchRequest):
    """Start a deep research session."""
    chat_id = request.chat_id
    if not chat_id:
        chat_id = create_chat(f"Research: {request.query[:40]}")

    add_message(chat_id, "user", request.query, "chat")

    research_id = str(uuid.uuid4())[:8]
    thread_id = f"thread_{research_id}"

    initial_state = create_initial_state(
        user_query=request.query,
        chat_id=chat_id,
        model=request.model,
        doc_ids=request.doc_ids,
    )
    initial_state["research_id"] = research_id

    research_sessions[research_id] = {
        "state": initial_state,
        "thread_id": thread_id,
        "chat_id": chat_id,
        "config": {"configurable": {"thread_id": thread_id}},
    }

    return StartResearchResponse(
        research_id=research_id,
        chat_id=chat_id,
        status="analyzing",
        ws_url=f"/ws/deep-research/{research_id}",
    )


# ============================================================
# DEEP RESEARCH - CLARIFY (now chat-based)
# ============================================================
@app.post("/api/v1/deep-research/{research_id}/clarify")
async def submit_clarification(research_id: str, request: ClarifyRequest):
    if research_id not in research_sessions:
        raise HTTPException(404, "Session not found")

    session = research_sessions[research_id]

    # Append user's chat message to the clarification conversation log
    conversation = session["state"].get("clarification_conversation", [])
    conversation.append({"role": "user", "content": request.message})
    session["state"]["clarification_conversation"] = conversation

    await _send_ws(research_id, {"type": "progress", "message": "💬 Got your response..."})
    asyncio.create_task(_resume_graph(research_id))
    return {"status": "processing"}


# ============================================================
# DEEP RESEARCH - CONFIRM PLAN
# ============================================================
@app.post("/api/v1/deep-research/{research_id}/confirm-plan")
async def confirm_plan(research_id: str, request: PlanActionRequest):
    if research_id not in research_sessions:
        raise HTTPException(404, "Session not found")

    session = research_sessions[research_id]

    if request.action == "cancel":
        session["state"]["status"] = "cancelled"
        await _send_ws(research_id, {"type": "cancelled"})
        return {"status": "cancelled"}

    if request.action == "edit":
        session["state"]["plan_edits"] = request.edits

    session["state"]["plan_approved"] = True
    await _send_ws(research_id, {"type": "progress", "message": "✅ Plan approved"})
    asyncio.create_task(_resume_graph(research_id))
    return {"status": "researching"}


# ============================================================
# DEEP RESEARCH - STATUS (polling fallback)
# ============================================================
@app.get("/api/v1/deep-research/{research_id}/status")
async def research_status(research_id: str):
    if research_id not in research_sessions:
        raise HTTPException(404, "Session not found")

    state = research_sessions[research_id]["state"]
    progress_map = {
        "analyzing": 10, "clarifying": 15, "planning": 20,
        "awaiting_confirmation": 25, "researching": 50,
        "synthesizing": 70, "writing": 85, "completed": 100,
    }

    return {
        "research_id": research_id,
        "status": state.get("status", ""),
        "progress": progress_map.get(state.get("status", ""), 0),
        "progress_messages": state.get("progress", []),
        "todos": state.get("todos", []),
    }


# ============================================================
# DEEP RESEARCH - GET REPORT
# ============================================================
@app.get("/api/v1/deep-research/{research_id}/report")
async def get_report(research_id: str):
    if research_id not in research_sessions:
        raise HTTPException(404, "Session not found")

    state = research_sessions[research_id]["state"]
    if state.get("status") != "completed":
        raise HTTPException(400, f"Not completed. Status: {state.get('status')}")

    sources = []
    seen = set()
    for s in state.get("all_sources", []):
        url = s.get("url", "")
        if url and url not in seen:
            seen.add(url)
            sources.append({"url": url, "title": s.get("title", "")})

    return {
        "report": state.get("report", ""),
        "sources": sources,
        "todos": state.get("todos", []),
        "metadata": {
            "research_loops": state.get("research_loop_count", 0),
            "model": state.get("model", "default"),
        },
    }


# ============================================================
# WEBSOCKET
# ============================================================
@app.websocket("/ws/deep-research/{research_id}")
async def ws_endpoint(websocket: WebSocket, research_id: str):
    await websocket.accept()

    if research_id not in research_sessions:
        await websocket.send_json({"type": "error", "message": "Session not found"})
        await websocket.close()
        return

    ws_connections[research_id] = websocket

    try:
        # Start the graph
        asyncio.create_task(_run_graph(research_id))

        while True:
            try:
                data = await asyncio.wait_for(websocket.receive_json(), timeout=300)
                msg_type = data.get("type", "")

                if msg_type == "clarify":
                    session = research_sessions[research_id]
                    user_message = data.get("message", "")

                    # Append to clarification conversation
                    conversation = session["state"].get("clarification_conversation", [])
                    conversation.append({"role": "user", "content": user_message})
                    session["state"]["clarification_conversation"] = conversation

                    asyncio.create_task(_resume_graph(research_id))

                elif msg_type == "confirm_plan":
                    session = research_sessions[research_id]
                    if data.get("action") == "cancel":
                        session["state"]["status"] = "cancelled"
                        await websocket.send_json({"type": "cancelled"})
                        break
                    session["state"]["plan_edits"] = data.get("edits", {})
                    session["state"]["plan_approved"] = True
                    asyncio.create_task(_resume_graph(research_id))

            except asyncio.TimeoutError:
                await websocket.send_json({"type": "heartbeat"})

    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"[WS ERROR] {e}")
    finally:
        ws_connections.pop(research_id, None)


# ============================================================
# GRAPH EXECUTION
# ============================================================
async def _run_graph(research_id: str):
    """Run the LangGraph from start."""
    session = research_sessions.get(research_id)
    if not session:
        return

    async def rt_progress(msg):
        if isinstance(msg, dict):
            await _send_ws(research_id, {"type": "research_event", **msg})
        else:
            await _send_ws(research_id, {"type": "progress", "message": msg})
    register_progress_callback(research_id, rt_progress)

    graph = get_graph()
    config = session["config"]
    prev_count = 0

    print(f"[GRAPH] Starting research {research_id}")

    try:
        event_count = 0
        async for event in graph.astream(session["state"], config=config, stream_mode="values"):
            event_count += 1
            session["state"] = event
            status = event.get("status", "")
            print(f"[GRAPH] Event #{event_count}, status: {status}")

            await _stream_progress(research_id, event, prev_count)
            prev_count = len(event.get("progress", []))

            # HITL: clarification needed (now sends a chat message)
            if status == "clarifying" and event.get("clarification_message"):
                # Add the LLM's clarification message to the conversation log
                conversation = event.get("clarification_conversation", [])
                conversation.append({"role": "assistant", "content": event["clarification_message"]})
                event["clarification_conversation"] = conversation
                session["state"] = event

                await _send_ws(research_id, {
                    "type": "clarification_needed",
                    "message": event["clarification_message"],
                })
                return

            if status == "awaiting_confirmation" and event.get("plan"):
                await _send_ws(research_id, {
                    "type": "plan_ready",
                    "plan": event["plan"],
                    "todos": event.get("todos", []),
                })
                return

            if status == "completed":
                chat_id = session.get("chat_id", "")
                if chat_id:
                    add_message(chat_id, "assistant", event.get("report", ""), "research_report")

                await _send_ws(research_id, {
                    "type": "report_complete",
                    "report": event.get("report", ""),
                    "sources": event.get("all_sources", []),
                    "todos": event.get("todos", []),
                })
                return

            if status == "error":
                await _send_ws(research_id, {
                    "type": "error",
                    "message": event.get("error", "Unknown error"),
                })
                return

    except Exception as e:
        import traceback
        traceback.print_exc()
        await _send_ws(research_id, {"type": "error", "message": str(e)})

    print(f"[GRAPH] _run_graph ended for {research_id}")


async def _resume_graph(research_id: str):
    """Resume graph after HITL interrupt."""
    session = research_sessions.get(research_id)
    if not session:
        return

    async def rt_progress(msg):
        mtype = msg.get("event", "?") if isinstance(msg, dict) else "text"
        print(f"[RT_PROGRESS] type={mtype}, ws={research_id in ws_connections}")
        if isinstance(msg, dict):
            await _send_ws(research_id, {"type": "research_event", **msg})
        else:
            await _send_ws(research_id, {"type": "progress", "message": msg})
    register_progress_callback(research_id, rt_progress)

    graph = get_graph()
    config = session["config"]
    current_state = session["state"]
    resuming_from_status = current_state.get("status", "")

    print(f"[RESUME] Resuming {research_id}, status: {resuming_from_status}")

    # Determine which node produced the current state so LangGraph
    # knows where to resume from. interrupt_before pauses BEFORE the
    # target node, so the last executed node is the one before it.
    # - "clarifying" status → last node was "analyze_query"
    # - "awaiting_confirmation" status → last node was "generate_plan"
    as_node = None
    if resuming_from_status == "clarifying":
        as_node = "analyze_query"
    elif resuming_from_status == "awaiting_confirmation":
        as_node = "generate_plan"

    try:
        graph.update_state(config, current_state, as_node=as_node)
        print(f"[RESUME] update_state OK, as_node={as_node}")
    except Exception as e:
        print(f"[RESUME] update_state error: {e}")

    prev_count = len(current_state.get("progress", []))

    # Track the previous clarification_message so we can detect NEW ones
    # (vs replayed old ones from checkpoint)
    prev_clarification_msg = current_state.get("clarification_message", "")

    try:
        event_count = 0
        async for event in graph.astream(None, config=config, stream_mode="values"):
            event_count += 1
            session["state"] = event
            status = event.get("status", "")
            print(f"[RESUME] Event #{event_count}, status: {status}")

            await _stream_progress(research_id, event, prev_count)
            prev_count = len(event.get("progress", []))

            # Skip stale checkpoint replays (first event often replays old state)
            if event_count == 1 and status == resuming_from_status:
                print(f"[RESUME] Skipping stale replay event #{event_count}")
                continue

            # HITL: clarification needed again (LLM wants more info)
            if status == "clarifying" and event.get("clarification_message"):
                new_msg = event["clarification_message"]

                # Only send if this is a genuinely new message from the LLM
                if new_msg and new_msg != prev_clarification_msg:
                    conversation = event.get("clarification_conversation", [])
                    conversation.append({"role": "assistant", "content": new_msg})
                    event["clarification_conversation"] = conversation
                    session["state"] = event

                    await _send_ws(research_id, {
                        "type": "clarification_needed",
                        "message": new_msg,
                    })
                    return
                else:
                    # Stale replay of old clarification — skip
                    print(f"[RESUME] Skipping stale clarification replay")
                    continue

            # HITL: plan ready for approval
            if status == "awaiting_confirmation" and event.get("plan"):
                await _send_ws(research_id, {
                    "type": "plan_ready",
                    "plan": event["plan"],
                    "todos": event.get("todos", []),
                })
                return

            if status == "completed":
                chat_id = session.get("chat_id", "")
                if chat_id:
                    add_message(chat_id, "assistant", event.get("report", ""), "research_report")
                await _send_ws(research_id, {
                    "type": "report_complete",
                    "report": event.get("report", ""),
                    "sources": event.get("all_sources", []),
                    "todos": event.get("todos", []),
                })
                return

            if status == "error":
                await _send_ws(research_id, {
                    "type": "error", "message": event.get("error", "Unknown error"),
                })
                return

        print(f"[RESUME] Stream ended, {event_count} events total")

    except Exception as e:
        import traceback
        traceback.print_exc()
        await _send_ws(research_id, {"type": "error", "message": str(e)})


async def _stream_progress(research_id: str, event: dict, prev_count: int):
    """Stream new progress messages to WebSocket."""
    progress = event.get("progress", [])
    for msg in progress[prev_count:]:
        await _send_ws(research_id, {"type": "progress", "message": msg})


async def _send_ws(research_id: str, event: dict):
    ws = ws_connections.get(research_id)
    if ws:
        try:
            await ws.send_json(event)
        except Exception:
            pass


# ============================================================
# RUN
# ============================================================
def run():
    import uvicorn
    uvicorn.run(app, host=HOST, port=
