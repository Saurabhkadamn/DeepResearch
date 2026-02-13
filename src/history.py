"""
Deep Research - Chat History
Real JSON file-based storage. Multiple conversations by chat_id.
"""

import json
import os
import uuid
from datetime import datetime, timezone
from .config import CHAT_HISTORY_FILE, DATA_DIR


def _ensure_file():
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(CHAT_HISTORY_FILE):
        with open(CHAT_HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump({}, f)


def _load() -> dict:
    _ensure_file()
    try:
        with open(CHAT_HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        return {}


def _save(data: dict):
    _ensure_file()
    with open(CHAT_HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def create_chat(title: str = "New Chat") -> str:
    """Create a new chat conversation. Returns chat_id."""
    data = _load()
    chat_id = str(uuid.uuid4())[:8]
    data[chat_id] = {
        "title": title,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "messages": [],
    }
    _save(data)
    return chat_id


def add_message(chat_id: str, role: str, content: str, msg_type: str = "chat") -> dict:
    """
    Add a message to a chat.
    role: "user" | "assistant"
    msg_type: "chat" | "research_report" | "research_plan" | "system"
    """
    data = _load()
    if chat_id not in data:
        data[chat_id] = {
            "title": "Chat",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "messages": [],
        }

    msg = {
        "role": role,
        "content": content,
        "type": msg_type,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    data[chat_id]["messages"].append(msg)

    # Auto-update title from first user message
    if role == "user" and len(data[chat_id]["messages"]) == 1:
        data[chat_id]["title"] = content[:50]

    _save(data)
    return msg


def get_messages(chat_id: str, limit: int = 50) -> list[dict]:
    """Get messages for a chat, most recent `limit` messages."""
    data = _load()
    if chat_id not in data:
        return []
    return data[chat_id]["messages"][-limit:]


def get_messages_for_context(chat_id: str, limit: int = 10) -> list[dict]:
    """
    Get recent messages formatted for LLM context.
    Returns [{"role": "user", "content": "..."}, ...]
    Strips metadata, only role + content.
    """
    messages = get_messages(chat_id, limit)
    return [{"role": m["role"], "content": m["content"]} for m in messages]


def list_chats() -> list[dict]:
    """List all chats with metadata."""
    data = _load()
    chats = []
    for chat_id, chat_data in data.items():
        chats.append({
            "chat_id": chat_id,
            "title": chat_data.get("title", "Untitled"),
            "created_at": chat_data.get("created_at", ""),
            "message_count": len(chat_data.get("messages", [])),
        })
    # Sort by created_at descending
    chats.sort(key=lambda x: x["created_at"], reverse=True)
    return chats


def delete_chat(chat_id: str) -> bool:
    """Delete a chat."""
    data = _load()
    if chat_id in data:
        del data[chat_id]
        _save(data)
        return True
    return False