"""
Deep Research - Todo Planning Tool
Borrowed from deepagents pattern.
A structured planning tool that forces the agent to think before acting.
The todo list is stored in state, not in an external system.
"""


def write_todos(todos: list[dict], state_todos: list[dict]) -> list[dict]:
    """
    Write/update the todo list.
    Each todo: {"id": str, "task": str, "status": "pending"|"in_progress"|"done"}

    This is essentially a no-op from a system perspective —
    the real value is forcing the LLM to plan and track progress.
    """
    return todos


def read_todos(state_todos: list[dict]) -> list[dict]:
    """Read the current todo list."""
    return state_todos


def update_todo_status(todo_id: str, status: str, state_todos: list[dict]) -> list[dict]:
    """Mark a specific todo as done/in_progress."""
    for todo in state_todos:
        if todo["id"] == todo_id:
            todo["status"] = status
            break
    return state_todos


def format_todos_for_prompt(todos: list[dict]) -> str:
    """Format todos for inclusion in LLM prompts."""
    if not todos:
        return "No tasks planned yet."

    lines = ["## Current Task List"]
    for t in todos:
        status_icon = {"pending": "⬜", "in_progress": "🔄", "done": "✅"}.get(t.get("status", "pending"), "⬜")
        lines.append(f"{status_icon} [{t['id']}] {t['task']}")
    return "\n".join(lines)
