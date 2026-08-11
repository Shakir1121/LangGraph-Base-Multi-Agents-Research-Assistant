import os
from typing import Dict

from langchain_core.chat_history import InMemoryChatMessageHistory

from research_module.memory.db_chat_history import get_sql_chat_history


# Default backend is SQLite; set RESEARCH_CHAT_HISTORY_BACKEND=memory to
# use an in-memory chat history instead.
_BACKEND = os.getenv("RESEARCH_CHAT_HISTORY_BACKEND", "sql").lower().strip()

# In-process fallback store keyed by session_id.
_STORE: Dict[str, InMemoryChatMessageHistory] = {}


def get_session_history(session_id: str):
    """Return a LangChain ChatMessageHistory for the given session_id."""
    session_id = (session_id or "default").strip() or "default"

    if _BACKEND == "memory":
        if session_id not in _STORE:
            _STORE[session_id] = InMemoryChatMessageHistory()
        return _STORE[session_id]

    return get_sql_chat_history(session_id)


def normalize_session_id(session_id: str) -> str:
    """Normalize session_id to a filesystem-safe token."""
    session_id = (session_id or "").strip()
    if not session_id:
        return "default"
    return session_id.replace("/", "_").replace("\\", "_")
