import os

from langchain_core.chat_history import InMemoryChatMessageHistory

from research_module.memory.db_chat_history import get_sql_chat_history


_BACKEND = os.getenv(
    "RESEARCH_CHAT_HISTORY_BACKEND",
    "sql",
).strip().lower()

_STORE = {}


def get_session_history(session_id: str):
    session_id = (session_id or "default").strip() or "default"

    if _BACKEND == "memory":
        if session_id not in _STORE:
            _STORE[session_id] = InMemoryChatMessageHistory()

        return _STORE[session_id]

    return get_sql_chat_history(session_id)


def normalize_session_id(session_id: str) -> str:
    session_id = (session_id or "").strip()

    if not session_id:
        return "default"

    return session_id.replace("/", "_").replace("\\", "_")