import os
from typing import Optional

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import SQLChatMessageHistory


def get_sql_chat_history(
    session_id: str,
    db_url: Optional[str] = None,
) -> BaseChatMessageHistory:
    """Return persistent chat history backed by SQLChatMessageHistory.

    Uses SQLite by default (zero setup, no cloud charges).
    """
    session_id = session_id or "default"

    if not db_url:
        # Put DB under research_module/memory so it ships with the project.
        base_dir = os.path.dirname(__file__)
        db_path = os.path.join(base_dir, "chat_history.sqlite3")
        db_url = f"sqlite:///{db_path}"

    return SQLChatMessageHistory(
        session_id=session_id,
        connection_string=db_url,
        # When using SQLite, SQLChatMessageHistory creates required tables.
    )

