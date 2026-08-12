import os
from typing import Optional

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import SQLChatMessageHistory


def get_sql_chat_history(
    session_id: str,
    db_url: Optional[str] = None,
) -> BaseChatMessageHistory:
    session_id = session_id or "default"

    if db_url is None:
        db_path = os.path.join(
            os.path.dirname(__file__),
            "chat_history.sqlite3",
        )
        db_url = f"sqlite:///{db_path}"

    return SQLChatMessageHistory(
        session_id=session_id,
        connection_string=db_url,
    )