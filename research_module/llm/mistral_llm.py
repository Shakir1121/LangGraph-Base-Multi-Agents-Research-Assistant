import os

from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI

load_dotenv()

MODEL = os.getenv("MISTRAL_MODEL", "open-mistral-nemo")


def get_llm():
    """Return a ChatMistralAI instance configured from the environment.

    ``streaming=True`` lets LangGraph's ``astream_events`` emit per-token
    ``on_chat_model_stream`` events even when the runnable is invoked via the
    blocking ``.invoke()`` call inside graph nodes. This lets the UI stream the
    response token-by-token (ChatGPT-style typing effect) without changing the
    final output text.
    """
    return ChatMistralAI(
        model=MODEL,
        temperature=0.3,
        streaming=True,
        api_key=os.getenv("MISTRAL_API_KEY"),
    )


def llm(prompt: str) -> str:
    """Single-turn Mistral completion (no memory)."""
    try:
        response = get_llm().invoke(prompt)
        return response.content
    except Exception as e:
        return f"[MISTRAL ERROR] {e}"


def _truncate_prompt(text: str, max_chars: int = 3500) -> str:
    """Hard truncate prompt to avoid LLM request-too-large errors."""
    if not text:
        return ""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n\n[TRUNCATED: input too large]"


def _build_messages(prompt: str, session_id: str | None = None) -> list[dict]:
    """Build the full message list for a session, prepending chat history.

    Returns a list of ``{"role": ..., "content": ...}`` dicts suitable for
    both the Mistral client and LangChain message objects.
    """
    from langchain_core.messages import AIMessage, HumanMessage
    from research_module.memory.session_memory import get_session_history, normalize_session_id

    history = get_session_history(normalize_session_id(session_id or "default"))

    prior_messages = []
    for m in getattr(history, "messages", []):
        role = None
        content = getattr(m, "content", None)
        if isinstance(m, HumanMessage):
            role = "user"
        elif isinstance(m, AIMessage):
            role = "assistant"
        else:
            role = "user" if getattr(m, "type", "") == "human" else "assistant"
        if content:
            prior_messages.append({"role": role, "content": content})

    # For the idea-generator pipeline, later agents (gaps/methodology/proposal/
    # critic) copy the format of earlier sections when full history is injected.
    # Cap to only the most recent turn(s) so each node stays on-task and does
    # not echo the previous section's boilerplate.
    prior_messages = prior_messages[-2:]

    return prior_messages + [{"role": "user", "content": prompt}]


def _persist_turn(prompt: str, response: str, session_id: str | None = None) -> None:
    """Persist a user/assistant turn to the session's chat history."""
    from research_module.memory.session_memory import get_session_history, normalize_session_id

    history = get_session_history(normalize_session_id(session_id or "default"))
    history.add_user_message(prompt)
    history.add_ai_message(response)


def llm_with_memory(prompt: str, session_id: str | None = None) -> str:
    """Invoke Mistral while prepending LangChain chat history for the session.

    Reads prior messages from history for this session_id, sends them as part
    of the LLM `messages` payload, then persists the new turn.
    """
    prompt = _truncate_prompt(prompt)
    messages = _build_messages(prompt, session_id)

    try:
        response = get_llm().invoke(messages)
        text = response.content
        _persist_turn(prompt, text, session_id)
        return text
    except Exception as e:
        return f"[MISTRAL ERROR] {e}"


def stream_llm_with_memory(prompt: str, session_id: str | None = None):
    """Stream Mistral tokens while prepending chat history for the session.

    Yields each text token as it arrives and persists the completed turn to
    the session's chat history once the stream finishes.
    """
    prompt = _truncate_prompt(prompt)
    messages = _build_messages(prompt, session_id)

    try:
        text = ""
        for chunk in get_llm().stream(messages):
            token = getattr(chunk, "content", "")
            if token:
                text += token
                yield token
        _persist_turn(prompt, text, session_id)
    except Exception as e:
        yield f"[MISTRAL ERROR] {e}"
