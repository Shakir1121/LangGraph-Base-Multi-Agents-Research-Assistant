import os

from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI


load_dotenv()


MODEL = os.getenv(
    "MISTRAL_MODEL",
    "open-mistral-nemo",
)

API_KEY = os.getenv("MISTRAL_API_KEY")


def get_llm() -> ChatMistralAI:
    if not API_KEY:
        raise ValueError(
            "MISTRAL_API_KEY is not configured."
        )

    return ChatMistralAI(
        model=MODEL,
        temperature=0.3,
        api_key=API_KEY,
        streaming=True,
    )


def llm(prompt: str) -> str:
    prompt = str(prompt or "").strip()

    if not prompt:
        return ""

    try:
        response = get_llm().invoke(prompt)
        return str(
            getattr(response, "content", "")
            or ""
        ).strip()

    except Exception as exc:
        raise RuntimeError(
            f"Mistral request failed: {exc}"
        ) from exc


def stream_llm(prompt: str):
    prompt = str(prompt or "").strip()

    if not prompt:
        return

    try:
        for chunk in get_llm().stream(prompt):
            content = getattr(
                chunk,
                "content",
                "",
            )

            if content:
                yield content

    except Exception as exc:
        raise RuntimeError(
            f"Mistral streaming failed: {exc}"
        ) from exc