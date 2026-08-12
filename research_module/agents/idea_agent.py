import logging
import re

from research_module.llm.chains import ideas_chain


logger = logging.getLogger(__name__)

MAX_PAPERS = 6
MAX_ABSTRACT_CHARS = 500


def idea_agent(state: dict) -> dict:
    """Generate research ideas from the topic and retrieved evidence."""
    topic = str(state.get("query", "")).strip()
    papers = state.get("retrieved_docs") or []

    papers_context = _format_papers(papers)
    web_context = str(state.get("web_context") or "")[:3000]

    logger.info("Generating research ideas.")

    result = ideas_chain.invoke(
        {
            "topic": topic,
            "papers_context": papers_context,
            "web_context": web_context,
        }
    )

    ideas = _clean_idea_output(str(result or "").strip())

    logger.info("Research idea generation completed.")

    return {
        "ideas": ideas
    }


def _format_papers(papers: list) -> str:
    """Format retrieved papers into a compact prompt context."""
    if not papers:
        return "(No papers retrieved.)"

    formatted = []

    for index, paper in enumerate(papers[:MAX_PAPERS], start=1):
        if isinstance(paper, dict):
            title = str(paper.get("title") or "Unknown title").strip()
            abstract = str(
                paper.get("content")
                or paper.get("abstract")
                or ""
            ).strip()
            url = str(paper.get("url") or "").strip()

            abstract = _truncate(abstract, MAX_ABSTRACT_CHARS)

            text = (
                f"[Paper {index}]\n"
                f"Title: {title}\n"
                f"Abstract: {abstract}"
            )

            if url:
                text += f"\nURL: {url}"

        else:
            text = _truncate(str(paper).strip(), MAX_ABSTRACT_CHARS)
            text = f"[Paper {index}]\n{text}"

        formatted.append(text)

    return "\n\n".join(formatted)


def _truncate(text: str, max_length: int) -> str:
    """Limit text length while keeping the output readable."""
    if len(text) <= max_length:
        return text

    return text[:max_length] + "..."


def _clean_idea_output(text: str) -> str:
    """Remove extra ideas and duplicate blocks from the model output."""
    if not text:
        return ""

    text = _keep_first_ten_ideas(text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    blocks = re.split(r"\n\s*\n", text)

    seen = set()
    cleaned = []

    for block in blocks:
        block = block.strip()

        if not block:
            continue

        normalized = re.sub(r"\s+", " ", block).lower()

        if normalized in seen:
            continue

        seen.add(normalized)
        cleaned.append(block)

    return "\n\n".join(cleaned).strip()


def _keep_first_ten_ideas(text: str) -> str:
    """Discard content starting from Idea 11."""
    match = re.search(
        r"(?im)^\s*(?:#{1,6}\s*)?"
        r"(?:\*\*)?\s*idea\s+1[1-9]"
        r"\s*[:.)-]",
        text,
    )

    if match:
        return text[:match.start()].rstrip()

    return text