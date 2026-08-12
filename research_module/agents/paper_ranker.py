import logging
from typing import Any

from research_module.llm.chains import ranker_chain


logger = logging.getLogger(__name__)


MAX_PAPERS_TO_RANK = 15
MAX_ABSTRACT_CHARS = 700
MAX_CONTEXT_CHARS = 10000
MAX_RANKED_PAPERS = 10


def _text(value: Any) -> str:
    return str(value or "").strip()


def _paper_fields(paper: Any) -> tuple[str, str, str]:
    if isinstance(paper, dict):
        title = _text(paper.get("title"))
        abstract = _text(
            paper.get("content")
            or paper.get("abstract")
            or paper.get("snippet")
            or paper.get("summary")
            or paper.get("text")
        )
        url = _text(
            paper.get("url")
            or paper.get("link")
            or paper.get("pdf_url")
            or paper.get("paper_url")
        )
        return title, abstract, url

    page_content = getattr(paper, "page_content", None)

    if page_content is not None:
        metadata = getattr(paper, "metadata", {})
        metadata = metadata if isinstance(metadata, dict) else {}

        title = _text(metadata.get("title"))
        url = _text(
            metadata.get("url")
            or metadata.get("source")
        )

        return title, _text(page_content), url

    paper_text = _text(paper)

    if not paper_text:
        return "", "", ""

    lines = [
        line.strip()
        for line in paper_text.splitlines()
        if line.strip()
    ]

    title = lines[0] if lines and len(lines[0]) <= 150 else ""

    return title, paper_text, ""


def _truncate(text: str, limit: int) -> str:
    text = _text(text)

    if len(text) <= limit:
        return text

    text = text[:limit]

    if " " in text:
        text = text.rsplit(" ", 1)[0]

    return text + "..."


def _build_paper_context(papers: list[Any]) -> str:
    if not papers:
        return "(No papers available to rank.)"

    blocks = []
    total_chars = 0

    for index, paper in enumerate(
        papers[:MAX_PAPERS_TO_RANK],
        start=1,
    ):
        title, abstract, url = _paper_fields(paper)

        if not title:
            title = _truncate(abstract, 100) or "No title available"

        abstract = (
            _truncate(abstract, MAX_ABSTRACT_CHARS)
            or "No abstract/content available."
        )

        url = url or "(not available)"

        block = (
            f"[{index}] {title}\n"
            f"URL: {url}\n"
            f"Abstract: {abstract}"
        )

        if total_chars + len(block) > MAX_CONTEXT_CHARS:
            break

        blocks.append(block)
        total_chars += len(block)

    return "\n\n".join(blocks) or "(No papers available to rank.)"


def _fallback_ranking(papers: list[Any]) -> str:
    results = []

    for index, paper in enumerate(papers, start=1):
        title, _, url = _paper_fields(paper)

        title = title or "Paper without title"
        url = url or "(URL not available)"

        results.append(
            f"{index}. {title} — {url}"
        )

    return "\n".join(results)


def paper_ranker_agent(state: dict):
    arxiv_papers = state.get("arxiv_papers", []) or []
    openalex_papers = state.get("openalex_papers", []) or []
    tavily_papers = state.get("tavily_papers", []) or []

    papers = (
        list(arxiv_papers)
        + list(openalex_papers)
        + list(tavily_papers)
    )

    logger.info(
        "Ranking papers: arXiv=%d, OpenAlex=%d, Tavily=%d",
        len(arxiv_papers),
        len(openalex_papers),
        len(tavily_papers),
    )

    if not papers:
        logger.warning("No papers available for ranking.")
        return {
            "ranked_papers": (
                "No papers were available for ranking."
            )
        }

    unique_papers = []
    seen = set()

    for paper in papers:
        title, _, url = _paper_fields(paper)
        identity = (title or url).lower()

        if identity and identity in seen:
            continue

        if identity:
            seen.add(identity)

        unique_papers.append(paper)

    papers = unique_papers

    if len(papers) <= 2:
        return {
            "ranked_papers": _fallback_ranking(papers)
        }

    query = _text(state.get("query")) or "Research topic not specified"
    context = _build_paper_context(papers)

    num_to_rank = min(
        MAX_RANKED_PAPERS,
        len(papers),
    )

    logger.info(
        "Selecting top %d papers for '%s'",
        num_to_rank,
        query,
    )

    try:
        result = ranker_chain.invoke(
            {
                "query": query,
                "num_to_rank": num_to_rank,
                "compact_papers": context,
            }
        )
    except Exception:
        logger.exception("Paper ranking failed.")
        return {
            "ranked_papers": _fallback_ranking(
                papers[:num_to_rank]
            )
        }

    if hasattr(result, "content"):
        result = result.content

    result = _text(result)

    if not result:
        result = _fallback_ranking(
            papers[:num_to_rank]
        )

    logger.info("Paper ranking completed.")

    return {
        "ranked_papers": result
    }