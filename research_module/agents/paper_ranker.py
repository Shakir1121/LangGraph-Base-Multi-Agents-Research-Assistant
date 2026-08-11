import logging

from research_module.llm.chains import ranker_chain

logger = logging.getLogger(__name__)

_MAX_PAPERS_TO_RANK = 15
_MAX_ABSTRACT_CHARS = 600


def _compact(papers):
    """Trim each paper down to title + truncated abstract so we don't blow
    the LLM context window when the corpus is large."""
    compact = []
    for i, p in enumerate(papers[:_MAX_PAPERS_TO_RANK], start=1):
        if isinstance(p, dict):
            title = (p.get("title") or "").strip()
            abstract = (p.get("content") or p.get("abstract") or "").strip()
            url = (p.get("url") or "").strip()
        else:
            # Plain string (e.g., from Chroma) - extract first line as title
            paper_str = str(p).strip()
            lines = paper_str.split('\n')
            first_line = lines[0] if lines else ""
            title = first_line if len(first_line) < 100 else first_line[:100]
            abstract = paper_str
            url = ""

        if len(abstract) > _MAX_ABSTRACT_CHARS:
            abstract = abstract[:_MAX_ABSTRACT_CHARS].rsplit(" ", 1)[0] + "…"

        title_str = title if title else "(No title available)"
        url_str = f"URL: {url}" if url else "URL: (not available)"
        compact.append(f"[{i}] {title_str}\n{url_str}\nAbstract: {abstract}")

    if not compact:
        return "(No papers to rank)"
    return "\n\n".join(compact)


def paper_ranker_agent(state):
    """Rank papers from hybrid retrieval results (arxiv, openalex, tavily)"""

    # Collect all papers from hybrid retrieval sources
    arxiv_papers = state.get("arxiv_papers", []) or []
    openalex_papers = state.get("openalex_papers", []) or []
    tavily_papers = state.get("tavily_papers", []) or []

    # Combine all papers
    all_papers = arxiv_papers + openalex_papers + tavily_papers

    # If no papers available, skip ranking
    if not all_papers:
        logger.warning("⚠️ paper_ranker: no papers to rank from hybrid retrieval — skipping.")
        return {
            "ranked_papers": (
                "**No papers available for ranking.** The research will proceed with web context and generated ideas."
            )
        }

    # If very few papers (<=2), provide simple message instead of LLM ranking
    if len(all_papers) <= 2:
        simple_ranking = "\n".join([
            f"{i}. {p.get('title', p.get('content', 'Paper without title')[:100])}"
            for i, p in enumerate(all_papers, 1)
        ])
        logger.info(f"✅ Listed {len(all_papers)} paper(s) (too few to rank)")
        return {"ranked_papers": simple_ranking}

    compact_papers = _compact(all_papers)
    num_to_rank = min(10, len(all_papers))
    query = state.get("query", "(no query)")

    result = ranker_chain.invoke(
        {
            "query": query,
            "num_to_rank": num_to_rank,
            "compact_papers": compact_papers,
            "session_id": state.get("session_id"),
        }
    )

    logger.info(f"✅ Ranked {num_to_rank} papers from {len(all_papers)} total papers retrieved")
    return {"ranked_papers": result}
