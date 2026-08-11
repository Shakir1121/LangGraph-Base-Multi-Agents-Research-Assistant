import re

from research_module.llm.chains import query_planner_chain

# Cap the number of search queries we actually run. Each query triggers a
# multi-source retrieval call (ArXiv + OpenAlex + Tavily) that can take 10-20s.
# Keeping this small is the single biggest lever for reducing total wait time.
# Use a single best query so retrieval completes quickly and the LLM sections
# (the actual visible output) start streaming within a few seconds.
_MAX_QUERIES = 1


def _clean_queries(result) -> list:
    """Convert the LLM's free-form output into a tiny list of clean queries.

    The LLM frequently returns a JSON-ish blob (e.g. ``"has_abstract": true``,
    ``"exclude_terms": [...]``) instead of plain queries. Running retrieval for
    each of those fragments is what makes the pipeline slow. We aggressively
    reject anything that is not a clean, natural-language search query.
    """
    if isinstance(result, list):
        raw_lines = [str(x) for x in result]
    else:
        raw_lines = str(result).split("\n")

    queries = []
    for line in raw_lines:
        line = line.strip()
        if not line:
            continue

        # Strip markdown heading / bold markers.
        line = re.sub(r"^#+\s*", "", line)
        line = re.sub(r"\*\*|__", "", line)
        # Strip leading bullets / numbering ("- ", "1. ", "1) ").
        line = re.sub(r"^[\s\-\*\u2022\d\.\)\[]+\s*", "", line)
        # Tolerate a small trailing markdown artifact.
        if line.endswith("*"):
            line = line[:-1].strip()

        # Hard reject: any JSON / API scaffolding. A real search query never
        # contains quotes, braces, brackets, colons, or JSON keys.
        if any(ch in line for ch in "\"{}[]:,"):
            continue
        if line.startswith(("query", "filters", "include_terms", "exclude_terms",
                            "citation_count", "sort_by", "purpose", "max_results",
                            "api", "the api", "here", "these are", "the following",
                            "for the", "focus on", "output", "prompt", "below")):
            continue

        # Count words; a real query is a short noun phrase (2-8 words).
        words = re.findall(r"[A-Za-z0-9][A-Za-z0-9\-_']*", line)
        if len(words) < 2 or len(words) > 8:
            continue

        queries.append(line)
        if len(queries) >= _MAX_QUERIES:
            break

    if not queries:
        # Last-resort fallback: use the raw original user query.
        return []

    return queries[:_MAX_QUERIES]


def query_planner_agent(state):
    query = state["query"]
    web_context = state.get("web_context", "")

    result = query_planner_chain.invoke(
        {
            "query": query,
            "web_context": web_context,
            "session_id": state.get("session_id"),
        }
    )

    queries = _clean_queries(result)

    return {"search_queries": queries}
