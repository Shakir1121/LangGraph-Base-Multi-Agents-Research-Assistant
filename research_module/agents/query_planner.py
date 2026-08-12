import re

from research_module.llm.chains import query_planner_chain


MAX_QUERIES = 1


def _clean_queries(result):
    if isinstance(result, list):
        lines = [str(item) for item in result]
    else:
        lines = str(result or "").splitlines()

    queries = []

    for line in lines:
        line = line.strip()

        if not line:
            continue

        line = re.sub(
            r"^[\s\-\*\u2022\d\.\)\[]+\s*",
            "",
            line,
        )

        line = re.sub(r"\*\*|__", "", line)

        if any(char in line for char in "\"{}[]:,"):
            continue

        words = re.findall(
            r"[A-Za-z0-9][A-Za-z0-9\-_']*",
            line,
        )

        if not 2 <= len(words) <= 10:
            continue

        queries.append(line)

        if len(queries) >= MAX_QUERIES:
            break

    return queries


def query_planner_agent(state):
    query = str(state.get("query", "")).strip()

    web_context = str(
        state.get("web_context", "")
    ).strip()[:2000]

    result = query_planner_chain.invoke(
        {
            "query": query,
            "web_context": web_context,
        }
    )

    queries = _clean_queries(result)

    if not queries and query:
        queries = [query]

    return {
        "search_queries": queries[:MAX_QUERIES]
    }