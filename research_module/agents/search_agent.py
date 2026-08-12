from research_module.tools.tavily_tool import tavily_search


def search_agent(state):
    query = str(state.get("query", "")).strip()

    if not query:
        return {
            "web_context": "No research query was provided."
        }

    try:
        results = tavily_search(query) or []
    except Exception as exc:
        return {
            "web_context": f"Search failed: {exc}"
        }

    context_parts = []

    for result in results:
        if not isinstance(result, dict):
            continue

        content = (
            result.get("content")
            or result.get("snippet")
            or result.get("text")
            or result.get("answer")
        )

        if content:
            context_parts.append(str(content).strip())

    web_context = "\n\n".join(
        part for part in context_parts if part
    )

    return {
        "web_context": (
            web_context
            if web_context
            else "No relevant web context found."
        )
    }