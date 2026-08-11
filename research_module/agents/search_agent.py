from research_module.tools.tavily_tool import tavily_search


def search_agent(state):
    query = state.get("query", "").strip()

    try:
        results = tavily_search(query) or []
    except Exception as e:
        return {"web_context": f"Search failed: {str(e)}"}

    context_parts = []

    for r in results:
        if not isinstance(r, dict):
            continue

        content = (
            r.get("content")
            or r.get("snippet")
            or r.get("text")
            or r.get("answer")
        )

        if content:
            context_parts.append(content)

    web_context = "\n\n".join(context_parts).strip()

    if not web_context:
        web_context = "No relevant web context found."

    return {"web_context": web_context}