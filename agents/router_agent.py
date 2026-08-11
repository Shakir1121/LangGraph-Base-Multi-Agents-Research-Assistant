def route_query(query):

    query = query.lower()

    if "summary" in query or "summarize" in query:
        return "summary"

    elif "gap" in query or "limitation" in query:
        return "gap"

    elif "method" in query or "methodology" in query:
        return "methodology"

    elif "literature" in query or "related work" in query:
        return "literature_review"

    else:
        return "qa"