from research_module.llm.chains import ideas_chain


def idea_agent(state):
    topic = state["query"]
    papers = state.get("retrieved_docs", [])
    ranked_papers = state.get("ranked_papers", "")

    # Format papers with abstracts for better context.
    papers_context = _format_papers_with_abstracts(papers, ranked_papers)

    result = ideas_chain.invoke(
        {
            "topic": topic,
            "papers_context": papers_context,
            "web_context": state.get("web_context", ""),
            "session_id": state.get("session_id"),
        }
    )

    return {"ideas": result}


def _format_papers_with_abstracts(papers, ranked_papers):
    """Format papers with abstracts for better context in prompts"""
    if not papers:
        return "(No papers retrieved - ideas will be based on topic and web context)"

    formatted = []
    for i, paper in enumerate(papers[:10], start=1):  # Limit to first 10 papers
        if isinstance(paper, dict):
            title = paper.get("title", "Unknown Title")
            abstract = paper.get("content") or paper.get("abstract") or "No abstract available"
            url = paper.get("url", "")
            authors = paper.get("authors", "")

            # Truncate abstract if too long
            if len(abstract) > 300:
                abstract = abstract[:300] + "..."

            entry = f"[Paper {i}]\nTitle: {title}"
            if authors:
                entry += f"\nAuthors: {authors}"
            entry += f"\nAbstract: {abstract}"
            if url:
                entry += f"\nURL: {url}"
            formatted.append(entry)
        else:
            # Handle string papers
            paper_str = str(paper).strip()
            if len(paper_str) > 300:
                paper_str = paper_str[:300] + "..."
            formatted.append(f"[Paper {i}]\n{paper_str}")

    return "\n\n".join(formatted) if formatted else "(No papers retrieved - ideas will be based on topic and web context)"
