from typing import Any, TypedDict


class ResearchState(TypedDict, total=False):
    query: str
    session_id: str

    search_queries: list[str]
    web_context: str

    arxiv_papers: list[dict[str, Any]]
    openalex_papers: list[dict[str, Any]]
    tavily_papers: list[dict[str, Any]]

    retrieved_docs: list[Any]

    ranked_papers: str
    ideas: str
    selected_idea: str
    gaps: str
    methodology: str
    proposal: str
    review: str

    final_report: str
    final_output: str