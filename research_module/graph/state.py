from typing import TypedDict, List


class ResearchState(TypedDict, total=False):
    query: str
    session_id: str
    search_queries: str
    web_context: str
    arxiv_papers: List[dict]
    retrieved_docs: List[dict]
    ranked_papers: str
    ideas: str
    selected_idea: str
    gaps: str
    methodology: str
    proposal: str
    review: str
    final_report: str
    final_output: str
