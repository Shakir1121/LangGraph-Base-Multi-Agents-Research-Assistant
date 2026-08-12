import logging

from research_module.llm.chains import gaps_chain


logger = logging.getLogger(__name__)


def gap_agent(state: dict) -> dict:
    """Generate research gaps from the selected idea and supporting evidence."""
    topic = str(state.get("query", "")).strip()
    selected_idea = str(state.get("selected_idea", "")).strip()

    papers = state.get("retrieved_docs") or []
    web_context = str(state.get("web_context", ""))

    logger.info("Identifying research gaps.")

    result = gaps_chain.invoke(
        {
            "topic": topic,
            "selected_idea": selected_idea[:7000],
            "papers": str(papers)[:7000],
            "web_context": web_context[:3000],
        }
    )

    return {
        "gaps": str(result or "").strip()
    }