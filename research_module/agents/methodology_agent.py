import logging

from research_module.llm.chains import methodology_chain


logger = logging.getLogger(__name__)


def methodology_agent(state):
    logger.info("Generating methodology.")

    response = methodology_chain.invoke(
        {
            "topic": str(state.get("query", "")).strip(),
            "selected_idea": str(
                state.get("selected_idea", "")
            ).strip()[:7000],
            "gaps": str(
                state.get("gaps", "")
            ).strip()[:8000],
            "papers": str(
                state.get("retrieved_docs", [])
            )[:7000],
        }
    )

    return {
        "methodology": str(response or "").strip()
    }