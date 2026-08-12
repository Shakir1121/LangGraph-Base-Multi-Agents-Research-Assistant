import logging

from research_module.llm.chains import proposal_chain


logger = logging.getLogger(__name__)


def proposal_agent(state):
    logger.info("Generating research proposal.")

    response = proposal_chain.invoke(
        {
            "topic": str(state.get("query", "")).strip(),
            "selected_idea": str(
                state.get("selected_idea", "")
            ).strip()[:8000],
            "gaps": str(
                state.get("gaps", "")
            ).strip()[:8000],
            "methodology": str(
                state.get("methodology", "")
            ).strip()[:14000],
            "ranked_papers": str(
                state.get("ranked_papers", "")
            ).strip()[:7000],
        }
    )

    return {
        "proposal": str(response or "").strip()
    }