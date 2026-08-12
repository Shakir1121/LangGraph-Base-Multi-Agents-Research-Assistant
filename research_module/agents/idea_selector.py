import logging

from research_module.llm.chains import selector_chain


logger = logging.getLogger(__name__)


def selector_agent(state):
    topic = str(state.get("query", "")).strip()
    ideas = str(state.get("ideas", "")).strip()

    if not ideas:
        logger.warning("No research ideas available for selection.")
        return {
            "selected_idea": "No research idea was generated."
        }

    response = selector_chain.invoke(
        {
            "topic": topic,
            "ideas": ideas[:14000],
        }
    )

    return {
        "selected_idea": str(response or "").strip()
    }