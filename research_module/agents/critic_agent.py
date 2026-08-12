import logging

from research_module.llm.chains import critic_chain


logger = logging.getLogger(__name__)


def critic_agent(state: dict) -> dict:
    """Review the generated research proposal."""
    proposal = str(state.get("proposal", "")).strip()

    if not proposal:
        return {
            "review": "No proposal was generated for review."
        }

    logger.info("Reviewing research proposal.")

    review = critic_chain.invoke(
        {
            "proposal": proposal[:16000]
        }
    )

    return {
        "review": str(review or "").strip()
    }