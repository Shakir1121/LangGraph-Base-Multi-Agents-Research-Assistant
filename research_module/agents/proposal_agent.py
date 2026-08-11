from research_module.llm.chains import proposal_chain


def proposal_agent(state):
    response = proposal_chain.invoke(
        {
            "topic": state["query"],
            "selected_idea": state.get("selected_idea", ""),
            "gaps": state.get("gaps", ""),
            "methodology": state.get("methodology", ""),
            "ranked_papers": state.get("ranked_papers", ""),
            "session_id": state.get("session_id"),
        }
    )

    return {"proposal": response}
