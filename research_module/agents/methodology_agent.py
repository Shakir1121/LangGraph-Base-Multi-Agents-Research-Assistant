from research_module.llm.chains import methodology_chain


def methodology_agent(state):
    response = methodology_chain.invoke(
        {
            "topic": state["query"],
            "selected_idea": state.get("selected_idea", ""),
            "gaps": state.get("gaps", ""),
            "papers": state.get("retrieved_docs", []),
            "session_id": state.get("session_id"),
        }
    )

    return {"methodology": response}
