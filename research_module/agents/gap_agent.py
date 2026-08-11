from research_module.llm.chains import gaps_chain


def gap_agent(state):
    result = gaps_chain.invoke(
        {
            "topic": state["query"],
            "selected_idea": state.get("selected_idea", ""),
            "papers": state.get("retrieved_docs", []),
            "web_context": state.get("web_context", ""),
            "session_id": state.get("session_id"),
        }
    )

    return {"gaps": result}
