from research_module.llm.chains import selector_chain


def selector_agent(state):
    topic = state["query"]
    ideas = state["ideas"]

    response = selector_chain.invoke(
        {
            "topic": topic,
            "ideas": ideas,
            "session_id": state.get("session_id"),
        }
    )

    return {"selected_idea": response}
