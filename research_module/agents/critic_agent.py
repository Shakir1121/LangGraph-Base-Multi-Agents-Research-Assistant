from research_module.llm.chains import critic_chain


def critic_agent(state):
    review = critic_chain.invoke(
        {
            "proposal": state.get("proposal", ""),
            "session_id": state.get("session_id"),
        }
    )

    return {"review": review}
