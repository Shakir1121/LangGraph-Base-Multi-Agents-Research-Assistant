from langgraph.graph import StateGraph, END

from langgraph_flow.state import ResearchState

from langgraph_flow.nodes import (
    router_node,
    summary_node,
    methodology_node,
    gap_node,
    literature_node,
    qa_node
)



# ROUTING LOGIC


def route_decision(state):

    return state["route"]



# BUILD GRAPH


def build_workflow():

    workflow = StateGraph(ResearchState)

    workflow.add_node("router", router_node)

    workflow.add_node("summary", summary_node)

    workflow.add_node("methodology", methodology_node)

    workflow.add_node("gap", gap_node)

    workflow.add_node("literature", literature_node)

    workflow.add_node("qa", qa_node)

    workflow.set_entry_point("router")

    workflow.add_conditional_edges(
        "router",
        route_decision,
        {
            "summary": "summary",
            "methodology": "methodology",
            "gap": "gap",
            "literature_review": "literature",
            "qa": "qa"
        }
    )

    workflow.add_edge("summary", END)

    workflow.add_edge("methodology", END)

    workflow.add_edge("gap", END)

    workflow.add_edge("literature", END)

    workflow.add_edge("qa", END)

    return workflow.compile()

