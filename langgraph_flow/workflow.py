from langgraph.graph import END, StateGraph

from langgraph_flow.nodes import (
    gap_node,
    literature_node,
    methodology_node,
    qa_node,
    router_node,
    summary_node,
)
from langgraph_flow.state import ResearchState


def route_decision(state):
    return state["route"]


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
            "qa": "qa",
        },
    )

    workflow.add_edge("summary", END)
    workflow.add_edge("methodology", END)
    workflow.add_edge("gap", END)
    workflow.add_edge("literature", END)
    workflow.add_edge("qa", END)

    return workflow.compile()