from agents.router_agent import route_query

from agents.summarizer_agent import summarize_paper
from agents.methodology_agent import methodology_agent
from agents.gap_agent import find_research_gaps
from agents.literature_review_agent import literature_review_agent
from agents.qa_agent import qa_agent

# ROUTER NODE


def router_node(state):

    route = route_query(state["query"])

    return {
        "route": route
    }



# SUMMARY NODE


def summary_node(state):

    response = summarize_paper(
        state["sections"]
    )

    return {
        "response": response
    }



# METHODOLOGY NODE


def methodology_node(state):

    response = methodology_agent(
        state["sections"]
    )

    return {
        "response": response
    }



# GAP NODE


def gap_node(state):

    response = find_research_gaps(
        state["sections"]
    )

    return {
        "response": response
    }



# LITERATURE NODE


def literature_node(state):

    response = literature_review_agent(
        state["sections"]
    )

    return {
        "response": response
    }



# QA NODE


def qa_node(state):

    response = qa_agent(
        state["query"],
        state["vectorstore"]
    )

    return {
        "response": response
    }
