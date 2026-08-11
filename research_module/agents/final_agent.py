def final_agent(state):
    report = f"""


FINAL STRUCTURED RESEARCH REPORT


RESEARCH TOPIC

{state['query']}


SEARCH QUERIES


{state.get('search_queries', '')}


RESEARCH IDEAS


{state.get('ideas', '')}


SELECTED IDEA


{state.get('selected_idea', '')}

RESEARCH GAPS


{state.get('gaps', '')}


METHODOLOGY


{state.get('methodology', '')}


FINAL RESEARCH PROPOSAL


{state.get('proposal', '')}


CRITIC REVIEW


{state.get('review', '')}
"""

    return {"final_report": report}