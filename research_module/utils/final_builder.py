from research_module.utils.output_cleaner import OutputCleaner


def build_final_report(state):
    cleaner = OutputCleaner()

    final_report = state.get("final_report", "")

    if not final_report:
        final_report = f"""
# FINAL STRUCTURED RESEARCH REPORT

## Topic
{state.get('query', '')}

## Selected Idea
{state.get('selected_idea', '')}

## Research Gaps
{state.get('gaps', '')}

## Methodology
{state.get('methodology', '')}

## Research Proposal
{state.get('proposal', '')}

## Critic Review
{state.get('review', '')}
"""

    rag_count = len(state.get("retrieved_docs", []))

    wrapped = f"""
{final_report}

## RAG Status
Retrieved documents used: {rag_count}
"""

    cleaned = cleaner.clean(wrapped)

    return {"final_output": cleaned if cleaned else wrapped}
