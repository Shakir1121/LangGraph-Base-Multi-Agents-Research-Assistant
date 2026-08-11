from research_module.llm.chains import gaps_paper_chain


def find_research_gaps(sections, llm=None):
    introduction = sections.get("introduction", "")
    methodology = sections.get("methodology", "")
    results = sections.get("results", "")
    conclusion = sections.get("conclusion", "")

    context = f"""
INTRODUCTION:
{introduction}

METHODOLOGY:
{methodology}

RESULTS:
{results}

CONCLUSION:
{conclusion}
"""

    return gaps_paper_chain.invoke({"context": context})
