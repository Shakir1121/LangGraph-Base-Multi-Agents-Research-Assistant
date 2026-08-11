from research_module.llm.chains import methodology_paper_chain


def methodology_agent(sections, llm=None):
    methodology = sections.get("methodology", "")
    dataset = sections.get("dataset", "")
    results = sections.get("results", "")

    context = f"""
DATASET:
{dataset}

METHODOLOGY:
{methodology}

RESULTS:
{results}
"""

    return methodology_paper_chain.invoke({"context": context})
