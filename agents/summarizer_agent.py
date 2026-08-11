from research_module.llm.chains import summary_chain


def summarize_paper(sections, llm=None):
    abstract = sections.get("abstract", "")
    introduction = sections.get("introduction", "")
    conclusion = sections.get("conclusion", "")

    context = f"""
ABSTRACT:
{abstract}

INTRODUCTION:
{introduction}

CONCLUSION:
{conclusion}
"""

    return summary_chain.invoke({"context": context})
