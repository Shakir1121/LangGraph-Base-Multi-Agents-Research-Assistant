from research_module.llm.chains import literature_chain


def literature_review_agent(sections, llm=None):
    introduction = sections.get("introduction", "")
    methodology = sections.get("methodology", "")
    references = sections.get("references", "")

    context = f"""
INTRODUCTION:
{introduction}

METHODOLOGY:
{methodology}

REFERENCES:
{references}
"""

    return literature_chain.invoke({"context": context})
