from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

from research_module.llm.mistral_llm import get_llm


def _build_chain(prompt: ChatPromptTemplate):
    return prompt | get_llm() | StrOutputParser()


_query_planner_template = ChatPromptTemplate.from_template(
    """
You are an academic research search-query planner.

Research topic:
{query}

Web context:
{web_context}

Generate exactly ONE strong academic search query.

Rules:
- One line only.
- 3-10 meaningful words.
- No numbering.
- No bullets.
- No JSON.
- No explanation.
- Focus on the research problem and technical domain.

Output ONLY the query.
"""
)


_ideas_template = ChatPromptTemplate.from_template(
    """
You are a senior academic researcher and research supervisor.

Research Topic:
{topic}

Relevant Research Papers:
{papers_context}

Current Web Context:
{web_context}

TASK
Generate EXACTLY 10 distinct and publication-oriented research ideas.

The ideas must be meaningfully different from one another.

For EVERY idea use exactly this structure:

## Idea 1: <Research Title>

### Problem Statement
<2-4 sentences explaining the specific real-world or scientific problem,
why it matters, what existing systems fail to solve, and what gap motivates
this research.>

### Proposed Research Direction
<2-4 sentences explaining the proposed research direction and the main
technical concept.>

### Dataset Direction
<Name a realistic dataset type, source, or dataset characteristics that
would be appropriate. If a specific public dataset is uncertain, describe
the required dataset characteristics instead of inventing a dataset.>

### Novelty
<1-2 sentences describing what could make this research different from
existing work.>

### Expected Contribution
<1-2 sentences describing the expected scientific or practical contribution.>

### Feasibility
<1 sentence explaining why the project is realistically implementable.>

Repeat the same structure for Idea 2 through Idea 10.

STRICT RULES:

1. EXACTLY 10 ideas.
2. Never generate Idea 11.
3. Every idea must be distinct.
4. Do not repeat the same research problem with different wording.
5. Do not copy paragraphs between ideas.
6. Do not create a generic introduction.
7. Do not create a conclusion.
8. Do not generate a complete methodology for every idea.
9. Do not generate a complete proposal for every idea.
10. Do not fabricate specific dataset statistics.
11. Dataset information must be realistic and clearly qualified.
12. Keep each idea detailed but concise.
13. Focus on research novelty and publication potential.
14. Stop immediately after Idea 10.
"""
)


_selector_template = ChatPromptTemplate.from_template(
    """
You are a strict academic research supervisor.

Research Topic:
{topic}

Candidate Research Ideas:
{ideas}

Select exactly ONE strongest research idea.

Evaluate:

- Novelty
- Research gap
- Technical depth
- Dataset availability
- Feasibility
- Experimental potential
- Publication potential
- Reproducibility

Return ONLY:

# Selected Research Idea

## Title
<exact selected title>

## Selection Rationale
<one concise paragraph>

## Why This Idea Is Strong

### Novelty
<short explanation>

### Technical Depth
<short explanation>

### Dataset Feasibility
<short explanation>

### Publication Potential
<short explanation>

## Recommended Research Direction
<short explanation>

IMPORTANT:
- Select only ONE idea.
- Do not reproduce all ten ideas.
- Do not write the complete proposal.
- Do not repeat large portions of the candidate ideas.
"""
)


_gaps_template = ChatPromptTemplate.from_template(
    """
You are an expert academic researcher.

Research Topic:
{topic}

Selected Research Idea:
{selected_idea}

Relevant Papers:
{papers}

Web Context:
{web_context}

Identify the most important research gaps that directly justify the
selected research idea.

Return exactly 4-6 numbered gaps.

For each gap use:

### Gap 1
**Existing Limitation:** <specific limitation>

**Why It Matters:** <why the limitation is scientifically important>

**Opportunity:** <how the proposed research can address it>

STRICT RULES:
- Every gap must be specific.
- Do not repeat the selected idea.
- Do not repeat the same gap using different wording.
- Do not write methodology.
- Do not write a proposal.
- Do not write a conclusion.
- Do not invent unsupported claims.
"""
)


_methodology_template = ChatPromptTemplate.from_template(
    """
You are an expert AI/ML research scientist.

Research Topic:
{topic}

Selected Research Idea:
{selected_idea}

Research Gaps:
{gaps}

Relevant Papers:
{papers}

Create a detailed and implementable methodology.

Use EXACTLY these sections:

# Methodology

## 1. Dataset and Data Sources

Explain:
- Dataset type
- Possible public datasets
- Required variables/features
- Target variable if applicable
- Expected data format
- Dataset size requirements
- Train/validation/test split

Do not invent exact dataset statistics unless supported by the supplied
context.

## 2. Data Preprocessing

Explain:
- Cleaning
- Missing values
- Duplicates
- Outliers
- Encoding
- Normalization/scaling
- Class imbalance if applicable

## 3. Feature Engineering

Explain the important features and transformations.

## 4. Proposed Architecture

Describe the complete system architecture step by step.

## 5. Baseline Models

List suitable baseline approaches and explain why they are needed.

## 6. Proposed Model

Explain the proposed model and how it differs from the baselines.

## 7. Training Strategy

Explain:
- Loss function
- Optimizer
- Hyperparameter tuning
- Cross-validation
- Early stopping where appropriate

## 8. Experimental Design

Explain the experiments required to test the research hypothesis.

## 9. Evaluation Metrics

Select metrics appropriate for the task and explain each metric.

## 10. Ablation and Comparative Experiments

Explain what components should be removed or changed to verify the
contribution of the proposed approach.

## 11. Reproducibility

Explain seeds, configuration management, experiment tracking and
documentation.

## 12. Deployment or Practical Validation

Explain how the final system could be evaluated in a realistic setting.

## 13. Challenges and Risk Mitigation

Identify realistic technical risks and corresponding solutions.

IMPORTANT:
- Do not repeat the full research proposal.
- Do not write a generic conclusion.
- Avoid repeating the same paragraph.
- Keep the methodology technically actionable.
"""
)


_proposal_template = ChatPromptTemplate.from_template(
    """
You are an academic researcher writing a Master's-level research proposal.

Research Topic:
{topic}

Selected Research Idea:
{selected_idea}

Research Gaps:
{gaps}

Methodology:
{methodology}

Relevant Ranked Papers:
{ranked_papers}

Write ONE coherent research proposal focused ONLY on the selected idea.

Use EXACTLY this order:

# Research Proposal

## 1. Research Title

## 2. Abstract

## 3. Background and Context

## 4. Detailed Problem Statement

Explain:
- The real-world/scientific problem
- Why it matters
- Current limitations
- Who/what is affected
- Why existing approaches are insufficient

## 5. Research Aim

## 6. Research Objectives

Provide 4-6 measurable objectives.

## 7. Research Questions

Provide 3-5 research questions.

## 8. Literature Gap

Connect the research gaps to the proposed work without copying the
entire earlier gap section.

## 9. Proposed Solution

Explain the high-level solution.

## 10. Dataset and Data Requirements

Explain:
- Dataset source/type
- Features
- Target
- Data requirements
- Preprocessing requirements
- Train/validation/test strategy

Do not fabricate exact statistics.

## 11. Proposed Methodology

Summarize the detailed methodology in a proposal-friendly form.

## 12. Experimental Design

Describe the experiments and comparisons.

## 13. Evaluation Metrics

Explain the selected metrics.

## 14. Expected Results

Describe expected outcomes without claiming results that have not
actually been obtained.

## 15. Expected Research Contributions

List the scientific and practical contributions.

## 16. Limitations

Describe realistic limitations.

## 17. Future Work

Describe realistic extensions.

## 18. Conclusion

Provide a concise proposal conclusion.

IMPORTANT:
- Do not reproduce the ten original ideas.
- Do not repeat the methodology word-for-word.
- Do not claim experiments have already been completed.
- Do not invent numerical results.
- Do not fabricate dataset statistics.
- Avoid duplicate paragraphs.
"""
)


_critic_template = ChatPromptTemplate.from_template(
    """
You are a senior academic research reviewer.

Research Proposal:
{proposal}

Review the proposal constructively.

Return EXACTLY:

## Strengths

1. <point>
2. <point>
3. <point>

## Weaknesses

1. <point>
2. <point>
3. <point>

## Specific Improvements

1. <point>
2. <point>
3. <point>

## Overall Score

<score>/10

## Publication Readiness

<Low / Moderate / High> — <one sentence explanation>

Rules:
- Do not rewrite the proposal.
- Do not repeat large portions of it.
- Do not invent experimental results.
"""
)


_ranker_template = ChatPromptTemplate.from_template(
    """
You are ranking academic papers for:

Research Topic:
{query}

Candidate Papers:
{compact_papers}

Rank the top {num_to_rank} most relevant papers.

Criteria:
- Relevance: 50%
- Novelty/recency: 25%
- Technical depth: 25%

Return:

1. Title — short reason — URL
2. Title — short reason — URL

Continue for the requested number.

Do not invent paper titles or URLs.
Do not ask for additional papers.
"""
)


query_planner_chain = _build_chain(_query_planner_template)
ideas_chain = _build_chain(_ideas_template)
selector_chain = _build_chain(_selector_template)
gaps_chain = _build_chain(_gaps_template)
methodology_chain = _build_chain(_methodology_template)
proposal_chain = _build_chain(_proposal_template)
critic_chain = _build_chain(_critic_template)
ranker_chain = _build_chain(_ranker_template)


_qa_template = ChatPromptTemplate.from_template(
    """
You are an AI Research Assistant.

Answer the question ONLY from the supplied research paper context.

Context:
{context}

Question:
{query}

Rules:
- Use only the supplied context.
- If the answer is unavailable, clearly say so.
- Do not invent information.
- Give a clear and accurate answer.
"""
)


_summary_template = ChatPromptTemplate.from_template(
    """
You are an AI Research Assistant.

Summarize the research paper using ONLY the supplied context.

Include:

1. Main Problem
2. Proposed Method
3. Key Contributions
4. Main Results
5. Conclusion

Paper:
{context}
"""
)


_methodology_paper_template = ChatPromptTemplate.from_template(
    """
You are an AI research analyst.

Extract the methodology from the supplied research paper.

Include:

- Model architecture
- Algorithms
- Training approach
- Dataset usage
- Evaluation process

Paper:
{context}
"""
)


_gaps_paper_template = ChatPromptTemplate.from_template(
    """
You are an expert AI research analyst.

Analyze the research paper and identify:

1. Research limitations
2. Methodology weaknesses
3. Missing experiments
4. Future work opportunities
5. Potential improvements

Return concise bullet points.

Paper:
{context}
"""
)


_literature_template = ChatPromptTemplate.from_template(
    """
You are an expert academic researcher.

Generate a literature review based ONLY on the supplied paper context.

Include:

1. Existing approaches
2. Previous methods
3. Research gaps
4. Comparison with proposed work

Paper:
{context}
"""
)


qa_chain = _build_chain(_qa_template)
summary_chain = _build_chain(_summary_template)
methodology_paper_chain = _build_chain(_methodology_paper_template)
gaps_paper_chain = _build_chain(_gaps_paper_template)
literature_chain = _build_chain(_literature_template)


def _format_docs(docs) -> str:
    if not docs:
        return ""

    return "\n\n".join(
        doc.page_content
        for doc in docs
        if getattr(doc, "page_content", None)
    )


def build_rag_chain(retriever):
    return (
        {
            "context": retriever | RunnableLambda(_format_docs),
            "query": RunnablePassthrough(),
        }
        | _qa_template
        | get_llm()
        | StrOutputParser()
    )