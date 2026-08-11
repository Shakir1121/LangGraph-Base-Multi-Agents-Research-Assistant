"""LangChain components for the research pipeline.

Centralises every prompt as a built-in :class:`ChatPromptTemplate` and
exposes each agent as an LCEL runnable (``prompt | llm | parser``).

Two flavours are provided:

* **Memory-backed chains** (``*_chain``) — used by the Research Idea
  Generator. They inject the session's chat history via the existing
  ``llm_with_memory`` helper and return a plain string. Created with
  :func:`build_memory_chain`.
* **Structured chains** (``paper_*_chain``) — used by the Paper QA
  pipeline. They are pure ``prompt | llm | StrOutputParser`` runnables.
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

from research_module.llm.mistral_llm import (
    get_llm,
    _build_messages,
    _persist_turn,
    _truncate_prompt,
)



# Generic runnable builder (memory-backed, streaming-capable)

def build_memory_chain(template: ChatPromptTemplate):
    """Wrap a prompt template into a streaming LCEL runnable with chat history.

    The runnable accepts a dict of template variables plus an optional
    ``session_id`` key. It formats the prompt, injects the session's chat
    history, and composes the ``ChatMistralAI`` model directly in the LCEL
    pipe. Because the model is part of the runnable tree, LangGraph's
    ``astream_events`` can emit real ``on_chat_model_stream`` token events,
    giving the UI ChatGPT-style streaming output. The completed turn is
    persisted to history once streaming finishes.
    """

    def _prepare(inputs: dict) -> dict:
        vars_ = {k: v for k, v in inputs.items() if k != "session_id"}
        prompt_text = _truncate_prompt(template.format(**vars_))
        session_id = inputs.get("session_id")
        return {
            "prompt": prompt_text,
            "session_id": session_id,
            "messages": _build_messages(prompt_text, session_id),
        }

    def _pick_messages(payload: dict):
        return payload["messages"]

    def _persist(payload: dict) -> str:
        _persist_turn(payload["prompt"], payload["response"], payload["session_id"])
        return payload["response"]

    return (
        RunnableLambda(_prepare)
        | RunnablePassthrough.assign(
            response=RunnableLambda(_pick_messages) | get_llm() | StrOutputParser()
        )
        | RunnableLambda(_persist)
    )



# Research Idea Generator — prompt templates


_query_planner_template = ChatPromptTemplate.from_template(
    """You are a research planner generating queries for the arXiv and Semantic
Scholar APIs.

Research topic:
{query}

Existing web context (use only as background, do not quote):
{web_context}

Produce EXACTLY 4 search queries, one per line. The first 3 are academic
search queries, the 4th targets a relevant dataset.

Hard constraints — these APIs reject malformed input:
- Plain text only. No bullets, numbers, asterisks, quotes, colons, or labels.
- Each line <= 12 words and <= 120 characters.
- Use keywords and noun phrases, not full sentences or boolean operators.
- No newlines inside a query.

Output: 4 lines, nothing else."""
)

_ideas_template = ChatPromptTemplate.from_template(
    """You are a senior research scientist generating novel research ideas.

RESEARCH TOPIC: {topic}

RELATED PAPERS (with abstracts for reference):
{papers_context}

WEB CONTEXT & CURRENT STATE:
{web_context}

TASK: Generate 10 original research ideas grounded in the topic and papers above.

For EACH idea, provide a title followed by EXACTLY these three headings, in this order:
**Title**: Clear, specific research title

**Problem Statement**: A short, eye-catching 1-2 sentence text that grabs attention and states the core problem this idea solves.

**Abstract**: A 3-4 sentence summary of the research idea in simple, easy words that a researcher can quickly understand. Explain what the idea is, the approach, and why it is novel.

**Methodology**: Provide the technical steps AND the dataset details for this idea. Name 2-3 concrete, real datasets (e.g., CNN/DailyMail, XSum, PubMed, SAMSum, FEVER, ESG, SummEval) with a short note on how each will be used. If a public dataset link is available, include the URL so the researcher can access it directly. Also give the key evaluation metrics (e.g., ROUGE, BERTScore, FactCC).

STRICT OUTPUT RULES:
- Output the ideas directly. Do NOT add a long intro paragraph, a "Cross-Cutting
  Research Directions" section, "Key Questions for Future Work", or "Next Steps"
  at the end. Stop after idea 10.
- Make each idea distinct - do not repeat the same phrasing or approach across ideas.
- Ground each idea in the topic and the related papers/web context above.
Format each idea with a blank line between them."""
)

_selector_template = ChatPromptTemplate.from_template(
    """You are a strict AI research paper reviewer.

Task: Select ONLY ONE best idea.

Research topic:
{topic}

Ideas:
{ideas}

Return ONLY this compact format:

SELECTED IDEA:
<improved, fully-written idea in 4-6 sentences>

WHY SELECTED:
- Novelty
- Technical Depth
- Feasibility
- Publication Strength

Keep it short. Do NOT include a long reviewer essay, scoring tables, "Final
Verdict", or "Suggested Next Steps"."""
)

_gaps_template = ChatPromptTemplate.from_template(
    """Research Topic:
{topic}

Selected Idea:
{selected_idea}

Papers Context:
{papers}

Web Context:
{web_context}

Identify the key research gaps directly related to the selected idea above.

STRICT OUTPUT RULES:
- Output ONLY a concise bulleted list of concrete, distinct research gaps.
- Each gap must be a real, specific deficiency in the literature/area, stated
  as one short bullet (1-2 sentences max).
- NEVER repeat the selected idea, the papers, or any proposal.
- NEVER include headings, headings text like "Novelty", "Technical Depth",
  "Dataset Feasibility", "Publication Potential", "Why This Is the Best Idea",
  scores, evaluations, or reviews.
- NEVER include an introduction, conclusion, roadmap, timeline, or any
  narrative. Just the bullets, nothing else.
Iterate the 3-5 most important gaps as a simple bullet list."""
)

_methodology_template = ChatPromptTemplate.from_template(
    """Research Topic:
{topic}

Selected Idea:
{selected_idea}

Research Gaps:
{gaps}

Retrieved Papers:
{papers}

Create a strict technical methodology for the selected idea above.

STRICT OUTPUT RULES:
- Output ONLY the methodology section: the concrete technical steps to build
  the system for the selected idea.
- Structure it with clear sub-sections: Dataset, Models, Training Strategy,
  Augmentation, Metrics, Deployment Plan, Challenges.
- Focus ONLY on the how-to-implement for the selected idea. Do NOT restate the
  selected idea or the research gaps verbatim.
- NEVER include headings like "Novelty", "Technical Depth", "Dataset
  Feasibility", "Publication Potential", "Why This Is the Best Idea", any
  scores, evaluations, reviews, roadmap, or timeline.
- NEVER add an introduction or conclusion.
Output strictly the methodology content."""
)

_proposal_template = ChatPromptTemplate.from_template(
    """You are an academic researcher writing ONE cohesive research proposal.

Topic:
{topic}

Selected Idea:
{selected_idea}

Research Gaps:
{gaps}

Methodology:
{methodology}

Ranked Papers:
{ranked_papers}

Generate a single full research proposal for the selected idea.

STRICT OUTPUT RULES:
- Present ONE coherent proposal titled by its research title.
- Sections to include, in order:
  1. Title
  2. Abstract
  3. Problem Statement
  4. Literature Gap
  5. Methodology
  6. Dataset
  7. Models
  8. Metrics
  9. Timeline
  10. Expected Results
- Write each section's content concretely and distinctively.
- NEVER repeat or echo the ideas/overall review boilerplate, scoring rubrics
  (Novelty/Technical Depth/Dataset Feasibility/Publication Potential), "Why This
  Is the Best Idea", "Implementation Roadmap", or "Final Evaluation".
- Do NOT start with "Final Evaluation & Selection" or any review framing.
Output strictly the proposal."""
)

_critic_template = ChatPromptTemplate.from_template(
    """You are a senior research paper reviewer.

Review the following research proposal strictly as a CONSTRUCTIVE critic.

Proposal:
{proposal}

STRICT OUTPUT RULES:
- Provide ONLY a concise, focused critique with these bullet sections:
  * Strengths (3 bullets)
  * Weaknesses (3 bullets)
  * Specific Improvements (3 bullets)
- Give a short Overall Score line at the end (e.g. "Overall: 8/10").
- Do NOT re-add the proposal's own sections or content.
- Do NOT include "Novelty (1-10)", "Technical Depth", "Dataset Feasibility",
  "Publication Potential" rubrics, "Why This Idea Wins", "Implementation
  Roadmap", "Key Recommendations", or any "Final Evaluation & Selection"
  framing.
- Keep it distinct and short. Output strictly the critique."""
)

_ranker_template = ChatPromptTemplate.from_template(
    """You are ranking academic papers for a research project on: {query}

IMPORTANT: Below are the candidate papers that YOU MUST RANK. Do NOT ask for papers - they are already provided.

Your task: Rank the top {num_to_rank} most relevant papers by this weighted criteria:
- Relevance to the topic (highest weight - 50%)
- Novelty/recency (25%)
- Technical depth (25%)

Return results as a numbered list with:
  Number. Title — One-sentence reason explaining the ranking — URL (if available)

CANDIDATE PAPERS TO RANK:
{compact_papers}

IMPORTANT: Return ONLY the ranked papers in the format above. Do not ask for more information or say papers are missing."""
)


# Research Idea Generator — runnables
#
# All memory-backed chains share the same builder, so they are created from a
# simple (variable_name, template) registry instead of eight repetitive calls.
_MEMORY_CHAINS = [
    ("query_planner_chain", _query_planner_template),
    ("ideas_chain", _ideas_template),
    ("selector_chain", _selector_template),
    ("gaps_chain", _gaps_template),
    ("methodology_chain", _methodology_template),
    ("proposal_chain", _proposal_template),
    ("critic_chain", _critic_template),
    ("ranker_chain", _ranker_template),
]

for _name, _template in _MEMORY_CHAINS:
    globals()[_name] = build_memory_chain(_template)



# Paper QA — prompt templates + pure LCEL runnables


_qa_template = ChatPromptTemplate.from_template(
    """You are an AI Research Assistant.

Answer the question ONLY from the research paper.

Context:
{context}

Question:
{query}

Answer clearly and accurately."""
)

_summary_template = ChatPromptTemplate.from_template(
    """You are an AI Research Assistant.

Summarize this paper.

Return:

1. Main Problem
2. Proposed Method
3. Key Contributions
4. Main Results
5. Conclusion

Paper:
{context}"""
)

_methodology_paper_template = ChatPromptTemplate.from_template(
    """You are an AI research analyst.

Extract the methodology from the paper.

Include:
- Model architecture
- Algorithms
- Training approach
- Dataset usage
- Evaluation process

Paper:
{context}"""
)

_gaps_paper_template = ChatPromptTemplate.from_template(
    """You are an expert AI research analyst.

Analyze this paper and identify:

1. Research limitations
2. Weaknesses in methodology
3. Missing experiments
4. Future work opportunities
5. Potential improvements

Return concise bullet points.

Paper:
{context}"""
)

_literature_template = ChatPromptTemplate.from_template(
    """You are an expert academic researcher.

Generate a literature review.

Include:
1. Existing approaches
2. Previous methods
3. Research gaps
4. Comparison with proposed work

Paper:
{context}"""
)


def _build_lcel_chain(template: ChatPromptTemplate) -> RunnableLambda:
    """Build a pure ``prompt | llm | StrOutputParser`` runnable for Paper QA."""
    return template | get_llm() | StrOutputParser()


qa_chain = _build_lcel_chain(_qa_template)
summary_chain = _build_lcel_chain(_summary_template)
methodology_paper_chain = _build_lcel_chain(_methodology_paper_template)
gaps_paper_chain = _build_lcel_chain(_gaps_paper_template)
literature_chain = _build_lcel_chain(_literature_template)



# Proper RAG chain (retriever-grounded) — used by the Paper QA pipeline


def _format_docs(docs) -> str:
    """Join retrieved LangChain documents into a single context string."""
    return "\n\n".join(doc.page_content for doc in docs)


def build_rag_chain(retriever):
    """Build a canonical LCEL RAG chain: ``retrieve -> format -> prompt -> llm -> parser``.

``retriever`` is any LangChain retriever (e.g. ``vectorstore.as_retriever(k=...)``).
    The chain accepts a raw question string and streams the grounded answer.
    """
    return (
        {
            "context": retriever | RunnableLambda(_format_docs),
            "query": RunnablePassthrough(),
        }
        | _qa_template
        | get_llm()
        | StrOutputParser()
    )
