from research_module.utils.output_cleaner import OutputCleaner


def build_final_report(state):
    cleaner = OutputCleaner()

    topic = str(state.get("query", "")).strip()

    search_queries = state.get("search_queries", [])
    if isinstance(search_queries, list):
        search_queries = "\n".join(f"- {q}" for q in search_queries)

    ideas = str(state.get("ideas", "")).strip()
    selected_idea = str(state.get("selected_idea", "")).strip()
    gaps = str(state.get("gaps", "")).strip()
    methodology = str(state.get("methodology", "")).strip()
    proposal = str(state.get("proposal", "")).strip()
    review = str(state.get("review", "")).strip()
    ranked_papers = str(state.get("ranked_papers", "")).strip()

    retrieved_docs = state.get("retrieved_docs", []) or []

    report_parts = [
        "# 🧠 AI Research Copilot Report",
        f"## 1. Research Topic\n\n{topic}",
    ]

    if search_queries:
        report_parts.append(
            f"## 2. Research Search Strategy\n\n{search_queries}"
        )

    if ideas:
        report_parts.append(
            f"## 3. Research Ideas\n\n{ideas}"
        )

    if selected_idea:
        report_parts.append(
            f"## 4. Selected Research Idea\n\n{selected_idea}"
        )

    if gaps:
        report_parts.append(
            f"## 5. Research Gaps\n\n{gaps}"
        )

    if methodology:
        report_parts.append(
            f"## 6. Detailed Methodology\n\n{methodology}"
        )

    if ranked_papers:
        report_parts.append(
            f"## 7. Relevant Ranked Literature\n\n{ranked_papers}"
        )

    if proposal:
        report_parts.append(
            f"## 8. Complete Research Proposal\n\n{proposal}"
        )

    if review:
        report_parts.append(
            f"## 9. Critical Review\n\n{review}"
        )

    report_parts.append(
        "## 10. Research Evidence Status\n\n"
        f"Retrieved research documents used: **{len(retrieved_docs)}**"
    )

    final_report = "\n\n".join(report_parts)
    cleaned = cleaner.clean(final_report)

    return {
        "final_output": cleaned or final_report,
        "final_report": cleaned,
    }