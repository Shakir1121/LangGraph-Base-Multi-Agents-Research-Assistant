import logging
import re


logger = logging.getLogger(__name__)


def _clean_section(text: object) -> str:
    """Normalize generated text before adding it to the report."""
    if text is None:
        return ""

    text = str(text).strip()
    if not text:
        return ""

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text


def _add_section(
    sections: list[tuple[str, str]],
    title: str,
    content: object,
) -> None:
    """Add non-empty content unless the same content is already present."""
    content = _clean_section(content)
    if not content:
        return

    normalized = re.sub(r"\s+", " ", content).lower()

    if any(
        normalized == re.sub(r"\s+", " ", existing_content).lower()
        for _, existing_content in sections
    ):
        return

    sections.append((title, content))


def final_agent(state: dict) -> dict:
    """Build the final research report from the completed pipeline state."""
    sections: list[tuple[str, str]] = []

    topic = _clean_section(state.get("query"))
    if topic:
        _add_section(
            sections,
            "## 🔬 Research Topic",
            topic,
        )

    search_queries = state.get("search_queries", [])

    if isinstance(search_queries, str):
        search_queries = [
            query.strip()
            for query in search_queries.splitlines()
            if query.strip()
        ]

    if isinstance(search_queries, list):
        queries = [
            str(query).strip()
            for query in search_queries
            if str(query).strip()
        ]

        if queries:
            _add_section(
                sections,
                "## 🔎 Search Queries",
                "\n".join(f"- {query}" for query in queries),
            )

    report_sections = [
        ("## 💡 Research Ideas", "ideas"),
        ("## ⭐ Selected Research Idea", "selected_idea"),
        ("## 🧩 Research Gaps", "gaps"),
        ("## 🛠️ Proposed Methodology", "methodology"),
        ("## 📄 Research Proposal", "proposal"),
        ("## 🧾 Critic Review", "review"),
    ]

    for title, state_key in report_sections:
        _add_section(
            sections,
            title,
            state.get(state_key),
        )

    if not sections:
        logger.warning("No sections were generated for the final report.")

        return {
            "final_report": (
                "No research report was generated. "
                "Please try again with a more specific research topic."
            )
        }

    report = "\n\n---\n\n".join(
        f"{title}\n\n{content}"
        for title, content in sections
    )

    return {
        "final_report": report.strip()
    }