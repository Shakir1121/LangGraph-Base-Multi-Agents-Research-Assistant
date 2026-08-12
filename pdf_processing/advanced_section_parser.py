import re


def normalize_section_name(line):
    line = line.lower().strip()

    mappings = {
        "abstract": "abstract",
        "introduction": "introduction",
        "methodology": "methodology",
        "methods": "methodology",
        "materials and methods": "methodology",
        "dataset": "dataset",
        "data": "dataset",
        "experimental setup": "dataset",
        "results": "results",
        "experiments": "results",
        "evaluation": "results",
        "discussion": "discussion",
        "conclusion": "conclusion",
        "conclusions": "conclusion",
        "references": "references",
        "bibliography": "references",
    }

    for key, section in mappings.items():
        if key in line and len(line) < 50:
            return section

    return None


def extract_sections(docs):
    text = "\n".join(
        doc.page_content
        for doc in docs
    )

    sections = {
        "title": ""
    }

    current_section = "title"

    for line in text.split("\n"):
        clean_line = line.strip()

        if not clean_line:
            continue

        lower_line = clean_line.lower()

        if (
            "page " in lower_line
            or "journal" in lower_line
            or len(clean_line) < 2
        ):
            continue

        detected = normalize_section_name(
            clean_line
        )

        if detected:
            current_section = detected
            sections.setdefault(
                current_section,
                "",
            )
            continue

        sections[current_section] += (
            clean_line + "\n"
        )

    return sections