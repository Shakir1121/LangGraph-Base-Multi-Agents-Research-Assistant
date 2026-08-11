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

    for key in mappings:

        if key in line and len(line) < 50:
            return mappings[key]

    return None


def extract_sections(docs):

    text = "\n".join([d.page_content for d in docs])

    lines = text.split("\n")

    sections = {}

    current_section = "title"

    sections[current_section] = ""

    for line in lines:

        clean_line = line.strip()

        if not clean_line:
            continue

        # remove page noise
        if "page " in clean_line.lower():
            continue

        if "journal" in clean_line.lower():
            continue

        if len(clean_line) < 2:
            continue

        # detect section
        detected = normalize_section_name(clean_line)

        if detected:

            current_section = detected

            if current_section not in sections:
                sections[current_section] = ""

            continue

        # append content
        sections[current_section] += clean_line + "\n"

    return sections