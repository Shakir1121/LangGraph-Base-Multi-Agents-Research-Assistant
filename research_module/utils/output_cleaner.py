import re


class OutputCleaner:
    """Clean and normalize LLM-generated research reports."""

    def clean(self, text: str) -> str:
        if not text:
            return ""

        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = self._normalize_whitespace(text)
        text = self._remove_duplicate_paragraphs(text)
        text = self._remove_duplicate_lines(text)
        text = self._remove_duplicate_headings(text)

        return text.strip()

    def _normalize_whitespace(self, text: str) -> str:
        text = re.sub(r"[ \t]+", " ", text)
        return re.sub(r"\n{3,}", "\n\n", text)

    def _remove_duplicate_paragraphs(self, text: str) -> str:
        blocks = re.split(r"\n\s*\n", text)
        seen = set()
        output = []

        for block in blocks:
            block = block.strip()

            if not block:
                continue

            if "```" in block or block.startswith("|"):
                output.append(block)
                continue

            normalized = re.sub(r"\s+", " ", block.lower())

            if normalized in seen:
                continue

            seen.add(normalized)
            output.append(block)

        return "\n\n".join(output)

    def _remove_duplicate_lines(self, text: str) -> str:
        lines = text.split("\n")
        output = []

        for line in lines:
            if output and line.strip() == output[-1].strip():
                continue

            output.append(line)

        return "\n".join(output)

    def _remove_duplicate_headings(self, text: str) -> str:
        lines = text.split("\n")
        output = []
        seen = set()

        for line in lines:
            stripped = line.strip()

            if stripped.startswith("#") and len(stripped) < 150:
                normalized = re.sub(
                    r"^#+\s*",
                    "",
                    stripped.lower(),
                )

                if normalized in seen:
                    continue

                seen.add(normalized)

            output.append(line)

        return "\n".join(output)