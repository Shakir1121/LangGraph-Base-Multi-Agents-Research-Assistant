import re


class OutputCleaner:
    """Clean LLM-generated research output for consistent display.

    Fixes:
      • Excessive blank lines (collapses 2+ consecutive blank lines to one).
      • Blank lines immediately above markdown tables (removed).
      • Duplicate consecutive repeated line blocks (adjacent duplicate tables /
        paragraphs get collapsed).
      • Duplicate blank-line-separated blocks.
      • Duplicate consecutive lines and repeated ALL-CAPS section headers.
    """

    def clean(self, text: str) -> str:
        if not text:
            return ""

        # Normalize CRLF/CR to LF
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        text = self._collapse_blank_lines(text)
        text = self._remove_spaces_before_tables(text)
        text = self._remove_repeated_line_groups(text)
        text = self._remove_duplicate_blocks(text)
        text = self._remove_duplicate_lines(text)
        text = self._normalize_lines(text)
        text = self._remove_repeated_frames(text)
        return text.strip()

    def _collapse_blank_lines(self, text: str) -> str:
        """Collapse 2+ consecutive blank lines down to a single blank line."""
        return re.sub(r"\n[ \t]*\n(?:[ \t]*\n)+", "\n\n", text)

    def _remove_spaces_before_tables(self, text: str) -> str:
        """Remove any blank lines immediately preceding a markdown table row.

        A table row starts with ``|`` (optionally followed by a separator row
        like ``|---|---|``). Stripping blanks right before the table keeps the
        table snug against the preceding paragraph.
        """
        lines = text.split("\n")
        out = []
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("|"):
                # Drop trailing blank lines from the accumulated output
                while out and not out[-1].strip():
                    out.pop()
            out.append(line)
        return "\n".join(out)

    def _remove_repeated_line_groups(self, text: str) -> str:
        """Remove a second occurrence when a multi-line block repeats exactly.

        Catches adjacent duplicate markdown tables / paragraphs that end up
        back-to-back after blank-line collapsing.
        """
        lines = text.split("\n")
        result = list(lines)
        changed = True
        while changed:
            changed = False
            n = len(result)
            # Start from larger groups first so whole tables are removed as one
            for size in range(n // 2, 1, -1):
                for start in range(0, n - 2 * size + 1):
                    a = result[start:start + size]
                    b = result[start + size:start + 2 * size]
                    if a == b:
                        del result[start + size:start + 2 * size]
                        changed = True
                        break
                if changed:
                    break
        return "\n".join(result)

    def _remove_duplicate_blocks(self, text: str) -> str:
        """Remove repeated blank-line-separated blocks (paragraphs/tables)."""
        blocks = re.split(r"\n\s*\n", text)
        seen = set()
        out = []
        for block in blocks:
            key = block.strip().lower()
            if not key:
                continue
            if key in seen:
                continue
            seen.add(key)
            out.append(block.strip())
        return "\n\n".join(out)

    def _remove_duplicate_lines(self, text: str) -> str:
        """Remove consecutive duplicate lines."""
        lines = text.split("\n")
        out = []
        for line in lines:
            if out and out[-1] == line:
                continue
            out.append(line)
        return "\n".join(out)

    def _normalize_lines(self, text: str) -> str:
        """Deduplicate ALL-CAPS section headers."""
        lines = text.split("\n")
        cleaned = []
        seen_headers = set()

        for line in lines:
            stripped = line.strip()

            if stripped.isupper() and len(stripped) < 80:
                if stripped in seen_headers:
                    continue
                seen_headers.add(stripped)

            cleaned.append(line)

        return "\n".join(cleaned)


    def _remove_repeated_frames(self, text: str) -> str:
        """Remove common repeated review/evaluation framing blocks.

        Gaps/methodology/proposal/critic may echo "Final Evaluation & Selection",
        "Why This Idea Wins", "Implementation Roadmap", "Key Recommendations",
        "Key Advantages", or "Final Thoughts". When a later section repeats a
        frame that was already shown, this collapses it so the report stays
        distinct and non-duplicative.
        """
        frames = [
            "final evaluation & selection",
            "why this idea wins",
            "why these ideas stand out",
            "implementation roadmap",
            "key recommendations",
            "key advantages",
            "final thoughts",
            "final note",
            "overall score",
            "overall score:",
        ]
        lines = text.split("\n")
        seen_frames = set()
        out = []
        for line in lines:
            lowered = line.strip().lower()
            is_frame = any(lowered.startswith(f) or lowered == f for f in frames)
            if is_frame:
                if lowered in seen_frames:
                    # skip this line and everything that repeats the frame body
                    continue
                seen_frames.add(lowered)
            out.append(line)
        return "\n".join(out)
