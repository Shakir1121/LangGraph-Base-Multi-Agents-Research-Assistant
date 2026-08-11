import streamlit as st
import sys
import os
import tempfile
import re
import threading


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pdf_processing.parser import load_pdf
from pdf_processing.advanced_section_parser import extract_sections

from vector_store.section_vector_store import create_section_vectorstore

from langgraph_flow.workflow import build_workflow

# Real streaming for the Research Idea Generator
from research_module.graph.research_graph import stream_research


# Markdown → styled HTML rendering
_SECTION_ICONS = {
    "RESEARCH IDEAS": "💡 Research Ideas (ranked)",
    "RESEARCH IDEAS (RANKED)": "💡 Research Ideas (ranked)",
    "SELECTED IDEA": "⭐ Selected Idea",
    "RESEARCH GAPS": "🧩 Research Gaps",
    "METHODOLOGY": "🛠️ Methodology",
    "RESEARCH PROPOSAL": "📄 Final Research Proposal",
    "FINAL RESEARCH PROPOSAL": "📄 Final Research Proposal",
    "CRITIC REVIEW": "🧾 Critic Review",
    "RANKED PAPERS": "📊 Ranked Papers",
    "SEARCH QUERIES": "🔍 Search Queries",
    "RESEARCH TOPIC": "🎯 Research Topic",
}


def _inline(text):
    """Convert inline markdown to HTML: links, **bold**, *italic*, `code`.

    Order matters — links are converted first so that **bold** / *italic*
    markers *inside* a link label are still processed afterwards.
    """
    text = re.sub(
        r"\[([^\]]+)\]\(([^)\s]+)\)",
        r'<a href="\2" target="_blank" rel="noopener noreferrer">\1</a>',
        text,
    )
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"(?<!\*)\*([^*\n]+)\*(?!\*)", r"<em>\1</em>", text)
    text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
    return text


def _is_table_row(line):
    s = line.strip()
    return "\t" in s or (s.startswith("|") and "|" in s[1:])


def _split_cells(row):
    s = row.strip()
    if s.startswith("|"):
        s = s.strip("|")
        return [c.strip() for c in s.split("|")]
    return [c.strip() for c in re.split(r"\t+", s)]


_SECTION_HEADER_RE = re.compile(r"^\s*#{1,3}\s+(.+?)\s*$")
_SUB_HEADER_RE = re.compile(r"^\s*#{4,6}\s+(.+?)\s*$")
_BULLET_RE = re.compile(r"^\s*[-*•]\s+(.+)$")
_NUM_RE = re.compile(r"^\s*(\d+)[.)]\s+(.+)$")
_BANNER_TITLE_RE = re.compile(
    r"^[=\-]{3,}\s*\n\s*([A-Z0-9_ ()&]+?)\s*\n[=\-]{3,}\s*$",
    re.MULTILINE,
)


def _md_to_html(text):
    """Turn raw LLM markdown/heavy-text into a compact styled HTML string."""
    if not text:
        return ""

    # Convert banner-style headings into markdown before parsing.
    def _banner_repl(match):
        title = match.group(1).strip()
        label = _SECTION_ICONS.get(title.upper().replace("(RANKED)", "").strip(), title)
        return f"## {label}"

    text = _BANNER_TITLE_RE.sub(_banner_repl, text)

    lines = text.split("\n")
    html = []
    i = 0
    n = len(lines)

    while i < n:
        ln = lines[i].rstrip()
        s = ln.strip()

        # Skip stray separator lines made of = or - only.
        if s and re.fullmatch(r"[=\-_\s]+", s):
            i += 1
            continue

        # Code fence.
        if s.startswith("```"):
            j = i + 1
            code = []
            while j < n and not lines[j].strip().startswith("```"):
                code.append(lines[j])
                j += 1
            html.append("<pre class='code-block'>" + _inline("\n".join(code)) + "</pre>")
            i = j + 1
            continue

        # Section header (## / ###).
        m = _SECTION_HEADER_RE.match(ln)
        if m:
            title = m.group(1).strip()
            label = _SECTION_ICONS.get(title.upper().replace("(RANKED)", "").strip(), title)
            html.append(f"<div class='sec-h'>{_inline(label)}</div>")
            i += 1
            continue

        # Sub-header (#### or more).
        m = _SUB_HEADER_RE.match(ln)
        if m:
            html.append(f"<div class='sub-h'>{_inline(m.group(1).strip())}</div>")
            i += 1
            continue

        # Table block (consecutive rows separated by tabs or pipes).
        if _is_table_row(ln):
            rows = []
            while i < n and _is_table_row(lines[i]):
                rows.append(_split_cells(lines[i]))
                i += 1
            # First row is header.
            if rows:
                html.append(_table_to_html(rows))
            continue

        # Bullet list.
        m = _BULLET_RE.match(ln)
        if m:
            items = []
            while i < n:
                mm = _BULLET_RE.match(lines[i].rstrip())
                if not mm:
                    break
                items.append(_inline(mm.group(1)))
                i += 1
            html.append("<ul class='md-list'>" + "".join(f"<li>{it}</li>" for it in items) + "</ul>")
            continue

        # Numbered list.
        m = _NUM_RE.match(ln)
        if m:
            items = []
            while i < n:
                mm = _NUM_RE.match(lines[i].rstrip())
                if not mm:
                    break
                items.append(_inline(mm.group(2)))
                i += 1
                # Fold following indented / continuation lines into this item
                while i < n:
                    nxt = lines[i].rstrip()
                    if not nxt.strip():
                        break
                    if _NUM_RE.match(nxt) or _SECTION_HEADER_RE.match(nxt):
                        break
                    if not _is_table_row(nxt):
                        items[-1] = items[-1] + " " + _inline(nxt.strip())
                        i += 1
            html.append(
                "<ol class='md-list'>" + "".join("<li>{0}</li>".format(it) for it in items) + "</ol>"
            )
            continue

        # Regular paragraph: gather consecutive non-empty, non-special lines.
        para = []
        while i < n:
            cur = lines[i].rstrip()
            cs = cur.strip()
            if not cs:
                break
            if cs.startswith("```") or _is_table_row(cur) or _SECTION_HEADER_RE.match(cur) \
               or _SUB_HEADER_RE.match(cur) or _BULLET_RE.match(cur) or _NUM_RE.match(cur):
                break
            para.append(cur)
            i += 1
        if para:
            html.append("<p>" + _inline(" ".join(para)) + "</p>")
            continue

        i += 1

    return "\n".join(html)


_ELLIPSIS_RE = re.compile(r"[\s.]*\.\.\.+[\s.]*")


def _clean_cell(text):
    """Trim excessive ellipses and collapse repeated words in a table cell."""
    t = _inline(str(text or ""))
    t = _ELLIPSIS_RE.sub("...", t)
    t = re.sub(r"(\b\w+\b)(?: \1\b)+", r"\1", t)
    return t


_SEP_ROW_RE = re.compile(r"^[\s:|-]+$")


def _is_sep_row(cells):
    """A row is a markdown separator (e.g. |---|---| ) if every cell is only
    dashes/colons/whitespace/pipes."""
    if not cells:
        return True
    return all(_SEP_ROW_RE.fullmatch(str(c or "").strip()) for c in cells)


def _table_to_html(rows):
    header = rows[0]
    body = rows[1:]
    # Evict rows that are just a separator (e.g. |---|---| ).
    body = [r for r in body if not _is_sep_row(r)]

    th = "".join(f"<th>{_clean_cell(c)}</th>" for c in header)
    trs = "".join(
        "<tr>" + "".join(f"<td>{_clean_cell(c)}</td>" for c in r) + "</tr>"
        for r in body
    )
    return (
        "<table class='md-table'><thead><tr>"
        + th
        + "</tr></thead><tbody>"
        + trs
        + "</tbody></table>"
    )


def _pretty_print_report(text):
    """Render raw graph output as fully formatted, styled HTML."""
    body = _md_to_html(text or "")
    # Wrap in a container; CSS handles justification, spacing and colors.
    return f"<div class='report-pre'>{body}</div>"


st.set_page_config(
    page_title="AI Research Copilot",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# Global styles kept in a single constant for readability.
_STYLES = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');

.stApp {
    background: linear-gradient(135deg, #0b0f2a 0%, #141948 40%, #1a1f4e 70%, #0d1025 100%);
    font-family: 'Inter', sans-serif;
}

.stApp::before {
    content: "";
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    background:
        radial-gradient(ellipse 80% 60% at 10% 20%, rgba(34, 211, 238, 0.08) 0%, transparent 60%),
        radial-gradient(ellipse 60% 50% at 90% 80%, rgba(124, 58, 237, 0.08) 0%, transparent 60%),
        radial-gradient(ellipse 40% 40% at 50% 50%, rgba(16, 185, 129, 0.04) 0%, transparent 50%);
    pointer-events: none;
    z-index: 0;
}

h1 {
    background: linear-gradient(135deg, #22d3ee, #7c3aed);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 800;
    font-size: 3.2rem !important;
    text-align: center;
    padding: 1.2rem 0 0.3rem 0;
    letter-spacing: -0.5px;
}

h2, h3, .stMarkdown, p, label, .stTextInput label, .stSelectbox label {
    color: #e5e7eb !important;
}

.stSelectbox div[data-baseweb="select"] {
    background: rgba(255, 255, 255, 0.07) !important;
    border: 1px solid rgba(255, 255, 255, 0.18) !important;
    border-radius: 14px !important;
    padding: 8px 6px !important;
    box-shadow: 0 8px 24px rgba(0,0,0,0.25) !important;
    backdrop-filter: blur(8px) !important;
}

.stSelectbox div[data-baseweb="select"]:hover {
    border-color: rgba(34, 211, 238, 0.6) !important;
    box-shadow: 0 0 20px rgba(34, 211, 238, 0.15) !important;
}

.stSelectbox ul {
    background: rgba(15, 20, 50, 0.95) !important;
    backdrop-filter: blur(14px) !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    border-radius: 14px !important;
}

.stTextInput input {
    background: rgba(255, 255, 255, 0.06) !important;
    border: 1px solid rgba(255, 255, 255, 0.15) !important;
    border-radius: 14px !important;
    padding: 14px 18px !important;
    color: #e5e7eb !important;
    font-size: 16px !important;
    box-shadow: 0 6px 20px rgba(0,0,0,0.2) !important;
    backdrop-filter: blur(6px) !important;
}

.stTextInput input:focus {
    border-color: #22d3ee !important;
    box-shadow: 0 0 0 3px rgba(34, 211, 238, 0.2) !important;
}

.stFileUploader div[data-testid="stFileUploader"] {
    background: rgba(255, 255, 255, 0.04) !important;
    border: 2px dashed rgba(255, 255, 255, 0.20) !important;
    border-radius: 18px !important;
    padding: 18px !important;
    text-align: center !important;
    backdrop-filter: blur(6px) !important;
}

.stFileUploader div[data-testid="stFileUploader"]:hover {
    border-color: #22d3ee !important;
    background: rgba(34, 211, 238, 0.06) !important;
}

.stSpinner > div {
    background: rgba(0,0,0,0) !important;
}

.st-info, .stAlert {
    background: rgba(34, 211, 238, 0.08) !important;
    border: 1px solid rgba(34, 211, 238, 0.20) !important;
    border-radius: 14px !important;
    color: #e5e7eb !important;
}

.report-pre {
    text-align: justify;
    line-height: 1.9;
    font-size: 15px;
    color: #d7dbe5;
}

.report-pre p {
    margin: 10px 0;
}

.report-pre strong {
    color: #7dd3fc;
    font-weight: 700;
}

.report-pre em {
    color: #c4b5fd;
    font-style: italic;
}

.report-pre a {
    color: #67e8f9;
    text-decoration: none;
    border-bottom: 1px dashed rgba(103, 232, 249, 0.5);
    transition: color 0.15s ease, border-color 0.15s ease;
}

.report-pre a:hover {
    color: #a5f3fc;
    border-bottom-color: #a5f3fc;
}

.report-pre code {
    background: rgba(34, 211, 238, 0.12);
    color: #a5f3fc;
    padding: 1px 5px;
    border-radius: 4px;
    font-size: 0.9em;
}

.sec-h {
    color: #ffffff;
    font-weight: 800;
    font-size: 21px;
    letter-spacing: 0.3px;
    margin: 26px 0 14px 0;
    padding: 14px 20px;
    border-radius: 14px;
    background: linear-gradient(135deg, rgba(34, 211, 238, 0.45), rgba(124, 58, 237, 0.45));
    border: 1px solid rgba(147, 197, 253, 0.55);
    box-shadow: 0 0 20px rgba(34, 211, 238, 0.22);
    text-align: left;
}

.sub-h {
    color: #67e8f9;
    font-weight: 700;
    font-size: 17px;
    letter-spacing: 0.2px;
    margin: 18px 0 8px 0;
    padding: 6px 0;
    border-bottom: 1px solid rgba(34, 211, 238, 0.25);
    text-align: left;
}

.md-list {
    margin: 8px 0 12px 0;
    padding-left: 24px;
    line-height: 1.8;
}

.md-list li {
    margin: 4px 0;
}

.md-table {
    width: 100%;
    border-collapse: collapse;
    margin: 14px 0 18px 0;
    font-size: 13.5px;
    color: #e5e7eb;
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(34, 211, 238, 0.25);
    border-radius: 10px;
    overflow: hidden;
    box-shadow: 0 6px 18px rgba(0,0,0,0.25);
}

.md-table th {
    background: linear-gradient(135deg, rgba(34, 211, 238, 0.35), rgba(124, 58, 237, 0.35));
    color: #ffffff;
    font-weight: 700;
    text-align: left;
    padding: 11px 14px;
    border-bottom: 1px solid rgba(255,255,255,0.20);
}

.md-table td {
    padding: 9px 14px;
    border-bottom: 1px solid rgba(255,255,255,0.07);
    vertical-align: top;
}

.md-table tbody tr:nth-child(even) {
    background: rgba(34, 211, 238, 0.05);
}

.md-table tbody tr:hover {
    background: rgba(34, 211, 238, 0.10);
}

.md-table tr:last-child td {
    border-bottom: none;
}

.code-block {
    white-space: pre-wrap;
    background: rgba(0,0,0,0.35) !important;
    border: 1px solid rgba(34, 211, 238, 0.30);
    border-radius: 10px;
    padding: 14px 16px;
    font-family: 'Consolas', 'Monaco', monospace;
    font-size: 13px;
    color: #a5f3fc;
    margin: 12px 0;
}

.mode-label {
    text-align: center;
    font-size: 18px;
    font-weight: 600;
    color: rgba(229, 231, 235, 0.85);
    margin-bottom: 8px;
}

.loading {
    text-align: center;
    font-size: 16px;
    font-weight: 600;
    color: #a5f3fc;
    padding: 22px 16px;
    margin-top: 12px;
    border-radius: 14px;
    background: rgba(34, 211, 238, 0.08);
    border: 1px solid rgba(34, 211, 238, 0.25);
    animation: pulse 1.4s ease-in-out infinite;
}

@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.55; }
}

.gen-toolbar {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 14px;
    margin: 10px 0 6px 0;
    background: rgba(15, 20, 50, 0.6);
    border: 1px solid rgba(34, 211, 238, 0.35);
    border-radius: 40px;
    padding: 12px 20px;
    backdrop-filter: blur(10px);
    box-shadow: 0 0 24px rgba(34, 211, 238, 0.18);
}

.gen-loader {
    display: flex;
    align-items: center;
    gap: 10px;
    font-size: 14px;
    font-weight: 600;
    color: #a5f3fc;
    letter-spacing: 0.3px;
}

.spinner-ring {
    width: 20px;
    height: 20px;
    border: 3px solid rgba(34, 211, 238, 0.25);
    border-top-color: #22d3ee;
    border-radius: 50%;
    animation: spin 0.7s linear infinite;
}

@keyframes spin {
    to { transform: rotate(360deg); }
}

.gen-stop {
    border: 1px solid rgba(248, 113, 113, 0.6);
    background: linear-gradient(135deg, rgba(248, 113, 113, 0.20), rgba(239, 68, 68, 0.20));
    color: #fecaca;
    font-weight: 700;
    font-size: 14px;
    padding: 8px 16px;
    border-radius: 30px;
    cursor: pointer;
    transition: all 0.2s ease;
    height: 36px;
    line-height: 1;
}

.gen-stop:hover {
    background: linear-gradient(135deg, rgba(248, 113, 113, 0.45), rgba(239, 68, 68, 0.45));
    color: #ffffff;
    border-color: #f87171;
    box-shadow: 0 0 18px rgba(248, 113, 113, 0.4);
}

.cancelled-banner {
    text-align: center;
    color: #fecaca;
    background: rgba(239, 68, 68, 0.10);
    border: 1px solid rgba(248, 113, 113, 0.35);
    border-radius: 14px;
    padding: 12px 16px;
    font-weight: 600;
    font-size: 15px;
}

.result-box {
    background: rgba(255, 255, 255, 0.04);
    border: 1px solid rgba(255, 255, 255, 0.10);
    border-radius: 16px;
    padding: 24px;
    margin-top: 20px;
    box-shadow: 0 12px 32px rgba(0,0,0,0.2);
    backdrop-filter: blur(6px);
}
</style>
"""

st.markdown(_STYLES, unsafe_allow_html=True)

# Stable session id per browser tab (kept internally for memory keying).
if "session_id" not in st.session_state:
    st.session_state["session_id"] = (
        str(st.session_state.get("_created_at", 0)) + "-" + str(len(st.session_state))
    )

st.markdown("<h1>🧠 AI Research Copilot</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align:center; color:rgba(229,231,235,0.7); font-size:1.1rem; margin-bottom:2rem;'>"
    "Supercharge your research with AI — upload papers, ask questions, generate ideas.</p>",
    unsafe_allow_html=True,
)

st.markdown("<div class='mode-label'>Choose your mode</div>", unsafe_allow_html=True)

mode_option = st.selectbox(
    "",
    options=[
        "📄 Paper QA — Upload a PDF and ask questions",
        "💡 Research Idea Generator — Explore new research directions",
    ],
    index=0,
)

mode = "paper" if "Paper QA" in mode_option else "ideas"

_, center_col, _ = st.columns([0.08, 0.84, 0.08])

# Mode 1: Paper QA (upload pdf -> ask questions)
if mode == "paper":
    with center_col:
        st.subheader("📄 Paper QA")

    with center_col:
        uploaded_file = st.file_uploader(
            "Upload Research Paper PDF",
            type=["pdf"],
        )

    if uploaded_file is None:
        with center_col:
            st.info("Upload a PDF to enable Paper QA.")
        st.stop()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        temp_pdf_path = tmp_file.name

    with center_col:
        with st.spinner("📄 Loading PDF..."):
            docs = load_pdf(temp_pdf_path)

        sections = extract_sections(docs)
        vectorstore = create_section_vectorstore(sections)
        workflow = build_workflow()

        st.success("✅ PDF loaded successfully! Ask your question below.")

        query = st.text_input("Ask a question about the uploaded paper")
        if query:
            with st.spinner("🤖 AI Agents Thinking (Paper QA)..."):
                result = workflow.invoke(
                    {
                        "query": query,
                        "sections": sections,
                        "vectorstore": vectorstore,
                        "session_id": st.session_state["session_id"],
                    }
                )
                response = result.get("response") or result.get("final_output") or result.get("final_report") or result.get("final_message") or ""

            if hasattr(response, "content"):
                response = response.content
            plain = str(response or "")

            st.markdown("<div class='result-box'>", unsafe_allow_html=True)
            st.subheader("✅ AI Response")
            stream_placeholder = st.empty()
            chunk_size = 60
            shown = ""
            for j in range(0, len(plain), chunk_size):
                shown = plain[: j + chunk_size]
                stream_placeholder.write(shown)
            st.markdown("</div>", unsafe_allow_html=True)

# Mode 2: Research Idea Generator (ask topic only)
else:
    with center_col:
        st.subheader("💡 Research Idea Generator")

    with center_col:
        idea_prompt = st.text_input(
            "Enter a research topic to generate ideas",
        )

    if not idea_prompt:
        with center_col:
            st.info("Enter a topic first to generate ideas.")
        st.stop()

    with center_col:
        # A stop event lets the user cancel an in-progress generation. It is
        # kept in session_state so a Stop button click (which triggers a rerun)
        # can set it and abort the background worker. It is reset whenever the
        # user enters a new topic.
        if st.session_state.get("last_topic") != idea_prompt:
            st.session_state["last_topic"] = idea_prompt
            st.session_state["stop_event"] = threading.Event()
        stop_event = st.session_state["stop_event"]

        status_placeholder = st.empty()

        # Floating toolbar with a live spinner + an interactive Stop button.
        toolbar_placeholder = st.empty()
        with toolbar_placeholder.container():
            tcol1, tcol2 = st.columns([3, 1], vertical_alignment="center")
            with tcol1:
                st.markdown(
                    "<div class='gen-loader'>"
                    "<div class='spinner-ring'></div>"
                    "<span>Generating your research report…</span>"
                    "</div>",
                    unsafe_allow_html=True,
                )
            with tcol2:
                if st.button("⏹ Stop", key="stop_gen_btn", use_container_width=True):
                    stop_event.set()

        st.markdown("<div class='result-box'>", unsafe_allow_html=True)
        st.subheader("✅ Research Output")
        loading_placeholder = st.empty()
        output_placeholder = st.empty()
        st.markdown("</div>", unsafe_allow_html=True)

        full_output = ""
        active_section_header = None
        cancelled = False

        loading_placeholder.markdown(
            "<div class='loading'>🤖 Generating research ideas... "
            "The AI agents are searching, ranking, and writing your report. "
            "This can take a minute or two.</div>",
            unsafe_allow_html=True,
        )

        for event in stream_research(
            idea_prompt,
            session_id=st.session_state["session_id"],
            stop_event=stop_event,
        ):
            # If the user pressed Stop, abort the loop immediately.
            if stop_event.is_set():
                cancelled = True
                break

            ev_type = event["type"]

            if ev_type == "node_start":
                status_placeholder.info(f"{event['icon']} {event['label']}")

            elif ev_type == "section_open":
                # A section just started streaming tokens - show its header
                # immediately so the user sees progress right away.
                loading_placeholder.empty()
                active_section_header = event.get("header", "")
                if active_section_header and active_section_header not in full_output:
                    full_output += "\n\n" + active_section_header + "\n"
                output_placeholder.markdown(
                    _pretty_print_report(full_output),
                    unsafe_allow_html=True,
                )

            elif ev_type == "token":
                # Append the streamed token so text appears token-by-token
                # (ChatGPT-style typing effect) instead of all at once.
                full_output += event.get("token", "")
                output_placeholder.markdown(
                    _pretty_print_report(full_output),
                    unsafe_allow_html=True,
                )

            elif ev_type == "section":
                # A section finished generating - replace any partial streamed
                # content with the full, clean section text.
                loading_placeholder.empty()
                header = event["header"]
                content = event["content"]
                # Remove the partial "header + streamed tokens" block
                # so we can append the full clean section text once.
                cut = full_output.find(header)
                if cut >= 0:
                    full_output = full_output[:cut].rstrip()
                full_output += "\n\n" + header + "\n" + content
                active_section_header = None
                output_placeholder.markdown(
                    _pretty_print_report(full_output),
                    unsafe_allow_html=True,
                )

            elif ev_type == "done":
                loading_placeholder.empty()
                status_placeholder.success("✅ Research complete!")
                output_placeholder.markdown(
                    _pretty_print_report(full_output or event.get("final_output", "")),
                    unsafe_allow_html=True,
                )

        # Hide the floating toolbar after generation finishes (success or cancel).
        toolbar_placeholder.empty()

        if cancelled:
            status_placeholder.warning("⏹️ Generation stopped by user.")
