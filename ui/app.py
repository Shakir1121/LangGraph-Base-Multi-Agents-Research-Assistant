import os
import sys
import tempfile
import hashlib
import threading

import streamlit as st


PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


from pdf_processing.parser import load_pdf
from pdf_processing.advanced_section_parser import extract_sections
from vector_store.section_vector_store import create_section_vectorstore
from langgraph_flow.workflow import build_workflow
from research_module.graph.research_graph import stream_research


st.set_page_config(
    page_title="AI Research Copilot",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
<style>
.top-header {
    width: 100%;
    text-align: center;
    padding: 28px 30px;
    margin-bottom: 30px;
    border-radius: 22px;
    background: linear-gradient(
        135deg,
        rgba(75, 85, 180, 0.30),
        rgba(18, 25, 65, 0.75)
    );
    border: 1px solid rgba(255, 255, 255, 0.12);
    box-shadow: 0 15px 40px rgba(0, 0, 0, 0.25);
}

.top-header h1 {
    margin: 0;
    font-size: 44px;
    font-weight: 800;
    letter-spacing: -1px;
}

.top-header p {
    margin-top: 10px;
    margin-bottom: 0;
    color: #aeb8d5;
    font-size: 17px;
    line-height: 1.6;
}

.stApp {
    background: radial-gradient(
        circle at top left,
        #18235c 0%,
        #0b1029 40%,
        #070a19 100%
    );
}

.block-container {
    max-width: 1450px;
    padding-top: 2rem;
    padding-bottom: 4rem;
}

.hero {
    padding: 28px 32px;
    border-radius: 24px;
    margin-bottom: 25px;
    background: linear-gradient(
        135deg,
        rgba(75, 85, 180, .28),
        rgba(18, 25, 65, .72)
    );
    border: 1px solid rgba(255, 255, 255, .12);
    box-shadow: 0 20px 50px rgba(0, 0, 0, .25);
}

.hero-title {
    font-size: 44px;
    font-weight: 800;
    margin: 0;
}

.hero-subtitle {
    color: #aeb8d5;
    font-size: 17px;
    margin-top: 8px;
}

.research-card {
    padding: 24px;
    margin-top: 18px;
    border-radius: 20px;
    background: rgba(255, 255, 255, .045);
    border: 1px solid rgba(255, 255, 255, .10);
    box-shadow: 0 12px 35px rgba(0, 0, 0, .18);
}

.status-card {
    padding: 14px 18px;
    border-radius: 14px;
    background: rgba(70, 80, 150, .18);
    border: 1px solid rgba(120, 130, 210, .20);
    margin-top: 12px;
}

.idea-card {
    padding: 20px;
    margin: 15px 0;
    border-radius: 16px;
    background: rgba(255, 255, 255, .035);
    border: 1px solid rgba(255, 255, 255, .08);
}

.stButton > button {
    border-radius: 12px;
    font-weight: 700;
    min-height: 44px;
}

section[data-testid="stSidebar"] {
    background: rgba(5, 8, 25, .94);
}

.stDownloadButton > button {
    border-radius: 12px;
    font-weight: 700;
}
</style>
""",
    unsafe_allow_html=True,
)


defaults = {
    "session_id": None,
    "idea_output": "",
    "idea_topic": "",
    "paper_hash": None,
    "paper_sections": None,
    "paper_vectorstore": None,
    "stop_event": None,
    "is_generating": False,
    "generation_stopped": False,
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


st.markdown(
    """
    <div style="text-align:center;">
        <h1>🧠 AI Research Copilot</h1>
      
    </div>
    """,
    unsafe_allow_html=True,
)


with st.sidebar:
    st.markdown("## 🧭 Research Workspace")

    mode = st.radio(
        "Choose workflow",
        [
            "💡 Research Idea Generator",
            "📄 Research Paper Q&A",
        ],
    )

    st.divider()

    st.markdown(
        """
### Pipeline

🔍 Search  
📋 Query Planning  
📚 Paper Retrieval  
📊 Paper Ranking  
💡 10 Research Ideas  
⭐ Idea Selection  
🧩 Research Gaps  
🛠️ Methodology  
📄 Proposal  
🧾 Critic Review  
✅ Final Report
"""
    )

    st.divider()

    if st.button(
        "🗑️ Clear Generated Output",
        use_container_width=True,
    ):
        st.session_state.idea_output = ""
        st.session_state.idea_topic = ""
        st.session_state.generation_stopped = False
        st.session_state.is_generating = False
        st.session_state.stop_event = None
        st.rerun()


if mode == "📄 Research Paper Q&A":

    st.header("📄 Research Paper Q&A")

    st.caption(
        "Upload a research paper and ask questions using the RAG pipeline."
    )

    uploaded_file = st.file_uploader(
        "Upload research paper",
        type=["pdf"],
    )

    if uploaded_file is None:
        st.info("📄 Upload a PDF to begin.")
        st.stop()

    pdf_bytes = uploaded_file.getvalue()

    pdf_hash = hashlib.sha256(pdf_bytes).hexdigest()

    if st.session_state.paper_hash != pdf_hash:

        with st.spinner("📚 Processing research paper..."):

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".pdf",
            ) as tmp:
                tmp.write(pdf_bytes)
                temp_pdf_path = tmp.name

            try:
                docs = load_pdf(temp_pdf_path)
                sections = extract_sections(docs)

                vectorstore = create_section_vectorstore(
                    sections
                )

                st.session_state.paper_hash = pdf_hash
                st.session_state.paper_sections = sections
                st.session_state.paper_vectorstore = vectorstore

            finally:
                try:
                    os.remove(temp_pdf_path)
                except OSError:
                    pass

    st.success("✅ Research paper loaded successfully.")

    with st.form("paper_question_form"):

        question = st.text_input(
            "Ask a question",
            placeholder=(
                "Example: What methodology does this paper use?"
            ),
        )

        ask_button = st.form_submit_button(
            "🔎 Ask Question",
            use_container_width=True,
        )

    if ask_button:

        if not question.strip():
            st.warning("Please enter a question.")

        else:

            workflow = build_workflow()

            with st.spinner(
                "🤖 Analyzing research paper..."
            ):
                result = workflow.invoke(
                    {
                        "query": question.strip(),
                        "sections": st.session_state.paper_sections,
                        "vectorstore": st.session_state.paper_vectorstore,
                        "session_id": st.session_state.session_id,
                    }
                )

            response = (
                result.get("response")
                or result.get("final_output")
                or result.get("final_report")
                or result.get("final_message")
                or ""
            )

            if hasattr(response, "content"):
                response = response.content

            st.markdown(
                '<div class="research-card">',
                unsafe_allow_html=True,
            )

            st.subheader("🤖 AI Answer")
            st.markdown(str(response))

            st.markdown(
                "</div>",
                unsafe_allow_html=True,
            )


else:

    st.header("💡 Research Idea Generator")

    st.write(
        """
Enter your research topic and the multi-agent research pipeline will
discover literature, generate 10 distinct research ideas, select the
strongest idea, identify research gaps, build a methodology, write a
complete proposal and critically review it.
"""
    )

    with st.form("research_idea_form"):

        topic = st.text_input(
            "Research Topic",
            value=st.session_state.idea_topic,
            placeholder=(
                "Example: AI-based Bank Loan Default Prediction"
            ),
        )

        col1, col2 = st.columns([3, 1])

        with col1:
            generate_button = st.form_submit_button(
                "🚀 Generate Research",
                use_container_width=True,
            )

        with col2:
            clear_button = st.form_submit_button(
                "🧹 Clear",
                use_container_width=True,
            )

    if clear_button:
        st.session_state.idea_output = ""
        st.session_state.idea_topic = ""
        st.session_state.generation_stopped = False
        st.session_state.is_generating = False
        st.session_state.stop_event = None
        st.rerun()

    if generate_button:

        topic = topic.strip()

        if not topic:
            st.warning("Please enter a research topic.")

        else:

            st.session_state.idea_output = ""
            st.session_state.idea_topic = topic
            st.session_state.generation_stopped = False
            st.session_state.is_generating = True
            st.session_state.stop_event = threading.Event()

            st.session_state.session_id = hashlib.sha256(
                (
                    topic
                    + str(os.urandom(16))
                ).encode()
            ).hexdigest()[:24]

            status_placeholder = st.empty()
            output_placeholder = st.empty()
            stop_placeholder = st.empty()

            with stop_placeholder:
                stop_clicked = st.button(
                    "🛑 Stop Generation",
                    type="secondary",
                    use_container_width=True,
                    key="stop_generation_button",
                )

            if stop_clicked:

                if st.session_state.stop_event:
                    st.session_state.stop_event.set()

                st.session_state.generation_stopped = True

            full_output = ""

            try:

                for event in stream_research(
                    topic,
                    session_id=st.session_state.session_id,
                    stop_event=st.session_state.stop_event,
                ):

                    event_type = event.get("type")

                    if event_type == "node_start":

                        icon = event.get("icon", "⚙️")
                        label = event.get("label", "Working...")

                        status_placeholder.markdown(
                            f"""
<div class="status-card">
    <b>{icon} {label}</b>
</div>
""",
                            unsafe_allow_html=True,
                        )

                    elif event_type == "section_open":

                        header = event.get("header", "")

                        if header:
                            full_output += (
                                f"\n\n{header}\n\n"
                            )

                            output_placeholder.markdown(
                                full_output
                            )

                    elif event_type == "token":

                        token = event.get("token", "")
                        full_output += token

                        if (
                            len(full_output) % 100
                            < len(token)
                        ):
                            output_placeholder.markdown(
                                full_output
                            )

                    elif event_type == "section":

                        header = event.get("header", "")
                        content = event.get("content", "")

                        marker = header + "\n"

                        if marker in full_output:

                            before = full_output.split(
                                marker,
                                1,
                            )[0]

                            full_output = (
                                before.rstrip()
                                + "\n\n"
                                + header
                                + "\n\n"
                                + content
                            )

                        else:

                            full_output += (
                                "\n\n"
                                + header
                                + "\n\n"
                                + content
                            )

                        output_placeholder.markdown(
                            full_output
                        )

                    elif event_type == "stopped":

                        st.session_state.generation_stopped = True

                        status_placeholder.warning(
                            "🛑 Generation stopped."
                        )

                    elif event_type == "error":

                        status_placeholder.error(
                            "❌ Research generation failed."
                        )

                        st.error(
                            event.get(
                                "message",
                                "Unknown error.",
                            )
                        )

                    elif event_type == "done":

                        final_output = event.get(
                            "final_output",
                            "",
                        )

                        stopped = event.get(
                            "stopped",
                            False,
                        )

                        if final_output:

                            full_output = final_output

                            output_placeholder.markdown(
                                full_output
                            )

                        if stopped:

                            st.session_state.generation_stopped = True

                            status_placeholder.warning(
                                "🛑 Research generation stopped by user."
                            )

                        else:

                            status_placeholder.success(
                                "✅ Research generation completed successfully."
                            )

                if full_output:
                    st.session_state.idea_output = full_output

                st.session_state.is_generating = False

                st.rerun()

            except Exception as exc:

                st.session_state.is_generating = False

                status_placeholder.error(
                    "❌ Research generation failed."
                )

                st.exception(exc)


if (
    mode == "💡 Research Idea Generator"
    and st.session_state.idea_output
    and not st.session_state.is_generating
):

    st.markdown(
        '<div class="research-card">',
        unsafe_allow_html=True,
    )

    st.subheader("📑 Research Report")

    st.markdown(
        st.session_state.idea_output
    )

    st.markdown(
        "</div>",
        unsafe_allow_html=True,
    )

    st.download_button(
        label="⬇️ Download Research Report",
        data=st.session_state.idea_output,
        file_name="research_report.md",
        mime="text/markdown",
        use_container_width=True,
    )