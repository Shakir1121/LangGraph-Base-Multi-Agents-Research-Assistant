import logging
import queue
import re
import threading
from typing import Generator

from langchain_core.callbacks import BaseCallbackHandler
from langgraph.graph import END, StateGraph

from research_module.agents.critic_agent import critic_agent
from research_module.agents.final_agent import final_agent
from research_module.agents.gap_agent import gap_agent
from research_module.agents.idea_agent import idea_agent
from research_module.agents.idea_selector import selector_agent
from research_module.agents.methodology_agent import methodology_agent
from research_module.agents.paper_ranker import paper_ranker_agent
from research_module.agents.proposal_agent import proposal_agent
from research_module.agents.query_planner import query_planner_agent
from research_module.agents.search_agent import search_agent
from research_module.graph.state import ResearchState
from research_module.retrievers.hybrid_retriever import hybrid_search
from research_module.utils.cache_manager import get_global_cache
from research_module.vectorstore.chroma_store import retrieve, store_papers


logger = logging.getLogger(__name__)


AGENTS = {
    "planner": query_planner_agent,
    "ranker": paper_ranker_agent,
    "ideas": idea_agent,
    "select": selector_agent,
    "gaps": gap_agent,
    "methodology": methodology_agent,
    "proposal": proposal_agent,
    "critic": critic_agent,
}

PIPELINE = [
    "search",
    "planner",
    "hybrid_retrieval",
    "store",
    "retrieve",
    "ranker",
    "ideas",
    "select",
    "gaps",
    "methodology",
    "proposal",
    "critic",
    "final",
]


NODE_LABELS = {
    "search": ("🔍", "Searching web..."),
    "planner": ("📋", "Planning search query..."),
    "hybrid_retrieval": (
        "📚",
        "Retrieving papers from ArXiv, OpenAlex and Tavily...",
    ),
    "store": ("🗄️", "Storing research papers..."),
    "retrieve": ("🔎", "Retrieving relevant papers..."),
    "ranker": ("📊", "Ranking research papers..."),
    "ideas": ("💡", "Generating 10 research ideas..."),
    "select": ("⭐", "Selecting the strongest idea..."),
    "gaps": ("🧩", "Identifying research gaps..."),
    "methodology": ("🛠️", "Building methodology..."),
    "proposal": ("📄", "Writing research proposal..."),
    "critic": ("🧾", "Reviewing proposal..."),
    "final": ("✅", "Research generation completed."),
}


STREAM_SECTIONS = {
    "ideas": ("## 💡 Research Ideas", "ideas"),
    "select": ("## ⭐ Selected Research Idea", "selected_idea"),
    "gaps": ("## 🧩 Research Gaps", "gaps"),
    "methodology": ("## 🛠️ Proposed Methodology", "methodology"),
    "proposal": ("## 📄 Research Proposal", "proposal"),
    "critic": ("## 🧾 Critic Review", "review"),
}


_GRAPH_APP = None


def _get_session_id(state):
    session_id = state.get("session_id")

    if session_id:
        return session_id

    query = str(state.get("query", "")).strip()

    return str(abs(hash(query)))


def _search_node(state):
    state = {
        **state,
        "session_id": _get_session_id(state),
    }

    return search_agent(state)


def _hybrid_retrieval_node(state):
    queries = state.get("search_queries", [])

    if isinstance(queries, str):
        queries = [
            query.strip()
            for query in queries.splitlines()
            if query.strip()
        ]

    if not isinstance(queries, list):
        queries = []

    queries = queries[:1]

    if not queries:
        query = str(state.get("query", "")).strip()

        if query:
            queries = [query]

    cache = get_global_cache()

    arxiv_papers = []
    openalex_papers = []
    tavily_papers = []

    for query in queries:
        query = str(query).strip()

        if not query:
            continue

        cache_key = f"hybrid_search:v2:{query.lower()}"
        results = cache.get(cache_key)

        if results is None:
            results = hybrid_search(
                query,
                arxiv_limit=3,
                openalex_limit=3,
                tavily_limit=2,
            )
            cache.set(cache_key, results)

        if not isinstance(results, dict):
            continue

        arxiv_papers.extend(
            results.get("arxiv_papers", [])
        )
        openalex_papers.extend(
            results.get("openalex_papers", [])
        )
        tavily_papers.extend(
            results.get("tavily_papers", [])
        )

    return {
        "arxiv_papers": _deduplicate(arxiv_papers),
        "openalex_papers": _deduplicate(openalex_papers),
        "tavily_papers": _deduplicate(tavily_papers),
    }


def _deduplicate(papers):
    seen = set()
    unique = []

    for paper in papers:
        if isinstance(paper, dict):
            identity = str(
                paper.get("title")
                or paper.get("url")
                or ""
            ).strip().lower()
        else:
            identity = str(paper).strip().lower()

        if not identity or identity in seen:
            continue

        seen.add(identity)
        unique.append(paper)

    return unique


def _store_node(state):
    papers = (
        state.get("arxiv_papers", [])
        + state.get("openalex_papers", [])
        + state.get("tavily_papers", [])
    )

    if papers:
        store_papers(papers)
        logger.info("Stored %d papers.", len(papers))

    return {}


def _retrieve_node(state):
    query = str(state.get("query", "")).strip()

    if not query:
        return {"retrieved_docs": []}

    try:
        docs = retrieve(query, top_k=5) or []
    except Exception:
        logger.exception("RAG retrieval failed.")
        return {"retrieved_docs": []}

    logger.info("Retrieved %d documents.", len(docs))

    return {"retrieved_docs": docs}


def _final_node(state):
    try:
        result = final_agent(state)

        if isinstance(result, dict):
            return result

    except Exception:
        logger.exception("Final agent failed.")

    return {}


def build_graph():
    global _GRAPH_APP

    if _GRAPH_APP is not None:
        return _GRAPH_APP

    graph = StateGraph(ResearchState)

    graph.add_node("search", _search_node)
    graph.add_node("hybrid_retrieval", _hybrid_retrieval_node)
    graph.add_node("store", _store_node)
    graph.add_node("retrieve", _retrieve_node)
    graph.add_node("final", _final_node)

    for name, agent in AGENTS.items():
        graph.add_node(name, agent)

    graph.set_entry_point("search")

    for current, next_node in zip(
        PIPELINE,
        PIPELINE[1:],
    ):
        graph.add_edge(current, next_node)

    graph.add_edge("final", END)

    _GRAPH_APP = graph.compile()

    return _GRAPH_APP


def _remove_html(text):
    if not text:
        return ""

    text = re.sub(
        r"<[^>]+>",
        "",
        str(text),
    )

    return text.strip()


def _clean_output(text):
    text = _remove_html(text)

    if not text:
        return ""

    text = re.sub(
        r"\n*##\s*10\.\s*Research Evidence Status.*?(?=\n##\s|\Z)",
        "",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )

    text = re.sub(
        r"\n*Retrieved research documents used:\s*\**\d+\**.*",
        "",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text,
    )

    return text.strip()


class StopGeneration(Exception):
    pass


class ResearchStreamHandler(BaseCallbackHandler):

    def __init__(self, output_queue, stop_event=None):
        self.output_queue = output_queue
        self.stop_event = stop_event
        self.final_state = {}
        self.active_node = None
        self.sections = {}

    def on_chain_start(
        self,
        serialized,
        inputs,
        *,
        run_id=None,
        **kwargs,
    ):
        metadata = kwargs.get("metadata") or {}
        node = metadata.get("langgraph_node")

        if not node or node == self.active_node:
            return

        self.active_node = node

        icon, label = NODE_LABELS.get(
            node,
            ("⚙️", f"Running {node}..."),
        )

        self.output_queue.put({
            "type": "node_start",
            "node": node,
            "icon": icon,
            "label": label,
        })

        if node in STREAM_SECTIONS:
            header, _ = STREAM_SECTIONS[node]

            self.sections[node] = {
                "header": header,
                "open": False,
            }

    def on_llm_new_token(
        self,
        token,
        *,
        run_id=None,
        **kwargs,
    ):
        if not token:
            return

        section = self.sections.get(
            self.active_node
        )

        if section is None:
            return

        if not section["open"]:
            section["open"] = True

            self.output_queue.put({
                "type": "section_open",
                "node": self.active_node,
                "header": section["header"],
            })

        self.output_queue.put({
            "type": "token",
            "node": self.active_node,
            "token": token,
        })

    def on_chain_end(
        self,
        output,
        *,
        run_id=None,
        **kwargs,
    ):
        metadata = kwargs.get("metadata") or {}
        node = metadata.get("langgraph_node")

        if isinstance(output, dict):
            self.final_state.update(output)

        if node in STREAM_SECTIONS:
            header, state_key = STREAM_SECTIONS[node]

            content = None

            if isinstance(output, dict):
                content = output.get(state_key)

            if not content:
                content = self.final_state.get(state_key)

            if content:
                content = _clean_output(str(content))

                if content:
                    self.output_queue.put({
                        "type": "section",
                        "header": header,
                        "content": content,
                    })

            self.sections.pop(node, None)

        if self.stop_event and self.stop_event.is_set():
            raise StopGeneration()


def _worker(
    graph_app,
    initial_state,
    output_queue,
    stop_event,
):
    handler = ResearchStreamHandler(
        output_queue,
        stop_event,
    )

    try:
        graph_app.invoke(
            initial_state,
            config={
                "callbacks": [handler],
            },
        )

    except StopGeneration:
        logger.info("Research generation stopped.")

    except Exception as exc:
        logger.exception("Research graph failed.")

        output_queue.put({
            "type": "error",
            "message": str(exc),
        })

    output_queue.put({
        "type": "__final__",
        "state": handler.final_state,
    })

    output_queue.put(None)


def stream_research(
    query: str,
    session_id: str | None = None,
    stop_event: threading.Event | None = None,
) -> Generator[dict, None, None]:

    query = str(query or "").strip()

    if not query:
        yield {
            "type": "error",
            "message": "Research topic is empty.",
        }
        return

    graph_app = build_graph()

    session_id = session_id or str(abs(hash(query)))

    initial_state: ResearchState = {
        "query": query,
        "session_id": session_id,
    }

    output_queue = queue.Queue()

    worker = threading.Thread(
        target=_worker,
        args=(
            graph_app,
            initial_state,
            output_queue,
            stop_event,
        ),
        daemon=True,
    )

    worker.start()

    final_state = {}

    while True:
        item = output_queue.get()

        if item is None:
            break

        if item.get("type") == "__final__":
            final_state = item.get("state") or {}
            continue

        yield item

    worker.join(timeout=30)

    yield {
        "type": "done",
        "final_output": "",
        "state": final_state,
    }