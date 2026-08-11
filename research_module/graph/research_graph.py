import logging
import queue
import threading
from typing import Generator

from langgraph.graph import StateGraph, END

from research_module.graph.state import ResearchState

from research_module.agents.search_agent import search_agent
from research_module.agents.query_planner import query_planner_agent
from research_module.agents.paper_ranker import paper_ranker_agent
from research_module.agents.idea_agent import idea_agent
from research_module.agents.idea_selector import selector_agent
from research_module.agents.gap_agent import gap_agent
from research_module.agents.methodology_agent import methodology_agent
from research_module.agents.proposal_agent import proposal_agent
from research_module.agents.critic_agent import critic_agent
from research_module.agents.final_agent import final_agent

from research_module.retrievers.hybrid_retriever import hybrid_search

from research_module.vectorstore.chroma_store import store_papers, retrieve

from research_module.utils.final_builder import build_final_report
from research_module.utils.cache_manager import get_global_cache

logger = logging.getLogger(__name__)


# Nodes that simply delegate to a single agent function (state -> dict).
_AGENT_NODES = {
    "planner": query_planner_agent,
    "ranker": paper_ranker_agent,
    "ideas": idea_agent,
    "select": selector_agent,
    "gaps": gap_agent,
    "methodology": methodology_agent,
    "proposal": proposal_agent,
    "critic": critic_agent,
}

# The research pipeline runs in this exact order.
_WORKFLOW_SEQUENCE = [
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


def _ensure_session_id(state):
    """Fall back to a deterministic id derived from the query when none is set."""
    sid = state.get("session_id")
    if sid:
        return sid
    query = state.get("query", "")
    return str(abs(hash(query))) if query is not None else "default"


def _search_node(cache):
    def search_node(state):
        state = {**state, "session_id": _ensure_session_id(state)}
        return search_agent(state)
    return search_node


def _hybrid_retrieval_node(cache):
    """Multi-source retrieval: ArXiv, OpenAlex, and Tavily in parallel."""

    def hybrid_retrieval_node(state):
        queries = state.get("search_queries")
        if isinstance(queries, str):
            queries = [q.strip() for q in queries.split("\n") if q.strip()]
        elif not isinstance(queries, list):
            queries = []
        if not queries:
            queries = [state["query"]]

        all_arxiv, all_openalex, all_tavily = [], [], []

        for query in queries:
            logger.info(f"🔄 Hybrid retrieval starting for: {query}")

            cache_key = f"hybrid_search:{query}"
            results = cache.get(cache_key)
            if results:
                logger.info(f"📦 Using cached results for: {query}")
            else:
                results = hybrid_search(
                    query, arxiv_limit=3, openalex_limit=3, tavily_limit=2
                )
                cache.set(cache_key, results)

            all_arxiv.extend(results.get("arxiv_papers", []))
            all_openalex.extend(results.get("openalex_papers", []))
            all_tavily.extend(results.get("tavily_papers", []))

        def deduplicate(papers):
            seen, unique = set(), []
            for p in papers:
                title = p.get("title", "") if isinstance(p, dict) else str(p)
                if title not in seen:
                    seen.add(title)
                    unique.append(p)
            return unique

        all_arxiv = deduplicate(all_arxiv)
        all_openalex = deduplicate(all_openalex)
        all_tavily = deduplicate(all_tavily)

        logger.info(
            f"📥 Hybrid retrieval complete: "
            f"ArXiv={len(all_arxiv)}, OpenAlex={len(all_openalex)}, Tavily={len(all_tavily)}"
        )

        return {
            "arxiv_papers": all_arxiv,
            "openalex_papers": all_openalex,
            "tavily_papers": all_tavily,
        }

    return hybrid_retrieval_node


def _store_node(state):
    """Persist all retrieved papers into the vector store."""
    papers = (
        state.get("arxiv_papers", [])
        + state.get("openalex_papers", [])
        + state.get("tavily_papers", [])
    )
    if papers:
        store_papers(papers)
        logger.info(f"🗄️ store_node: stored {len(papers)} paper(s) in Chroma")
    else:
        logger.warning("🗄️ store_node: no papers to store")
    return {}


def _retrieve_node(state):
    """Retrieve the most relevant stored documents for the research query."""
    docs = retrieve(state["query"], top_k=5) or []
    if not docs:
        logger.warning(
            "🔍 retrieve_node: vector store returned 0 docs — "
            "downstream ranking/ideation will be ungrounded."
        )
    else:
        logger.info(f"🔍 retrieve_node: {len(docs)} doc(s) from vector store")
    return {"retrieved_docs": docs}


def _final_node(state):
    """Assemble the final report from all produced sections."""
    final_state = {**state, **final_agent(state)}
    final_state.update(build_final_report(final_state))
    return final_state


def build_graph():
    """Build and compile the LangGraph research workflow."""
    cache = get_global_cache()

    graph = StateGraph(ResearchState)

    graph.add_node("search", _search_node(cache))
    graph.add_node("hybrid_retrieval", _hybrid_retrieval_node(cache))
    graph.add_node("store", _store_node)
    graph.add_node("retrieve", _retrieve_node)
    graph.add_node("final", _final_node)

    for name, agent in _AGENT_NODES.items():
        graph.add_node(name, agent)

    graph.set_entry_point("search")

    for a, b in zip(_WORKFLOW_SEQUENCE, _WORKFLOW_SEQUENCE[1:]):
        graph.add_edge(a, b)
    graph.add_edge(_WORKFLOW_SEQUENCE[-1], END)

    return graph.compile()


# Map each node to an icon + status label shown in the UI.
_NODE_LABELS = {
    "search": ("🔍", "Searching web for context..."),
    "planner": ("📋", "Planning search queries..."),
    "hybrid_retrieval": ("📚", "Retrieving papers (ArXiv, OpenAlex, Tavily)..."),
    "store": ("🗄️", "Storing papers..."),
    "retrieve": ("🔎", "Retrieving relevant documents..."),
    "ranker": ("📊", "Ranking papers by relevance..."),
    "ideas": ("💡", "Generating research ideas..."),
    "select": ("⭐", "Selecting the best idea..."),
    "gaps": ("🧩", "Identifying research gaps..."),
    "methodology": ("🛠️", "Building methodology..."),
    "proposal": ("📄", "Writing research proposal..."),
    "critic": ("🧾", "Reviewing proposal..."),
    "final": ("✅", "Assembling final report..."),
}

# Map each streamable node to its display header and state key.
_NODE_STREAM_KEY = {
    "ideas": ("## Research Ideas", "ideas"),
    "select": ("## Selected Idea", "selected_idea"),
    "gaps": ("## Research Gaps", "gaps"),
    "methodology": ("## Methodology", "methodology"),
    "proposal": ("## Research Proposal", "proposal"),
    "critic": ("## Critic Review", "review"),
}


def _assemble_report(state: dict, fallback_query: str) -> str:
    """Build a human-readable report from the final graph state."""
    sections = []

    q = state.get("query") or fallback_query
    if q:
        sections.append(f"# Research Topic\n{q}")

    if state.get("search_queries"):
        sq = state["search_queries"]
        if isinstance(sq, list):
            sq = "\n".join(f"- {x}" for x in sq)
        sections.append(f"## Search Queries\n{sq}")

    for header, key in _NODE_STREAM_KEY.values():
        if state.get(key):
            sections.append(f"{header}\n{state[key]}")

    if not sections:
        return "No output generated. Please try again with a more specific research topic."

    return "\n\n".join(sections)


from langchain_core.callbacks import BaseCallbackHandler


class _StopGeneration(Exception):
    """Raised inside the callback handler when the user cancels generation."""


class _ResearchStreamHandler(BaseCallbackHandler):
    """Capture LLM tokens + node progress during a blocking graph invoke.

    ``stop_event`` lets the UI cancel generation: once set, the handler raises
    :class:`_StopGeneration` at the next node boundary, aborting the remaining
    graph steps.
    """

    def __init__(self, out_queue: "queue.Queue", stop_event: threading.Event | None = None):
        self.out_queue = out_queue
        self.final_state: dict = {}
        self.active_node = None
        self.stop_event = stop_event
        self._sections: dict = {}

    def on_chain_start(self, serialized, inputs, *, run_id=None, **kwargs):
        metadata = kwargs.get("metadata") or {}
        node = metadata.get("langgraph_node")
        if not node or node == self.active_node:
            return
        self.active_node = node
        icon, label = _NODE_LABELS.get(node, ("⚙️", f"Running {node}..."))
        self.out_queue.put(
            {"type": "node_start", "node": node, "icon": icon, "label": label}
        )
        if node in _NODE_STREAM_KEY:
            header, _key = _NODE_STREAM_KEY[node]
            self._sections[node] = {"header": header, "open": False}

    def on_llm_new_token(self, token, *, run_id=None, **kwargs):
        if not token:
            return
        node = self.active_node
        info = self._sections.get(node)
        if info is None:
            return
        if not info["open"]:
            info["open"] = True
            self.out_queue.put(
                {"type": "section_open", "node": node, "header": info["header"]}
            )
        self.out_queue.put({"type": "token", "node": node, "token": token})

    def on_chain_end(self, output, *, run_id=None, **kwargs):
        metadata = kwargs.get("metadata") or {}
        node = metadata.get("langgraph_node")
        if isinstance(output, dict):
            self.final_state.update(output)
        if node in _NODE_STREAM_KEY:
            header, key = _NODE_STREAM_KEY[node]
            content = output.get(key) if isinstance(output, dict) else None
            if not content:
                content = self.final_state.get(key)
            if content:
                self.out_queue.put(
                    {"type": "section", "header": header, "content": str(content).strip()}
                )
            self._sections.pop(node, None)
        if node and self.stop_event and self.stop_event.is_set():
            raise _StopGeneration()


def _worker(
    graph_app,
    initial_state: dict,
    out_queue: "queue.Queue",
    stop_event: threading.Event | None = None,
):
    """Run the graph via a blocking invoke with a streaming callback handler."""
    handler = _ResearchStreamHandler(out_queue, stop_event)
    try:
        graph_app.invoke(initial_state, config={"callbacks": [handler]})
    except _StopGeneration:
        logger.info("stream_research: generation cancelled by user")
    except Exception as exc:
        logger.error("stream_research invoke failed: %s", exc)
        out_queue.put({"type": "error", "message": str(exc)})
    out_queue.put({"type": "__final__", "state": handler.final_state})
    out_queue.put(None)


def stream_research(
    query: str,
    session_id: str | None = None,
    stop_event: threading.Event | None = None,
) -> Generator[dict, None, None]:
    """Run the research graph, streaming LLM tokens live for a typing effect.

    ``stop_event`` may be set by the caller to cancel generation early.

    Yields dicts:
        {"type": "node_start" | "section_open" | "token" | "section" | "done",
         ...}
    """
    graph_app = build_graph()
    session_id = session_id or str(abs(hash(query)))

    initial_state: ResearchState = {
        "query": query,
        "session_id": session_id,
    }

    out_queue: "queue.Queue" = queue.Queue()
    thread = threading.Thread(
        target=_worker,
        args=(graph_app, initial_state, out_queue, stop_event),
        daemon=True,
    )
    thread.start()

    final_state: dict = {}
    while True:
        item = out_queue.get()
        if item is None:
            break
        if item.get("type") == "__final__":
            final_state = item.get("state") or {}
            continue
        yield item

    thread.join(timeout=30)

    raw_output = (
        final_state.get("final_output")
        or final_state.get("final_report")
        or _assemble_report(final_state, query)
    )
    raw_output = str(raw_output or "").strip()

    yield {"type": "done", "final_output": raw_output}
