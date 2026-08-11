import os
import logging

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

logger = logging.getLogger(__name__)


for _noisy in (
    "httpx",
    "httpcore",
    "huggingface_hub",
    "huggingface_hub.utils._http",
    "sentence_transformers",
    "sentence_transformers.base.model",
    "sentence_transformers.SentenceTransformer",
    "transformers",
    "transformers.modeling_utils",
    "urllib3",
    "filelock",
    "chromadb",
):
    logging.getLogger(_noisy).setLevel(logging.ERROR)

MODEL_NAME = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./research_db/chroma")


def _get_embeddings():
    """Return a shared LangChain embedding model."""
    return HuggingFaceEmbeddings(
        model_name=MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


_vectorstore = Chroma(
    collection_name="research_papers",
    embedding_function=_get_embeddings(),
    persist_directory=_PERSIST_DIR,
)


def store_papers(papers):
    """Index papers into the LangChain Chroma vector store."""
    if not papers:
        logger.warning("🗄️ store_papers: no papers to store")
        return

    ids = []
    docs = []
    metas = []

    for i, p in enumerate(papers):
        content = p.get("content") or p.get("summary") or ""
        if not content:
            continue
        title = p.get("title", "")
        url = p.get("url", "")
        docs.append(content)
        metas.append({"title": title, "url": url})
        # Stable id per paper to avoid duplicate inserts on re-runs.
        ids.append(f"{i}-{abs(hash(title or content[:50]))}")

    if docs:
        _vectorstore.add_texts(texts=docs, metadatas=metas, ids=ids)
        logger.info(f"🗄️ store_papers: stored {len(docs)} paper(s) in Chroma")


def retrieve(query, top_k=5):
    """Retrieve documents from Chroma, returning dicts for downstream agents."""
    docs = _vectorstore.similarity_search(query, k=top_k)
    return [
        {
            "title": d.metadata.get("title", ""),
            "content": d.page_content,
            "url": d.metadata.get("url", ""),
        }
        for d in docs
    ]


def as_retriever(top_k=5):
    """Return a LangChain retriever backed by the Chroma vector store."""
    return _vectorstore.as_retriever(search_kwargs={"k": top_k})
