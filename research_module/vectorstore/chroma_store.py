import hashlib
import logging
import os

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


logger = logging.getLogger(__name__)

for module in (
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
    logging.getLogger(module).setLevel(logging.ERROR)


MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2",
)

PERSIST_DIR = os.getenv(
    "CHROMA_PERSIST_DIR",
    "./research_db/chroma",
)


def _get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


_vectorstore = Chroma(
    collection_name="research_papers",
    embedding_function=_get_embeddings(),
    persist_directory=PERSIST_DIR,
)


def _paper_id(title: str, content: str) -> str:
    value = f"{title}|{content[:500]}"
    return hashlib.md5(value.encode("utf-8")).hexdigest()


def store_papers(papers):
    if not papers:
        logger.warning("No papers to store.")
        return

    ids = []
    documents = []
    metadatas = []

    for paper in papers:
        content = (
            paper.get("content")
            or paper.get("summary")
            or paper.get("abstract")
            or ""
        ).strip()

        if not content:
            continue

        title = paper.get("title", "").strip()
        url = paper.get("url", "").strip()

        documents.append(content)
        metadatas.append({
            "title": title,
            "url": url,
        })
        ids.append(_paper_id(title, content))

    if not documents:
        logger.warning("No valid paper content found.")
        return

    _vectorstore.add_texts(
        texts=documents,
        metadatas=metadatas,
        ids=ids,
    )

    logger.info(
        f"Stored {len(documents)} paper(s) in Chroma."
    )


def retrieve(query, top_k=5):
    docs = _vectorstore.similarity_search(
        query,
        k=top_k,
    )

    return [
        {
            "title": doc.metadata.get("title", ""),
            "content": doc.page_content,
            "url": doc.metadata.get("url", ""),
        }
        for doc in docs
    ]


def as_retriever(top_k=5):
    return _vectorstore.as_retriever(
        search_kwargs={"k": top_k}
    )