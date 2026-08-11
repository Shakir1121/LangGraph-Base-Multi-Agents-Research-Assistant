import os

from langchain_chroma import Chroma
from embeddings.embedding_model import get_embedding_model

# Single shared Chroma database (same persist dir as the research idea
# generator's vector store). Sections are stored in their own collection so
# Paper QA and the idea generator coexist in one Chroma database.
_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./research_db/chroma")
_COLLECTION = "paper_sections"


def create_section_vectorstore(sections):
    """Index paper sections into the shared Chroma database.

    Returns a Chroma vector store supporting ``.as_retriever()`` and
    ``.similarity_search()``, which the Paper QA pipeline uses.
    """
    embedding = get_embedding_model()

    texts = []
    metadatas = []
    ids = []

    for section_name, content in sections.items():
        if not content.strip():
            continue

        texts.append(content)
        metadatas.append({"section": section_name})
        ids.append(f"section-{section_name}-{abs(hash(content[:50]))}")

    vectorstore = Chroma(
        collection_name=_COLLECTION,
        embedding_function=embedding,
        persist_directory=_PERSIST_DIR,
    )

    if texts:
        vectorstore.add_texts(texts=texts, metadatas=metadatas, ids=ids)

    return vectorstore
