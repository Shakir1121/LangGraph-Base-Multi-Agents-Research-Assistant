import os

from langchain_chroma import Chroma
from embeddings.embedding_model import get_embedding_model


_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./research_db/chroma")
_COLLECTION = "paper_sections"


def create_section_vectorstore(sections):
    """Index paper sections into the shared Chroma database."""

    embedding = get_embedding_model()

    texts = []
    metadatas = []
    ids = []

    for section_name, content in sections.items():
        if not content.strip():
            continue

        texts.append(content)
        metadatas.append({"section": section_name})
        ids.append(
            f"section-{section_name}-{abs(hash(content[:50]))}"
        )

    vectorstore = Chroma(
        collection_name=_COLLECTION,
        embedding_function=embedding,
        persist_directory=_PERSIST_DIR,
    )

    if texts:
        vectorstore.add_texts(
            texts=texts,
            metadatas=metadatas,
            ids=ids,
        )

    return vectorstore