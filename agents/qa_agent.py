from research_module.llm.chains import build_rag_chain


def qa_agent(query, vectorstore, llm=None):
    
    # Canonical LangChain RAG: build a retriever from the vector store and
    # ground the answer with it via the LCEL RAG chain.
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    rag_chain = build_rag_chain(retriever)

    return rag_chain.invoke(query)
