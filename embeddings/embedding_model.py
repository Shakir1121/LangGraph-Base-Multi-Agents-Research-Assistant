from langchain_huggingface import HuggingFaceEmbeddings
import streamlit as st


@st.cache_resource
def get_embedding_model():

    model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    return model