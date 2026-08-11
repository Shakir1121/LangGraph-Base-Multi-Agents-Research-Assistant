# 🧠 Multi-Agent Research Assistant

An AI-powered research assistant built with **LangGraph, LangChain, LLMs, RAG, and multiple specialized AI agents**.

The system helps researchers analyze research papers and generate complete research proposals from research ideas.

---

## 🚀 Features

### 📄 1. Research Paper Q&A

Upload a research paper (PDF) and ask questions about it.

The AI agents can:
- 📖 Understand the research paper
- 🔎 Retrieve relevant information
- 💬 Answer multiple questions
- 🧠 Generate context-aware answers

---

### 💡 2. Research Idea Generator

Enter a research topic or idea and the system generates **10 ranked research ideas**.

Each idea can include:
- Problem/Research Area
- Key Innovation
- Research Questions
- Potential Impact
- Technical Direction

The system then selects the **best research idea** and develops it further.

---

### 📝 3. Complete Research Proposal

After selecting the best idea, multiple AI agents work together to generate a detailed research plan including:

- Introduction
- Problem Statement
- Research Objectives
- Research Questions
- Methodology
- Dataset Details
- Expected Results
- Research Gaps
- Complete Research Proposal
- Critical Review

---

## 🤖 Multi-Agent Architecture

The project uses specialized agents for different research tasks:

```text
User
 │
 ▼
Query Planner
 │
 ▼
Search & Retrieval
 │
 ├── ArXiv
 ├── OpenAlex
 └── Tavily
 │
 ▼
Paper Ranking
 │
 ▼
Research Ideas
 │
 ▼
Best Idea Selection
 │
 ▼
Research Gap
 │
 ▼
Methodology
 │
 ▼
Proposal Generation
 │
 ▼
Critic Review
 │
 ▼
Final Research Report
## 🛠️ Technologies

- Python
- LangGraph
- LangChain
- LLMs
- RAG
- ChromaDB
- FAISS
- ArXiv
- OpenAlex
- Tavily
- Streamlit
- Sentence Transformers

---

## 🎯 Example

### Input

```text
LLM-based Text Summarization
