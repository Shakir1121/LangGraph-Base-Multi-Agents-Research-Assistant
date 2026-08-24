# 🧠 Multi-Agent AI Research Assistant

An AI-powered **Multi-Agent Research Assistant** designed to help students and researchers analyze research papers, generate research ideas, identify research gaps, develop methodologies, and create complete research proposals.

The project is built using **LangGraph, LangChain, RAG, LLMs, ChromaDB, FAISS, Sentence Transformers, ArXiv, OpenAlex, Tavily, and Streamlit**.

---

## 🚀 Project Overview

This project provides **two main AI-powered workflows** through a Streamlit user interface.

The user can select the required functionality from a dropdown menu.

### 1. 📄 Research Paper Q&A

The user can upload a research paper in PDF format.

After uploading the research paper, a search bar becomes available.

The user can ask multiple questions related to the uploaded research paper, and the AI system retrieves relevant information and generates answers.

### 2. 💡 Research Idea & Proposal Generator

The user can provide a research topic or research idea.

The system generates **10 ranked research ideas** related to the user's input.

After generating the ideas, an AI agent selects the best research idea.

The selected idea is then passed through multiple specialized agents that generate:

- Research Introduction
- Problem Statement
- Research Objectives
- Research Questions
- Research Gaps
- Methodology
- Dataset Details
- Expected Results
- Research Contributions
- Complete Research Proposal
- Critic Review
- Final Research Report

---

# ✨ Features

## 📄 Research Paper Upload

Users can upload research papers in PDF format.

The system processes the document and makes its content available for question answering.

### User Workflow

```text
Upload Research Paper
        ↓
PDF Processing
        ↓
Document Processing
        ↓
Vector Representation
        ↓
Relevant Information Retrieval
        ↓
Ask Questions
        ↓
AI Generated Answers
```

---

## 💬 Multiple Questions About Research Papers

After uploading a paper, users can ask multiple questions such as:

```text
What is the main objective of this research?

What methodology was used?

What dataset was used?

What are the limitations of the research?

What research gap has been identified?

```

The system retrieves relevant information and generates answers based on the research paper.

---

# 💡 Research Idea Generation

Users can provide a research topic.

### Example Input

```text
LLM-based Text Summarization
```

The system generates **10 ranked research ideas**.

Each research idea can contain:

- Research concept
- Technical direction
- Key innovation
- Research questions
- Potential impact
- Possible applications
- Research motivation

### Example Output

```text
1. Dynamic Knowledge Graph-Augmented Summarization

2. Personalized LLM-Based Summarization

3. Explainable LLM-Based Summarization

4. Hallucination-Aware Text Summarization

5. Domain-Adaptive LLM Summarization

...
```

---

# ⭐ Best Research Idea Selection

After generating 10 research ideas, a dedicated **Idea Selector Agent** analyzes the generated ideas and selects the most promising research idea.

The selected idea is then used by the remaining research agents.

---

# 🧩 Research Gap Identification

The **Gap Agent** analyzes the selected research idea and identifies potential research gaps.

The system can analyze:

- Existing research limitations
- Current problems
- Missing approaches
- Limitations of existing methods
- Opportunities for improvement
- Possible future research directions

---

# 🛠️ Methodology Generation

The **Methodology Agent** develops a detailed methodology for the selected research idea.

The methodology can include:

- Proposed approach
- System architecture
- Data processing
- Model selection
- Training approach
- Experimental setup
- Evaluation strategy
- Evaluation metrics

---

# 📊 Dataset Details

The system can provide possible dataset information related to the selected research idea.

Dataset analysis can include:

- Dataset name
- Dataset purpose
- Possible data sources
- Data requirements
- Dataset usage
- Data preprocessing
- Evaluation considerations

---

# 📝 Research Proposal Generation

The **Proposal Agent** generates a detailed research proposal based on the selected research idea, research gap, and methodology.

The generated proposal can contain:

- Introduction
- Background
- Problem Statement
- Research Objectives
- Research Questions
- Research Gap
- Proposed Methodology
- Dataset Details
- Experimental Design
- Expected Results
- Research Contributions
- Limitations
- Future Work

---

# 🧾 Critic Review

The **Critic Agent** reviews the generated research proposal.

It analyzes:

- Research quality
- Research gap
- Methodology
- Technical feasibility
- Potential weaknesses
- Limitations
- Possible improvements

The reviewed content is then used to prepare the final research output.

---

# 🤖 Multi-Agent Architecture

The project uses multiple specialized AI agents.

Each agent is responsible for a specific research task.

```text
                         USER
                           │
                           ▼
                  ┌─────────────────┐
                  │  Query Planner  │
                  └────────┬────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │  Search Agent   │
                  └────────┬────────┘
                           │
                           ▼
                 ┌───────────────────┐
                 │ Hybrid Retrieval  │
                 └─────────┬─────────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
          ┌───────┐   ┌─────────┐   ┌────────┐
          │ ArXiv │   │ OpenAlex│   │ Tavily │
          └───┬───┘   └────┬────┘   └───┬────┘
              │            │             │
              └────────────┼─────────────┘
                           ▼
                  ┌─────────────────┐
                  │  Paper Ranker   │
                  └────────┬────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │ Research Ideas  │
                  └────────┬────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │  Idea Selector  │
                  └────────┬────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │ Research Gaps   │
                  └────────┬────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │   Methodology   │
                  └────────┬────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │    Proposal     │
                  └────────┬────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │  Critic Agent   │
                  └────────┬────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │  Final Report   │
                  └─────────────────┘
```

---

# 🔄 Research Workflow

The main research workflow runs in the following order:

```text
Search
  ↓
Query Planning
  ↓
Hybrid Retrieval
  ↓
Store Papers
  ↓
Retrieve Relevant Documents
  ↓
Paper Ranking
  ↓
Research Idea Generation
  ↓
Best Idea Selection
  ↓
Research Gap Identification
  ↓
Methodology Generation
  ↓
Research Proposal
  ↓
Critic Review
  ↓
Final Research Report
```

---

# 🔎 Hybrid Retrieval

The project uses a hybrid retrieval system to collect research papers from multiple sources.

### Research Sources

- 🔬 ArXiv
- 📚 OpenAlex
- 🔎 Tavily

The system searches multiple sources and combines the retrieved results.

Duplicate papers are removed before the results are passed to the downstream agents.

The retrieved papers can also be stored in **ChromaDB** for later retrieval.

---

# 🧠 RAG Architecture

The project uses **Retrieval-Augmented Generation (RAG)** to retrieve relevant information before generating AI responses.

```text
User Query
    ↓
Search
    ↓
Retrieve Documents
    ↓
Store Documents
    ↓
Vector Database
    ↓
Retrieve Relevant Documents
    ↓
LLM
    ↓
Generated Response
```

RAG helps the system provide responses using retrieved research information instead of relying only on the LLM's internal knowledge.

---

# 🗃️ Vector Store

The project contains vector-store code for storing and retrieving research information.

The project uses:

- ChromaDB
- Sentence Transformers

The `vector_store/` directory contains Python source code and is part of the project.

```text
vector_store/
└── section_vector_store.py
```

---

# 🗄️ ChromaDB

ChromaDB is used as a vector database for storing research papers and retrieving relevant documents.

The workflow includes:

```text
Research Papers
      ↓
Embedding
      ↓
ChromaDB
      ↓
Similarity Retrieval
      ↓
Relevant Documents
```

---

# 🧠 Embeddings

The project uses **Sentence Transformers** for creating vector representations of research content.

These embeddings allow the system to perform semantic similarity search.

---

# 💬 Streaming AI Responses

The research workflow supports streaming output.

Instead of waiting for the complete response, the application can display generated content progressively.

The streaming workflow uses:

```text
Graph Node Started
       ↓
Section Started
       ↓
LLM Tokens
       ↓
Next Node
       ↓
More Tokens
       ↓
Final Output
```

This provides a more interactive experience similar to modern AI chat applications.

---

# 🛠️ Technologies

- Python
- LangGraph
- LangChain
- Large Language Models (LLMs)
- Retrieval-Augmented Generation (RAG)
- ChromaDB
- FAISS
- Sentence Transformers
- ArXiv
- OpenAlex
- Tavily
- Streamlit

---

# 📂 Project Structure

```text
research_copilot/
│
├── agents/
│   ├── __init__.py
│   ├── gap_agent.py
│   ├── literature_review_agent.py
│   ├── methodology_agent.py
│   ├── qa_agent.py
│   ├── router_agent.py
│   └── summarizer_agent.py
│
├── data/
│   └── sample.pdf
│
├── embeddings/
│   └── embedding_model.py
│
├── langgraph_flow/
│   ├── __init__.py
│   ├── nodes.py
│   ├── state.py
│   └── workflow.py
│
├── pdf_processing/
│   ├── advanced_section_parser.py
│   └── parser.py
│
├── research_module/
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── critic_agent.py
│   │   ├── final_agent.py
│   │   ├── gap_agent.py
│   │   ├── idea_agent.py
│   │   ├── idea_selector.py
│   │   ├── methodology_agent.py
│   │   ├── paper_ranker.py
│   │   ├── proposal_agent.py
│   │   ├── query_planner.py
│   │   └── search_agent.py
│   │
│   ├── graph/
│   │   ├── __init__.py
│   │   ├── research_graph.py
│   │   └── state.py
│   │
│   ├── llm/
│   │   ├── __init__.py
│   │   ├── chains.py
│   │   └── mistral_llm.py
│   │
│   ├── memory/
│   │   ├── __init__.py
│   │   ├── db_chat_history.py
│   │   └── session_memory.py
│   │
│   ├── retrievers/
│   │   ├── __init__.py
│   │   ├── arxiv_retriever.py
│   │   ├── hybrid_retriever.py
│   │   └── openalex_retriever.py
│   │
│   ├── tools/
│   │   ├── __init__.py
│   │   └── tavily_tool.py
│   │
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── cache_manager.py
│   │   ├── final_builder.py
│   │   └── output_cleaner.py
│   │
│   └── vectorstore/
│       ├── __init__.py
│       └── chroma_store.py
│
├── vector_store/
│   └── section_vector_store.py
│
├── ui/
│   └── app.py
│
├── .env
├── .gitignore
├── requirements.txt
└── README.md
```

---

# ▶️ Run the Project Locally

## 1. Clone the Repository

```bash
git clone https://github.com/Shakir1121/LangGraph-Base-Multi-Agents-Research-Assistant.git
```

```bash
cd LangGraph-Base-Multi-Agents-Research-Assistant
```

---

## 2. Create Virtual Environment

```bash
python -m venv venv
```

---

## 3. Activate Virtual Environment

### Windows

```bash
venv\Scripts\activate
```

### Linux / macOS

```bash
source venv/bin/activate
```

---

## 4. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 🔑 Environment Variables

Create a `.env` file in the project root.

Example:

```env
MISTRAL_API_KEY=your_mistral_api_key
TAVILY_API_KEY=your_tavily_api_key
```

Add any other API keys required by your configuration.

> **Important:** Never upload your real API keys or `.env` file to GitHub.

---

# ▶️ Run Streamlit

Start the application with:

```bash
streamlit run ui/app.py
```

Streamlit will provide a local URL in the terminal.

Open the URL in your browser to use the application.

---

# 🎓 Use Cases

This project can be useful for:

- MS or PhD students
- AI/ML researchers
- Academic researchers
- Research paper analysis
- Literature research
- Research idea generation
- Research gap identification
- Methodology development
- Research proposal generation
- Research paper Q&A

---

# 🌟 Project Goal

The main goal of this project is to create an **AI Research Copilot** that assists researchers throughout the research process.

```text
Research Paper Analysis
        ↓
Literature Search
        ↓
Research Idea Generation
        ↓
Best Idea Selection
        ↓
Research Gap
        ↓
Methodology
        ↓
Dataset
        ↓
Research Proposal
        ↓
Critic Review
        ↓
Final Research Report
```

Instead of using a single AI model for every task, the system uses specialized agents for different research activities.

---

# 🔮 Future Improvements

Future versions can include:

- More research databases
- Automatic citation generation
- Automatic reference management
- Research paper comparison
- Improved paper ranking
- Advanced long-document RAG
- Automatic experiment planning
- Better research evaluation
- User authentication
- Cloud deployment
- Research history and project management

---

