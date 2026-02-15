# AI Leadership Insight Agent (RAG) — Adobe Document Q&A + Executive Insights

## 📌 Project Overview

This project implements **Task 1 — AI Leadership Insight Agent** using a Retrieval-Augmented Generation (RAG) pipeline.

It enables leadership teams to ask natural-language questions over large corporate documents such as:
- Annual Reports
- 10-K / 10-Q filings
- Strategy documents
- Investor updates

The system retrieves relevant evidence and generates grounded answers with citations.

---

## 🎯 What We Are Building

Leadership teams often struggle to quickly extract insights from large enterprise documents.  
This solution provides:

- Semantic search over company documents
- Evidence-based answers
- Executive-level summaries
- Citation-backed responses
- A chatbot UI for interactive Q&A

---

## 🧠 Architecture

1. Document Loader (PDF/DOCX/TXT)
2. Text Chunking
3. Embeddings (SentenceTransformers)
4. FAISS Vector Index
5. Retrieval (Top-K semantic search)
6. LLM-based Answer Generation (OpenAI / Ollama)
7. Streamlit Chat UI

---

## 📂 Sample Input Documents (Adobe)

Place these in the `docs/` folder:

- adbe-2024-annual-report.pdf
- adbe-10k-fy25-final.pdf
- adbe10qq125unofficialpdf.pdf

These are used for testing leadership questions.

---

## ⚙️ Setup Instructions

### 1️⃣ Create Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🔑 LLM Configuration

Copy environment template:

```bash
cp .env.example .env
```

### Option A — Use OpenAI

```
OPENAI_API_KEY=your_key_here
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4o-mini
```

### Option B — Use Ollama (Recommended)

Start Ollama:

```bash
ollama serve
ollama pull llama3
```

Set `.env`:

```
OPENAI_API_KEY=dummy
OPENAI_BASE_URL=http://127.0.0.1:11434/v1
OPENAI_MODEL=llama3
```

---

## 🏗️ Build the Index

```bash
python -m leadership_agent.cli index --docs ./docs --out ./indices/company
```

This step:
- Reads documents
- Chunks text
- Creates embeddings
- Builds FAISS index

---

## 💬 Ask Questions via CLI

```bash
python -m leadership_agent.cli ask --index ./indices/company --question "What is Adobe’s mission and core business focus?"
```

---

## 🖥️ Run Streamlit Chat UI

```bash
python -m streamlit run streamlit_app.py
```

Features:
- Chat-style Q&A interface
- Build index button
- Clear chat option
- Uses same backend RAG pipeline

---

## 🧪 Demo Questions

Use these for testing:

1. What is Adobe’s mission and core business focus?
2. What are Adobe’s reportable business segments?
3. How is Adobe using AI in its products?
4. What risks are mentioned in the filings?
5. What is Adobe’s long-term strategy?

---

## 📊 Output Format

Each response includes:

- Executive Summary
- Key Findings
- Evidence citations
- Confidence score
- Retrieved chunk references

---

## 📌 Assumptions

- Documents are the source of truth
- LLM must answer only from retrieved context
- If not found → returns "Not found in provided documents"

---

## 🧯 Troubleshooting

### Streamlit using wrong Python
Run:

```bash
python -m streamlit run streamlit_app.py
```

### Missing modules
Install:

```bash
pip install -r requirements.txt
```

### Ollama model missing
Check:

```bash
ollama list
ollama pull llama3
```

---

## 📁 Folder Structure

```
leadership_insight_agent/
│
├── docs/
├── indices/
├── leadership_agent/
├── streamlit_app.py
├── requirements.txt
├── README.md
└── .env
```

---

## 🚀 Highlights

- Works offline with Ollama
- Evidence-based responses
- Executive-friendly summaries
- Interactive chatbot UI
- Production-ready architecture

---
