# 🚀 Prod-RAG: Production-Grade Intelligent RAG Pipeline

A robust, end-to-end, production-ready Retrieval-Augmented Generation (RAG) system. It intelligently routes complex PDFs (charts, tables, handwriting) through dual-path extraction, auto-tunes Matryoshka embedding dimensions to save vector database costs, and delivers highly accurate, citation-backed answers using a Custom Hybrid Search (RRF) and LangGraph state machine.

---

## 🌟 Key Features

### 📄 Intelligent PDF Ingestion & Smart Routing
* **Dual-Path Extraction**: Automatically analyzes PDF pages and routes them to the optimal pipeline:
  * **Fast Path (PyMuPDF)**: For pure text pages (extracts in milliseconds).
  * **Rich Path (Docling)**: For complex pages with charts, tables, diagrams, or handwritten ink annotations (uses advanced OCR, layout analysis, and table structure extraction).
* **Resilience**: Built-in `CheckpointManager` allows the pipeline to resume exactly where it left off if interrupted, preventing wasted compute.

### 🧠 Auto-Tuning Matryoshka Embeddings
* **Cost & Performance Optimization**: Automatically tests embedding dimensions (`256`, `512`, `768`) against a ground-truth Q&A dataset.
* **Smart Caching**: Uses `nomic-ai/nomic-embed-text-v1.5` with Matryoshka Representation Learning to truncate and L2-normalize vectors. This saves massive amounts of Pinecone storage costs without sacrificing retrieval accuracy.

### 🔍 Custom Hybrid Retrieval (RRF)
* **Bulletproof Search**: Bypasses fragile native LangChain hybrid implementations by building a custom `CustomHybridRetriever`.
* **Reciprocal Rank Fusion (RRF)**: Seamlessly blends **Pinecone Dense Vector Search** (semantic) with local **BM25 Sparse Search** (keyword) using the formula: `score = sum(1 / (k + rank))`.
* **Metadata Extraction**: Automatically extracts and cleans page numbers from Markdown markers for precise source citations.

### 🤖 LangGraph Generation & Strict Grounding
* **Stateful Chat**: Powered by LangGraph for structured, stateful conversational flows (`retrieve` -> `generate` -> `format_sources`).
* **Strict Grounding**: System prompts are tuned to prevent hallucinations, forcing the LLM to rely *only* on retrieved context.
* **Beautiful Citations**: Automatically formats and displays exact source files and page numbers in the CLI.

### 📊 End-to-End RAG Evaluation
* **LLM-as-a-Judge**: Automatically evaluates the pipeline using a ground-truth dataset, measuring:
  * **Retrieval Metrics**: Source Hit Rate (Recall@5) and Mean Reciprocal Rank (MRR).
  * **Generation Metrics**: Faithfulness (groundedness) and Relevance (1-5 scale).

---

## 🏗️ Architecture & Pipeline Flow

```text
[ PDFs in /data ] 
       │
       ▼
┌─────────────────────────────────────────┐
│  1. PDF INGESTION (Smart Routing)       │
│  • Analyze visual density & ink         │
│  • Route to PyMuPDF (Text) or Docling   │
│  • Concurrent extraction + Checkpoints  │
└─────────────────────────────────────────┘
       │ (Outputs clean Markdown with <!-- page: X --> markers)
       ▼
┌─────────────────────────────────────────┐
│  2. CHUNKING & EMBEDDING                │
│  • Auto-evaluate optimal dimension      │
│  • Matryoshka truncation & L2 Norm      │
│  • Resumable upsert to Pinecone         │
└─────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────┐
│  3. EVALUATION (Automated)              │
│  • Test Retrieval (Recall/MRR)          │
│  • Test Generation (Faithfulness/Rel.)  │
└─────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────┐
│  4. QUERY PIPELINE (LangGraph)          │
│  • Custom Hybrid Retriever (RRF)        │
│  • Groq LLM (Llama 3.3 70B)           │
│  • Interactive CLI with Citations       │
└─────────────────────────────────────────┘


###📂 Project Structure

git_rag/
├── combined_pipeline/      # Master orchestrator (Auth -> Ingest -> Embed -> Eval -> Chat)
├── pdf_ingestion/          # Smart PDF parsing (PyMuPDF + Docling)
├── chunking_and_embedding/ # Chunking, Matryoshka Embeddings, Pinecone Upsert
├── query_pipeline/         # LangGraph Hybrid Search & LLM Generation
├── evaluation/             # End-to-End RAG Evaluation (LLM-as-a-Judge)
├── data/                   # Place your raw PDFs here
├── output/                 # Generated Markdown files
├── logs/                   # Process-safe rotating logs per user
├── .env                    # Environment variables (API Keys)
└── README.md               # You are here!

