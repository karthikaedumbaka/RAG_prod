import sys
import numpy as np
from pathlib import Path
from typing import List, TypedDict, Any, Dict

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_openai import ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langgraph.graph import StateGraph, START, END
from rank_bm25 import BM25Okapi

# Add project root to path to import your custom modules
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from chunking_and_embedding.embedder import create_embedder
from chunking_and_embedding.chunker import load_markdown_files, chunk_documents
from .config import QueryConfig

# ==============================================================================
# 1. DEFINE THE GRAPH STATE
# ==============================================================================
class GraphState(TypedDict):
    """Represents the state of our RAG graph."""
    messages: List[Any]       # Chat history (HumanMessage, AIMessage)
    question: str             # The latest user query
    context: List[Document]   # Retrieved chunks from Hybrid Search
    generation: str           # The LLM's text response
    sources: List[Dict]       # Formatted citations (source, page)

# ==============================================================================
# 2. CUSTOM HYBRID RETRIEVER (RRF)
# ==============================================================================
class CustomHybridRetriever(BaseRetriever):
    """
    A bulletproof Custom Hybrid Retriever using Reciprocal Rank Fusion (RRF).
    Bypasses LangChain's fragile import paths by using rank-bm25 directly.
    """
    dense_retriever: Any
    documents: List[Document]
    k: int = 5
    bm25: Any = None
    
    def __init__(self, dense_retriever: Any, documents: List[Document], k: int = 5, **kwargs):
        super().__init__(dense_retriever=dense_retriever, documents=documents, k=k, **kwargs)
        # Pre-compute the BM25 index for lightning-fast local search
        corpus = [doc.page_content.split() for doc in documents]
        self.bm25 = BM25Okapi(corpus)

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> List[Document]:
        # 1. Dense Retrieval (Pinecone)
        dense_docs = self.dense_retriever.invoke(query)
        
        # 2. Sparse Retrieval (Local BM25)
        tokenized_query = query.split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        top_k_indices = np.argsort(bm25_scores)[::-1][:self.k]
        bm25_docs = [self.documents[i] for i in top_k_indices]
        
        # 3. Reciprocal Rank Fusion (RRF)
        # RRF formula: score = sum(1 / (k + rank))
        rrf_k = 60
        scored_docs = {}
        
        # Score Dense results
        for rank, doc in enumerate(dense_docs):
            # Create a unique key for the document to handle duplicates
            key = (doc.metadata.get("source"), doc.metadata.get("page"), doc.page_content[:50])
            if key not in scored_docs:
                scored_docs[key] = {"doc": doc, "score": 0.0}
            scored_docs[key]["score"] += 1.0 / (rrf_k + rank + 1)
            
        # Score BM25 results
        for rank, doc in enumerate(bm25_docs):
            key = (doc.metadata.get("source"), doc.metadata.get("page"), doc.page_content[:50])
            if key not in scored_docs:
                scored_docs[key] = {"doc": doc, "score": 0.0}
            scored_docs[key]["score"] += 1.0 / (rrf_k + rank + 1)
            
        # Sort by combined RRF score and return top K
        sorted_docs = sorted(scored_docs.values(), key=lambda x: x["score"], reverse=True)
        return [item["doc"] for item in sorted_docs[:self.k]]

# ==============================================================================
# 3. INITIALIZE COMPONENTS (LLM & HYBRID RETRIEVER)
# ==============================================================================
def get_llm(config: QueryConfig):
    """Initializes the Groq LLM via OpenAI-compatible API."""
    if not config.LLM_API_KEY:
        raise ValueError("Missing GROQ_API_KEY in .env")
        
    return ChatOpenAI(
        model=config.LLM_MODEL,
        api_key=config.LLM_API_KEY,
        base_url=config.LLM_BASE_URL,
        temperature=0.3,
    )

def get_hybrid_retriever(config: QueryConfig):
    """Initializes the Custom Hybrid Retriever (BM25 + Pinecone Dense)."""
    
    # 1. DENSE RETRIEVER (Pinecone)
    embedder = create_embedder(
        model=config.EMBEDDING_MODEL, 
        output_dimensionality=config.EMBEDDING_DIMENSION
    )
    vector_store = PineconeVectorStore(
        embedding=embedder,
        index_name=config.PINECONE_INDEX_NAME,
        pinecone_api_key=config.PINECONE_API_KEY
    )
    
    dense_retriever = vector_store.as_retriever(
        search_type="mmr" if config.USE_MMR else "similarity",
        search_kwargs={"k": config.RETRIEVAL_K, "fetch_k": config.MMR_FETCH_K}
    )
    
    # 2. Load local markdown files for BM25
    docs = load_markdown_files(config.INPUT_DIR)
    chunks = chunk_documents(docs, chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP)
    
    if not chunks:
        print("⚠️ Warning: No local markdown files found for BM25. Falling back to Dense-only retrieval.")
        return dense_retriever
        
    # 3. Return our Custom RRF Hybrid Retriever
    return CustomHybridRetriever(
        dense_retriever=dense_retriever, 
        documents=chunks, 
        k=config.RETRIEVAL_K
    )

# ==============================================================================
# 4. DEFINE GRAPH NODES
# ==============================================================================
def retrieve_node(state: GraphState, config: QueryConfig):
    """Node 1: Retrieve relevant documents using Custom Hybrid Search."""
    print("🔍 [Graph] Retrieving context via Custom Hybrid Search (RRF: BM25 + Pinecone)...")
    retriever = get_hybrid_retriever(config)
    question = state["question"]
    
    docs = retriever.invoke(question)
    return {"context": docs}

def generate_node(state: GraphState, config: QueryConfig):
    """Node 2: Generate an answer using the LLM and retrieved context."""
    print("🧠 [Graph] Generating answer via Groq LLM...")
    llm = get_llm(config)
    
    # Format the context with source citations
    context_str = ""
    for i, doc in enumerate(state["context"], 1):
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "N/A")
        context_str += f"[Source {i}: {source} | Page {page}]\n{doc.page_content}\n\n"

    # Prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", 
         "You are an expert research assistant. Answer the user's question using ONLY the provided context. "
         "If the answer is not in the context, politely state that you don't know. "
         "When referencing facts, include the Source and Page number in brackets, e.g., [Source: paper.md, Page: 2]."
        ),
        MessagesPlaceholder(variable_name="messages"), # Inject chat history
        ("human", 
         "Context:\n{context}\n\nUser Question: {question}\n\nProvide a comprehensive answer:"
        )
    ])

    # Invoke LLM
    chain = prompt | llm
    response = chain.invoke({
        "messages": state["messages"],
        "context": context_str,
        "question": state["question"]
    })
    
    return {"generation": response.content, "messages": state["messages"] + [response]}

def format_sources_node(state: GraphState, config: QueryConfig):
    """Node 3: Extract and format metadata for the UI/CLI to display."""
    print("📑 [Graph] Formatting source citations...")
    sources = []
    seen = set()
    
    for doc in state["context"]:
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "N/A")
        
        # Deduplicate sources
        key = f"{source}_{page}"
        if key not in seen:
            seen.add(key)
            sources.append({"source": source, "page": page})
            
    return {"sources": sources}

# ==============================================================================
# 5. BUILD AND COMPILE THE GRAPH
# ==============================================================================
def build_rag_graph(config: QueryConfig):
    """Constructs the LangGraph workflow."""
    workflow = StateGraph(GraphState)

    # Add nodes (passing config via lambda)
    workflow.add_node("retrieve", lambda state: retrieve_node(state, config))
    workflow.add_node("generate", lambda state: generate_node(state, config))
    workflow.add_node("format_sources", lambda state: format_sources_node(state, config))

    # Define the flow
    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", "format_sources")
    workflow.add_edge("format_sources", END)

    # Compile the graph
    return workflow.compile()