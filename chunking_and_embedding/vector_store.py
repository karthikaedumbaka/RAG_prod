
from typing import List, Optional
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec


def init_pinecone_index(api_key: str, index_name: str, cloud: str = "aws", region: str = "us-east-1", dimension: int = 768):
    """Initialize Pinecone index (create if doesn't exist)"""
    pc = Pinecone(api_key=api_key)

    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud=cloud, region=region)
        )

    return pc.Index(index_name)


def store_in_pinecone(chunks: List[Document], embedder: GoogleGenerativeAIEmbeddings, index_name: str, api_key: str):
    """Store document chunks in Pinecone vector database"""
    vector_store = PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embedder,
        index_name=index_name,
        pinecone_api_key=api_key
    )
    return vector_store


def load_pinecone_vector_store(embedder: GoogleGenerativeAIEmbeddings, index_name: str, api_key: str):
    """Load existing Pinecone vector store"""
    vector_store = PineconeVectorStore(
        embedding=embedder,
        index_name=index_name,
        pinecone_api_key=api_key
    )
    return vector_store
