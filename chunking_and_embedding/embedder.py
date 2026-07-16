
from typing import List
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document


def create_embedder(google_api_key: str, model: str = "models/text-embedding-004"):
    """Create a Google Generative AI embeddings instance"""
    return GoogleGenerativeAIEmbeddings(model=model, google_api_key=google_api_key)


def embed_chunks(chunks: List[Document], embedder) -> List[Document]:
    """Embed chunks using the provided embedder (optional, vector stores usually do this)"""
    # Note: Most vector stores handle embedding internally, but this is here for reference
    return chunks
