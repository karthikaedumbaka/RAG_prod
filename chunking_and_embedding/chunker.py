
from pathlib import Path
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


def load_markdown_files(input_dir: str) -> List[Document]:
    """Load all markdown files from input directory"""
    docs = []
    input_path = Path(input_dir)
    for md_file in input_path.glob("*.md"):
        with open(md_file, "r", encoding="utf-8") as f:
            content = f.read()
        docs.append(Document(
            page_content=content,
            metadata={"source": md_file.name, "file_path": str(md_file)}
        ))
    return docs


def chunk_documents(docs: List[Document], chunk_size: int = 1000, chunk_overlap: int = 200, separators: List[str] = None) -> List[Document]:
    """Split documents into chunks using LangChain's RecursiveCharacterTextSplitter"""
    if separators is None:
        separators = ["\n\n", "\n", " ", ""]

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators,
        length_function=len,
    )

    chunks = text_splitter.split_documents(docs)
    return chunks
