import re
from pathlib import Path
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

def clean_markdown_text(text: str) -> str:
    """
    Cleans raw markdown extracted from PDFs. 
    Converts page markers to a distinct format for later metadata extraction.
    """
    # 1. Convert page markers to a distinct format: <!-- page: 1 --> -> \n[PAGE:1]\n
    # This allows us to easily extract them later for metadata.
    text = re.sub(r'<!--\s*page:\s*(\d+)\s*-->', r'\n[PAGE:\1]\n', text)
    
    # 2. Remove all other HTML comments (e.g., <!-- image -->, <!-- formula-not-decoded -->)
    text = re.sub(r'<!--.*?-->', '', text)
    
    # 3. Remove empty markdown headers (e.g., lines that are just "## " or "### ")
    text = re.sub(r'^#+\s*$', '', text, flags=re.MULTILINE)
    
    # 4. Remove markdown image syntax if it's broken or empty
    text = re.sub(r'!\[\]\(\)', '', text)
    
    # 5. Collapse 3 or more newlines down to 2 (standard paragraph break)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # 6. Strip leading/trailing whitespace
    return text.strip()

def load_markdown_files(input_dir: str) -> List[Document]:
    """Load and clean all markdown files from input directory"""
    docs = []
    input_path = Path(input_dir)
    
    for md_file in input_path.glob("*.md"):
        with open(md_file, "r", encoding="utf-8") as f:
            raw_content = f.read()
            
        # 🧹 CLEAN THE TEXT BEFORE CHUNKING
        clean_content = clean_markdown_text(raw_content)
        
        # Skip files that are completely empty after cleaning
        if not clean_content:
            continue 
            
        docs.append(Document(
            page_content=clean_content,
            metadata={"source": md_file.name, "file_path": str(md_file)}
        ))
    return docs

def extract_and_clean_page_metadata(chunks: List[Document]) -> List[Document]:
    """
    Extracts [PAGE:X] markers from chunk content, adds them to metadata,
    and removes the markers from the final text so the LLM gets clean data.
    """
    page_pattern = re.compile(r'\[PAGE:(\d+)\]')
    
    for chunk in chunks:
        # Find all page numbers in the chunk
        pages = page_pattern.findall(chunk.page_content)
        
        if pages:
            # Convert to integers and get unique sorted pages
            unique_pages = sorted(list(set(int(p) for p in pages)))
            
            # Add to metadata (Pinecone accepts: string, number, boolean, list of strings)
            chunk.metadata["page"] = unique_pages[0]          # Single integer ✅
            chunk.metadata["pages"] = [str(p) for p in unique_pages]  # List of strings ✅
            
            # Remove the markers from the text so it's clean for the LLM
            chunk.page_content = page_pattern.sub('', chunk.page_content).strip()
        else:
            # 🚨 FIX: Pinecone does NOT accept None/null or empty lists.
            # Simply remove the keys entirely so they are never sent to Pinecone.
            chunk.metadata.pop("page", None)
            chunk.metadata.pop("pages", None)
            
            # Clean up any leftover markers just in case
            chunk.page_content = page_pattern.sub('', chunk.page_content).strip()
            
    return chunks


def chunk_documents(docs: List[Document], chunk_size: int = 1000, chunk_overlap: int = 200, separators: List[str] = None) -> List[Document]:
    """Split documents into chunks using LangChain's RecursiveCharacterTextSplitter"""
    if separators is None:
        # Prioritize splitting by double newlines (paragraphs) first
        separators = ["\n\n", "\n", " ", ""]
        
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators,
        length_function=len,
    )
    chunks = text_splitter.split_documents(docs)
    
    #  EXTRACT PAGE NUMBERS INTO METADATA & CLEAN TEXT
    chunks = extract_and_clean_page_metadata(chunks)
    
    return chunks