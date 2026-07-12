from docling.document_converter import DocumentConverter
from pathlib import Path
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
pdf = next(DATA_DIR.glob("*.pdf"))
def covert_files(source):
    converter = DocumentConverter()
    result = converter.convert(source)  
    print(result.document.export_to_markdown())

if __name__ == "__main__":
    print("Started pdf_to_text")
    
    # source = r"C:\Users\karth\Desktop\rag\rag_project_1\data\SDA_INDIA_2023_24.pdf"
    covert_files(pdf)
    print("completed the pdf_to_text")
