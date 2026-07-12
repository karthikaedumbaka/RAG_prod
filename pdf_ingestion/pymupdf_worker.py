import fitz  # PyMuPDF
from pathlib import Path
from logger import setup_logger

def process_text_batch(batch_info: dict, output_dir: str):
    """
    Top 1% Move: The Fast Path.
    Extracts native text using PyMuPDF's C-bindings. 
    Bypasses Docling entirely for text-heavy pages.
    """
    batch_path = batch_info["path"]
    batch_name = Path(batch_path).stem
    log = setup_logger(f"pymupdf-{batch_name}")
    
    try:
        doc = fitz.open(batch_path)
        markdown_content = []
        
        for page in doc:
            # PyMuPDF >= 1.23.0 supports direct markdown extraction
            # Fallback to text if markdown fails
            try:
                text = page.get_text("markdown")
            except Exception:
                text = page.get_text("text")
            markdown_content.append(text)
            
        doc.close()
        
        # Write to disk
        out_path = Path(output_dir) / f"{batch_name}.md"
        out_path.write_text("\n\n".join(markdown_content), encoding="utf-8")
        
        log.info(f" {batch_name} (Text) extracted in milliseconds.")
        return {
            "batch": batch_name,
            "type": "text",
            "status": "success",
            "output": str(out_path),
            "pages": batch_info["pages"]
        }
    except Exception as e:
        log.exception(f" PyMuPDF failed on {batch_name}: {e}")
        return {"batch": batch_name, "type": "text", "status": "error", "error": str(e)}