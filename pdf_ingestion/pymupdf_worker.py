import fitz  # PyMuPDF
from pathlib import Path

try:
    from .logger import setup_logger
except ImportError:
    from logger import setup_logger

def _extract_annotation_notes(page) -> str:
    """
    Extracts typed sticky notes, FreeText boxes, stamps, etc.
    Ignores Ink (hand-drawn) annotations as they are routed to OCR.
    """
    notes = []
    try:
        for annot in page.annots() or []:
            type_id, type_name = annot.type[0], annot.type[1]
            if type_name == "Ink":
                continue
            content = (annot.info.get("content") or "").strip()
            if content:
                notes.append(f"- [{type_name} annotation] {content}")
    except Exception:
        pass
    return "\n".join(notes)

def process_text_batch(batch_info: dict, output_dir: str, user_id: str = "unknown"):
    """
    Extracts native text using PyMuPDF's C-bindings.
    Injects invisible page markers for RAG chunking.
    """
    batch_path = batch_info["path"]
    batch_name = Path(batch_path).stem
    log = setup_logger(f"pymupdf-{batch_name}", user_id)
    
    try:
        # 🛠️ FIX: Use context manager to prevent file handle leaks
        with fitz.open(batch_path) as doc:
            markdown_content = []
            
            for local_idx, page in enumerate(doc):
                original_page_num = batch_info["pages"][local_idx] + 1 
                
                try:
                    text = page.get_text("markdown")
                except Exception:
                    text = page.get_text("text")
                    
                page_marker = f"\n\n<!-- page: {original_page_num} -->\n\n"
                annotation_notes = _extract_annotation_notes(page)
                
                if annotation_notes:
                    text = text + "\n\n" + annotation_notes
                    
                markdown_content.append(page_marker + text)
                
        # Write to disk (outside the 'with' block, so the file is closed)
        out_path = Path(output_dir) / f"{batch_name}.md"
        out_path.write_text("\n\n".join(markdown_content), encoding="utf-8")
        
        log.info(f" {batch_name} (Text) extracted in milliseconds.")
        return {
            "batch": batch_name, "type": "text", "status": "success",
            "output": str(out_path), "pages": batch_info["pages"]
        }
        
    except Exception as e:
        log.exception(f" PyMuPDF failed on {batch_name}: {e}")
        return {"batch": batch_name, "type": "text", "status": "error", "error": str(e)}