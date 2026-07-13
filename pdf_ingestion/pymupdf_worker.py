import fitz  # PyMuPDF
from pathlib import Path
from logger import setup_logger


def _extract_annotation_notes(page) -> str:
    """
    get_text() only returns the page's programmatic text layer. It never
    sees annotation content - typed sticky notes, FreeText boxes, stamps,
    etc. On the fast text path those would otherwise vanish silently. Ink
    (hand-drawn) annotations are excluded here on purpose: they have no
    text to read and are instead forced onto the OCR path by analyzer.py.
    """
    notes = []
    try:
        for annot in page.annots() or []:
            type_id, type_name = annot.type[0], annot.type[1]
            if type_name == "Ink":
                continue
            content = (annot.info.get("content") or "").strip()
            if content:
                notes.append(f"- **[{type_name} annotation]** {content}")
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
        doc = fitz.open(batch_path)
        markdown_content = []
        
        # batch_info["pages"] contains the ORIGINAL page numbers for this batch
        for local_idx, page in enumerate(doc):
            # Get the original 1-based page number for RAG metadata
            original_page_num = batch_info["pages"][local_idx] + 1 
            
            # PyMuPDF >= 1.23.0 supports direct markdown extraction
            try:
                text = page.get_text("markdown")
            except Exception:
                text = page.get_text("text")
            
            # INJECT INVISIBLE PAGE MARKER
            # We use an HTML comment so the LLM ignores it, but the RAG chunker can parse it.
            page_marker = f"\n\n<!-- page: {original_page_num} -->\n\n"

            annotation_notes = _extract_annotation_notes(page)
            if annotation_notes:
                text = text + "\n\n" + annotation_notes

            markdown_content.append(page_marker + text)
            
        doc.close()
        
        # Write to disk
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