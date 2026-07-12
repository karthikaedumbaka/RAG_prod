from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import fitz  # PyMuPDF
import json
import os

from docling.document_converter import DocumentConverter

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
PDF_PATH = next(DATA_DIR.glob("*.pdf"))

# PDF_PATH = Path("large.pdf")
OUTPUT_DIR = Path("output")
CHECKPOINT = OUTPUT_DIR / "completed_pages.json"

OUTPUT_DIR.mkdir(exist_ok=True)

TEXT_THRESHOLD = 200      # Characters required to consider page "text"
MAX_WORKERS = os.cpu_count()


def load_checkpoint():
    if CHECKPOINT.exists():
        return set(json.loads(CHECKPOINT.read_text()))
    return set()


def save_checkpoint(completed):
    CHECKPOINT.write_text(json.dumps(sorted(list(completed))))


def page_has_text(pdf_path, page_number):
    doc = fitz.open(pdf_path)

    text = doc[page_number].get_text("text")

    doc.close()

    return len(text.strip()) > TEXT_THRESHOLD


def save_page_pdf(pdf_path, page_number):

    src = fitz.open(pdf_path)

    dst = fitz.open()

    dst.insert_pdf(src, from_page=page_number, to_page=page_number)

    temp_file = OUTPUT_DIR / f"page_{page_number}.pdf"

    dst.save(temp_file)

    dst.close()
    src.close()

    return temp_file


def process_page(page_number):

    converter = DocumentConverter()

    if page_has_text(PDF_PATH, page_number):

        doc = fitz.open(PDF_PATH)

        text = doc[page_number].get_text("text")

        doc.close()

    else:

        temp_pdf = save_page_pdf(PDF_PATH, page_number)

        result = converter.convert(temp_pdf)

        text = result.document.export_to_markdown()

        temp_pdf.unlink()

    output_file = OUTPUT_DIR / f"{page_number:05d}.md"

    with open(output_file, "w", encoding="utf8") as f:
        f.write(text)

    return page_number


def merge_results(total_pages):

    final = OUTPUT_DIR / "final_document.md"

    with open(final, "w", encoding="utf8") as outfile:

        for page in range(total_pages):

            page_file = OUTPUT_DIR / f"{page:05d}.md"

            if page_file.exists():

                with open(page_file, encoding="utf8") as f:

                    outfile.write(f.read())

                    outfile.write("\n\n")


def main():
    print("-- started the test pdf to test fast ---")
    pdf = fitz.open(PDF_PATH)

    total_pages = len(pdf)

    pdf.close()

    completed = load_checkpoint()

    remaining = [
        i for i in range(total_pages)
        if i not in completed
    ]

    print(f"Remaining Pages : {len(remaining)}")

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:

        futures = {
            executor.submit(process_page, page): page
            for page in remaining
        }

        for future in as_completed(futures):

            page = future.result()

            completed.add(page)

            save_checkpoint(completed)

            print(f"Completed Page {page}")

    merge_results(total_pages)

    print("Finished")


if __name__ == "__main__":
    main()
    print("-- END the test pdf to test fast ---")