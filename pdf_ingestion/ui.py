import gradio as gr
import socket
import shutil
import time
import uuid
from pathlib import Path
from typing import Union

try:
    from .config import PipelineConfig
    from .auth import authenticate_or_register, init_db
    from .analyzer import analyze_pdf
    from .batcher import create_batches
    from .extractor import extract_batches
    from .merger import merge_outputs
    from .cleaner import cleanup_artifacts
    from .utils import ensure_dir
except ImportError:
    from config import PipelineConfig
    from auth import authenticate_or_register, init_db
    from analyzer import analyze_pdf
    from batcher import create_batches
    from extractor import extract_batches
    from merger import merge_outputs
    from cleaner import cleanup_artifacts
    from utils import ensure_dir


def _resolve_uploaded_pdf(pdf_file: Union[str, Path, object]) -> Path:
    if isinstance(pdf_file, Path):
        return pdf_file
    if isinstance(pdf_file, str):
        return Path(pdf_file)
    if hasattr(pdf_file, "name"):
        return Path(pdf_file.name)
    raise TypeError("Unsupported upload payload received from Gradio.")


def _get_custom_css() -> str:
    return """
    .gradio-container { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
    .header { background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%); color: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; text-align: center; }
    .header h1 { margin: 0; font-size: 2em; font-weight: 600; }
    .header p { margin: 5px 0 0 0; opacity: 0.9; }
    .auth-box { border: 1px solid #e0e0e0; border-radius: 8px; padding: 15px; background: #f9f9f9; }
    .log-box { font-family: 'Consolas', 'Courier New', monospace; font-size: 0.9em; background: #1e1e1e; color: #d4d4d4; border-radius: 4px; }
    """


def _find_available_port(host: str = "127.0.0.1", preferred_port: int = 7860, search_limit: int = 20) -> int:
    for port in range(preferred_port, preferred_port + search_limit):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind((host, port))
                return port
            except OSError:
                continue
    raise OSError(
        f"Could not find a free port between {preferred_port} and {preferred_port + search_limit - 1}."
    )

def run_pipeline_with_auth(user_id: str, password: str, pdf_file, progress=gr.Progress()):
    """
    Main pipeline function with authentication and progress tracking.
    """
    # 1. Authenticate user
    user_id = user_id.strip()
    if not user_id or not password:
        return "User ID and password are required.", None, ""

    auth_status = authenticate_or_register(user_id, password)
    if auth_status == "failed":
        return f"Authentication failed. Incorrect password for user '{user_id}'.", None, ""

    status_msg = f"Authenticated as {user_id}. "
    if auth_status == "registered":
        status_msg += "Account created. "
        
    if pdf_file is None:
        return status_msg + "Please upload a PDF file.", None, ""
        
    # Setup paths
    PROJECT_ROOT = Path(__file__).parent.parent
    uploaded_pdf = _resolve_uploaded_pdf(pdf_file)
    
    # 🛠️ FIX: Create isolated directories for this specific session to prevent race conditions
    session_id = uuid.uuid4().hex[:8]
    
    temp_upload = PROJECT_ROOT / f"temp_upload_{session_id}.pdf"
    output_dir = PROJECT_ROOT / "output" / user_id
    temp_dir = PROJECT_ROOT / "temp_batches" / f"{user_id}_{session_id}"
    
    # Save uploaded file
    shutil.copy2(str(uploaded_pdf), temp_upload)
    
    # Initialize config
    config = PipelineConfig()
    config.user_id = user_id
    config.output_dir = str(output_dir)
    config.temp_dir = str(temp_dir)
    ensure_dir(output_dir)
    ensure_dir(temp_dir)
    
    log_output = []
    try:
        # 2. Analyze PDF
        progress(0.1, desc="Analyzing PDF structure...")
        log_output.append(f"[{time.strftime('%H:%M:%S')}] Analyzing {uploaded_pdf.name}...")
        analysis = analyze_pdf(str(temp_upload), config)
        log_output.append(f"[{time.strftime('%H:%M:%S')}] Total pages: {analysis['total_pages']}")
        log_output.append(f"[{time.strftime('%H:%M:%S')}] Text pages: {analysis['text_count']}")
        log_output.append(f"[{time.strftime('%H:%M:%S')}] OCR needed: {analysis['ocr_count']}")
        
        # 3. Create batches
        progress(0.2, desc="Creating batches...")
        log_output.append(f"[{time.strftime('%H:%M:%S')}] Creating smart batches...")
        batches = create_batches(str(temp_upload), analysis, config)
        total_batches = len(batches['text_batches']) + len(batches['ocr_batches'])
        log_output.append(f"[{time.strftime('%H:%M:%S')}] Created {total_batches} batches")
        
        # 4. Extract (with progress updates)
        progress(0.3, desc="Extracting text and running OCR...")
        log_output.append(f"[{time.strftime('%H:%M:%S')}] Starting extraction...")
        extract_batches(batches, config)
        
        progress(0.8, desc="Merging results...")
        log_output.append(f"[{time.strftime('%H:%M:%S')}] Merging outputs...")
        
        # 5. Merge
        final_md = merge_outputs(uploaded_pdf.stem, batches, config)
        
        # 6. Cleanup
        log_output.append(f"[{time.strftime('%H:%M:%S')}] Cleaning up...")
        cleanup_artifacts(config)
        
        progress(1.0, desc="Complete!")
        log_output.append(f"[{time.strftime('%H:%M:%S')}] Pipeline completed successfully!")
        log_output.append(f"[{time.strftime('%H:%M:%S')}] Output: {final_md}")
        
        return status_msg + "Processing complete!", str(final_md), "\n".join(log_output)
        
    except Exception as e:
        error_msg = f"[{time.strftime('%H:%M:%S')}] ERROR: {str(e)}"
        log_output.append(error_msg)
        return f"Error: {str(e)}", None, "\n".join(log_output)
    finally:
        if temp_upload.exists():
            temp_upload.unlink()

def create_ui():
    """Create the Gradio interface."""
    with gr.Blocks(title="RAG PDF Ingestion") as demo:
        gr.HTML("""
        <div class="header">
            <h1>RAG PDF Ingestion Pipeline</h1>
            <p>Enterprise-grade PDF to Markdown conversion with intelligent OCR</p>
        </div>
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Authentication")
                with gr.Group():
                    user_id_input = gr.Textbox(label="User ID", placeholder="Enter your user ID", elem_classes="auth-box")
                    password_input = gr.Textbox(label="Password", type="password", placeholder="Enter your password", elem_classes="auth-box")
                
                gr.Markdown("### Upload PDF")
                pdf_upload = gr.File(label="Select PDF File", file_types=[".pdf"], type="filepath")
                
                with gr.Row():
                    submit_btn = gr.Button("Start Processing", variant="primary", size="lg")
                    clear_btn = gr.Button("Clear", variant="secondary")
                    
                gr.Markdown("### System Info")
                with gr.Group():
                    gr.Markdown("""
                    - **Engine**: Docling + PyMuPDF
                    - **OCR**: RapidOCR (ONNX)
                    - **Output**: Markdown
                    - **Auto-batching**: Enabled
                    - **GPU Acceleration**: Auto-detect
                    """)
                    
            with gr.Column(scale=2):
                gr.Markdown("### Processing Status")
                status_output = gr.Textbox(label="Status", interactive=False, lines=2)
                
                gr.Markdown("### Live Logs")
                log_output = gr.Textbox(label="Pipeline Logs", interactive=False, lines=15, max_lines=20, elem_classes="log-box")
                
                gr.Markdown("### Download Result")
                download_output = gr.File(label="Download Markdown", interactive=False)

        submit_btn.click(
            fn=run_pipeline_with_auth,
            inputs=[user_id_input, password_input, pdf_upload],
            outputs=[status_output, download_output, log_output]
        )
        clear_btn.click(
            fn=lambda: ("", None, ""),
            inputs=[],
            outputs=[status_output, download_output, log_output]
        )
        
    return demo

def launch_ui():
    """Launch the Gradio web interface."""
    init_db()
    demo = create_ui()
    host = "127.0.0.1"
    port = _find_available_port(host=host, preferred_port=7860)
    print(f"Starting UI at http://{host}:{port}")
    demo.launch(
        css=_get_custom_css(),
        server_name=host,
        server_port=port,
        share=False,
        show_error=True
    )

if __name__ == "__main__":
    launch_ui()