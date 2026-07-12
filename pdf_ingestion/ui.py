import gradio as gr
import os
import shutil
import time
import uuid
import bcrypt
from pathlib import Path
from typing import Dict, Any
import threading
import queue

# Import your existing backend modules
from config import PipelineConfig
from auth import authenticate_or_register, init_db
from analyzer import analyze_pdf
from batcher import create_batches
from extractor import extract_batches
from merger import merge_outputs
from cleaner import cleanup_artifacts
from utils import ensure_dir, find_pdfs

# Global state for tracking progress
progress_queue = queue.Queue()
current_user_id = None

def run_pipeline_with_auth(user_id: str, password: str, pdf_file, progress=gr.Progress()):
    """
    Main pipeline function with authentication and progress tracking.
    """
    global current_user_id
    
    # 1. Authenticate user
    auth_status = authenticate_or_register(user_id, password)
    
    if auth_status == "failed":
        return f"Authentication failed. Incorrect password for user '{user_id}'.", None, ""
    
    current_user_id = user_id
    status_msg = f"Authenticated as {user_id}. "
    if auth_status == "registered":
        status_msg += "Account created. "
    
    if pdf_file is None:
        return status_msg + "Please upload a PDF file.", None, ""
    
    # Setup paths
    PROJECT_ROOT = Path(__file__).parent.parent
    temp_upload = PROJECT_ROOT / "temp_upload.pdf"
    output_dir = PROJECT_ROOT / "output"
    temp_dir = PROJECT_ROOT / "temp_batches"
    
    # Save uploaded file
    shutil.copy(pdf_file.name, temp_upload)
    
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
        log_output.append(f"[{time.strftime('%H:%M:%S')}] Analyzing {Path(pdf_file.name).name}...")
        
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
        final_md = merge_outputs(Path(temp_upload).stem, batches, config)
        
        # 6. Cleanup
        log_output.append(f"[{time.strftime('%H:%M:%S')}] Cleaning up...")
        cleanup_artifacts(config)
        
        progress(1.0, desc="Complete!")
        log_output.append(f"[{time.strftime('%H:%M:%S')}] Pipeline completed successfully!")
        log_output.append(f"[{time.strftime('%H:%M:%S')}] Output: {final_md}")
        
        # Clean up temp upload
        if temp_upload.exists():
            temp_upload.unlink()
        
        return status_msg + "Processing complete!", str(final_md), "\n".join(log_output)
        
    except Exception as e:
        error_msg = f"[{time.strftime('%H:%M:%S')}] ERROR: {str(e)}"
        log_output.append(error_msg)
        return f"Error: {str(e)}", None, "\n".join(log_output)

def create_ui():
    """
    Create the Gradio interface.
    """
    
    # Custom CSS for classical, professional look
    custom_css = """
    .gradio-container {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .header {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        padding: 20px;
        border-radius: 8px;
        margin-bottom: 20px;
        text-align: center;
    }
    .header h1 {
        margin: 0;
        font-size: 2em;
        font-weight: 600;
    }
    .header p {
        margin: 5px 0 0 0;
        opacity: 0.9;
    }
    .auth-box {
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 15px;
        background: #f9f9f9;
    }
    .log-box {
        font-family: 'Consolas', 'Courier New', monospace;
        font-size: 0.9em;
        background: #1e1e1e;
        color: #d4d4d4;
        border-radius: 4px;
    }
    """
    
    with gr.Blocks(css=custom_css, title="RAG PDF Ingestion") as demo:
        
        # Header
        gr.HTML("""
        <div class="header">
            <h1>RAG PDF Ingestion Pipeline</h1>
            <p>Enterprise-grade PDF to Markdown conversion with intelligent OCR</p>
        </div>
        """)
        
        with gr.Row():
            # Left panel - Controls
            with gr.Column(scale=1):
                gr.Markdown("### Authentication")
                with gr.Group():
                    user_id_input = gr.Textbox(
                        label="User ID",
                        placeholder="Enter your user ID",
                        elem_classes="auth-box"
                    )
                    password_input = gr.Textbox(
                        label="Password",
                        type="password",
                        placeholder="Enter your password",
                        elem_classes="auth-box"
                    )
                
                gr.Markdown("### Upload PDF")
                pdf_upload = gr.File(
                    label="Select PDF File",
                    file_types=[".pdf"],
                    type="filepath"
                )
                
                with gr.Row():
                    submit_btn = gr.Button(
                        "Start Processing",
                        variant="primary",
                        size="lg"
                    )
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
            
            # Right panel - Output
            with gr.Column(scale=2):
                gr.Markdown("### Processing Status")
                status_output = gr.Textbox(
                    label="Status",
                    interactive=False,
                    lines=2
                )
                
                gr.Markdown("### Live Logs")
                log_output = gr.Textbox(
                    label="Pipeline Logs",
                    interactive=False,
                    lines=15,
                    max_lines=20,
                    elem_classes="log-box"
                )
                
                gr.Markdown("### Download Result")
                download_output = gr.File(
                    label="Download Markdown",
                    interactive=False
                )
        
        # Event handlers
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
    """
    Launch the Gradio web interface.
    """
    demo = create_ui()
    
    # Launch on local network
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,  # Set to True to create a public link
        show_error=True
    )

if __name__ == "__main__":
    launch_ui()