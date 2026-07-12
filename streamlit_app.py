import streamlit as st
import sys
import time
import shutil
from pathlib import Path
import logging
import os

# Top 1% Move: Suppress Hugging Face Transformers deprecation spam
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
logging.getLogger("transformers").setLevel(logging.ERROR)

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT / "pdf_ingestion"))

try:
    from config import PipelineConfig
    from auth import authenticate_or_register, init_db
    from analyzer import analyze_pdf
    from batcher import create_batches
    from extractor import extract_batches
    from merger import merge_outputs
    from cleaner import cleanup_artifacts
    from utils import ensure_dir
except ImportError as e:
    st.error(f"Import Error: {e}. Make sure you are running this from the project root.")
    st.stop()

st.set_page_config(
    page_title="RAG Ingestion",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Minimal, safe CSS for dark mode enhancement
st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
        color: #fafafa;
    }
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        background: linear-gradient(90deg, #667eea, #764ba2);
        color: white;
        font-weight: bold;
        border: none;
    }
    .stButton>button:hover {
        opacity: 0.9;
    }
    .stMetric {
        background-color: #1e2127;
        border: 1px solid #262730;
        border-radius: 10px;
        padding: 15px;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if 'user_id' not in st.session_state:
        st.session_state.user_id = None
    if 'processing_complete' not in st.session_state:
        st.session_state.processing_complete = False
    if 'output_file' not in st.session_state:
        st.session_state.output_file = None
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None
    if 'processing_time' not in st.session_state:
        st.session_state.processing_time = None

def show_login():
    st.title("📚 RAG Ingestion Pipeline")
    st.markdown("#### Enterprise Document Processing")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("---")
        with st.form("login_form"):
            st.subheader("🔐 Authentication")
            user_id = st.text_input("User ID", placeholder="Enter your username")
            password = st.text_input("Password", type="password", placeholder="Enter your password")
            submitted = st.form_submit_button("Sign In / Register", use_container_width=True)
            
            if submitted:
                if not user_id or not password:
                    st.error("Please enter both User ID and Password.")
                else:
                    with st.spinner("Authenticating..."):
                        status = authenticate_or_register(user_id, password)
                        if status in ["authenticated", "registered"]:
                            st.session_state.authenticated = True
                            st.session_state.user_id = user_id
                            st.success(f"Welcome, {user_id}!")
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.error("Authentication failed.")
        st.markdown("---")

def show_main():
    st.title(f"📚 RAG Ingestion | {st.session_state.user_id}")
    
    with st.sidebar:
        st.header("⚙️ System")
        st.info("Docling + PyMuPDF\nRapidOCR (ONNX)\nGPU Auto-Detect")
        if st.button("🚪 Sign Out", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("📄 Upload Document")
        uploaded_file = st.file_uploader("Drop PDF here", type=['pdf'])
        
        if uploaded_file:
            st.success(f"**{uploaded_file.name}**\n({uploaded_file.size/1024/1024:.2f} MB)")
            
            if st.button("🚀 Start Processing", use_container_width=True, type="primary"):
                process_document(uploaded_file)
        else:
            st.info("Please upload a PDF to begin.")

    with col2:
        st.subheader("📊 Status & Results")
        
        if st.session_state.processing_complete:
            show_results()
        else:
            st.markdown("### Ready to Process")
            st.write("Upload a document and click **Start Processing** to begin the intelligent extraction pipeline.")

def process_document(uploaded_file):
    st.session_state.processing_complete = False
    st.session_state.output_file = None
    start_time = time.time()
    
    temp_dir = PROJECT_ROOT / "temp_upload"
    output_dir = PROJECT_ROOT / "output"
    temp_batches = PROJECT_ROOT / "temp_batches"
    
    ensure_dir(str(temp_dir))
    ensure_dir(str(output_dir))
    ensure_dir(str(temp_batches))
    
    temp_pdf = temp_dir / uploaded_file.name
    with open(temp_pdf, "wb") as f:
        f.write(uploaded_file.getbuffer())
        
    config = PipelineConfig()
    config.user_id = st.session_state.user_id
    config.output_dir = str(output_dir)
    config.temp_dir = str(temp_batches)
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        status_text.text("🔍 Analyzing...")
        progress_bar.progress(10)
        analysis = analyze_pdf(str(temp_pdf), config)
        st.session_state.analysis_results = analysis
        
        status_text.text("📦 Batching...")
        progress_bar.progress(30)
        batches = create_batches(str(temp_pdf), analysis, config)
        
        status_text.text("⚡ Extracting (This may take a few minutes)...")
        progress_bar.progress(50)
        extract_batches(batches, config)
        
        status_text.text("🔗 Merging...")
        progress_bar.progress(85)
        pdf_name = Path(uploaded_file.name).stem
        final_md = merge_outputs(pdf_name, batches, config)
        
        status_text.text("🧹 Cleaning...")
        progress_bar.progress(95)
        cleanup_artifacts(config)
        if temp_pdf.exists(): temp_pdf.unlink()
        
        elapsed = time.time() - start_time
        st.session_state.processing_time = elapsed
        st.session_state.processing_complete = True
        st.session_state.output_file = str(final_md)
        
        progress_bar.progress(100)
        status_text.text("✅ Complete!")
        time.sleep(1)
        st.rerun()
        
    except Exception as e:
        st.error(f"Processing failed: {e}")
        import traceback
        st.code(traceback.format_exc())

def show_results():
    analysis = st.session_state.analysis_results
    output_file = st.session_state.output_file
    processing_time = st.session_state.processing_time
    
    st.success("✅ Processing Complete!")
    
    if analysis:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Pages", analysis['total_pages'])
        c2.metric("Text Pages", analysis['text_count'])
        c3.metric("OCR Pages", analysis['ocr_count'])
        c4.metric("Time", f"{processing_time:.1f}s")
        
    if output_file and Path(output_file).exists():
        with open(output_file, 'rb') as f:
            st.download_button("📥 Download Markdown", f, file_name=Path(output_file).name, mime="text/markdown", use_container_width=True)
            
        with st.expander("👁️ Preview"):
            with open(output_file, 'r', encoding='utf-8') as f:
                st.code(f.read(2000), language='markdown')

def main():
    initialize_session_state()
    try:
        init_db()
    except Exception as e:
        st.error(f"Database Error: {e}")
        st.stop()
        
    if not st.session_state.authenticated:
        show_login()
    else:
        show_main()

if __name__ == "__main__":
    main()