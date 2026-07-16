
# Plan: Connect PDF Ingestion and Chunking/Embedding for RAG Project

## Repo Research Conclusion
The RAG project currently has two separate modules:
1. **`pdf_ingestion`**: Takes PDFs from `data/` directory, processes them, and outputs markdown files to `output/` directory.
2. **`chunking_and_embedding`**: Takes markdown files from `output/` directory, chunks them, embeds using Google's API, and stores in Pinecone vector DB.

Both modules are complete and functional, but they need to be connected into a single pipeline.

## Files and Modules to Edit
1. **Create new `combined_pipeline` module** with the following files:
   - `__init__.py`: Module initializer
   - `main.py`: The main combined pipeline that orchestrates both steps

## Steps for Modifications or New Features
1. **Create `combined_pipeline` directory**:
   - Location: `c:\Users\karth\Desktop\rag\rag_project_1\combined_pipeline`

2. **Implement `combined_pipeline/main.py`**:
   - Import necessary functions from `pdf_ingestion`
   - Import necessary functions from `chunking_and_embedding`
   - Implement `run_combined_pipeline()` function that:
     1. Authenticates user
     2. Runs PDF ingestion pipeline
     3. Runs chunking and embedding pipeline

3. **Ensure all imports work correctly** with both relative and absolute imports.

## Potential Dependencies or Considerations
- All dependencies are already listed in `pyproject.toml`, so no new dependencies are needed.
- User needs to have `.env` file with API keys (GOOGLE_API_KEY and PINECONE_API_KEY) set up.
- PDFs must be placed in the `data/` directory.

## Risk Handling
- If PDF ingestion fails for any PDF, the pipeline will skip chunking/embedding if no successful outputs are generated.
- Fallback to running individual pipelines is still available.
