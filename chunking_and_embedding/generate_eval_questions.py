"""
Automatically generate evaluation questions for RAG using Kimi or Gemini API.
Usage:
uv run chunking_and_embedding\generate_eval_questions.py
"""
import json
import os
from pathlib import Path
from dotenv import load_dotenv

try:
    from openai import OpenAI
except ImportError:
    print("ERROR: Please install the OpenAI package: uv pip install openai")
    raise

# Load environment variables
PROJECT_ROOT = Path(__file__).parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# Configuration
INPUT_DIR = PROJECT_ROOT / "output"  # Where your merged markdown files are
OUTPUT_JSON = Path(__file__).parent / "EvalQuestions.json"

# API Configuration (Supports both Kimi and Gemini via OpenAI-compatible endpoints)
KIMI_KEY = os.getenv("KIMI_API_KEY")
GEMINI_KEY = os.getenv("GEMINI_API_KEY")

if KIMI_KEY:
    API_KEY = KIMI_KEY
    BASE_URL = "https://api.moonshot.cn/v1"
    MODEL = "moonshot-v1-8k"
elif GEMINI_KEY:
    API_KEY = GEMINI_KEY
    BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
    MODEL = "gemini-2.0-flash-lite"
else:
    print("ERROR: Neither KIMI_API_KEY nor GEMINI_API_KEY found in .env file.")
    raise SystemExit(1)

# Initialize client
client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
)

def generate_questions_with_kimi(input_dir: str, output_path: Path, num_questions_per_file: int = 3):
    """
    Reads markdown files, asks the LLM to generate Q&A pairs, and saves to JSON.
    """
    md_files = list(Path(input_dir).glob("*.md"))
    if not md_files:
        raise FileNotFoundError(f"❌ No .md files found in {input_dir}")
        
    print(f"📄 Found {len(md_files)} markdown file(s). Generating questions via LLM...")
    all_questions = []
    
    for md_file in md_files:
        with open(md_file, "r", encoding="utf-8") as f:
            # Read first 4000 chars to stay within cheap/fast token limits
            text = f.read()[:4000] 
            
        prompt = f"""You are an expert in creating evaluation datasets for RAG systems.
Based on the following text, generate exactly {num_questions_per_file} specific, factual questions.
Output STRICTLY as a valid JSON array of objects with keys: "question", "expected_source", "expected_content_snippet".
Do not include markdown formatting (like ```json) or explanations outside the JSON array.

Text snippet:
---
{text}
---
"""
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
            )
            content = response.choices[0].message.content
            # Clean up potential markdown wrappers
            content = content.replace("```json", "").replace("```", "").strip()
            data = json.loads(content)
            
            if isinstance(data, dict) and "questions" in data:
                all_questions.extend(data["questions"])
            elif isinstance(data, list):
                all_questions.extend(data)
            print(f"   ✅ {md_file.name}: Generated {num_questions_per_file} questions.")
        except Exception as e:
            print(f"   ⚠️ Error generating questions for {md_file.name}: {e}")
            
    # Save to disk
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_questions, f, indent=2, ensure_ascii=False)
        
    print(f"\n🎉 Successfully generated {len(all_questions)} questions and saved to {output_path.name}")
    return all_questions

def main():
    try:
        generate_questions_with_kimi(str(INPUT_DIR), OUTPUT_JSON)
    except Exception as e:
        print(f"Failed: {e}")

if __name__ == "__main__":
    main()