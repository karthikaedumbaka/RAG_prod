import json
import sys
import re
from pathlib import Path
from typing import List, Dict

#  UPDATED PATH LOGIC: Go up two levels (evaluation -> git_rag)
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from query_pipeline.config import QueryConfig
from query_pipeline.graph import get_hybrid_retriever, get_llm
from langchain_core.documents import Document

# Path to the ground-truth dataset (relative to the actual project root)
EVAL_QUESTIONS_PATH = PROJECT_ROOT / "chunking_and_embedding" / "EvalQuestions.json"

def load_eval_questions() -> List[Dict]:
    """Load the ground-truth Q&A dataset."""
    if not EVAL_QUESTIONS_PATH.exists():
        print(f" Error: {EVAL_QUESTIONS_PATH} not found.")
        sys.exit(1)
    with open(EVAL_QUESTIONS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def evaluate_retrieval(retrieved_docs: List[Document], expected_source: str, expected_snippet: str) -> Dict:
    """Evaluate if the retriever found the correct source and content."""
    hit_source = False
    hit_snippet = False
    rank_of_hit = None

    for rank, doc in enumerate(retrieved_docs, start=1):
        source = doc.metadata.get("source", "")
        content = doc.page_content.lower()
        
        if expected_source.lower() in source.lower():
            hit_source = True
            if expected_snippet.lower() in content:
                hit_snippet = True
                rank_of_hit = rank
                break # Found the best match

    return {
        "hit_source": hit_source,
        "hit_snippet": hit_snippet,
        "rank": rank_of_hit,
        "mrr": 1.0 / rank_of_hit if rank_of_hit else 0.0
    }

def evaluate_generation(question: str, context: str, answer: str, llm) -> Dict:
    """Use LLM-as-a-Judge to grade Faithfulness and Relevance (1-5 scale)."""
    judge_prompt = f"""You are an expert evaluator of RAG systems. Grade the AI's answer based on the provided context and question.

Question: {question}
Retrieved Context: {context[:1000]}... (truncated for brevity)
AI Answer: {answer}

Evaluate on two metrics (score 1 to 5):
1. Faithfulness: Is the answer grounded ONLY in the provided context? (1=Hallucinates, 5=Perfectly grounded)
2. Relevance: Does the answer directly address the user's question? (1=Irrelevant, 5=Perfectly relevant)

Output STRICTLY in this JSON format:
{{
  "faithfulness_score": <1-5>,
  "relevance_score": <1-5>,
  "reasoning": "<brief 1-sentence explanation>"
}}
"""
    try:
        response = llm.invoke(judge_prompt)
        # Extract JSON from response (handles markdown code blocks)
        json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
        return {"faithfulness_score": 0, "relevance_score": 0, "reasoning": "Failed to parse JSON"}
    except Exception as e:
        return {"faithfulness_score": 0, "relevance_score": 0, "reasoning": f"LLM Judge Error: {e}"}

def main():
    print("=" * 70)
    print(" END-TO-END RAG EVALUATION STARTING")
    print("=" * 70)
    
    config = QueryConfig()
    if not config.LLM_API_KEY:
        print(" Error: LLM_API_KEY (GROQ_API_KEY) missing in .env")
        return

    questions = load_eval_questions()
    print(f" Loaded {len(questions)} evaluation questions.\n")

    # Initialize components ONCE
    print("️ Initializing Hybrid Retriever and LLM Judge...")
    retriever = get_hybrid_retriever(config)
    llm = get_llm(config)
    print(" Ready!\n")

    # Metrics tracking
    total_hit_source = 0
    total_hit_snippet = 0
    total_mrr = 0.0
    total_faithfulness = 0.0
    total_relevance = 0.0

    for i, item in enumerate(questions, 1):
        q = item["question"]
        expected_source = item["expected_source"]
        expected_snippet = item["expected_content_snippet"]
        
        print(f"[{i}/{len(questions)}] Q: {q[:60]}...")
        
        # 1. RETRIEVAL
        docs = retriever.invoke(q)
        ret_metrics = evaluate_retrieval(docs, expected_source, expected_snippet)
        
        total_hit_source += 1 if ret_metrics["hit_source"] else 0
        total_hit_snippet += 1 if ret_metrics["hit_snippet"] else 0
        total_mrr += ret_metrics["mrr"]
        
        # 2. GENERATION
        context_str = "\n\n".join([f"[Source: {d.metadata.get('source')}] {d.page_content}" for d in docs[:3]])
        gen_prompt = f"Answer the question based ONLY on this context:\n\nContext:\n{context_str}\n\nQuestion: {q}\n\nAnswer:"
        answer = llm.invoke(gen_prompt).content
        
        # 3. JUDGE
        judge_metrics = evaluate_generation(q, context_str, answer, llm)
        total_faithfulness += judge_metrics.get("faithfulness_score", 0)
        total_relevance += judge_metrics.get("relevance_score", 0)
        
        print(f"   ↳ Retrieval: Source={'' if ret_metrics['hit_source'] else ''} | Snippet={'' if ret_metrics['hit_snippet'] else ''} (Rank: {ret_metrics['rank']})")
        print(f"   ↳ Generation: Faithfulness={judge_metrics.get('faithfulness_score', 'N/A')}/5 | Relevance={judge_metrics.get('relevance_score', 'N/A')}/5")
        print(f"   ↳ Judge Reasoning: {judge_metrics.get('reasoning', 'N/A')}\n")

    # FINAL SCORECARD
    n = len(questions)
    print("=" * 70)
    print(" FINAL EVALUATION SCORECARD")
    print("=" * 70)
    print(f" Total Questions Evaluated: {n}")
    print("-" * 70)
    print(" RETRIEVAL METRICS:")
    print(f"   • Source Hit Rate (Recall@5):  {(total_hit_source / n) * 100:.1f}%")
    print(f"   • Snippet Hit Rate:            {(total_hit_snippet / n) * 100:.1f}%")
    print(f"   • Mean Reciprocal Rank (MRR):  {total_mrr / n:.3f}")
    print("-" * 70)
    print(" GENERATION METRICS (LLM-as-a-Judge):")
    print(f"   • Average Faithfulness:        {total_faithfulness / n:.2f} / 5.0")
    print(f"   • Average Relevance:           {total_relevance / n:.2f} / 5.0")
    print("=" * 70)

if __name__ == "__main__":
    main()