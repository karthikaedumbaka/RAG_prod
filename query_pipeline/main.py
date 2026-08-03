import sys
from pathlib import Path
from langchain_core.messages import HumanMessage

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from query_pipeline.config import QueryConfig
from query_pipeline.graph import build_rag_graph, GraphState

def run_chat_interface():
    """Runs the interactive CLI chat loop powered by LangGraph & Hybrid Search."""
    print("=" * 60)
    print(" HYBRID RAG CHAT INTERFACE INITIALIZING...")
    print("=" * 60)
    
    config = QueryConfig()
    
    if not config.PINECONE_API_KEY or not config.LLM_API_KEY:
        print(" Error: Missing API keys in .env file.")
        return

    # Initialize the graph globally for this session
    # Note: This will load the BM25 index into memory once at startup
    print("️ Building Hybrid Retriever (Loading BM25 & Pinecone)...")
    rag_graph = build_rag_graph(config)
    print(" Retriever ready!")
    
    # Initialize the graph state with an empty message history
    state: GraphState = {
        "messages": [],
        "question": "",
        "context": [],
        "generation": "",
        "sources": []
    }
    
    print("\n Ready to chat! (Type 'quit', 'exit', or 'clear' to stop/clear history)")
    print("=" * 60)

    while True:
        try:
            user_query = input("\n You: ").strip()
            
            if user_query.lower() in ['quit', 'exit', 'q']:
                print(" Goodbye!")
                break
                
            if user_query.lower() == 'clear':
                state["messages"] = []
                print(" Chat history cleared.")
                continue
                
            if not user_query:
                continue

            # Update state with the new question and user message
            state["question"] = user_query
            state["messages"].append(HumanMessage(content=user_query))

            #  INVOKE THE LANGGRAPH
            final_state = rag_graph.invoke(state)
            
            # Update state with the graph's output
            state = final_state

            # Print the AI's response
            print("\n" + "=" * 60)
            print(" AI Assistant:")
            print("=" * 60)
            print(state["generation"])
            
            # Print the extracted sources beautifully
            if state["sources"]:
                print("\n Sources Referenced:")
                for src in state["sources"]:
                    print(f"   • {src['source']} (Page {src['page']})")
            print("=" * 60)
            
        except KeyboardInterrupt:
            print("\n\n Exiting chat.")
            break
        except Exception as e:
            print(f"\n An unexpected error occurred: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    run_chat_interface()