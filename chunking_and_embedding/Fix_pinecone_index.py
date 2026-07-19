"""
Pinecone Index Manager: Inspect, fix, or dynamically delete indexes.
Usage:
uv run chunking_and_embedding\fix_pinecone_index.py
"""
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from pinecone import Pinecone

PROJECT_ROOT = Path(__file__).parent.parent
load_dotenv(PROJECT_ROOT / ".env")

API_KEY = os.getenv("PINECONE_API_KEY")
DEFAULT_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "rag-index-3072")

if not API_KEY:
    print("ERROR: PINECONE_API_KEY not found in .env file.")
    raise SystemExit(1)

pc = Pinecone(api_key=API_KEY)

def get_all_indexes() -> list:
    """Fetch all index names in the current Pinecone project."""
    return pc.list_indexes().names()

def inspect_and_delete_index(index_name: str):
    """Show stats for a specific index and prompt for deletion."""
    available_indexes = get_all_indexes()
    
    if index_name not in available_indexes:
        print(f"\n Index '{index_name}' doesn't exist in this project.")
        return

    desc = pc.describe_index(index_name)
    index = pc.Index(index_name)
    stats = index.describe_index_stats()
    vector_count = stats.get("total_vector_count", 0)
    
    print(f"\n{'='*60}")
    print(f"Index Name: {index_name}")
    print(f"Dimension:  {desc.dimension}")
    print(f"Metric:     {desc.metric}")
    print(f"Vectors:    {vector_count}")
    print(f"{'='*60}")
    
    if vector_count == 0:
        print("STATUS: This index is EMPTY.")
    else:
        print(f"  WARNING: This index contains {vector_count} vectors.")
        print("Deleting it will PERMANENTLY destroy this data.")
        
    confirm = input(f"\nDelete '{index_name}' now? [y/N]: ")
    if confirm.strip().lower() == 'y':
        print(f"  Deleting index '{index_name}'...")
        pc.delete_index(index_name)
        print(f" Successfully deleted '{index_name}'.")
    else:
        print(" Deletion cancelled.")

def list_and_delete_dynamically():
    """List all indexes in the project and let the user pick one to delete."""
    indexes = get_all_indexes()
    
    if not indexes:
        print("\n No indexes found in this Pinecone project.")
        return
        
    print(f"\nFound {len(indexes)} index(es) in your Pinecone project:")
    print("-" * 50)
    for i, name in enumerate(indexes, start=1):
        print(f"  [{i}] {name}")
    print("-" * 50)
    
    while True:
        choice = input("\nEnter the number of the index to delete (or 'q' to go back): ")
        
        if choice.lower() == 'q':
            return
            
        if choice.isdigit() and 1 <= int(choice) <= len(indexes):
            selected_index = indexes[int(choice) - 1]
            inspect_and_delete_index(selected_index)
            break
        else:
            print(f" Invalid input. Please enter a number between 1 and {len(indexes)}, or 'q'.")

def main_menu():
    """Display the main interactive menu."""
    print("=" * 60)
    print("🌲 PINECONE INDEX MANAGER")
    print("=" * 60)
    print(f"Default Index (from .env): {DEFAULT_INDEX_NAME}")
    print("-" * 60)
    print("1. Inspect and optionally delete the DEFAULT index")
    print("2. List ALL indexes and select one to delete dynamically")
    print("3. Exit")
    print("-" * 60)
    
    choice = input("Select an option [1-3]: ")
    
    if choice == '1':
        inspect_and_delete_index(DEFAULT_INDEX_NAME)
    elif choice == '2':
        list_and_delete_dynamically()
    elif choice == '3':
        print("Exiting.")
        sys.exit(0)
    else:
        print("Invalid option. Exiting.")
        sys.exit(1)

if __name__ == "__main__":
    main_menu()