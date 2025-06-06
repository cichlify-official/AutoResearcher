import faiss
import numpy as np
import ollama
import sys # For exiting on critical errors

# Configuration constants (mirroring app.py for consistency)
OLLAMA_MODEL = "mistral"
FAISS_INDEX_PATH = "papers.index"
PAPERS_TEXT_PATH = "papers.txt"
DEFAULT_SEARCH_K = 3

def get_embedding(text):
    """Generates embeddings using Ollama with error handling."""
    try:
        response = ollama.embeddings(model=OLLAMA_MODEL, prompt=text)
        if "embedding" not in response:
            print("Error: Embedding service returned an unexpected response format.")
            sys.exit(1) # Exit if embedding fails critically for a script
        return np.array(response["embedding"], dtype=np.float32)
    except ollama.ResponseError as e:
        print(f"Error: Ollama embedding service error: {e.error}")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred while generating embeddings: {str(e)}")
        sys.exit(1)

def search_papers(query):
    try:
        index = faiss.read_index(FAISS_INDEX_PATH)
        with open(PAPERS_TEXT_PATH, "r") as f:
            papers_content = f.readlines()
    except FileNotFoundError:
        print(f"Error: Index file '{FAISS_INDEX_PATH}' or papers file '{PAPERS_TEXT_PATH}' not found. Please run store_papers.py first.")
        return []

    query_vector = np.array([get_embedding(query)], dtype=np.float32)
    k = min(DEFAULT_SEARCH_K, index.ntotal)
    if k == 0: # No items in index
        return []

    _, indices = index.search(query_vector, k)
    results = [papers_content[i].strip() for i in indices[0] if i < len(papers_content)]
    return results

if __name__ == "__main__":
    query = input("Enter your research topic: ")
    results = search_papers(query)

    print("\nTop Matching Papers:\n")
    for res in results:
        print(res)
