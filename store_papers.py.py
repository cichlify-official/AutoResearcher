import numpy as np
import faiss
import ollama
import sys
from fetch_papers import fetch_papers
from summarize_papers import summarize_text

# Configuration constants
OLLAMA_MODEL = "mistral"
FAISS_INDEX_PATH = "papers.index"
PAPERS_TEXT_PATH = "papers.txt"

def get_embedding(text):
    """Generates embeddings using Ollama with error handling."""
    try:
        response = ollama.embeddings(model=OLLAMA_MODEL, prompt=text)
        if "embedding" not in response:
            print("Error: Embedding service returned an unexpected response format.")
            sys.exit(1)
        return np.array(response["embedding"], dtype=np.float32)
    except ollama.ResponseError as e:
        print(f"Error: Ollama embedding service error: {e.error}")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred while generating embeddings: {str(e)}")
        sys.exit(1)

def store_papers(query="machine learning", max_results=3):
    papers = fetch_papers(query, max_results)
    vectors = []
    texts = []

    for paper in papers:
        try:
            summary = summarize_text(paper["summary"])
            embedding = get_embedding(summary)
            vectors.append(embedding)
            texts.append(f"{paper['title']} - {summary}")
        except Exception as e:
            print(f"Skipping paper '{paper.get('title', 'Unknown Title')}' due to error: {e}")
            continue

    if not vectors:
        print("No papers processed or no embeddings generated. Nothing to store.")
        return

    dimension = len(vectors[0])
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(vectors, dtype=np.float32))

    faiss.write_index(index, FAISS_INDEX_PATH)
    print(f"FAISS index saved to {FAISS_INDEX_PATH}")

    with open(PAPERS_TEXT_PATH, "w") as f:
        for text in texts:
            f.write(text + "\n")
    print(f"Paper texts saved to {PAPERS_TEXT_PATH}")

if __name__ == "__main__":
    store_papers()
