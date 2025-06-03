from fastapi import FastAPI, Query, HTTPException
import faiss
import numpy as np
import ollama
from fetch_papers import fetch_papers
from summarize_papers import summarize_text

app = FastAPI()
OLLAMA_MODEL = "mistral"
FAISS_INDEX_PATH = "papers.index"
PAPERS_TEXT_PATH = "papers.txt"
DEFAULT_SEARCH_K = 3

def get_embedding(text):
    try:
        response = ollama.embeddings(model=OLLAMA_MODEL, prompt=text)
        return np.array(response["embedding"], dtype=np.float32)
    except Exception as e:
        # Log the error in a real application
        # logger.error(f"Ollama embedding error: {e}")
        raise HTTPException(status_code=503, detail=f"Ollama embedding service error: {str(e)}")


@app.get("/fetch_papers")
def fetch_papers_api(query: str = "machine learning", max_results: int = 3):
    papers = fetch_papers(query, max_results)
    return {"papers": papers}

@app.get("/summarize")
def summarize_paper(text: str):
    try:
        summary = summarize_text(text) # Assumes summarize_text might raise exceptions
        return {"summary": summary}
    except Exception as e:
        # Log the error in a real application
        # logger.error(f"Ollama summarization error: {e}")
        raise HTTPException(status_code=503, detail=f"Ollama summarization service error: {str(e)}")

@app.post("/store_papers")
def store_papers(query: str = "machine learning", max_results: int = 3):
    papers = fetch_papers(query, max_results)
    vectors = []
    texts = []

    for paper in papers:
        summary = summarize_text(paper["summary"])
        embedding = get_embedding(summary)
        vectors.append(embedding)
        texts.append(f"{paper['title']} - {summary}")

    if vectors:
        dimension = len(vectors[0])
        index = faiss.IndexFlatL2(dimension)
        index.add(np.array(vectors, dtype=np.float32))
        faiss.write_index(index, FAISS_INDEX_PATH)

        with open(PAPERS_TEXT_PATH, "w") as f:
            for text in texts:
                f.write(text + "\n")
    elif not papers:
        return {"message": "No papers found for the given query. Nothing stored."}

    return {"message": "Papers stored successfully"}

@app.get("/search")
def search_papers_api(query: str):
    try:
        index = faiss.read_index(FAISS_INDEX_PATH)
        with open(PAPERS_TEXT_PATH, "r") as f:
            papers_content = f.readlines()
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Data files ('{FAISS_INDEX_PATH}', '{PAPERS_TEXT_PATH}') not found. Please run /store_papers first.")
    except Exception as e:
        # Log the error in a real application
        # logger.error(f"Error loading data for search: {e}")
        raise HTTPException(status_code=500, detail=f"Error loading data: {str(e)}")

    query_vector = np.array([get_embedding(query)], dtype=np.float32)
    
    # Ensure k is not greater than the number of items in the index
    k = min(DEFAULT_SEARCH_K, index.ntotal)
    if k == 0:
        return {"results": []} # Or some other appropriate response if index is empty

    _, indices = index.search(query_vector, k)

    results = [papers_content[i].strip() for i in indices[0] if i < len(papers_content)]
    return {"results": results}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
