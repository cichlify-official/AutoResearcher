from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import faiss
import numpy as np
import ollama
from fetch_papers import fetch_papers
from summarize_papers import summarize_text

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your website domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OLLAMA_MODEL = "mistral"
FAISS_INDEX_PATH = "papers.index"
PAPERS_TEXT_PATH = "papers.txt"
DEFAULT_SEARCH_K = 3

# Pydantic model for POST /search
class SearchRequest(BaseModel):
    query: str

def get_embedding(text):
    try:
        response = ollama.embeddings(model=OLLAMA_MODEL, prompt=text)
        return np.array(response["embedding"], dtype=np.float32)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Ollama embedding service error: {str(e)}")

@app.get("/fetch_papers")
def fetch_papers_api(query: str = "machine learning", max_results: int = 3):
    papers = fetch_papers(query, max_results)
    return {"papers": papers}

@app.get("/summarize")
def summarize_paper(text: str):
    try:
        summary = summarize_text(text)
        return {"summary": summary}
    except Exception as e:
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

@app.post("/search")
def search_papers_api(request: SearchRequest):
    query = request.query

    try:
        index = faiss.read_index(FAISS_INDEX_PATH)
        with open(PAPERS_TEXT_PATH, "r") as f:
            papers_content = f.readlines()
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Data files ('{FAISS_INDEX_PATH}', '{PAPERS_TEXT_PATH}') not found. Please run /store_papers first.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading data: {str(e)}")

    query_vector = np.array([get_embedding(query)], dtype=np.float32)

    # Ensure k is not greater than number of items in the index
    k = min(DEFAULT_SEARCH_K, index.ntotal)
    if k == 0:
        return {"results": []}

    _, indices = index.search(query_vector, k)
    results = [papers_content[i].strip() for i in indices[0] if i < len(papers_content)]
    return {"results": results}

# Optional: Uvicorn entry point if run directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
