import os
import faiss
import numpy as np
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel
from dotenv import load_dotenv
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from fastapi.responses import JSONResponse
from fetch_papers import fetch_papers
from summarize_papers import summarize_text # Note: summarize_papers might also need review if it uses local models
import requests # Add requests import
import secrets # For secrets.compare_digest

# Initialize FastAPI
app = FastAPI()

# CORS settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rate limiter
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.exception_handler(RateLimitExceeded)
def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(status_code=429, content={"detail": "Too many requests. Please slow down."})

# Basic Auth
security = HTTPBasic()

# Load .env file for local development
load_dotenv()

# Configuration from environment variables
APP_ADMIN_USERNAME = os.getenv("APP_ADMIN_USERNAME", "admin")
APP_ADMIN_PASSWORD = os.getenv("APP_ADMIN_PASSWORD", "password123")
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_FILE_PATH", "papers.index")
PAPERS_TEXT_PATH = os.getenv("PAPERS_TEXT_FILE_PATH", "papers.txt")

# Hugging Face Inference Endpoint Configuration
HF_INFERENCE_API_KEY = os.getenv("HF_INFERENCE_API_KEY")
# Default endpoint for sentence-transformers/all-MiniLM-L6-v2
HF_EMBEDDING_ENDPOINT_URL = os.getenv("HF_EMBEDDING_ENDPOINT_URL", "https://api-inference.huggingface.co/models/sentence-transformers/all-MiniLM-L6-v2")

def verify_credentials(credentials: HTTPBasicCredentials = Depends(security)):
    correct_username = secrets.compare_digest(credentials.username, APP_ADMIN_USERNAME)
    correct_password = secrets.compare_digest(credentials.password, APP_ADMIN_PASSWORD)
    if not (correct_username and correct_password):
        raise HTTPException(status_code=401, detail="Unauthorized")

# Load embedding model
# Removed local model loading

# Settings
DEFAULT_SEARCH_K = 3

# Pydantic schema
class SearchRequest(BaseModel):
    query: str

def get_embedding(text: str) -> np.ndarray:
    """Generates embeddings using HuggingFace transformers."""
    if not HF_INFERENCE_API_KEY:
         raise HTTPException(status_code=500, detail="Hugging Face Inference API key not configured.")

    headers = {"Authorization": f"Bearer {HF_INFERENCE_API_KEY}"}
    payload = {"inputs": text}

    try:
        response = requests.post(HF_EMBEDDING_ENDPOINT_URL, headers=headers, json=payload)
        response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
        embedding = response.json()
        # The response is typically a list of floats for a single input
        if isinstance(embedding, list) and all(isinstance(x, (int, float)) for x in embedding):
             return np.array(embedding, dtype=np.float32)
        else:
             raise ValueError("Unexpected response format from Inference Endpoint")
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Hugging Face Inference Endpoint request failed: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred during embedding generation: {str(e)}")

@app.get("/fetch_papers")
@limiter.limit("10/minute")
def fetch_papers_api(request: Request, query: str = "machine learning", max_results: int = 3):
    papers = fetch_papers(query, max_results)
    return {"papers": papers}

@app.get("/summarize")
@limiter.limit("10/minute")
def summarize_paper(request: Request, text: str):
    summary = summarize_text(text)
    return {"summary": summary}

@app.post("/store_papers")
@limiter.limit("5/minute")
def store_papers(request: Request, query: str = "machine learning", max_results: int = 3, _: HTTPBasicCredentials = Depends(verify_credentials)):
    papers = fetch_papers(query, max_results)
    vectors, texts = [], []

    for paper in papers:
        try:
            summary = summarize_text(paper["summary"])
            embedding = get_embedding(summary)
            vectors.append(embedding)
            texts.append(f"{paper['title']} - {summary}")
        except HTTPException as e:
             print(f"Skipping paper '{paper.get('title', 'Unknown Title')}' due to embedding error: {e.detail}")

    if vectors:
        dim = len(vectors[0])
        index = faiss.IndexFlatL2(dim)
        index.add(np.array(vectors, dtype=np.float32))
        faiss.write_index(index, FAISS_INDEX_PATH)
        with open(PAPERS_TEXT_PATH, "w") as f:
            f.write("\n".join(texts))
    else:
        if not papers:
             return {"message": "No papers found for the given query. Nothing stored."}
        else:
             raise HTTPException(status_code=500, detail="Failed to generate embeddings for any papers.")

    return {"message": "Papers stored successfully"}

@app.post("/search")
@limiter.limit("10/minute")
def search_papers_api(request: Request, search_request: SearchRequest):
    query = search_request.query
    try:
        index = faiss.read_index(FAISS_INDEX_PATH)
        with open(PAPERS_TEXT_PATH, "r") as f:
            papers_content = f.readlines()
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Data not found. Please run /store_papers first.")
    
    try:
        query_vector = np.array([get_embedding(query)], dtype=np.float32)
    except HTTPException as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate query embedding: {e.detail}")
    k = min(DEFAULT_SEARCH_K, index.ntotal)
    if k == 0:
        return {"results": []}

    _, indices = index.search(query_vector, k)
    results = [papers_content[i].strip() for i in indices[0] if i < len(papers_content)]
    return {"results": results}

# Uvicorn CLI entry
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)