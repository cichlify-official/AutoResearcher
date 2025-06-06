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
from summarize_papers import summarize_text
from transformers import AutoTokenizer, AutoModel
import torch
import secrets


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

def verify_credentials(credentials: HTTPBasicCredentials = Depends(security)):
    correct_username = secrets.compare_digest(credentials.username, "admin")
    correct_password = secrets.compare_digest(credentials.password, "password123")
    if not (correct_username and correct_password):
        raise HTTPException(status_code=401, detail="Unauthorized")
    return credentials

# Settings
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
FAISS_INDEX_PATH = "papers.index"
PAPERS_TEXT_PATH = "papers.txt"
DEFAULT_SEARCH_K = 3

# Load embedding model
tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)
model = AutoModel.from_pretrained(EMBEDDING_MODEL)

# Pydantic schema
class SearchRequest(BaseModel):
    query: str

def get_embedding(text: str) -> np.ndarray:
    """Generates embeddings using HuggingFace transformers."""
    try:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
            # Use mean pooling
            embeddings = outputs.last_hidden_state.mean(dim=1)
        return embeddings.numpy().flatten().astype(np.float32)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred while generating embeddings: {str(e)}")

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
        summary = summarize_text(paper["summary"])
        embedding = get_embedding(summary)
        vectors.append(embedding)
        texts.append(f"{paper['title']} - {summary}")

    if vectors:
        dim = len(vectors[0])
        index = faiss.IndexFlatL2(dim)
        index.add(np.array(vectors, dtype=np.float32))
        faiss.write_index(index, FAISS_INDEX_PATH)
        with open(PAPERS_TEXT_PATH, "w") as f:
            f.write("\n".join(texts))
    else:
        return {"message": "No papers found for the given query. Nothing stored."}

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
    
    query_vector = np.array([get_embedding(query)], dtype=np.float32)
    k = min(DEFAULT_SEARCH_K, index.ntotal)
    if k == 0:
        return {"results": []}

    _, indices = index.search(query_vector, k)
    results = [papers_content[i].strip() for i in indices[0] if i < len(papers_content)]
    return {"results": results}

# Uvicorn CLI entry
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)