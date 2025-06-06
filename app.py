import os
import requests
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
import secrets
import json

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
PAPERS_DATA_PATH = os.getenv("PAPERS_DATA_PATH", "papers_data.json")

def verify_credentials(credentials: HTTPBasicCredentials = Depends(security)):
    correct_username = secrets.compare_digest(credentials.username, APP_ADMIN_USERNAME)
    correct_password = secrets.compare_digest(credentials.password, APP_ADMIN_PASSWORD)
    if not (correct_username and correct_password):
        raise HTTPException(status_code=401, detail="Unauthorized")
    return credentials

# Settings
DEFAULT_SEARCH_K = 3

# Pydantic schema
class SearchRequest(BaseModel):
    query: str

def summarize_text_simple(text: str, max_sentences: int = 3) -> str:
    """Simple text summarization by taking first few sentences"""
    sentences = text.split('. ')
    summary_sentences = sentences[:max_sentences]
    return '. '.join(summary_sentences) + '.' if summary_sentences else text[:200]

def simple_text_similarity(query: str, text: str) -> float:
    """Simple text similarity using word overlap"""
    query_words = set(query.lower().split())
    text_words = set(text.lower().split())
    
    if not query_words or not text_words:
        return 0.0
    
    intersection = query_words.intersection(text_words)
    union = query_words.union(text_words)
    
    return len(intersection) / len(union) if union else 0.0

@app.get("/fetch_papers")
@limiter.limit("10/minute")
def fetch_papers_api(request: Request, query: str = "machine learning", max_results: int = 3):
    papers = fetch_papers(query, max_results)
    return {"papers": papers}

@app.get("/summarize")
@limiter.limit("10/minute")
def summarize_paper(request: Request, text: str):
    summary = summarize_text_simple(text)
    return {"summary": summary}

@app.post("/store_papers")
@limiter.limit("5/minute")
def store_papers(request: Request, query: str = "machine learning", max_results: int = 3, _: HTTPBasicCredentials = Depends(verify_credentials)):
    papers = fetch_papers(query, max_results)
    papers_data = []

    for paper in papers:
        summary = summarize_text_simple(paper["summary"])
        papers_data.append({
            "title": paper["title"],
            "summary": summary,
            "full_text": f"{paper['title']} - {summary}",
            "original_summary": paper["summary"]
        })

    if papers_data:
        with open(PAPERS_DATA_PATH, "w") as f:
            json.dump(papers_data, f, indent=2)
    else:
        return {"message": "No papers found for the given query. Nothing stored."}

    return {"message": f"Stored {len(papers_data)} papers successfully"}

@app.post("/search")
@limiter.limit("10/minute")
def search_papers_api(request: Request, search_request: SearchRequest):
    query = search_request.query
    
    try:
        with open(PAPERS_DATA_PATH, "r") as f:
            papers_data = json.load(f)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Data not found. Please run /store_papers first.")
    
    # Calculate similarity scores
    scored_papers = []
    for paper in papers_data:
        score = simple_text_similarity(query, paper["full_text"])
        scored_papers.append((score, paper["full_text"]))
    
    # Sort by score and get top results
    scored_papers.sort(key=lambda x: x[0], reverse=True)
    top_results = scored_papers[:DEFAULT_SEARCH_K]
    
    results = [paper[1] for paper in top_results if paper[0] > 0]
    return {"results": results}

@app.get("/")
def root():
    return {"message": "Paper Search API - Lightweight Version"}

@app.get("/health")
def health_check():
    return {"status": "healthy"}

# Uvicorn CLI entry
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000)) # Use PORT from env, default to 8000 for local
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False) # reload=False for production
    