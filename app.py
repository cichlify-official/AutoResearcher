import os
import json
import faiss
import numpy as np
import secrets
from typing import List
from fastapi import FastAPI, Depends, HTTPException, status, Request
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import logging
from fetch_papers import fetch_papers
import requests

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment config
EMBEDDING_MODEL_API_URL = os.getenv("EMBEDDING_MODEL_API_URL", "http://127.0.0.1:5000/embedding")
SUMMARIZATION_MODEL_API_URL = os.getenv("SUMMARIZATION_MODEL_API_URL", "http://127.0.0.1:5000/summarize")
APP_ADMIN_USERNAME = os.getenv("APP_ADMIN_USERNAME")
APP_ADMIN_PASSWORD = os.getenv("APP_ADMIN_PASSWORD")

# Security checks
if not APP_ADMIN_USERNAME or not APP_ADMIN_PASSWORD:
    raise RuntimeError("Admin credentials must be set via environment variables.")

# File paths
FAISS_INDEX_PATH = "paper_index.faiss"
PAPERS_CONTENT_PATH = "papers.json"

# Constants
DIMENSION = 384

# Auth
security = HTTPBasic()
def verify_credentials(credentials: HTTPBasicCredentials = Depends(security)):
    correct_username = secrets.compare_digest(credentials.username, APP_ADMIN_USERNAME)
    correct_password = secrets.compare_digest(credentials.password, APP_ADMIN_PASSWORD)
    if not (correct_username and correct_password):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials", headers={"WWW-Authenticate": "Basic"})
    return credentials

# Rate Limiting
limiter = Limiter(key_func=get_remote_address)
app = FastAPI()
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, lambda r, e: HTTPException(status_code=429, detail="Too many requests"))

# CORS
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Request models
class StoreRequest(BaseModel):
    query: str
    max_results: int

class SearchRequest(BaseModel):
    query: str
    top_k: int

class SummarizeRequest(BaseModel):
    text: str

# Helper functions
def get_embedding_from_hf_api(text):
    response = requests.post(EMBEDDING_MODEL_API_URL, json={"text": text})
    if response.status_code != 200:
        raise HTTPException(status_code=500, detail="Failed to get embedding")
    return response.json()["embedding"]

def get_summary_from_hf_api(text):
    response = requests.post(SUMMARIZATION_MODEL_API_URL, json={"text": text[:2048]})
    if response.status_code != 200:
        logger.warning("Summarization failed; using truncated text.")
        return text[:500]
    return response.json()["summary"]

def load_faiss_index():
    try:
        return faiss.read_index(FAISS_INDEX_PATH)
    except:
        return faiss.IndexFlatL2(DIMENSION)

def save_faiss_index(index):
    faiss.write_index(index, FAISS_INDEX_PATH)

def load_stored_data():
    try:
        with open(PAPERS_CONTENT_PATH, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return []

def save_stored_data(data):
    with open(PAPERS_CONTENT_PATH, "w") as f:
        json.dump(data, f, indent=2)

# Routes
@app.post("/store_papers")
@limiter.limit("10/minute")
def store_papers(req: StoreRequest, _: HTTPBasicCredentials = Depends(verify_credentials)):
    papers_fetched = fetch_papers(req.query, req.max_results)
    if not papers_fetched:
        raise HTTPException(status_code=404, detail="No papers found")

    stored_data = load_stored_data()
    start_id = len(stored_data)
    index = load_faiss_index()

    for i, paper in enumerate(papers_fetched):
        try:
            summary = get_summary_from_hf_api(paper['summary'])
            embedding = get_embedding_from_hf_api(summary)
            paper_data = {
                "id": start_id + i,
                "title": paper["title"],
                "summary": summary,
                "link": paper["link"],
                "published": paper["published"],
                "authors": paper["authors"]
            }
            index.add(np.array([embedding], dtype="float32"))
            stored_data.append(paper_data)
        except Exception as e:
            logger.warning("Skipping paper due to error: %s", e)

    save_faiss_index(index)
    save_stored_data(stored_data)
    return {"message": f"{len(papers_fetched)} papers processed and stored."}

@app.post("/search")
@limiter.limit("20/minute")
def search_papers(req: SearchRequest):
    stored_data = load_stored_data()
    if not stored_data:
        raise HTTPException(status_code=404, detail="No papers stored yet")

    index = load_faiss_index()
    embedding = get_embedding_from_hf_api(req.query)
    _, indices = index.search(np.array([embedding], dtype="float32"), req.top_k)
    result = [stored_data[i] for i in indices[0] if i < len(stored_data)]
    return result

@app.post("/summarize")
def summarize(req: SummarizeRequest):
    return {"summary": get_summary_from_hf_api(req.text)}

@app.get("/papers")
def get_all_papers():
    return load_stored_data()

@app.get("/")
def read_root():
    return {"message": "Welcome to AutoResearcher Backend"}

@app.post("/delete_index")
def delete_index(_: HTTPBasicCredentials = Depends(verify_credentials)):
    try:
        os.remove(FAISS_INDEX_PATH)
        os.remove(PAPERS_CONTENT_PATH)
    except FileNotFoundError:
        pass
    return {"message": "Index and stored data deleted"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)), reload=True)
