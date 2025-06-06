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
import faiss
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

HF_INFERENCE_API_KEY = os.getenv("HF_INFERENCE_API_KEY")
HF_EMBEDDING_ENDPOINT_URL = os.getenv("HF_EMBEDDING_ENDPOINT_URL", "https://api-inference.huggingface.co/models/sentence-transformers/all-MiniLM-L6-v2")
HF_SUMMARIZATION_ENDPOINT_URL = os.getenv("HF_SUMMARIZATION_ENDPOINT_URL", "https://api-inference.huggingface.co/models/facebook/bart-large-cnn")

FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "papers.index")
PAPERS_CONTENT_PATH = os.getenv("PAPERS_CONTENT_PATH", "papers_content.json")

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

def get_embedding_from_hf_api(text: str) -> np.ndarray:
    if not HF_INFERENCE_API_KEY:
        raise HTTPException(status_code=500, detail="Hugging Face Inference API key not configured.")
    if not HF_EMBEDDING_ENDPOINT_URL:
        raise HTTPException(status_code=500, detail="Hugging Face Embedding Endpoint URL not configured.")

    headers = {"Authorization": f"Bearer {HF_INFERENCE_API_KEY}"}
    payload = {"inputs": text, "options": {"wait_for_model": True}}

    try:
        response = requests.post(HF_EMBEDDING_ENDPOINT_URL, headers=headers, json=payload)
        response.raise_for_status()
        embedding_data = response.json()
        if isinstance(embedding_data, list) and embedding_data and isinstance(embedding_data[0], list) and isinstance(embedding_data[0][0], float):
            return np.array(embedding_data[0], dtype=np.float32)
        elif isinstance(embedding_data, list) and embedding_data and isinstance(embedding_data[0], float): # If API returns flat list for single input
             return np.array(embedding_data, dtype=np.float32)
        else:
            print(f"Unexpected embedding format: {embedding_data}")
            raise ValueError("Unexpected response format from Embedding Inference Endpoint")
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=503, detail=f"Embedding Inference Endpoint request failed: {str(e)}")
    except (ValueError, TypeError, IndexError) as e: # Added IndexError
        raise HTTPException(status_code=500, detail=f"Error processing embedding response: {str(e)}")

def get_summary_from_hf_api(text: str, max_length: int = 150, min_length: int = 30) -> str:
    if not HF_INFERENCE_API_KEY:
        raise HTTPException(status_code=500, detail="Hugging Face Inference API key not configured.")
    if not HF_SUMMARIZATION_ENDPOINT_URL:
        raise HTTPException(status_code=500, detail="Hugging Face Summarization Endpoint URL not configured.")

    headers = {"Authorization": f"Bearer {HF_INFERENCE_API_KEY}"}
    payload = {
        "inputs": [text[:1024]], # Truncate input for models like BART
        "parameters": {"max_length": max_length, "min_length": min_length, "do_sample": False},
        "options": {"wait_for_model": True}
    }
    try:
        response = requests.post(HF_SUMMARIZATION_ENDPOINT_URL, headers=headers, json=payload)
        response.raise_for_status()
        summary_data = response.json()
        if isinstance(summary_data, list) and summary_data and "summary_text" in summary_data[0]:
            return summary_data[0]["summary_text"]
        else:
            print(f"Unexpected summary format: {summary_data}. Falling back.")
            return text.split('. ')[0] + '.' if '. ' in text else text[:200] # Fallback
    except requests.exceptions.RequestException as e:
        print(f"Summarization Inference Endpoint request failed: {str(e)}. Falling back.")
        return text.split('. ')[0] + '.' if '. ' in text else text[:200] # Fallback
    except (ValueError, TypeError, IndexError) as e:
        print(f"Error processing summary response: {str(e)}. Falling back.")
        return text.split('. ')[0] + '.' if '. ' in text else text[:200] # Fallback

@app.get("/fetch_papers")
@limiter.limit("10/minute")
def fetch_papers_api(request: Request, query: str = "machine learning", max_results: int = 3):
    papers_fetched = fetch_papers(query, max_results)
    return {"papers": papers_fetched}

@app.get("/summarize")
@limiter.limit("10/minute")
def summarize_paper_api(request: Request, text: str):
    summary = get_summary_from_hf_api(text)
    return {"summary": summary}

@app.post("/store_papers")
@limiter.limit("5/minute")
def store_papers(request: Request, query: str = "machine learning", max_results: int = 3, _: HTTPBasicCredentials = Depends(verify_credentials)):
    papers_fetched = fetch_papers(query, max_results)
    
    embeddings_list = []
    content_for_store = []

    if not papers_fetched:
        return {"message": "No papers found for the given query. Nothing stored."}

    for i, paper_data in enumerate(papers_fetched):
        try:
            # You could choose to summarize the fetched summary again, or use it directly
            # For this example, let's use the fetched ArXiv summary for embedding
            text_to_embed_and_store = f"{paper_data['title']}. {paper_data['summary']}"
            
            embedding = get_embedding_from_hf_api(text_to_embed_and_store)
            embeddings_list.append(embedding)
            
            content_for_store.append({
                "id": i, 
                "title": paper_data["title"],
                "text_stored": text_to_embed_and_store,
                "original_arxiv_summary": paper_data["summary"],
                "authors": paper_data.get("authors", []),
                "pdf_url": paper_data.get("pdf_url", "")
            })
        except HTTPException as e:
            print(f"Skipping paper '{paper_data.get('title', 'Unknown Title')}' due to error: {e.detail}")
            continue

    if not embeddings_list:
        # This case means all papers failed embedding, or no papers were fetched initially.
        # The initial check for `not papers_fetched` handles the "no papers" case.
        # So this means all fetched papers failed processing.
        raise HTTPException(status_code=500, detail="Failed to generate embeddings for any fetched papers.")

    embeddings_np = np.array(embeddings_list).astype(np.float32)
    if embeddings_np.ndim == 1 and embeddings_np.size > 0: # Handle case of single embedding
        embeddings_np = embeddings_np.reshape(1, -1)
    
    if embeddings_np.shape[0] == 0:
        return {"message": "No papers could be processed for embeddings. Nothing stored."}

    dimension = embeddings_np.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings_np)
    
    faiss.write_index(index, FAISS_INDEX_PATH)
    with open(PAPERS_CONTENT_PATH, "w") as f:
        json.dump(content_for_store, f, indent=2)

    return {"message": f"Stored {index.ntotal} papers successfully with embeddings."}

@app.post("/search")
@limiter.limit("10/minute")
def search_papers_api(request: Request, search_request: SearchRequest):
    query = search_request.query
    
    try:
        index = faiss.read_index(FAISS_INDEX_PATH)
        with open(PAPERS_CONTENT_PATH, "r") as f:
            papers_content_list = json.load(f)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Data not found ({FAISS_INDEX_PATH} or {PAPERS_CONTENT_PATH}). Please run /store_papers first.")
    
    if index.ntotal == 0:
        return {"results": []}

    try:
        query_vector = get_embedding_from_hf_api(query)
    except HTTPException as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate query embedding: {e.detail}")

    query_vector_np = np.array([query_vector]).astype(np.float32)
    
    k = min(DEFAULT_SEARCH_K, index.ntotal)
    distances, indices = index.search(query_vector_np, k)
    
    results = []
    for i in range(len(indices[0])):
        idx = indices[0][i]
        if idx != -1 and idx < len(papers_content_list):
            paper_detail = papers_content_list[idx]
            # L2 distance to similarity: 1 / (1 + distance). Smaller distance = higher similarity.
            similarity = 1 / (1 + float(distances[0][i])) if distances[0][i] >= 0 else 0 
            results.append({
                "title": paper_detail["title"],
                "match_content": paper_detail["text_stored"],
                "original_arxiv_summary": paper_detail["original_arxiv_summary"],
                "authors": paper_detail.get("authors", []),
                "pdf_url": paper_detail.get("pdf_url", ""),
                "similarity_score": round(similarity, 4)
            })
            
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
    