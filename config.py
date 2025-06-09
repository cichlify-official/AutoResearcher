import os

# API Endpoints
EMBEDDING_MODEL_API_URL = os.getenv("EMBEDDING_MODEL_API_URL", "http://127.0.0.1:5000/embedding")
SUMMARIZATION_MODEL_API_URL = os.getenv("SUMMARIZATION_MODEL_API_URL", "http://127.0.0.1:5000/summarize")

# Admin Credentials
APP_ADMIN_USERNAME = os.getenv("APP_ADMIN_USERNAME")
APP_ADMIN_PASSWORD = os.getenv("APP_ADMIN_PASSWORD")

# File Paths
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "paper_index.faiss")
PAPERS_CONTENT_PATH = os.getenv("PAPERS_CONTENT_PATH", "papers.json")

# Model/FAISS Constants
DIMENSION = int(os.getenv("FAISS_DIMENSION", 384)) # Ensure DIMENSION is an int

# Server Port
PORT = int(os.getenv("PORT", 8000))

# HuggingFace API Key (if needed directly by config, otherwise app can get it)
HF_INFERENCE_API_KEY = os.getenv("HF_INFERENCE_API_KEY")