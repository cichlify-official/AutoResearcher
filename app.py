from fastapi import FastAPI, Query
import faiss
import numpy as np
import ollama
from fetch_papers import fetch_papers
from summarize_papers import summarize_text

app = FastAPI()

def get_embedding(text):
    response = ollama.embeddings(model="mistral", prompt=text)
    return np.array(response["embedding"], dtype=np.float32)

@app.get("/fetch_papers")
def fetch_papers_api(query: str = "machine learning", max_results: int = 3):
    papers = fetch_papers(query, max_results)
    return {"papers": papers}

@app.get("/summarize")
def summarize_paper(text: str):
    summary = summarize_text(text)
    return {"summary": summary}

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
        faiss.write_index(index, "papers.index")

        with open("papers.txt", "w") as f:
            for text in texts:
                f.write(text + "\n")

    return {"message": "Papers stored successfully"}

@app.get("/search")
def search_papers_api(query: str):
    index = faiss.read_index("papers.index")

    query_vector = np.array([get_embedding(query)], dtype=np.float32)
    _, indices = index.search(query_vector, 3)

    with open("papers.txt", "r") as f:
        papers = f.readlines()

    results = [papers[i].strip() for i in indices[0] if i < len(papers)]
    return {"results": results}
