import faiss
import numpy as np
import ollama
from fetch_papers import fetch_papers
from summarize_papers import summarize_text

def get_embedding(text):
    response = ollama.embeddings(model="mistral", prompt=text)
    return np.array(response["embedding"], dtype=np.float32)

def store_papers(query="machine learning", max_results=3):
    papers = fetch_papers(query, max_results)
    vectors = []
    texts = []

    for paper in papers:
        summary = summarize_text(paper["summary"])
        embedding = get_embedding(summary)
        vectors.append(embedding)
        texts.append(f"{paper['title']} - {summary}")

    dimension = len(vectors[0])
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(vectors, dtype=np.float32))

    faiss.write_index(index, "papers.index")

    with open("papers.txt", "w") as f:
        for text in texts:
            f.write(text + "\n")

if __name__ == "__main__":
    store_papers()