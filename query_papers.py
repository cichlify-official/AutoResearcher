import faiss
import numpy as np
import ollama

def get_embedding(text):
    response = ollama.embeddings(model="mistral", prompt=text)
    return np.array(response["embedding"], dtype=np.float32)

def search_papers(query):
    index = faiss.read_index("papers.index")

    query_vector = np.array([get_embedding(query)], dtype=np.float32)

    _, indices = index.search(query_vector, 3)

    with open("papers.txt", "r") as f:
        papers = f.readlines()

    results = [papers[i].strip() for i in indices[0] if i < len(papers)]
    return results

if __name__ == "__main__":
    query = input("Enter your research topic: ")
    results = search_papers(query)

    print("\nTop Matching Papers:\n")
    for res in results:
        print(res)
