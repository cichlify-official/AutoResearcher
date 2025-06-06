import requests
import xml.etree.ElementTree as ET

ARXIV_API_URL = "http://export.arxiv.org/api/query"

def fetch_papers(query, max_results=5):
    params = {
        "search_query": query,
        "start": 0,
        "max_results": max_results
    }
    response = requests.get(ARXIV_API_URL, params=params)
    
    if response.status_code == 200:
        root = ET.fromstring(response.text)
        papers = []
        
        for entry in root.findall("{http://www.w3.org/2005/Atom}entry"):
            title = entry.find("{http://www.w3.org/2005/Atom}title").text
            summary = entry.find("{http://www.w3.org/2005/Atom}summary").text
            link = entry.find("{http://www.w3.org/2005/Atom}id").text
            papers.append({"title": title, "summary": summary, "link": link})
        
        return papers
    else:
        return []

if __name__ == "__main__":
    query = "Data science"
    papers = fetch_papers(query)
    for i, paper in enumerate(papers, 1):
        print(f"{i}. {paper['title']}\n{paper['summary']}\n{paper['link']}\n")