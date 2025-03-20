import ollama

def summarize_text(text):
    """Summarizes the given text using Mistral via Ollama."""
    response = ollama.chat(model='mistral', messages=[
        {'role': 'system', 'content': 'You are an AI assistant that provides concise and informative summaries.'},
        {'role': 'user', 'content': f"Summarize this text: {text}"}
    ])
    return response['message']['content']

if __name__ == "__main__":
    paper = {
        "title": "Lecture Notes: Optimization for Machine Learning",
        "summary": "Lecture notes on optimization for machine learning, derived from a course at Princeton University and tutorials given in MLSS, Buenos Aires, as well as Simons Foundation, Berkeley."
    }

    print(f"Title: {paper['title']}")
    print(f"Original Summary: {paper['summary']}")
    
    try:
        ai_summary = summarize_text(paper['summary'])
        print(f"AI Summary: {ai_summary}")
    except Exception as e:
        print(f"Error generating summary: {e}")
