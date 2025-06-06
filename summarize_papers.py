from transformers import pipeline

# Initialize summarization pipeline
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def summarize_text(text: str, max_length: int = 150, min_length: int = 50) -> str:
    """
    Summarize text using BART model
    """
    try:
        # Truncate text if too long (BART has input limitations)
        if len(text) > 1000:
            text = text[:1000]
        
        summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
        return summary[0]['summary_text']
    except Exception as e:
        # Fallback: return first few sentences if summarization fails
        sentences = text.split('. ')[:3]
        return '. '.join(sentences) + '.' if sentences else text[:200]