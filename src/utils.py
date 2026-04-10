from typing import List, Tuple, Dict, Generator
import regex as re

def get_tokens_from_text(text: str) -> List[str]:
    text_clean = clean_text(text)
    tokens = tokenize(text_clean)
    return tokens

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return text

def tokenize(text: str) -> List[str]:
    return text.lower().split()

def get_sentiment_label(sentiment: str):
    return (1 if sentiment == "positive" else 0)