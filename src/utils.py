from typing import List, Tuple, Dict, Generator
import regex as re
import ast
from collections import Counter
from typing import Dict, List, Tuple

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
    return 1 if sentiment == "positive" else 0

def parse_serialized_list(value: str) -> List[str]:
    if value is None:
        return []

    value = value.strip()
    if not value:
        return []

    try:
        parsed = ast.literal_eval(value)
        if isinstance(parsed, list):
            return [str(x) for x in parsed]
    except Exception:
        pass

    value = value.strip("[]")
    if not value:
        return []

    return [item.strip().strip("'").strip('"') for item in value.split(",")]

def create_label_lookup_tables(label_sequences: List[List[str]]) -> Tuple[Dict[str, int], Dict[int, str]]:
    special_tokens = ["<PAD>"]
    label_counts = Counter(label for seq in label_sequences for label in seq)
    labels = special_tokens + sorted(label_counts.keys())

    label2idx = {label: i for i, label in enumerate(labels)}
    idx2label = {i: label for label, i in label2idx.items()}

    return label2idx, idx2label