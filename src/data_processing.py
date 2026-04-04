from typing import List, Tuple, Dict, Generator
from src.utils import clean_text, tokenize, get_sentiment_label
import csv
from collections import Counter

def load_and_preprocess_data(infile: str) -> List[Tuple[List[str], int]]:
    processed_data = []

    with open(infile,"r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            review = row["review"]
            sentiment = row["sentiment"]

            review_clean = clean_text(review)
            tokens = tokenize(review_clean)

            label = get_sentiment_label(sentiment)

            processed_data.append((tokens, label))

    return processed_data


def create_lookup_tables(words: List[str]) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Create lookup tables for vocabulary.

    Args:
        words (List[str]): A list of words from which to create vocabulary.

    Returns:
        A tuple containing two dictionaries. The first dictionary maps words to integers (vocab_to_int),
        and the second maps integers to words (int_to_vocab).
    """
    word_counts: Counter = Counter(words)
    sorted_vocab: List[int] = sorted(word_counts, key=word_counts.get, reverse=True)

    int_to_vocab: Dict[int, str] = {ii: word for ii, word in enumerate(sorted_vocab)}
    vocab_to_int: Dict[str, int] = {word: ii for ii, word in int_to_vocab.items()}

    return vocab_to_int, int_to_vocab