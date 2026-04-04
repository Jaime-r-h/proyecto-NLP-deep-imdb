from typing import List, Tuple, Dict, Generator
from src.utils import clean_text, tokenize, get_sentiment_label
import csv
from collections import Counter
import numpy as np

def get_tokens_and_labels(infile: str):
    tokens_list = []
    labels_list = []

    with open(infile,"r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            review = row["review"]
            sentiment = row["sentiment"]

            review_clean = clean_text(review)
            tokens = tokenize(review_clean)

            label = get_sentiment_label(sentiment)

            tokens_list.append(tokens)
            labels_list.append(label)

    return tokens_list, labels_list


def create_lookup_tables(texts: List[List[str]], min_freq=1) -> Tuple[Dict[str, int], Dict[int, str]]:

    special_tokens = ['<PAD>', '<UNK>', '<SOS>', '<EOS>']
    
    word_counts: Counter = Counter(tok for text in texts for tok in text)

    vocab_words = [w for w, f in word_counts.items() if f >= min_freq]

    vocab_complete = special_tokens + vocab_words
    word2idx = {vocab_complete[i]: i for i in range(len(vocab_complete))}
    idx2word = {i: w for w, i in word2idx.items()}
    
    return word2idx, idx2word

def text_to_indices(tokens: List[str], word2idx: Dict[str,int], max_len=None) -> List[int]:
    indices = [word2idx.get(tok, word2idx['<UNK>']) for tok in tokens]

    if max_len:
        indices = indices[:max_len] + [word2idx['<PAD>']] * max(0, max_len - len(indices))

    return indices

def create_embedding_matrix(word2idx: Dict[str,int], embedding_dim=100) -> np.ndarray:
    vocab_size = len(word2idx)
    embedding_matrix = np.random.normal(scale=0.6, size=(vocab_size, embedding_dim)).astype(np.float32)

    embedding_matrix[word2idx['<PAD>']] = np.zeros(embedding_dim, dtype=np.float32)
    return embedding_matrix


def load_processed_data_with_embeddings(infile: str, min_freq=2, max_len=None, embedding_dim=100):
    # Devuelve los textos de input X en forma de lista de índices, y los outpus Y en forma de 0 (negativo) y 1 (positivo)
    # Devuelve los diccionarios de conversión palabra - índice
    # Devuelve la matriz de embeddings, ya sea preentrenada o por entrenar

    tokens_list, labels_list = get_tokens_and_labels(infile)
    word2idx, idx2word = create_lookup_tables(tokens_list, min_freq=min_freq)

    X = [text_to_indices(tokens, word2idx, max_len=max_len) for tokens in tokens_list]
    y = labels_list
    embedding_matrix = create_embedding_matrix(word2idx, embedding_dim=embedding_dim)
    
    return np.array(X), np.array(y), word2idx, idx2word, embedding_matrix