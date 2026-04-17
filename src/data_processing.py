from typing import List, Tuple, Dict, Generator
import csv
from collections import Counter
import numpy as np
from gensim.models import Word2Vec


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

def train_word2vec(texts, embedding_dim=100, window=5, min_count=2):
    model = Word2Vec(
        sentences=texts,
        vector_size=embedding_dim,
        window=window,
        min_count=min_count,
        workers=4
    )
    return model

def create_embedding_matrix(word2idx, w2v_model, embedding_dim=100):
    vocab_size = len(word2idx)
    embedding_matrix = np.zeros((vocab_size, embedding_dim), dtype=np.float32)

    for word, idx in word2idx.items():
        if word in w2v_model.wv:
            embedding_matrix[idx] = w2v_model.wv[word]
        else:
            embedding_matrix[idx] = np.random.normal(scale=0.6, size=(embedding_dim,))

    embedding_matrix[word2idx['<PAD>']] = np.zeros(embedding_dim)

    return embedding_matrix

