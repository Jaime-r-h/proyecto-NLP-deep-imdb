from typing import List, Tuple, Dict, Generator
from src.utils import get_tokens_from_text, get_sentiment_label
from src.data_processing import create_lookup_tables, text_to_indices, create_embedding_matrix, train_word2vec
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
            
            tokens = get_tokens_from_text(review)

            label = get_sentiment_label(sentiment)

            tokens_list.append(tokens)
            labels_list.append(label)

    return tokens_list, labels_list

def load_processed_data_with_embeddings(infile: str, min_freq=2, max_len=None, embedding_dim=100):
    # Devuelve los textos de input X en forma de lista de índices, y los outpus Y en forma de 0 (negativo) y 1 (positivo)
    # Devuelve los diccionarios de conversión palabra - índice
    # Devuelve la matriz de embeddings, ya sea preentrenada o por entrenar

    tokens_list, labels_list = get_tokens_and_labels(infile)

    word2idx, idx2word = create_lookup_tables(tokens_list, min_freq=min_freq)

    w2v_model = train_word2vec(tokens_list, embedding_dim=100)

    X = [text_to_indices(tokens, word2idx, max_len=max_len) for tokens in tokens_list]
    y = labels_list
    embedding_matrix = create_embedding_matrix(word2idx, w2v_model)

    return np.array(X), np.array(y), word2idx, idx2word, embedding_matrix