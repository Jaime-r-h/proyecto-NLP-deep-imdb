import ast
import csv
import os
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.data_processing import (
    create_embedding_matrix,
    create_lookup_tables,
    text_to_indices,
    train_word2vec,
)

try:
    from src.utils import parse_serialized_list, create_label_lookup_tables
except ImportError:
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


def get_ner_tokens_and_labels(infile: str) -> Tuple[List[List[str]], List[List[str]]]:
    tokens_list = []
    ner_tags_list = []

    with open(infile, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row_num, row in enumerate(reader, start=1):
            tokens = parse_serialized_list(row["tokens"])
            ner_tags = parse_serialized_list(row["ner_tags"])

            if len(tokens) == 0 or len(ner_tags) == 0:
                continue

            if len(tokens) != len(ner_tags):
                print(f"[WARNING] Fila {row_num} ignorada: {len(tokens)} tokens vs {len(ner_tags)} etiquetas")
                continue

            tokens_list.append(tokens)
            ner_tags_list.append(ner_tags)

    if len(tokens_list) == 0:
        raise ValueError("No se pudieron cargar ejemplos válidos desde ner_dataset.csv")

    return tokens_list, ner_tags_list


def labels_to_indices(labels: List[str], label2idx: Dict[str, int], max_len: int = None) -> List[int]:
    indices = [label2idx[label] for label in labels]
    if max_len is not None:
        pad_idx = label2idx["<PAD>"]
        indices = indices[:max_len] + [pad_idx] * max(0, max_len - len(indices))
    return indices


def load_processed_ner_data_with_embeddings(
    infile: str,
    min_freq: int = 2,
    max_len: int = 100,
    embedding_dim: int = 100,
):
    tokens_list, ner_tags_list = get_ner_tokens_and_labels(infile)

    word2idx, idx2word = create_lookup_tables(tokens_list, min_freq=min_freq)
    label2idx, idx2label = create_label_lookup_tables(ner_tags_list)

    w2v_model = train_word2vec(tokens_list, embedding_dim=embedding_dim, min_count=min_freq)

    X = [text_to_indices(tokens, word2idx, max_len=max_len) for tokens in tokens_list]
    y = [labels_to_indices(tags, label2idx, max_len=max_len) for tags in ner_tags_list]

    embedding_matrix = create_embedding_matrix(word2idx, w2v_model, embedding_dim=embedding_dim)

    return (
        np.array(X, dtype=np.int64),
        np.array(y, dtype=np.int64),
        word2idx,
        idx2word,
        label2idx,
        idx2label,
        embedding_matrix,
    )


class NERDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class LSTMNER(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        output_dim: int,
        padding_idx: int,
        embedding_matrix: np.ndarray = None,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx
        )

        if embedding_matrix is not None:
            self.embedding.weight.data.copy_(torch.tensor(embedding_matrix, dtype=torch.float32))

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            bidirectional=False,
            batch_first=True,
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        emb = self.embedding(x)
        out, _ = self.lstm(emb)
        out = self.dropout(out)
        logits = self.fc(out)
        return logits


def split_data(X: np.ndarray, y: np.ndarray, train_ratio: float = 0.8, seed: int = 42):
    np.random.seed(seed)
    indices = np.arange(len(X))
    np.random.shuffle(indices)

    split_idx = int(len(X) * train_ratio)
    train_idx = indices[:split_idx]
    val_idx = indices[split_idx:]

    return X[train_idx], y[train_idx], X[val_idx], y[val_idx]


def token_accuracy(logits: torch.Tensor, labels: torch.Tensor, pad_idx: int) -> float:
    preds = torch.argmax(logits, dim=-1)
    mask = labels != pad_idx
    correct = (preds == labels) & mask

    total_tokens = mask.sum().item()
    if total_tokens == 0:
        return 0.0

    return correct.sum().item() / total_tokens


def train_one_epoch(model, dataloader, optimizer, criterion, device, pad_idx):
    model.train()
    total_loss = 0.0
    total_acc = 0.0

    for X_batch, y_batch in dataloader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        logits = model(X_batch)

        loss = criterion(logits.view(-1, logits.shape[-1]), y_batch.view(-1))
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_acc += token_accuracy(logits, y_batch, pad_idx)

    return total_loss / len(dataloader), total_acc / len(dataloader)


@torch.no_grad()
def evaluate(model, dataloader, criterion, device, pad_idx):
    model.eval()
    total_loss = 0.0
    total_acc = 0.0

    for X_batch, y_batch in dataloader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        logits = model(X_batch)
        loss = criterion(logits.view(-1, logits.shape[-1]), y_batch.view(-1))

        total_loss += loss.item()
        total_acc += token_accuracy(logits, y_batch, pad_idx)

    return total_loss / len(dataloader), total_acc / len(dataloader)


def train_ner_model(
    csv_path: str = "data/ner_dataset.csv",
    min_freq: int = 2,
    max_len: int = 100,
    embedding_dim: int = 100,
    hidden_dim: int = 128,
    batch_size: int = 32,
    epochs: int = 5,
    lr: float = 1e-3,
    train_ratio: float = 0.8,
    save_path: str = "models/ner_lstm_model.pt",
):
    print("Cargando y procesando dataset NER...")

    X, y, word2idx, idx2word, label2idx, idx2label, embedding_matrix = \
        load_processed_ner_data_with_embeddings(
            infile=csv_path,
            min_freq=min_freq,
            max_len=max_len,
            embedding_dim=embedding_dim,
        )

    print(f"Número de muestras: {len(X)}")
    print(f"Vocab size: {len(word2idx)}")
    print(f"Número de etiquetas: {len(label2idx)}")

    X_train, y_train, X_val, y_val = split_data(X, y, train_ratio=train_ratio)

    train_loader = DataLoader(NERDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(NERDataset(X_val, y_val), batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando device: {device}")

    pad_word_idx = word2idx["<PAD>"]
    pad_label_idx = label2idx["<PAD>"]

    model = LSTMNER(
        vocab_size=len(word2idx),
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        output_dim=len(label2idx),
        padding_idx=pad_word_idx,
        embedding_matrix=embedding_matrix,
    ).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=pad_label_idx)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device, pad_label_idx)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device, pad_label_idx)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"train_loss={train_loss:.4f} | train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} | val_acc={val_acc:.4f}"
        )

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "word2idx": word2idx,
        "idx2word": idx2word,
        "label2idx": label2idx,
        "idx2label": idx2label,
        "embedding_dim": embedding_dim,
        "hidden_dim": hidden_dim,
        "max_len": max_len,
        "history": history,
    }

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(checkpoint, save_path)

    print(f"Modelo guardado en: {save_path}")
    return model, history, word2idx, idx2word, label2idx, idx2label


def main():
    train_ner_model(
        csv_path="data/ner_dataset.csv",
        min_freq=2,
        max_len=100,
        embedding_dim=100,
        hidden_dim=128,
        batch_size=32,
        epochs=5,
        lr=1e-3,
        train_ratio=0.8,
        save_path="models/ner_lstm_model.pt",
    )


if __name__ == "__main__":
    main()