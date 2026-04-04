from src.data_processing import load_processed_data_with_embeddings
import torch 

def main():
    data_dir = "data/imdb.csv"

    X, y, word2idx, idx2word, embedding_matrix = load_processed_data_with_embeddings(
        data_dir, min_freq=2, max_len=100, embedding_dim=100
    )

    X_tensor = torch.tensor(X, dtype=torch.long)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    print(X_tensor[0])
    print(y_tensor[0])

    embedding_layer = torch.nn.Embedding.from_pretrained(
        torch.tensor(embedding_matrix, dtype=torch.float32),
        freeze=False 
    )


if __name__ == "__main__":
    main()