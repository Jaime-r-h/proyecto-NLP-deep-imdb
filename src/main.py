from pydoc import text
from src.data_processing_sentiment import load_processed_data_with_embeddings, get_tokens_and_labels
from src.create_ner_dataset import get_entities, entities_to_iob, generate_ner_dataset
from src.utils import get_tokens_from_text
import torch 

def main():
    data_dir = "data/imdb.csv"


    X, y, word2idx, idx2word, embedding_matrix = load_processed_data_with_embeddings(
        data_dir, min_freq=2, max_len=100, embedding_dim=100
    )

    # X_tensor = torch.tensor(X, dtype=torch.long)
    # y_tensor = torch.tensor(y, dtype=torch.float32)
    # print(X_tensor[0])
    # print(y_tensor[0])

    # embedding_layer = torch.nn.Embedding.from_pretrained(
    #     torch.tensor(embedding_matrix, dtype=torch.float32),
    #     freeze=False 
    # )

    for i in range(2, 6):
        print(idx2word[i])
        print(embedding_matrix[i])

    

    # text = "I loved Avatar and Sam Worthington"
    # tokens = get_tokens_from_text(text)
    # entities = get_entities(text)
    # iob_tags = entities_to_iob(tokens, entities)
    # print(iob_tags)
    
    token_list, label_list = get_tokens_and_labels(data_dir)
    data = list(zip(token_list, label_list)) 
    ner_dataset = generate_ner_dataset(data)
    print(ner_dataset[0])



if __name__ == "__main__":
    main()