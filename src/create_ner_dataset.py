from typing import List, Tuple, Dict, Generator
from src.utils import get_tokens_from_text, get_sentiment_label
import csv
from collections import Counter
import numpy as np
import spacy
import os
import pandas as pd

# nlp = spacy.load("en_core_web_trf")
nlp = spacy.load("en_core_web_trf")


def get_entities(text):
    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]

def entities_to_iob(tokens, entities):
    tags = ["O"] * len(tokens)
    
    for ent_text, ent_label in entities:
        ent_tokens = ent_text.lower().split()
        
        for i in range(len(tokens)):
            if tokens[i:i+len(ent_tokens)] == ent_tokens:
                tags[i] = "B-" + ent_label
                for j in range(1, len(ent_tokens)):
                    tags[i+j] = "I-" + ent_label
    
    return tags

def generate_ner_dataset(data, output_file='data/ner_dataset.csv'):
    ner_data = []

    for tokens, _ in data:
        text = " ".join(tokens)
        
        entities = get_entities(text)
        ner_tags = entities_to_iob(tokens, entities)
        
        ner_data.append((tokens, ner_tags))

    df = pd.DataFrame(ner_data, columns=['tokens', 'ner_tags'])

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_csv(output_file, index=False)

    print(f"Dataset guardado en {output_file}")

    return ner_data