import argparse
import os
import sys

SRC_DIR  = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(SRC_DIR, "..")
sys.path.insert(0, SRC_DIR)
sys.path.insert(0, ROOT_DIR)

import torch
from src.sentiment_analysis_NV import cargar_modelo as cargar_sa, entrenar_y_guardar as entrenar_sa
from src.prueba_LM import generar_alerta, guardar_modelo as guardar_lm
from src.ner import BiLSTMNER, predict_sentence, train_ner_model
from src.data_processing import create_lookup_tables, train_word2vec, create_embedding_matrix
from src.utils import get_tokens_from_text

NER_MODEL_PATH = os.path.join(ROOT_DIR, "models", "LSTM_model.pt")
NER_DATA_PATH  = os.path.join(ROOT_DIR, "data",   "ner_dataset.csv")

def cargar_ner():
    if not os.path.exists(NER_MODEL_PATH):
        print("[NER] No se encontró modelo guardado. Entrenando desde cero...")
        train_ner_model(
            csv_path=NER_DATA_PATH,
            save_path=NER_MODEL_PATH,
        )

    print(f"[NER] Cargando modelo desde: {NER_MODEL_PATH}")
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(NER_MODEL_PATH, map_location=device)

    word2idx       = checkpoint["word2idx"]
    idx2label      = checkpoint["idx2label"]
    label2idx      = checkpoint["label2idx"]
    embedding_dim  = checkpoint["embedding_dim"]
    hidden_dim     = checkpoint["hidden_dim"]
    max_len        = checkpoint["max_len"]

    model = BiLSTMNER(
        vocab_size     = len(word2idx),
        embedding_dim  = embedding_dim,
        hidden_dim     = hidden_dim,
        output_dim     = len(label2idx),
        padding_idx    = word2idx["<PAD>"],
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print("[NER] Modelo listo.")
    return model, word2idx, idx2label, max_len, device



# Etiquetas BIO que corresponden a entidades reales (no O ni PAD)
_ENTITY_TAGS = {"B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC",
                "B-MISC", "I-MISC", "B-per", "I-per", "B-org", "I-org",
                "B-loc", "I-loc", "B-gpe", "I-gpe", "B-geo", "I-geo"}

_TAG_TO_TYPE = {
    "PER":  "PER",  "per":  "PER",
    "ORG":  "ORG",  "org":  "ORG",
    "LOC":  "LOC",  "loc":  "LOC",  "geo": "LOC",
    "MISC": "MISC", "misc": "MISC",
    "GPE":  "LOC",  "gpe":  "LOC",
}


def _tag_to_type(tag: str) -> str:
    core = tag.split("-", 1)[-1] if "-" in tag else tag
    return _TAG_TO_TYPE.get(core, core.upper())


def ejecutar_ner(texto: str, model, word2idx, idx2label, max_len, device) -> str:
    tokens = get_tokens_from_text(texto)      
    if not tokens:
        return ""

    token_tag_pairs = predict_sentence(
        sentence_tokens=tokens,
        model=model,
        word2idx=word2idx,
        idx2label=idx2label,
        max_len=max_len,
        device=str(device),
    )

    menciones = []
    current_tokens = []
    current_type   = None

    for token, tag in token_tag_pairs:
        if tag.startswith("B-"):
            if current_tokens:
                menciones.append((" ".join(current_tokens), _tag_to_type(current_type)))
            current_tokens = [token]
            current_type   = tag
        elif tag.startswith("I-") and current_tokens:
            current_tokens.append(token)
        else:                                 
            if current_tokens:
                menciones.append((" ".join(current_tokens), _tag_to_type(current_type)))
            current_tokens = []
            current_type   = None

    if current_tokens:
        menciones.append((" ".join(current_tokens), _tag_to_type(current_type)))

    if not menciones:
        return "No entities found"

    return ", ".join(f"{ent} [{tipo}]" for ent, tipo in menciones)


def pipeline(titulo: str, texto: str, modelo_sa, ner_model, word2idx,
             idx2label, max_len, device) -> dict:

    print("\n[1/3] Ejecutando NER...")
    entidades = ejecutar_ner(texto, ner_model, word2idx, idx2label, max_len, device)
    print(f"      Entidades: {entidades}")

    print("[2/3] Ejecutando SA (Naive Bayes)...")
    sentimiento = modelo_sa.predecir(texto)
    print(f"      Sentimiento: {sentimiento}")

    print("[3/3] Generando alerta con LM (flan-t5)...")
    alerta = generar_alerta(
        entidades=entidades,
        sentimiento=sentimiento,
        titulo=titulo,
        texto=texto,
    )
    print(f"Alerta: {alerta}")

    return {
        "titulo":      titulo,
        "entidades":   entidades,
        "sentimiento": sentimiento,
        "alerta":      alerta,
    }


def modo_interactivo(modelo_sa, ner_model, word2idx, idx2label, max_len, device):
    print("\n" + "-" * 55)
    print("  Sistema de Generación Automática de Alertas")
    print("  NER (BiLSTM) + SA (Naive Bayes) + LM (flan-t5)")
    print("-" * 55)
    print("Escribe 'salir' en cualquier momento para terminar.\n")

    while True:
        titulo = input("Título del artículo / película: ").strip()
        if titulo.lower() == "salir":
            break
        if not titulo:
            print("El título no puede estar vacío.\n")
            continue

        print("Texto. Termina con una línea que solo contenga 'FIN':")
        lineas = []
        while True:
            linea = input()
            if linea.strip().upper() == "FIN" or linea.strip().lower() == "salir":
                break
            lineas.append(linea)
        texto = " ".join(lineas).strip()

        if not texto:
            print("El texto no puede estar vacío.\n")
            continue

        resultado = pipeline(
            titulo=titulo,
            texto=texto,
            modelo_sa=modelo_sa,
            ner_model=ner_model,
            word2idx=word2idx,
            idx2label=idx2label,
            max_len=max_len,
            device=device,
        )

        continuar = input("¿Procesar otro texto? (s/n): ").strip().lower()
        if continuar != "s":
            break

    print("\nHasta luego.")

def main():
    parser = argparse.ArgumentParser(description="Pipeline NER + SA + Alert Generation")
    parser.add_argument("--train-sa",  action="store_true",
                        help="Re-entrena el modelo SA desde imdb.csv y lo guarda")
    parser.add_argument("--train-ner", action="store_true",
                        help="Re-entrena el modelo NER desde ner_dataset.csv y lo guarda")
    parser.add_argument("--save-lm",   action="store_true",
                        help="Descarga flan-t5-large y lo guarda en models/flan_t5_large/")
    args = parser.parse_args()

    if args.train_sa:
        print("Re-entrenando modelo SA...")
        entrenar_sa()

    if args.train_ner:
        print("Re-entrenando modelo NER...")
        train_ner_model(csv_path=NER_DATA_PATH, save_path=NER_MODEL_PATH)

    if args.save_lm:
        token = os.environ.get("HF_TOKEN", "")
        if token:
            from huggingface_hub import login
            login(token)
        guardar_lm()

    modelo_sa = cargar_sa()
    ner_model, word2idx, idx2label, max_len, device = cargar_ner()

    modo_interactivo(modelo_sa, ner_model, word2idx, idx2label, max_len, device)


if __name__ == "__main__":
    main()
