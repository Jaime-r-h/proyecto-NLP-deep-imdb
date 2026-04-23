import math
import re
import random
import pickle
import os
from collections import defaultdict
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.join(BASE_DIR, "..", "data")
MODEL_DIR = os.path.join(BASE_DIR, "..", "models")
os.makedirs(MODEL_DIR, exist_ok=True)

SA_MODEL_PATH = os.path.join(MODEL_DIR, "sa_naive_bayes.pkl")

def limpiar_y_tokenizar(texto: str) -> list[str]:
    texto = texto.lower()
    texto = re.sub(r'[^\w\s]', '', texto)
    return texto.split()

class ClasificadorSentimientoArtesanal:
    def __init__(self):
        self.conteo_palabras            = defaultdict(lambda: defaultdict(int))
        self.total_palabras_por_clase   = defaultdict(int)
        self.documentos_por_clase       = defaultdict(int)
        self.vocabulario_global         = set()
        self.total_documentos           = 0

    def entrenar(self, datos: list[tuple[str, str]]):
        for texto, sentimiento in datos:
            self.total_documentos += 1
            self.documentos_por_clase[sentimiento] += 1
            for palabra in limpiar_y_tokenizar(str(texto)):
                self.conteo_palabras[sentimiento][palabra] += 1
                self.total_palabras_por_clase[sentimiento] += 1
                self.vocabulario_global.add(palabra)

    def predecir(self, texto: str) -> str:
        palabras = limpiar_y_tokenizar(texto)
        probabilidades = {}
        for sentimiento in self.documentos_por_clase:
            prob_log = math.log(
                self.documentos_por_clase[sentimiento] / self.total_documentos
            )
            vocab_size   = len(self.vocabulario_global)
            total_palabras = self.total_palabras_por_clase[sentimiento]
            for palabra in palabras:
                conteo = self.conteo_palabras[sentimiento][palabra]
                prob_log += math.log((conteo + 0.1) / (total_palabras + vocab_size))
            probabilidades[sentimiento] = prob_log
        return max(probabilidades, key=probabilidades.get)

    def guardar(self, path: str = SA_MODEL_PATH):
        
        data = {
            "conteo_palabras":          {k: dict(v) for k, v in self.conteo_palabras.items()},
            "total_palabras_por_clase": dict(self.total_palabras_por_clase),
            "documentos_por_clase":     dict(self.documentos_por_clase),
            "vocabulario_global":       self.vocabulario_global,
            "total_documentos":         self.total_documentos,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)
        print(f"[SA] Modelo guardado en: {path}")

    def cargar(self, path: str = SA_MODEL_PATH):
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.conteo_palabras = defaultdict(
            lambda: defaultdict(int),
            {k: defaultdict(int, v) for k, v in data["conteo_palabras"].items()}
        )
        self.total_palabras_por_clase = defaultdict(int, data["total_palabras_por_clase"])
        self.documentos_por_clase     = defaultdict(int, data["documentos_por_clase"])
        self.vocabulario_global       = data["vocabulario_global"]
        self.total_documentos         = data["total_documentos"]
        print(f"[SA] Modelo cargado desde: {path}")

def entrenar_y_guardar(csv_path: str = None) -> ClasificadorSentimientoArtesanal:
    """Entrena el modelo desde cero con el CSV de IMDb y lo guarda."""
    if csv_path is None:
        csv_path = os.path.join(DATA_DIR, "imdb.csv")

    df = pd.read_csv(csv_path)
    dataset = list(zip(df['review'], df['sentiment']))
    random.shuffle(dataset)

    split = int(len(dataset) * 0.8)
    train_data = dataset[:split]
    test_data  = dataset[split:]
    print(f"[SA] Train: {len(train_data)} | Test: {len(test_data)}")

    modelo = ClasificadorSentimientoArtesanal()
    print("[SA] Entrenando...")
    modelo.entrenar(train_data)

    correctos = sum(
        1 for texto, label in test_data if modelo.predecir(texto) == label
    )
    print(f"[SA] Accuracy: {correctos / len(test_data):.4f}")

    modelo.guardar()
    return modelo


def cargar_modelo() -> ClasificadorSentimientoArtesanal:
    """Carga el modelo desde disco (entrena si no existe)."""
    modelo = ClasificadorSentimientoArtesanal()
    if os.path.exists(SA_MODEL_PATH):
        modelo.cargar()
    else:
        print("[SA] No se encontró modelo guardado. Entrenando desde cero...")
        modelo = entrenar_y_guardar()
    return modelo


if __name__ == "__main__":
    entrenar_y_guardar()
