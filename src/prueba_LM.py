import os
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "..", "models", "flan_t5_large")
MODEL_ID  = "google/flan-t5-large"


_tokenizer = None
_model     = None


def _cargar_modelo():
    global _tokenizer, _model
    if _tokenizer is not None:
        return

    source = MODEL_DIR if os.path.isdir(MODEL_DIR) else MODEL_ID
    print(f"[AG] Cargando modelo desde: {source}")
    _tokenizer = AutoTokenizer.from_pretrained(source)
    _model     = AutoModelForSeq2SeqLM.from_pretrained(source)
    print("[AG] Modelo listo.")


def guardar_modelo(path: str = MODEL_DIR):
    """Persiste el tokenizador y el modelo en disco."""
    _cargar_modelo()
    os.makedirs(path, exist_ok=True)
    _tokenizer.save_pretrained(path)
    _model.save_pretrained(path)
    print(f"[AG] Modelo guardado en: {path}")


def generar_alerta(entidades: str, sentimiento: str, titulo: str, texto: str) -> str:
    _cargar_modelo()

    sin_entidades = (
        not entidades
        or entidades.strip().lower() in ("", "no entities found", "entity_placeholder [per]")
    )

    if sin_entidades:
        prompt = f"""Generate a concise, single-sentence  news alert based on the provided Title, Sentiment, and Review. 
    Rules:
    - Clearly reflect the given sentiment.
    - Use only the provided information.
    - Paraphrase the review in your own words.
    - Alert cannot have more than 10 words
    Examples:

    Title: The Batman
    Sentiment: Positive
    Review: A dark and visually stunning film with an outstanding lead performance and strong direction.
    Alert: 'The Batman' receives strong positive reception.

    Title: Ant-Man
    Sentiment: Negative
    Review: Weak script and unconvincing visual effects overshadow the lead's charm.
    Alert: 'Ant-Man' faces negative reception due to a weak script.
    
    Now generate the alert:

    Title: {titulo}
    Entities: {entidades}
    Sentiment: {sentimiento.capitalize()}
    Review: {texto}
    Alert:"""

    else:
        prompt = f"""Generate a concise, single-sentence news alert based on the provided Title, Entities, Sentiment, and Review. 
    Rules:
    - Incorporate as many of the listed entities as possible.
    - Clearly reflect the given sentiment.
    - Use only the provided information.
    - Paraphrase the review in your own words.
    - Alert cannot have more than 10 words
    

    Examples:

    Title: The Batman
    Entities: Robert Pattinson [PER], Matt Reeves [PER]
    Sentiment: Positive
    Review: Matt Reeves delivers a dark and visually stunning film, with Robert Pattinson giving an outstanding lead performance and strong direction throughout.
    Alert: Matt Reeves' 'The Batman' receives strong positive reception, highlighting Robert Pattinson's performance.

    Title: Ant-Man
    Entities: Marvel Studios [ORG], Paul Rudd [PER]
    Sentiment: Negative
    Review: Marvel Studios presents a film with a weak script and unconvincing visual effects, which overshadow Paul Rudd's otherwise charming performance.
    Alert: Marvel Studios' 'Ant-Man' faces negative reception,

    
    Now generate the alert:

    Title: {titulo}
    Entities: {entidades}
    Sentiment: {sentimiento.capitalize()}
    Review: {texto}
    Alert:"""
        
    inputs = _tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    outputs = _model.generate(**inputs, max_new_tokens=60, num_beams=5, early_stopping=True)
    return _tokenizer.decode(outputs[0], skip_special_tokens=True)
