# Automatic Alert Generation — NER + SA + LM

Sistema de generación automática de alertas a partir de artículos de noticias o reseñas.  
Combina un modelo **NER** (BiLSTM), un clasificador de **sentimiento** (Naive Bayes) y un **LM** (flan-t5-large) para producir alertas concisas.

---

## Instalación

> Requiere **Python 3.10+**. Se recomienda usar un entorno virtual.

## Instalar dependencias

```bash
pip install -r requirements.txt
```

## Descargar datos

Los datos para entrenar el SA (imdb.csv) ya están en la carpeta /data.

Los datos para entrenar el NER son demasiado grandes para subirlos a Github, así que el dataset deberá ser descargado accediendo a:

https://drive.google.com/file/d/1Kvkow999Di58hdBMBS3tBaGICqietamW/view?usp=sharing

Una vez descargado deberá ser añadido a /data.

Si, por alguna razón, este link no funcionara, se deberá ejecutar la función generate_ner_dataset() del fichero create_ner_dataset.py, utilizando como argumento los datos de imdb.csv (esto puede tardar mucho tiempo).


## Modelos pre-entrenados

Los modelos SA y NER **ya vienen entrenados**.

El modelo flan-t5-large se descarga automáticamente de HuggingFace la primera vez que se ejecuta el pipeline (requiere ~3 GB de espacio y conexión a internet).


Si tienes token de HuggingFace, expórtalo antes:


set HF_TOKEN=hf_xxxxxxxxxxxx  


---

## Uso — modo interactivo

Ejecuta siempre desde la **raíz del proyecto**:

```bash
python -m src.real_main
```

El sistema arranca y pide:

1. **Título** del artículo o película.
2. **Texto** (review o artículo), línea a línea. Escribe `FIN` para terminar el texto.

Es importante que la review se escriba en inglés. Hay varios ejemplos de reviews en el paper por si las quereis utilizar.
Después de esto te da el valor de NER y sentiment y la alerta generada.

### Re-entrenar SA y NER antes de arrancar el pipeline

```bash
python -m src.real_main --train-sa --train-ner
```
