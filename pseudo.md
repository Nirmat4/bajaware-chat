El siguiente es un seudocodigo para la generacion de un chat con historial dinamico:

En la primera iteracion:
    tomamos el primer mensaje y se forma el primer topico

En la segunda iteracion:
    Si la consulta no encaja con el topico anterior por medio de una paridad semantica,
    se crea un nuevo topico

    Pero si la pregunta es una aclaracion se une al topico mas reciente

En la tercera iteracion:
    si la consulta no encaja con algun topico se crea uno nuevo,

    Si la pregunta es una aclaracion se une al topico mas reciente

Hasta llegar al final de los mensajes


text=[
    "Que reportes se envian al IPAB?",
    "Cuales de ellos son por evento?",
    "Cuantos son en total",
    "Cuantas validaciones estan activas actualmente?",
    "Cuales de ellas son de tipo INTRA?",
    "Me recuerdas a cuales son los reportes del IPAB?",
    "Que reportes se son de BM?",
    "Algunos de ellos son trimestrales?"
]

Para llegar a la paridad semantica usamos el modelo "jina-reranker-m0", por medio de carga local

model=AutoModel.from_pretrained(
    "../models/jina-reranker-m0",
    trust_remote_code=True,
    torch_dtype=torch.float16,
    device_map="auto"
)
model.eval()

tokenizer=AutoTokenizer.from_pretrained(
    "../models/jina-reranker-m0",
    trust_remote_code=True
)

Y para la identificacion de intenciones podriamos usar el modelo "xlm-roberta-large-xnli" de manera local igual:

classifier = pipeline("zero-shot-classification", model="../models/xlm-roberta-large-xnli")


```Python
import math
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity

# Cargar DeBERTa-v3
tokenizer = AutoTokenizer.from_pretrained("../models/deberta-v3-large")
model = AutoModel.from_pretrained("../models/deberta-v3-large")

textos = [
    "Que reportes se envian al IPAB?",
    "Cuales de ellos son por evento?",
    "Cuantos son en total",
    # ... (resto de textos)
]

# Paso 1: Tokenizar y obtener embeddings
token_info = []  # Lista de (token, índice_texto)
embeddings = []

for idx, texto in enumerate(textos):
    inputs = tokenizer(texto, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    hidden_states = outputs.last_hidden_state.squeeze(0)
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0], skip_special_tokens=True)
    
    for token, embedding in zip(tokens, hidden_states):
        if token not in ["[CLS]", "[SEP]", "[PAD]"]:
            token_info.append((token, idx))
            embeddings.append(embedding.numpy())

# Paso 2 y 3: Espiral y factor temporal (igual que antes)
# ...

# Función de búsqueda modificada para tokens
def buscar_tokens_similares(query, spiral, token_info, n=5):
    # Procesar query
    inputs = tokenizer(query, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    query_embeddings = outputs.last_hidden_state.squeeze(0).numpy()
    
    # Ajustar embedding de la query
    t_max = int(math.sqrt(len(embeddings) - 1))
    t_query = t_max
    phi_query = (t_query ** 2) % (2 * t_query + 1)
    factor_query = generar_factor_temporal(t_query, phi_query, dimension=query_embeddings.shape[1])
    query_ajustada = np.mean(query_embeddings, axis=0) * factor_query
    
    # Calcular similitudes para TODOS los tokens
    similitudes = []
    for i, (token, idx_texto) in enumerate(token_info):
        t = int(math.sqrt(i))
        phi = i - t * t
        emb_ajustado = spiral.get((t, phi), np.zeros(model.config.hidden_size))
        sim = cosine_similarity([query_ajustada], [emb_ajustado])[0][0]
        similitudes.append((token, sim, idx_texto))  # Token + similitud + índice texto origen
    
    # Ordenar y filtrar
    similitudes.sort(key=lambda x: x[1], reverse=True)
    
    # Devolver top N tokens únicos (evitar duplicados)
    tokens_vistos = set()
    resultados = []
    for token, sim, idx in similitudes:
        if token not in tokens_vistos:
            resultados.append((token, sim))
            tokens_vistos.add(token)
        if len(resultados) == n:
            break
            
    return resultados

# Ejemplo de uso
query = "bajo que periodo se envian"
resultados = buscar_tokens_similares(query, spiral, token_info, n=5)
print(f"\nTop 5 tokens similares a '{query}':")
for token, sim in resultados:
    print(f"- {token} (similitud: {sim:.2f})")
```