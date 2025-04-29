import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
import sqlite3
import gc
import time
import torch
from commons import db_path, data, muest_empt
from rich import print
conn=sqlite3.connect(db_path)

def desc_search(prompt):
    transformer_model=SentenceTransformer("hiiamsid/sentence_similarity_spanish_es")
    embedding=transformer_model.encode(prompt.replace("reportes", "").replace("reporte", ""))
    # -- Buscamo el mas cercano --
    distances=[]
    for clave, info in data.items():
        dist=np.linalg.norm(np.array(info["EMBEDDING"])-np.array(embedding))
        distances.append((dist, info))
    # -- Ordenamos por distancia --
    distances.sort(key=lambda x: x[0])
    # -- Guardamos los 10 mas cercasnos (solo CLAVE_REP) --
    closest_ids=[]
    closest_ids=[info["CLAVE_REP"] for _, info in distances[:10]]
    # -- Con los IDs creamos el dataframe --
    response="SELECT CLAVE_REP, CLAVE_ENTIDADREGULADA, REPORTE, CLAVE_PERIODO, DESCRIPCION_ESP, VIGENTE FROM INVENTARIO_REPORTES WHERE CLAVE_REP IN ({})".format(",".join(f"'{id}'" for id in closest_ids))
    print(f"[bold cyan]query:[/] [cyan]{response}[/cyan]")
    query_result=pd.read_sql_query(response, conn)
    num_rows=len(query_result)
    context=query_result.sample(n=min(10, num_rows), random_state=42)
    muestreo, empty_message=muest_empt(num_rows, context)
    # -- Eliminacion de memoria --
    del transformer_model
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(1)
    return muestreo, empty_message, response, context.to_string(index=False)