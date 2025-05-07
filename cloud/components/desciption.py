import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
import sqlite3
import gc
import time
import torch
from components.commons import db_path
import faiss
from rich import print
import re
from tabulate import tabulate
conn=sqlite3.connect(db_path)
model=SentenceTransformer("sentence-transformers/LaBSE")

def clean_text(text):
    text=str(text).lower()
    text=re.sub(r'[^\w\s]', '', text)
    return text.strip()

df=pd.read_sql_query("SELECT * FROM INVENTARIO_REPORTES", conn)
df['clave']=df['CLAVE_REP'].apply(clean_text)
df['nombre_limpio']=df['REPORTE'].apply(clean_text)
df['descripcion_limpia']=df['DESCRIPCION_ESP'].apply(clean_text)

dimension=model.get_sentence_embedding_dimension()
index=faiss.IndexFlatL2(dimension)

row_ids=[]
for columna in ['clave', 'nombre_limpio', 'descripcion_limpia']:
    emb=model.encode(df[columna].tolist(), show_progress_bar=True).astype(np.float32)
    index.add(emb)
    row_ids.extend(df.index.tolist())

row_ids=np.array(row_ids)

def desc_search(prompt):
    q_emb=model.encode([clean_text(prompt)]).astype(np.float32)
    D, I=index.search(q_emb, 50)

    df_indices=row_ids[I[0]]
    similitudes=1 - D[0] / 4

    cols_a_omitir=['clave', 'nombre_limpio', 'descripcion_limpia']
    all_cols=[c for c in df.columns if c not in cols_a_omitir]

    resultados=df.loc[df_indices, all_cols].copy()
    resultados['similitud']=similitudes

    umbral=similitudes.mean()
    filtrados=resultados[resultados['similitud'] > umbral]

    context_df=(
        filtrados
        .sort_values(by='similitud', ascending=False)
        .drop(columns=['similitud'])
    )

    return tabulate(context_df, headers='keys', tablefmt='grid')
