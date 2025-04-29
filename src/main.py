#╔══════════════════════════╗ 
#║ Importacion de librerias ║ 
#╚══════════════════════════╝ 
import warnings
warnings.filterwarnings("ignore")
import torch
import time
import gc
import subprocess
import sqlite3
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel, AutoProcessor
import warnings
import pandas as pd
from datetime import datetime
from ollama import chat
import spacy
from spacy.language import Language
import re
import hjson
import pandas as pd
from sentence_transformers import SentenceTransformer
import pickle
import numpy as np
from colorama import init, Fore, Style, Back
# -- Ajustes de warnings y memoeria --
max_memory={0: "3500MB"}
torch.manual_seed(42)
warnings.filterwarnings("ignore", "Using a slow image processor as `use_fast` is unset", category=UserWarning, module="transformers.generation.utils")
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

#╔══════════════════════════════════════════╗ 
#║ Carga de diccionarios y rutas de modelos ║ 
#╚══════════════════════════════════════════╝ 
ollama_model="deepseek-r1:14b"
nlp=spacy.load("es_core_news_lg")
qbert_path="../models/llama-3-sqlcoder-8b"
jina_path="../models/jina-reranker-m0"
db_path='../database/bajaware.db'
conn=sqlite3.connect(db_path)
with open("../assets/replacements.hjson", "r") as file:
    replacements=hjson.load(file)
with open("../assets/embeddings.pkl", "rb") as f:
    data=pickle.load(f)

#╔══════════════════════════╗ 
#║ Caraga de modelos en CPU ║ 
#╚══════════════════════════╝ 
jina=AutoModel.from_pretrained(
    jina_path, 
    trust_remote_code=True, 
    torch_dtype=torch.float16, 
    local_files_only=True
).to("cpu")
tokenizer_jina=AutoTokenizer.from_pretrained(jina_path, trust_remote_code=True)

# -- Edicion de componente customizado --
@Language.component("replacer_component")
def replacer_component(doc):
    text=doc.text
    for k, v in replacements.items():
        pattern=r"\b" + re.escape(k) + r"\b"
        text=re.sub(pattern, v, text, flags=re.IGNORECASE)
    new_doc=nlp.make_doc(text)
    return new_doc
nlp.add_pipe("replacer_component", first=True)

#╔═══════════════════════════════════╗ 
#║ Definicion de tablas y parametros ║ 
#╚═══════════════════════════════════╝ 
with open("../assets/tables.hjson", "r") as file:
    tables=hjson.load(file)

table_docs={
    "INVENTARIO_REPORTES": f"{tables['table_reports'][0]}\n{tables['table_reports'][1]}",
    "INVENTARIO_VALIDACIONES": f"{tables['table_valid'][0]}\0{tables['table_valid'][1]}",
    "INVENTARIO_VALIDACIONES_COMPUESTO": f"{tables['table_report_valid'][0]}\n{tables['table_report_valid'][1]}",
    "CLIENTE": f"{tables['table_client'][0]}\n{tables['table_client'][1]}",
    "CONTRATOS": f"{tables['table_contra'][0]}\n{tables['table_contra'][1]}",
    "CONTRATOS_REPORTES": f"{tables['table_contra_report'][0]}\n{tables['table_contra_report'][1]}",
    "CONTRATOS_REPORTES_COMPUESTO": f"{tables['table_contra_report_report'][0]}\n{tables['table_contra_report_report'][1]}",
    "CONTRATOS_CLIENTES_REPORTES": f"{tables['table_client_contra_report'][0]}\n{tables['table_client_contra_report'][1]}",
}

table_desc={
    "INVENTARIO_REPORTES": tables['table_reports'][1],
    "INVENTARIO_VALIDACIONES": tables['table_valid'][1],
    "INVENTARIO_VALIDACIONES_COMPUESTO": tables['table_report_valid'][1],
    "CLIENTE": tables['table_client'][1],
    "CONTRATOS": tables['table_contra'][1],
    "CONTRATOS_REPORTES": tables['table_contra_report'][1],
    "CONTRATOS_REPORTES_COMPUESTO": tables['table_contra_report_report'][1],
    "CONTRATOS_CLIENTES_REPORTES": tables['table_client_contra_report'][1],
}

#╔══════════════════════════╗ 
#║ Funcion de analisis Jina ║ 
#╚══════════════════════════╝ 
def choose_table(query: str) -> str:
    pairs=[[query, schema] for schema in table_docs.values()]
    
    jina.to("cpu")
    with torch.no_grad():
        scores=jina.compute_score(pairs, max_length=1024, doc_type="text")

    jina.to("cpu")
    torch.cuda.empty_cache()
    best_idx=int(torch.argmax(torch.tensor(scores, device="cpu")))
    return list(table_docs.keys())[best_idx]

#╔════════════════════════════════════════╗ 
#║ Funcion loop y definicion de variables ║ 
#╚════════════════════════════════════════╝ 
query_history=""
response_history=""
menu=""
while menu!="S":
    # -- Ingreso de consulta y tratamiento --
    query=input(">> ")
    original_query=query
    if query=="S":
        break
    query=(nlp(query)).text
    flag=False
    if len(query_history)==0:
        flag=True
    if flag:
        query_history=query
    
    table=choose_table(query)
    print(Style.BRIGHT+Fore.YELLOW+table+Style.RESET_ALL)

    # -- Reemplazo de terminos --
    if table=="INVENTARIO_REPORTES":
        subsecuencias=["CLAVE_VALIDACION", "ID_VALIDACION_ANT", "TIPO_VALIDACION", "CAMPO"]
        if " -- VIGENTE=0" not in query:
            query+=" -- (VIGENTE=1)"
        for termino in subsecuencias:
            if termino in query:
                query=query.replace(" -- (VIGENTE=1)", "")
    if not flag:
        query_history=query_history.replace(" -- VIGENTE=0", "").replace(" -- (VIGENTE=1)", "")
        query_history+=f"\n{query}"
    print(Style.BRIGHT+Fore.MAGENTA+query_history+Style.RESET_ALL)

    # -- Preparacion de prompt qBERT --
    text=f"""
    ### Instruction:
    Your task is to generate valid SQL to answer the following question, given a database schema.

    ### Input:
    Here is the database schema that the SQL query will run on:
    {table_desc[table]}

    ### Question:
    {query_history}

    ### Response (use shorthand if possible):"""

    # -- Generacion de consulta SQL --
    sql_model="qbert"
    stream=chat(
        model=sql_model,
        messages=[{
            'role': 'user',
            'content': text
        }],
        stream=True
    )

    response=""
    for chunk in stream:
        print(Style.BRIGHT+Fore.GREEN+chunk['message']['content'], end='', flush=True)
        response+=(chunk['message']['content'])
    print(Style.RESET_ALL)
    subprocess.run(['ollama', 'stop', sql_model])
    time.sleep(1)

    # -- Extraccion de terminos en tabla --
    count=0
    values=[]
    for text in replacements.values():
        values.append(text)
    values=list(set(values))
    for text in values:
        if f" {text} " in query:
            count+=1

    # -- Generacion de df --
    if count>=0:
        try:
            query_result=pd.read_sql_query(response, conn)    
            num_rows=len(query_result)
            sub_df=query_result.sample(n=min(10, num_rows), random_state=42)

        except Exception as e:
            print(Fore.RED+f"error en la ejecucion de la consulta: {e}"+Style.RESET_ALL)

    elif table=="INVENTARIO_REPORTES" and count<0:
        transformer_model=SentenceTransformer("hiiamsid/sentence_similarity_spanish_es")
        embedding=transformer_model.encode(original_query.replace("reportes", "").replace("reporte", ""))
        # Buscamos el más cercano
        distances=[]
        for clave, info in data.items():
            dist=np.linalg.norm(np.array(info["EMBEDDING"])-np.array(embedding))
            distances.append((dist, info))
        # Ordenamos por distancia
        distances.sort(key=lambda x: x[0])
        print("Más cercanos:")
        # Guardamos los 10 mas cercasnos (solo CLAVE_REP)
        closest_ids=[]
        closest_ids=[info["CLAVE_REP"] for _, info in distances[:10]]
        print(closest_ids)
        # Con los IDs creamos el dataframe
        query="SELECT CLAVE_REP, CLAVE_ENTIDADREGULADA, REPORTE, CLAVE_PERIODO, DESCRIPCION_ESP, VIGENTE FROM INVENTARIO_REPORTES WHERE CLAVE_REP IN ({})".format(",".join(f"'{id}'" for id in closest_ids))
        query_result=pd.read_sql_query(query, conn)
        num_rows=len(query_result)
        sub_df=query_result.sample(n=min(10, num_rows), random_state=42)
        # -- Eliminacion de memoria --
        del transformer_model
        gc.collect()
        torch.cuda.empty_cache()
        time.sleep(1)

    # -- Valores de respuesta en prompt --
    if num_rows>9:
        muestreo="La muestra es parcial (10 elementos), menciona claramente que se trata de un subconjunto y sugiere descargar el CSV completo para mayor alcance."
    else:
        muestreo="Esta es la información completa de la consulta, no es necesario descargar el CSV completo o reformular la consulta."

    if sub_df.empty:
        empty_message="No se encontró información relacionada con la consulta o la consulta está mal escrita, ofrece disculpas breves y recomienda revisar o reformular la consulta."
    else:
        empty_message=""

    # -- Creacion del prompt final --
    context=f"""{sub_df.to_string(index=False)}"""
    # -- Formato del prompt --
    final_prompt = f"""
    <history>
        {response_history}
    </history>
    <instruction>
        Eres un asistente especializado en análisis de datos financieros mexicanos. 
        Utiliza únicamente la sección <context> para responder a la consulta; allí se encuentra toda la información necesaria. 
        Sigue estas pautas al pie de la letra:
        
        1. **Tono y estilo**  
        - Profesional y cercano.  
        - Terminología propia del sector financiero en México.  
        - Evita estructuras tabulares; presenta todo en párrafos narrativos con sangría.

        2. **Uso del muestreo**  
        - {muestreo}

        3. **Información no encontrada**  
        - {empty_message}

        4. **Claridad y precisión**  
        - Al mencionar valores, indica siempre la unidad o el contexto (p.ej., montos en MXN, fechas en formato DD/MM/AAAA).  
        - Refuerza las conclusiones con referencias a campos específicos del <context> cuando sea necesario.

        5. **Estructura de la respuesta**  
        - Introducción breve que enmarque la pregunta.  
        - Desarrollo con uno o varios párrafos que expliquen los hallazgos.  
        - Cierre con una recomendación o resumen final.
    </instruction>
    <context>
        {context}
    </context>
    <query>
        {original_query}
    </query>
    """

    print(Style.BRIGHT+Fore.BLUE+final_prompt+Style.RESET_ALL)

    model="phi4"
    stream=chat(
        model=model,
        messages=[{
            'role': 'user',
            'content': final_prompt
        }],
        stream=True,
    )

    # Generacion de respuesta
    ollama_response=""
    print("Respuesta de Ollama:")
    for chunk in stream:
        print(chunk['message']['content'], end='', flush=True)
        ollama_response+=(chunk['message']['content'])
    print()

    query_history+=f"\n{response}\n"
    clean_response=ollama_response.split('</think>')[-1].replace('\n\n', '\n')
    response_history+=f"<user>\n{original_query}\n</user>\n<deepsek>\n{clean_response}\n</deepsek>\n"

    # Eliminacion de la memoria y escritura de documentos
    subprocess.run(['ollama', 'stop', model])
    time.sleep(1)