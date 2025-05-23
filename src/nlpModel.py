#╔══════════════════════════╗ 
#║ Importacion de librerias ║ 
#╚══════════════════════════╝ 
import torch
from llm.ollama import GptClass
from log.logger import logger
import sqlite3
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
import warnings
import pandas as pd
import spacy
from spacy.language import Language
import re
import hjson
from sentence_transformers import SentenceTransformer
import pickle
import numpy as np
import time
import gc
# -- Desactivar advertencias de UserWarning --
warnings.filterwarnings("ignore", category=UserWarning, module="transformers.generation.utils")
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

class NlpModelClass:
    #╔══════════════════════╗ 
    #║ Historial para qBERT ║ 
    #╚══════════════════════╝
    prompts_history: str=""
    response_history: str=""
    #╔══════════════════════════╗ 
    #║ Caraga de modelos en CPU ║ 
    #╚══════════════════════════╝ 
    qbert_path="src/alphi/models/qbertsql"
    qbert=AutoModelForCausalLM.from_pretrained(
        qbert_path,
        torch_dtype=torch.bfloat16,
        local_files_only=True
    ).to("cpu")
    tokenizer_qbert=AutoTokenizer.from_pretrained(qbert_path)

    jina_path="src/alphi/models/jina-reranker-m0"
    jina=AutoModel.from_pretrained(
        jina_path, 
        trust_remote_code=True, 
        torch_dtype=torch.float16, 
        local_files_only=True
    ).to("cpu")
    tokenizer_jina=AutoTokenizer.from_pretrained(jina_path, trust_remote_code=True)

    #╔═══════════════════════════════════╗ 
    #║ Definicion de tablas y parametros ║ 
    #╚═══════════════════════════════════╝ 
    with open("database/tables.hjson", "r") as file:
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
        pairs=[[query, schema] for schema in NlpModelClass.table_docs.values()]
        
        NlpModelClass.jina.to(device)
        with torch.no_grad():
            scores=NlpModelClass.jina.compute_score(pairs, max_length=1024, doc_type="text")

        NlpModelClass.jina.to("cpu")
        torch.cuda.empty_cache()
        best_idx=int(torch.argmax(torch.tensor(scores, device="cpu")))
        return list(NlpModelClass.table_docs.keys())[best_idx]

    #╔═════════════════════╗ 
    #║ Ejecucion principal ║ 
    #╚═════════════════════╝ 
    @staticmethod
    def nlp_model(prompt, id_user, id_m, id_space):
        #╔══════════════╗ 
        #║ Carga de NLP ║ 
        #╚══════════════╝ 
        nlp=spacy.load("es_core_news_lg")
        with open("assets/replacements.hjson", "r") as file:
            replacements=hjson.load(file)
        with open("assets/embeddings.pkl", "rb") as f:
            data=pickle.load(f)

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
        original_prompt=prompt
        prompt=(nlp(prompt)).text
        flag=False
        if len(NlpModelClass.prompts_history)==0:
            flag=True
        if flag:
            NlpModelClass.prompts_history=prompt
        print(NlpModelClass.prompts_history)
        # -- Elegir tabla --
        table=NlpModelClass.choose_table(prompt)
        logger.system(f"[NlpModel/nlp_model] {table}")

        if table=="INVENTARIO_REPORTES":
            # -- Reemplazo de palabras --
            if " -- VIGENTE=0" not in prompt:
                prompt+=" -- (VIGENTE=1)"
        if not flag:
            NlpModelClass.prompts_history=NlpModelClass.prompts_history.replace(" -- VIGENTE=0", "").replace(" -- (VIGENTE=1)", "")
            NlpModelClass.prompts_history+=f"\n{prompt}"
        logger.user(f"[NlpModel/nlp_model] {prompt}")
        
        text=f"""
        ### Instruction:
        Your task is to generate valid SQLite3 to answer the following question, given a database schema.

        ### Input:
        Here is the database schema that the SQLite3 query will run on:
        {NlpModelClass.table_desc[table]}

        ### Question:
        {NlpModelClass.prompts_history}

        ### Response (use shorthand if possible):"""        
        
        # -- Generar SQL --
        inputs=NlpModelClass.tokenizer_qbert(text, return_tensors="pt")
        inputs={k: v.to(device) for k, v in inputs.items()}

        # -- Enviar a GPU --
        NlpModelClass.qbert.to(device)
        with torch.no_grad():
            generated_ids=NlpModelClass.qbert.generate(**inputs, max_new_tokens=256)
        NlpModelClass.qbert.to("cpu")
        torch.cuda.empty_cache()

        # -- Ejecutar SQL --
        response=NlpModelClass.tokenizer_qbert.decode(generated_ids[0], skip_special_tokens=True).replace(text, "").strip()
        NlpModelClass.prompts_history+=f"\n{response}\n"
        logger.system(f"[NlpModel/nlp_model] {response}")
        
        # -- Analizar query --
        count=0
        values=[]
        for text in replacements.values():
            values.append(text)
        values=list(set(values))
        for text in values:
            if f" {text} " in prompt:
                count+=1
        logger.system(f"[NlpModel/nlp_model] {count}")

        # -- Evaluacion del count y ejecucion de base de datos --
        db_path="database/bajaware.db"
        conn=sqlite3.connect(db_path)
        if count>=0:
            try:
                query_result=pd.read_sql_query(response, conn)
                num_rows=len(query_result)
                sub_df=query_result.sample(n=min(10, num_rows), random_state=42)
            except Exception as e:
                logger.system(f"[NlpModel/nlp_model] error en la ejecucion de la consulta: {e}")
                query_result=pd.DataFrame()
        elif table=="INVENTARIO_REPORTES" and count<0:
            transformer_model=SentenceTransformer("hiiamsid/sentence_similarity_spanish_es")
            embedding=transformer_model.encode(original_prompt.replace("reportes", "").replace("reporte", ""))
            # -- Buscamos el más cercano -- 
            distances=[]
            for clave, info in data.items():
                dist=np.linalg.norm(np.array(info["EMBEDDING"])-np.array(embedding))
                distances.append((dist, info))
            # -- Ordenamos por distancia -- 
            distances.sort(key=lambda x: x[0])
            # -- Guardamos los 10 mas cercasnos (solo CLAVE_REP) -- 
            closest_ids=[]
            closest_ids=[info["CLAVE_REP"] for _, info in distances[:10]]
            logger.system(f"[NlpModel/nlp_model] {closest_ids}")
            # -- Con los IDs creamos el dataframe -- 
            query="SELECT CLAVE_REP, CLAVE_ENTIDADREGULADA, REPORTE, CLAVE_PERIODO, DESCRIPCION_ESP, VIGENTE FROM INVENTARIO_REPORTES WHERE CLAVE_REP IN ({})".format(",".join(f"'{id}'" for id in closest_ids))
            query_result=pd.read_sql_query(query, conn)
            num_rows=len(query_result)
            sub_df=query_result.sample(n=min(10, num_rows), random_state=42)
            # -- Liberamos memoria -- 
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
        
        context=f"""{sub_df.to_string(index=False)}"""
        final_prompt=f"""
        <history>
            {NlpModelClass.response_history}
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
            {original_prompt}
        </query>
        """

        # -- Mostramos el prompt final --
        logger.system(f"[NlpModel/nlp_model] {final_prompt}")

        #╔═════════════════╗ 
        #║ Respuesta final ║ 
        #╚═════════════════╝ 
        nlp_response=GptClass.Ollama(final_prompt, "", id_user, id_m, id_space, prompt, query_result.to_string(index=False))
        logger.model(f"[NlpModel/nlp_model] Respuesta del modelo: {nlp_response}")
        if nlp_response[0]=="\n" and nlp_response[1]=="\n":
            nlp_response=nlp_response[2:]
        
        clean_response=nlp_response.split('</think>')[-1].replace('\n\n', '\n')
        NlpModelClass.response_history+=f"<user>\n\t{original_prompt}\n</user>\n<deepsek>\n\t{clean_response}\n</deepsek>\n"
        return nlp_response, query_result
