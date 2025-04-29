import pickle
import hjson

sql_model="qbert"
model="gemma3:12b"
db_path='../database/bajaware.db'
jina_path="../models/jina-reranker-m0"

with open("../assets/embeddings.pkl", "rb") as f:
    data=pickle.load(f)
with open("../assets/modules.hjson", "r") as file:
    modules=hjson.load(file)
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

def init_flag(query_history, ):
    flag=False
    if len(query_history)==0:
        flag=True
    return flag

def prompt_sql(prompt, table):
    text=f"""
    ### Task
    Generate a SQL query to answer [QUESTION]{prompt}[/QUESTION]

    ### Database Schema
    The query will run on a database with the following schema:
    {table}

    ### Answer
    Given the database schema, here is the SQL query that [QUESTION]{prompt}[/QUESTION]
    [SQL]
    """
    return text

def prompt_llm(history, muestreo, empty_message, original_prompt, context):
    final_prompt = f"""
    [HISTORY]
        {history}
    [/HISTORY]
    [INSTRUCTION]
        Eres un asistente especializado en análisis de datos financieros mexicanos. 
        Utiliza únicamente la sección [QUESTION][/QUESTION] para responder a la consulta; allí se encuentra toda la información necesaria. 
        Sigue estas pautas al pie de la letra:
        
        **Tono y estilo**  
        - Profesional y cercano.  
        - Terminología propia del sector financiero en México.  
        - Evita estructuras tabulares; presenta todo en párrafos narrativos con sangría.

        **Claridad y precisión**  
        - Al mencionar valores, indica siempre la unidad o el contexto (p.ej., montos en MXN, fechas en formato DD/MM/AAAA).  
        - Refuerza las conclusiones con referencias a campos específicos del <context> cuando sea necesario.

        **Estructura de la respuesta**  
        - Introducción breve que enmarque la pregunta.  
        - Desarrollo con uno o varios párrafos que expliquen los hallazgos.  
        - Cierre con una recomendación o resumen final.
        {muestreo}
        {empty_message}
    [/INSTRUCTION]
    [CONTEXT]
        {context}
    [/CONTEXT]
    [QUESTION]
        {original_prompt}
    [/QUESTION]
    """
    return final_prompt

def clean_chat(llm_response, llm_history, prompt):
    llm_response=llm_response.split('</think>')[-1].replace('\n\n', '\n')
    llm_history+=f"[USER]\n{prompt}\n[/USER]\n[ASSISTANT]\n{llm_response}\n[/ASSISTANT]\n"
    return llm_history

def muest_empt(num_rows, df):
    # -- Valores de respuesta en prompt --
    if num_rows>9:
        muestreo="""
        **Uso del muestreo**  
        - La muestra es parcial (10 elementos), menciona claramente que se trata de un subconjunto y sugiere descargar el CSV completo para mayor alcance.
        """
    else:
        muestreo="""
        **Uso del muestreo**  
        - Esta es la información completa de la consulta, no es necesario descargar el CSV completo o reformular la consulta.
        """
    if df.empty:
        empty_message="""
        **Información no encontrada**  
        - No se encontró información relacionada con la consulta o la consulta está mal escrita, ofrece disculpas breves y recomienda revisar o reformular la consulta.
        """
    else:
        empty_message=""
    return muestreo, empty_message