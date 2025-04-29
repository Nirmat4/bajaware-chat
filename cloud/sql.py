from ranker import choose_table
import subprocess
from ollama import chat
from colorama import Style, Fore
import time
import sqlite3
import pandas as pd
from commons import db_path, prompt_sql, sql_model, muest_empt
conn=sqlite3.connect(db_path)

def sql_search(prompt):
    table=choose_table(prompt)
    text=prompt_sql(prompt, table)
    # -- Generacion de consulta SQL --
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
    try:
        query_result=pd.read_sql_query(response, conn)    
        num_rows=len(query_result)
        context=query_result.sample(n=min(10, num_rows), random_state=42)
    except Exception as e:
        print(Fore.RED+f"error en la ejecucion de la consulta: {e}"+Style.RESET_ALL)
    muestreo, empty_message=muest_empt(num_rows, context)
    
    return muestreo, empty_message, response, context