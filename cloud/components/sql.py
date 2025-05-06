from ranker import choose_table
import subprocess
from ollama import chat
import time
import sqlite3
import pandas as pd
from components.commons import db_path, prompt_sql, sql_model, muest_empt, table_desc
from rich import print
conn=sqlite3.connect(db_path)

history_prompt=[]
history_table=[]
def sql_search(prompt):
    choose=choose_table(prompt)
    if choose=="GENERALES":
        prompt=f"{history_prompt[-1]}{prompt}"
        table=f"{history_table[-1]}"
        print(f"[bold purple]prompt:[/]\n[purple]{prompt}[/]")
        print(f"[bold orange1]tabla:[/]\n[orange1]{table}[/]")
        text=prompt_sql(prompt, table)
    else:
        table=table_desc[choose]
        prompt=f"{history_prompt[-1]}{prompt}"
        print(f"[bold purple]prompt:[/]\n[purple]{prompt}[/]")
        print(f"[bold orange1]tabla:[/]\n[orange1]{table}[/]")
        text=prompt_sql(prompt, table)


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
        print(f"[cyan]{chunk['message']['content']}[/cyan]", end='', flush=True)
        response+=(chunk['message']['content'])
    print()
    subprocess.run(['ollama', 'stop', sql_model])
    time.sleep(1)
    try:
        query_result=pd.read_sql_query(response, conn)    
        num_rows=len(query_result)
        context=query_result.sample(n=min(10, num_rows), random_state=42)
    except Exception as e:
        print(f"[red]error en la ejecucion de la consulta: {e}[/red]")
    muestreo, empty_message=muest_empt(num_rows, context)
    response=response.replace("\n", "")
    history_prompt.append(f"{prompt}\n{response}\n\n")
    history_table.append(table)
    
    return muestreo, empty_message, response, context.to_string(index=False)