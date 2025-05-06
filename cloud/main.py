from nlpModel import clean_prompt
from components.commons import init_flag
from components.sql import sql_search
from components.desciption import desc_search
import re
from components.commons import model, prompt_llm, clean_chat
from ollama import chat
import subprocess
import time
from rich import print
from rich.markup import escape

prompt_history, llm_history, menu="", "", ""
while menu!="S":
    print("[bold cyan]ingresa las fuentes de busqueda\nA - SQL\nB - JQL\nC - embeddings[/]")
    module=input(">> ")
    prompt=input(">> ")
    if prompt=="S": break
    format_prompt=clean_prompt(prompt)
    print(f"[bold bright_white]{format_prompt}[/]")
    flag=init_flag(prompt_history)
    if flag: prompt_history=format_prompt
    if not flag: prompt_history=re.sub(r' -- (VIGENTE=0|\(VIGENTE=1\))', '', prompt_history)+"\n"+format_prompt
    if "B" in module: print(module)
    if "C" in module: muestreo, empty_message, query, context=desc_search(format_prompt)
    if "A" in module: muestreo, empty_message, query, context=sql_search(prompt_history)
    prompt_history+=f"\n{query}\n"
    print(f"[bold magenta]hisotrial:[/]\n[magenta]{prompt_history}[/magenta]")
    final_prompt=prompt_llm(llm_history, muestreo, empty_message, prompt, context)
    print(f"[bold blue]prompt:[/]\n[blue]{escape(final_prompt)}[/blue]")
    stream=chat(
        model=model,
        messages=[{
            'role': 'user',
            'content': final_prompt
        }],
        stream=True,
    )
    # -- Generacion de respuesta --
    print(f"[bold light_steel_blue]respuesta llm:[/]")
    llm_response=""
    for chunk in stream:
        print(f"[light_steel_blue]{chunk['message']['content']}[/light_steel_blue]", end='', flush=True)
        llm_response+=(chunk['message']['content'])
    print()

    llm_history=clean_chat(llm_response, llm_history, prompt)

    # -- Eliminacion de la memoria --
    subprocess.run(['ollama', 'stop', model])
    time.sleep(1)