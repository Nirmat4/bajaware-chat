from nlpModel import clean_prompt
from commons import init_flag
from ranker import module_ranker
from sql import sql_search
from desciption import desc_search
import re
from commons import model, prompt_llm, clean_chat
from ollama import chat
import subprocess
import time

prompt_history, llm_history, menu="", "", ""
while menu!="S":
    prompt=input(">> ")
    if prompt=="S": break
    format_prompt=clean_prompt(prompt)
    flag=init_flag(prompt_history)
    if flag: prompt_history=format_prompt
    if not flag: prompt_history=re.sub(r' -- (VIGENTE=0|\(VIGENTE=1\))', '', prompt_history)+"\n"+format_prompt
    # -- seleccion de funcion jina --
    module=module_ranker(format_prompt)
    if module=="SQL":
        muestreo, empty_message, query, context=sql_search(prompt_history)
    if module=="JIR":
        print(module)
    if module=="DES":
        muestreo, empty_message, query, context=desc_search(format_prompt)
    prompt_history+=f"\n{query}\n"
    final_prompt=prompt_llm(llm_history, muestreo, empty_message, prompt, context)
    stream=chat(
        model=model,
        messages=[{
            'role': 'user',
            'content': final_prompt
        }],
        stream=True,
    )
    # -- Generacion de respuesta --
    llm_response=""
    for chunk in stream:
        print(chunk['message']['content'], end='', flush=True)
        llm_response+=(chunk['message']['content'])
    print()

    llm_history=clean_chat(llm_response, llm_history, prompt)

    # -- Eliminacion de la memoria --
    subprocess.run(['ollama', 'stop', model])
    time.sleep(1)