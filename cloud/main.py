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
    print("[bold cyan]ingresa tu consulta, recuerda colocar las siglas de las fuentes que quieres consultar\n[/][bold gold1]A - SQL\nB - JQL\nC - embeddings[/]")
    prompt=input(">> ")
    if prompt=="S": break
    prompt_secciones=prompt.split(" ", 1)
    fuentes_info=prompt_secciones[0]
    prompt=prompt_secciones[1]
    format_prompt=clean_prompt(prompt)
    print(f"[bold green]{format_prompt}[/]")
    context=""
    if "A" in fuentes_info:
        muestreo, empty_message, query, temp_context=sql_search(format_prompt)
        context+=f"{temp_context}\n"
    if "B" in fuentes_info: print("JQL")
    if "C" in fuentes_info: print("EMB")
    print(context)