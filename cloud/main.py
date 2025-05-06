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
    format_prompt=clean_prompt(prompt)
    print(f"[bold green]{format_prompt}[/]")
    if prompt=="S": break
    fuentes_info=prompt.split(" ", 1)
    context=""
    if "A" in fuentes_info[0]:
        muestreo, empty_message, query, temp_context=sql_search(format_prompt)
        context+=f"{temp_context}\n"
    if "B" in fuentes_info[0]: print("JQL")
    if "C" in fuentes_info[0]: print("EMB")
    print(context)