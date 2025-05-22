from flask import Flask, Response, request, jsonify
from flask_cors import CORS
from rich import print
from ollama import chat
import json
import os
import uuid
from rich.status import Status
import logging
from components.sql import sql_search
import subprocess
from datetime import datetime
import time
import random

app = Flask(__name__)
logging.getLogger("werkzeug").setLevel(logging.ERROR)
CORS(app)

CHATS_FILE = "database/chats.json"
MESSAGES_FILE = "database/messages.json"

if not os.path.exists(CHATS_FILE):
    with open(CHATS_FILE, "w") as f:
        json.dump({}, f)

if not os.path.exists(MESSAGES_FILE):
    with open(MESSAGES_FILE, "w") as f:
        json.dump({}, f)

models = {
    "Qwen3": "qwen3:30b-a3b",
    "DeepSeek-R1": "deepseek-r1:7b",
    "Phi-4": "phi4:14b",
}


@app.route("/api/prompt", methods=["POST"])
def handle_prompt():
    data = request.get_json()
    id_chat = data.get("id_chat", "")
    id_user = data.get("id_message", "")
    prompt = data.get("prompt", "")
    role = data.get("role", "")
    model = data.get("model", "")
    sql = data.get("sql", "")
    jql = data.get("jql", "")
    emb = data.get("emb", "")
    flags = {
        "sql": bool(sql),
        "jql": bool(jql),
        "emb": bool(emb),
    }
    tasks = [name for name, enabled in flags.items() if enabled]

    print(f"[bold medium_spring_green]chat: {id_chat} prompt: {prompt} model: {model} tasks: {tasks}[/]")
    date = datetime.today().strftime("%Y-%m-%d")
    receivedDataJson = {"id": id_user, "role": role, "content": prompt, "date": date}
    with open(MESSAGES_FILE, "r+") as f:
        messages_data = json.load(f)
    history = ""
    for message in messages_data[id_chat]:
        if message["role"] == "user":
            history += f"<user>\n{message['content']}\n</user>\n"
        if message["role"] == "assistant":
            history += f"<assistant>\n{message['content']}\n</assistant>\n"
    print(f"[bold purple]{history}[/]")
    messages_data[id_chat].append(receivedDataJson)
    context = ""
    if "sql" in tasks:
        print("[bold yellow1]ejecutando sql_search[/]")
        context += sql_search(prompt)

    if len(tasks) != 0:
        final_prompt = f"""
        {history}
        <system>
        Eres un asistente especializado en análisis de datos financieros mexicanos. 
        Tu tarea es **leer** el resultado crudo de una consulta SQL y volcarlo en lenguaje natural, sin transformarlo ni resumirlo antes de que se formule la pregunta.
        </system>

        <user>
        —INICIO_RESULTADO—
        {context}
        —FIN_RESULTADO—

        Pregunta: {prompt}
        </user>

        <assistant>
        Tono y estilo:
        • Profesional y cercano  
        • Terminología propia del sector financiero en México  
        • Evita estructuras tabulares; presenta todo en párrafos narrativos con sangría  

        Claridad y precisión:
        • Al mencionar valores, indica siempre la unidad o el contexto (p.ej., montos en MXN, fechas en formato DD/MM/AAAA).  
        • Refuerza las conclusiones con referencias a campos específicos del resultado cuando sea necesario.  

        Estructura de la respuesta:
        1. Introducción breve que enmarque la pregunta.  
        2. Desarrollo en uno o varios párrafos con los hallazgos.  
        3. Cierre con una recomendación o resumen final.

        Ahora responde a la pregunta basándote **solo** en lo que está entre «—INICIO_RESULTADO—» y «—FIN_RESULTADO—» ya que esa es la respuesta correcta pese a que no lo parezca.
        </assistant>"""
    else:
        final_prompt = f"""
        {history}
        <user>
        {prompt}
        </user>
        """

    print(f"[bold plum2]{final_prompt}[/]")
    def generate():
        response = ""
        stream = chat(
            model=models[model],
            messages=[{"role": "user", "content": final_prompt}],
            stream=True,
        )
        with Status(
            "[bold light_steel_blue]generating response...[/]",
            spinner="dots",
            spinner_style="light_steel_blue bold",
        ) as status:
            for chunk in stream:
                response += chunk["message"]["content"]
                yield chunk["message"]["content"]
        subprocess.run(["ollama", "stop", models[model]])
        print(f"[bold sky_blue2]{response}[/]")

        id_llm = format(int(time.time() * 1000), "x") + "".join(
            random.choices("abcdefghijklmnopqrstuvwxyz0123456789", k=4)
        )
        date = datetime.today().strftime("%Y-%m-%d")
        receivedDataJson = {
            "id": id_llm,
            "role": "assistant",
            "content": response,
            "date": date,
        }
        messages_data[id_chat].append(receivedDataJson)
        with open(MESSAGES_FILE, "w+") as f:
            json.dump(messages_data, f, indent=4, ensure_ascii=False)

    return Response(generate(), mimetype="text/plain")


@app.route("/api/new_chat", methods=["POST"])
def handle_new_chat():
    data = request.get_json()
    user_id = data.get("user_id")

    if not user_id:
        return jsonify({"error": "user_id es requerido"}), 400

    chat_id = str(uuid.uuid4())
    date = datetime.today().strftime("%Y-%m-%d")
    chat = {"id": chat_id, "name": f"New Chat", "date": f"{date}"}

    with open(CHATS_FILE, "r+") as f:
        chats_data = json.load(f)
        if user_id not in chats_data:
            chats_data[user_id] = []
        chats_data[user_id].append(chat)
        f.seek(0)
        json.dump(chats_data, f, indent=2)
        f.truncate()

    with open(MESSAGES_FILE, "r+") as f:
        messages_data = json.load(f)
        messages_data[chat_id] = []
        f.seek(0)
        json.dump(messages_data, f, indent=2)
        f.truncate()

    return (
        jsonify({"message": "Chat creado exitosamente", "chat": chat}),
        200,
    )


@app.route("/api/get_chats", methods=["POST"])
def get_chats():
    data = request.get_json()
    user_id = data.get("user_id")

    if not user_id:
        return jsonify({"error": "user_id requerido"}), 400

    if not os.path.exists(CHATS_FILE):
        with open(CHATS_FILE, "w") as f:
            json.dump({}, f)

    with open(CHATS_FILE, "r") as f:
        chats_data = json.load(f)

    user_chats = chats_data.get(user_id, [])
    return jsonify({"user_id": user_id, "chats": user_chats})


@app.route("/api/get_messages", methods=["POST"])
def get_messages():
    data = request.get_json()
    print(f"[bold gold1]Consulta de chat: {data}[/]")
    id_chat = data.get("id_chat")

    if not id_chat:
        return jsonify({"error": "id_chat requerido"}), 400

    with open(MESSAGES_FILE, "r+") as f:
        messages_data = json.load(f)

    messages = messages_data[id_chat]

    return jsonify({"messages": messages})

@app.route("/api/delete_chat", methods=["POST"])
def delete_chat():
    data = request.get_json()
    user_id = data.get("user_id")

    if not user_id:
        return jsonify({"error": "user_id es requerido"}), 400

    chat_id = data.get("id_chat")

    if not chat_id:
        return jsonify({"error": "id_chat es requerido"}), 400 

    with open(CHATS_FILE, "r+") as f:
        chats_data = json.load(f)

    with open(MESSAGES_FILE, "r+") as f:
        messages_data = json.load(f)

    del messages_data[chat_id]
    chats_data[user_id] = [d for d in chats_data[user_id] if d["id"] != chat_id]

    with open(CHATS_FILE, "w+") as f:
        json.dump(chats_data, f, indent=4)

    with open(MESSAGES_FILE, "w+") as f:
        json.dump(messages_data, f, indent=4)

    return (
        jsonify({"message": "Chat Eliminado exitosamente", "chat": chat_id}),
        200,
    )
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
