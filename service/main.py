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

app = Flask(__name__)
logging.getLogger('werkzeug').setLevel(logging.ERROR)
CORS(app)

SPACES_FILE = "database/spaces.json"
MESSAGES_FILE = "database/messages.json"

if not os.path.exists(SPACES_FILE):
    with open(SPACES_FILE, "w") as f:
        json.dump({}, f)

if not os.path.exists(MESSAGES_FILE):
    with open(MESSAGES_FILE, "w") as f:
        json.dump({}, f)


@app.route("/api/prompt", methods=["POST"])
def handle_prompt():
    data = request.get_json()
    prompt = data.get("prompt", "")
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

    print(f"[bold medium_spring_green]prompt: {prompt} model: {model} tasks: {tasks}[/]")
    context=""
    if "sql" in tasks: 
        print("[bold yellow1]🚀 ejecutando sql_search[/]")
        context+=sql_search(prompt)

    if len(tasks)!=0:
        prompt=f'''
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
        </assistant>'''
    
    def generate():
        stream = chat(
            model="qwen3:30b-a3b",
            messages=[{"role": "user", "content": prompt}],
            stream=True,
        )
        response=""
        with Status("[bold light_steel_blue]generating response...[/]", spinner="dots", spinner_style="light_steel_blue bold") as status:
            for chunk in stream:
                response+=(chunk['message']['content'])
                yield chunk["message"]["content"]
        subprocess.run(['ollama', 'stop', "qwen3:30b-a3b"])
        print(f"[bold sky_blue2]{response}[/]")
    
    return Response(generate(), mimetype="text/plain")


@app.route("/api/new_space", methods=["POST"])
def handle_new_space():
    data = request.get_json()
    user_id = data.get("user_id")

    if not user_id:
        return jsonify({"error": "user_id es requerido"}), 400

    space_id = str(uuid.uuid4())

    with open(SPACES_FILE, "r+") as f:
        spaces_data = json.load(f)
        if user_id not in spaces_data:
            spaces_data[user_id] = []
        spaces_data[user_id].append(space_id)
        f.seek(0)
        json.dump(spaces_data, f, indent=2)
        f.truncate()

    with open(MESSAGES_FILE, "r+") as f:
        messages_data = json.load(f)
        messages_data[space_id] = []
        f.seek(0)
        json.dump(messages_data, f, indent=2)
        f.truncate()

    return (
        jsonify({"message": "Espacio creado exitosamente", "space_id": space_id}),
        200,
    )


@app.route("/api/get_spaces", methods=["POST"])
def get_spaces():
    data = request.get_json()
    user_id = data.get("user_id")

    if not user_id:
        return jsonify({"error": "user_id requerido"}), 400

    if not os.path.exists(SPACES_FILE):
        with open(SPACES_FILE, "w") as f:
            json.dump({}, f)

    with open(SPACES_FILE, "r") as f:
        spaces_data = json.load(f)

    user_spaces = spaces_data.get(user_id, [])
    return jsonify({"user_id": user_id, "spaces": user_spaces})


if __name__ == "__main__":
    app.run(debug=True)