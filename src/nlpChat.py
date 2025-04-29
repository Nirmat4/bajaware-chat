# src/main/chatbot.py


from googleapiclient.http import MediaIoBaseDownload # pip install google-api-python-client
from google.auth.transport.requests import Request # pip install google-auth
from google.oauth2.credentials import Credentials # pip install google-auth
from google_auth_oauthlib.flow import InstalledAppFlow # pip install google-auth-oauthlib
from googleapiclient.http import MediaFileUpload # pip install google-api-python-client
from flask import Blueprint, request, jsonify # pip install Flask[async]
from googleapiclient.discovery import build # pip install google-api-python-client
from alphi.nlpModel import NlpModelClass
from log.logger import logger
from google.oauth2 import service_account # pip install google-auth
#from llm.ollama import GptClass #
from datetime import datetime
from tabulate import tabulate
import pandas as pd
import requests
import os
import re
import uuid
import io
import json
import threading # Para evitar 'El bot no responde'
# from main.consultor_reportes import TfIdfProcessor, NewFunctions, DBConnection
from contextlib import redirect_stdout
from googleapiclient.errors import HttpError # pip install google-api-python-client

chatbot_bp = Blueprint('chatbot', __name__)
SCOPES = [
    'https://www.googleapis.com/auth/chat.messages',
    'https://www.googleapis.com/auth/chat.bot'
]
UPLOAD_SCOPES = ["https://www.googleapis.com/auth/chat.messages.create"]
processed_messages = ""
history = ""

pat_header = r'^(#{2,3})\s+(.+?)$'
pattern = r'(?<!\\)\*\*(.*?)(?<!\\)\*\*'

def replace_markdown_with_backticks(text):
    text = text.replace("```markdown", "")
    text = text.replace("```", "")
    return text
                        
def limpiar_titulos_tabla(text):
    # Elimina líneas que sean exactamente "Tabla X:" (con o sin espacios)
    return re.sub(r'^\s*Tabla \d+:\s*$', '', text, flags=re.MULTILINE)

def format_header(match):
    content = match.group(2).strip()
    return f'<h>' + content + '</h>'

def format_bold(match):
    content = match.group(1)
    return f'<b>' + content + '</b>'

def format_table(text):
    lines = text.split('\n')
    formatted_lines = []
    table_lines = []
    in_table = False
    table_count = 0
    max_rows = 3  # Máximo de filas a mostrar por tabla

    for idx, line in enumerate(lines):
        line_stripped = line.strip()
        if line_stripped.startswith('|') and line_stripped.endswith('|'):
            if not in_table:
                in_table = True
            table_lines.append(line_stripped)
        elif in_table:
            in_table = False
            if table_lines:
                headers = []
                table_data = []
                for i, table_line in enumerate(table_lines):
                    if not table_line.strip():
                        continue
                    cells = [cell.strip() for cell in table_line.strip('|').split('|')]
                    if i == 0:
                        headers = cells
                    elif i == 1 and all(set(cell) <= set('- ') for cell in cells):
                        continue
                    else:
                        if len(cells) == len(headers):
                            table_data.append(cells)
                if headers and table_data:
                    table_count += 1
                    truncated = False
                    if len(table_data) > max_rows:
                        table_data = table_data[:max_rows]
                        truncated = True
                    tabla_str = tabla_como_listas(headers, table_data)
                    formatted_lines.append(f"<pre>{tabla_str}</pre>")
                    if truncated:
                        formatted_lines.append(f"<i>Mostrando solo las primeras {max_rows} filas.</i>")
                table_lines = []
            formatted_lines.append(line)
        else:
            formatted_lines.append(line)
    # Procesar si el texto termina con una tabla
    if table_lines:
        headers = []
        table_data = []
        for i, table_line in enumerate(table_lines):
            if not table_line.strip():
                continue
            cells = [cell.strip() for cell in table_line.strip('|').split('|')]
            if i == 0:
                headers = cells
            elif i == 1 and all(set(cell) <= set('- ') for cell in cells):
                continue
            else:
                if len(cells) == len(headers):
                    table_data.append(cells)
        if headers and table_data:
            table_count += 1
            truncated = False
            if len(table_data) > max_rows:
                table_data = table_data[:max_rows]
                truncated = True
            tabla_str = tabla_como_listas(headers, table_data)
            formatted_lines.append(f"<pre>{tabla_str}</pre>")
            if truncated:
                formatted_lines.append(f"<i>Mostrando solo las primeras {max_rows} filas.</i>")
    return '\n'.join(formatted_lines)

def tabla_como_listas(headers, rows):
    lines = []
    for idx, row in enumerate(rows, 1):
        lines.append(f"- Fila {idx}:")
        for h, v in zip(headers, row):
            lines.append(f"  - {h}: {v}")
    return "\n".join(lines)

@chatbot_bp.route('/oauth2callback')
def oauth2callback():
    from flask import request, redirect
    code = request.args.get('code')
    state = request.args.get('state', '{}')
    
    if not code:
        return "Código no proporcionado", 400

    try:
        # Recuperar el user_id del state
        state_data = json.loads(state)
        user_id = state_data.get('user_id')
        
        # Si no hay user_id en el state, usar un valor por defecto
        if not user_id:
            user_id = "default_user"
            logger.warning("No se encontró user_id en el estado, usando default_user")
            
        # Directorio para tokens de usuario
        tokens_dir = "user_tokens"
        os.makedirs(tokens_dir, exist_ok=True)
        
        # Ruta al token específico del usuario
        token_path = os.path.join(tokens_dir, f"token_{user_id}.json")
        client_secrets_path = 'client_secret_783922470241.json'
        
        flow = InstalledAppFlow.from_client_secrets_file(
            client_secrets_path, UPLOAD_SCOPES)
        flow.redirect_uri = "https://weasel-fast-repeatedly.ngrok-free.app/oauth2callback"
        flow.fetch_token(code=code)
        creds = flow.credentials
        
        # Guardar en el archivo específico del usuario
        with open(token_path, 'w') as token:
            token.write(creds.to_json())
            
        return "Autorización completada. Ahora puedes cerrar esta ventana y volver a exportar el CSV en la conversación de Google Chat."
    except Exception as e:
        logger.error(f"Error en el callback OAuth: {str(e)}", exc_info=True)
        return f"Error en la autorización: {str(e)}", 500


class Chatbot:
    credentials_file = "bwchatbot-e4fcd6ddebff.json"
    PROFILE_IMG = "https://pbs.twimg.com/profile_images/994625197360664576/bGJXyrw9_400x400.jpg"
    export_cache = {}
    app = None

    global history
    def __init__(self, credentials_file=credentials_file):
        logger.info(f"Inicializando Chatbot con credenciales de {credentials_file}")
        self.credentials_file = credentials_file
        logger.info("Configurando rutas del chatbot")
        self.setup_routes()
        logger.info("Chatbot inicializado correctamente")

    def setup_routes(self):
        chatbot_bp.route('/webhook', methods=['POST'])(self.handle_webhook)
        logger.info("Ruta '/webhook' registrada correctamente.")

    def handle_webhook(self):
        """Maneja todos los eventos entrantes al webhook"""
        logger.info("[Chatbot/handle_webhook] Evento entrante recibido en /webhook")
        data = request.get_json()
        event_type = data.get('type')
        logger.debug(f"[Chatbot/handle_webhook] Tipo de evento: {event_type}")
        logger.debug(f"[Chatbot/handle_webhook] Datos recibidos: {json.dumps(data, indent=2)}")

        if event_type == 'MESSAGE':
            # Procesar mensajes nuevos o comandos slash
            return self.on_message(data)
        elif event_type == 'CARD_CLICKED':
            # Procesar clics en botones de tarjetas
            return self.on_card_clicked(data)
        else:
            logger.warning(f"[Chatbot/handle_webhook] Tipo de evento no manejado: {event_type}")
            return '', 204 # Ignorar otros tipos de eventos

    
    def on_message(self, data):
        """Maneja las solicitudes entrantes al webhook"""
        logger.info("[Chatbot/on_message] Solicitud entrante recibida en /webhook")
        global processed_messages
        
        try:
            data = request.get_json()
            message_id = str(data['message']['name'])
            logger.warning(f"[Chatbot/on_message/received] Solicitud recibida: {message_id}")

            if message_id in processed_messages:
                logger.warning(f"[Chatbot/on_message/ignore] Solicitud ignorada: {message_id}")
                return jsonify({"status": "ignored"}), 204  # Cambiado a 204 para respuesta inmediata

            processed_messages = message_id
            logger.info(f"[Chatbot/on_message/success] Solicitud recibida: {data}")

            # Responder inmediatamente con 204 para evitar el mensaje "Bot no responde"
            threading.Thread(target=self.process_message_background, args=(data,)).start()  # Iniciar procesamiento en segundo plano
            return jsonify({"status": "processing"}), 204 # Respuesta inmediata con 204

        except Exception as e:
            logger.error(f"[Chatbot/on_message] Error al procesar la solicitud inicial: {str(e)}")
            return jsonify({"text": f"Error al procesar la solicitud inicial: {str(e)}"}), 500

    def process_message_background(self, data):
        """Procesa el mensaje en segundo plano después de responder con 204"""
        logger.info("[Chatbot/process_message_background] Iniciando procesamiento en segundo plano")
        global history
        try:
            if 'message' in data:
                prompt = data['message'].get('text', 'Sin texto')
                space_name = data['message'].get('space', {}).get('name')
                space_name_string = data['message'].get('space', {}).get('displayName', 'Chat directo')
                user_name_string = data['message'].get('sender', {}).get('displayName')
                user_name = data['message'].get('sender', {}).get('name')
                id_user = data['message'].get('sender', {}).get('name', 'Usuario desconocido').replace("users/", "")
                id_space = data['message'].get('space', {}).get('name', 'Chat directo').replace("spaces/", "")
                id_message=str(data['message']['name']).replace(f"/messages/", "").replace(f"spaces/{id_space}", "")
                logger.warning(f"[process_message_background/data]: Mensaje recibido de {user_name_string} en el espacio {space_name_string}:\n{prompt}")

                # Verificar si hay archivos adjuntos
                attachment_content = None
                attachments = data['message'].get('attachment', [])  # Obtener la lista de adjuntos

                if attachments:
                    logger.info(f"[Chatbot/process_message_background] Se encontraron {len(attachments)} archivos adjuntos.")
                    logger.info(f"[Chatbot/process_message_background] Los archivos adjuntos son: {attachments}")
                    for attachment in attachments:
                        attachment_content, attachment_name = self.get_attachment_content(attachment)
                        if attachment_content:
                            logger.info("Contenido del archivo adjunto obtenido correctamente.")
                            break  # Procesar solo el primer archivo adjunto válido
                else:
                    attachment_content = ''
                    attachment_name = 'Sin archivo adjunto'

                members = self.get_space_members(space_name)
                # Determinar el método a llamar basado en el comando
                command_name = data['message'].get('annotations', [{}])[0].get('slashCommand', {}).get('commandName', '')
                
                # --- Lógica específica del RAG --- 
                if command_name == "/rag":
                    prompt_cleaned = prompt.replace("/rag", "").strip()
                    edit_success, response_text = self.rag_message_processor(prompt_cleaned, id_user, id_space, user_name_string, space_name_string)
                    if not edit_success:
                        logger.error(f"[process_message_background] Error al editar la tarjeta: {response_text}")
                # --- Fin lógica específica del RAG --

                elif command_name == "/chat":
                    prompt_cleaned = prompt.replace("/chat", "").strip()
                    edit_success, response_text = self.chat_message_to_nlp(prompt_cleaned, id_user, id_message, id_space, user_name_string, space_name_string, data)
                    if not edit_success:
                        logger.error(f"[process_message_background] Error en chat_message_to_nlp: {response_text}")
                else:
                    logger.warning(f"[process_message_background] Comando no reconocido: {command_name}, ejecutando /rag")
                    edit_success, response_text = self.rag_message_processor(prompt, id_user, id_space, user_name_string, space_name_string)
                    if not edit_success:
                        logger.error(f"[process_message_background] Error al editar la tarjeta: {response_text}")

                logger.warning(f"[process_message_background] El nombre del espacio es: {space_name}")
                # No se envía el mensaje aquí directamente, se edita la tarjeta de carga

            else:
                logger.warning("[process_message_background] Solicitud recibida pero no se encontró el campo 'message'.")


        except Exception as e:
            logger.error(f"[process_message_background] Error al procesar la solicitud en segundo plano: {str(e)}")


    def rag_message_processor(self, prompt, id_user, id_space, user_name, space_name):
        """
        Procesa consultas usando el consultor de reportes

        Parámetros:
            prompt: str - La consulta del usuario
            id_user: str - El ID del usuario
            id_space: str - El ID del espacio
            user_name: str - El nombre del usuario
        """
        logger.info(f"[Chatbot/rag_message_processor] Procesando RAG para: {prompt}")
        try:
            # --- Paso 1: Enviar mensaje de carga inicial ---
            rag_loading_response = {
                "cardsV2": [{
                    "card": {
                        "header": {
                            "title": "⌛ Bajaware Bot - Consultor de Reportes",
                            "subtitle": "Iniciando consulta...",
                            "imageUrl": self.PROFILE_IMG
                        },
                        "sections": [{
                            "widgets": [{
                                "textParagraph": {"text": "<b>Estado:</b> Preparando consulta..."}
                            }]
                        }]
                    }
                }]
            }
            success, loading_response_data = self.send_message(f"spaces/{id_space}", rag_loading_response)
            if not success:
                logger.error(f"[Chatbot/rag_message_processor] Error al enviar el mensaje de carga: {loading_response_data}")
                return False, loading_response_data

            loading_message_name = loading_response_data.get('name')
            logger.info(f"[Chatbot/rag_message_processor] Mensaje de carga enviado: {loading_message_name}")

            # --- Paso 2: Actualizar tarjeta a "Procesando" ---
            processing_card_content = {
                "cardsV2": [{
                    "card": {
                        "header": {
                            "title": "🔄 Bajaware Bot - Consultor de Reportes",
                            "subtitle": "Procesando tu consulta...",
                            "imageUrl": self.PROFILE_IMG
                        },
                        "sections": [{
                            "widgets": [{
                                "textParagraph": {
                                    "text": f"<b>Usuario:</b> {user_name}<br><b>Consulta:</b> {prompt}<br><b>Estado:</b> Analizando y buscando datos..."
                                }
                            }]
                        }]
                    }
                }]
            }
            # Se intenta editar el mensaje de carga sin bloquear si falla.
            self.edit_message(loading_message_name, processing_card_content)

            # --- Paso 3: Ejecutar el modelo NLP ---
            output_text = "Error: No se puede obtener respuesta del consultor"
            df_resultado = None
            tipo_consulta = None

            logger.info(f"[Chatbot/rag_message_processor] Ejecutando nlp_model")
            
            output_text_funct, df_resultado = NlpModelClass.nlp_model(prompt, id_user, '', id_space)
            #output_text_funct = """
            output_text_funct_test = """
            Respuesta del modelo NLP
            
            
            | Columna1 | Columna2 |
            |---------|----------|
            | Dato1   | Dato3    |
            | Dato2   | Dato4    |
            
            
            | Columna1 | Columna2 |
            |---------|----------|
            | Dato1   | Dato3    |
            | Dato2   | Dato4    |
            
            """
            
            """ try:
                df_resultado = GptClass.Ollama("System: Escribe ÚNICAMENTE un JSON que pueda ser leído por pd.DataFrame para archivo csv con una estructura de columnas y filas\n" +  f"User: {output_text_funct}\nlang=es", "", id_user, "", id_space, prompt, "")
                logger.warning(f"[Chatbot/rag_message_processor] DataFrame obtenido: {df_resultado}")
                df_resultado = df_resultado.replace("```json", "").replace("```", "")
                df_dict = json.loads(df_resultado)
                df_resultado = pd.DataFrame(df_dict)
            except Exception as e:
                logger.error(f"[Chatbot/rag_message_processor] Error al convertir el DataFrame, devolviendo un dataframe harcodeado... {e}")
                df_resultado = pd.DataFrame({"Columna1": ["Dato1", "Dato2"], "Columna2": ["Dato3", "Dato4"]}) """
            
            #df_resultado = pd.DataFrame({"Columna1": ["Dato1", "Dato2"], "Columna2": ["Dato3", "Dato4"]})
            
            
            output_text_funct = limpiar_titulos_tabla(output_text_funct)
            
            output_text_funct = format_table(output_text_funct)
            
            lines = output_text_funct.split('\n')
            formatted_lines = []
            
            for line in lines:
                line = re.sub(pat_header, format_header, line)
                line = re.sub(pattern, format_bold, line)
                formatted_lines.append(line)
            formatted_output = '\n'.join(formatted_lines)
                
            formatted_output = formatted_output.replace("─", "-")\
                                                  .replace("📊", "")\
                                                  .replace("📋", "")\
                                                  .replace("📌", "")\
                                                  .replace("✅", "")\
                                                  .replace("🔌", "")\
                                                  .replace("🔍", "")
            formatted_output = replace_markdown_with_backticks(formatted_output)
            max_len = 3800
            if len(formatted_output) > max_len:
                logger.warning("[Chatbot/rag_message_processor] Cantidad máxima de caracteres excedida, truncando contenido...")
                formatted_output = formatted_output[:max_len] + "... (contenido truncado)"

            # --- Paso 4: Construir la tarjeta final de respuesta ---
            response_card_content = {
                "cardsV2": [{
                    "card": {
                        "header": {
                            "title": "✅ Bajaware Bot - Consultor de Reportes",
                            "subtitle": f"Resultado para: {user_name}",
                            "imageUrl": self.PROFILE_IMG
                        },
                        "sections": [
                            {"widgets": [{"textParagraph": {"text": f"<b>Consulta:</b> {prompt}"}}]},
                            {"widgets": [{"textParagraph": {"text": f"<br>{formatted_output}"}}]}
                        ]
                    }
                }]
            }

            # --- Paso 5: Añadir botón de exportación si existe DataFrame ---
            if df_resultado is not None and not df_resultado.empty:
                # tipo_consulta = GptClass.Ollama(f"System: Escribe ÚNICAMENTE una palabra que describa el tipo de contenido que se exportará al csv usando la respuesta de un modelo de lenguaje, User: {formatted_output}, Lang=es", "", id_user, "", id_space, prompt, "")
                tipo_consulta = "reporte"
                export_id = str(uuid.uuid4())  # Genera un ID único para la exportación
                self.export_cache[export_id] = df_resultado

                export_button = {
                    "buttonList": {
                        "buttons": [{
                            "text": f"📊 Exportar {tipo_consulta} a CSV",
                            "onClick": {
                                "action": {
                                    "function": "exportar_rag_csv",
                                    "parameters": [
                                        {"key": "tipo_consulta", "value": tipo_consulta},
                                        {"key": "consulta_original", "value": prompt},
                                        {"key": "export_id", "value": export_id},
                                        {"key": "user_name", "value": user_name},
                                        {"key": "id_space", "value": id_space}
                                    ]
                                }
                            }
                        }]
                    }
                }
                
                response_card_content["cardsV2"][0]["card"]["sections"].append({
                    "widgets": [export_button]
                })

            # --- Paso 6: Editar mensaje final con la tarjeta de respuesta ---
            edit_success, edit_response = self.edit_message(loading_message_name, response_card_content)
            if not edit_success:
                logger.error(f"[Chatbot/rag_message_processor] Error al actualizar mensaje final: {edit_response}")
                # Si falla la edición se envía un mensaje nuevo informando del problema
                self.send_message(f"spaces/{id_space}", {"text": "Error al actualizar la tarjeta de resultados. Contacta al administrador."})
                return False, edit_response

            return True, "Consulta RAG procesada y tarjeta actualizada."

        except Exception as e:
            logger.error(f"[Chatbot/rag_message_processor] Error GRANDE procesando consulta RAG: {str(e)}", exc_info=True)
            # En caso de error, se intenta actualizar la tarjeta con un mensaje de error
            error_card_content = {
                "cardsV2": [{
                    "card": {
                        "header": {
                            "title": "❌ Bajaware Bot - Error en Consulta RAG",
                            "subtitle": "No se pudo completar la solicitud",
                            "imageUrl": self.PROFILE_IMG
                        },
                        "sections": [{
                            "widgets": [{
                                "textParagraph": {
                                    "text": f"<b>Consulta:</b> {prompt}<br><b>Error:</b> Ocurrió un error inesperado durante el procesamiento.<br><i>Detalle: {str(e)}</i>"
                                }
                            }]
                        }]
                    }
                }]
            }
            self.edit_message(loading_message_name, error_card_content)
            return False, f"Error al procesar consulta RAG: {str(e)}"

    def on_card_clicked(self, data):
        """Maneja los eventos de clic en botones de tarjetas."""
        logger.info("[Chatbot/on_card_clicked] Procesando evento CARD_CLICKED")
        action = data.get('common', {}).get('invokedFunction')
        parameters = {p['key']: p['value'] for p in data.get('action', {}).get('parameters', [])}
        user_name = data.get('user', {}).get('displayName', 'Usuario desconocido')
        id_space = data.get('space', {}).get('name', 'Desconocido').replace("spaces/", "")
        
        # Obtener el ID del usuario que hizo clic
        id_user = data.get('user', {}).get('name', '').replace("users/", "")
        
        logger.info(f"[Chatbot/on_card_clicked] Acción: {action}, Usuario: {id_user}, Espacio: {id_space}")
        
        if action == "exportar_rag_csv":
            # Definir loading_msg_name antes de usarlo
            loading_msg_name = None
            
            tipo_consulta = parameters.get('tipo_consulta')
            consulta_original = parameters.get('consulta_original')
            export_id = parameters.get('export_id')
            df_to_export = self.export_cache.get(export_id)
            
            if df_to_export is None:
                self.send_message(f"spaces/{id_space}", {"text": "No se encontró DataFrame para exportar. Quizá expiró la caché o fue un ID inválido."})
                return jsonify({})
        
            export_success, export_msg = self.export_to_csv_and_send(
                tipo_consulta=tipo_consulta,
                datos_df=df_to_export,
                id_space=id_space,
                prompt=consulta_original,
                id_user=id_user  # Pasar el ID del usuario
            )
            
            return jsonify({}) # Devolver acción para actualizar tarjeta (o similar)


    def chat_message_to_nlp(self, prompt, id_user, id_message, id_space, user_name, space_name, event_data):
        """
        Procesa mensajes de usuario para el modelo de lenguaje
        
        Parámetros:
            prompt: str - El mensaje del usuario
            id_user: str - El ID del usuario
            id_space: str - El ID del espacio
            user_name: str - El nombre del usuario
            space_name: str - El nombre del espacio
            event_data: dict - Los datos del evento
        
        Devuelve:
            True, response_text - Si el mensaje se envió correctamente
            False, error_msg - Si hay un error
        """
        logger.info(f"[Chatbot/chat_message_to_nlp] NLP para: {prompt}")
        loading_message_name = None

        # Si es un comando slash, probablemente ya enviamos un mensaje de carga
        # Necesitamos obtener su nombre para editarlo. El evento original 'data' lo tiene.
        # Nota: Esto asume que la llamada viene de process_message_background después de enviar carga.
        #       Si se llama directamente, puede que no haya mensaje de carga.

        # --- Mensaje de carga (si no se envió antes) ---
        # Comprobar si ya existe un mensaje para editar (podría venir de un flujo anterior)
        # Por simplicidad, enviaremos siempre uno nuevo para NLP y lo editaremos.
        nlp_loading_response = {
             "cardsV2": [{
                 "card": {
                     "header": {"title": "⌛ Bajaware Bot (NLP)", "subtitle": "Procesando tu mensaje...", "imageUrl": self.PROFILE_IMG},
                     "sections": [{"widgets": [{"textParagraph": {"text": "<b>Estado:</b> Consultando modelo de lenguaje..."}}]}]
                 }
             }]
         }
        success, loading_response_data = self.send_message((f"spaces/{id_space}"), nlp_loading_response)
        if not success:
             logger.error(f"[Chatbot/chat_message_to_nlp] Error al enviar mensaje de carga NLP: {loading_response_data}")
             return False, loading_response_data
        loading_message_name = loading_response_data.get('name')
        logger.info(f"[Chatbot/chat_message_to_nlp] Mensaje de carga NLP enviado: {loading_message_name}")


        # --- Llamada al modelo NLP ---
        logger.warning(f"[Chatbot/chat_message_to_nlp/nlp_model] Enviando entrada al modelo de NLP")
        # Asegúrate que NlpModelClass.nlp_model NO intente enviar/editar mensajes por sí mismo
        # Debe solo devolver el texto de la respuesta.
        try:
             # Pasamos id_message=None porque no lo estamos usando directamente aquí
            nlp_response_text = NlpModelClass.nlp_model(prompt, id_user, None, id_space) # Ajustar firma si es necesario
             # Test con modelo de lenguaje
            # nlp_response_text = GptClass.Ollama(f"System: Eres un modelo de lenguaje para debuguear un sistema de chatbot dentro de google chat, tu respuesta siempre será autoconclusiva, el texto generado estará estilizado en formato markdown. User: {prompt},  Lang=es", "",  id_user, id_message, id_space, prompt, "")
            
            nlp_response_text = limpiar_titulos_tabla(nlp_response_text)
            nlp_response_text = format_table(nlp_response_text)
            
            lines = nlp_response_text.split('\n')
            formatted_lines = []
            
            for line in lines:
                line = re.sub(pat_header, format_header, line)
                line = re.sub(pattern, format_bold, line)
                formatted_lines.append(line)
            nlp_response_text = '\n'.join(formatted_lines)
            nlp_response_text = nlp_response_text.replace("─", "-")\
                                                  .replace("📊", "")\
                                                  .replace("📋", "")\
                                                  .replace("📌", "")\
                                                  .replace("✅", "")\
                                                  .replace("🔌", "")\
                                                  .replace("🔍", "")
            nlp_response_text = replace_markdown_with_backticks(nlp_response_text)
            max_len = 3800
            if len(nlp_response_text) > max_len:
                logger.warning("[Chatbot/chat_message_to_nlp] Cantidad máxima de caracteres excedida, truncando contenido...")
                nlp_response_text = nlp_response_text[:max_len] + "... (contenido truncado)"
            
        except Exception as nlp_error:
            logger.error(f"[Chatbot/chat_message_to_nlp/nlp_model/error] Error al llamar a nlp_model: {nlp_error}", exc_info=True)
            nlp_response_text = f"Error al obtener respuesta del modelo NLP: {nlp_error}"

        logger.debug(f"[Chatbot/chat_message_to_nlp/nlp_response_text] Respuesta del modelo NLP: {nlp_response_text}")
        
        
        # --- Construir y editar tarjeta final ---
        final_card_content = {
            "cardsV2": [{
                "card": {
                    "header": {
                        "title": "💬 Bajaware Bot (Respuesta NLP)",
                        "imageUrl": self.PROFILE_IMG,
                        "subtitle": f"Para: {user_name}"
                    },
                    "sections": [{
                        "widgets": [{
                            "textParagraph": {
                                "text": f"<b>Mensaje:</b> {prompt}<br><br><b>{nlp_response_text}</b>"
                            }
                        }]
                    }]
                }
            }]
        }

        if loading_message_name:
            logger.info(f"[Chatbot/chat_message_to_nlp/edit_message/info] Editando mensaje de carga NLP: {loading_message_name}")
            edit_success, edit_response = self.edit_message(loading_message_name, final_card_content)
            logger.info(f"[Chatbot/chat_message_to_nlp/edit_message/info] Mensaje de carga NLP editado: {edit_success}")
            return True, "Mensaje de carga NLP editado."
        elif not edit_success:
            logger.error(f"[Chatbot/chat_message_to_nlp/edit_message/error] Error al editar mensaje NLP: {edit_response}")
            # Intentar enviar como mensaje nuevo si falla la edición
            self.send_message(f"spaces/{id_space}", {"text": f"Respuesta NLP:\n{nlp_response_text}\n\n(Error al actualizar tarjeta)"})
            return False, edit_response
        else:
            # Si no había mensaje de carga, enviar como nuevo
            logger.warning(f"[Chatbot/chat_message_to_nlp/send_message/warning] Enviando mensaje final NLP")
            send_success, send_response = self.send_message(f"spaces/{id_space}", final_card_content)
            logger.info(f"[Chatbot/chat_message_to_nlp/send_message/info] Mensaje final NLP enviado: {send_success}")
            if not send_success:
                logger.error(f"[Chatbot/chat_message_to_nlp/send_message/error] Error al enviar mensaje final NLP: {send_response}")
                return False, send_response
            
            return True, "Mensaje NLP enviado."

    def send_message(self, space_name, message):
        """
        Se envía el mensaje al espacio
        """
        access_token = self.get_access_token()
        if not access_token:
            logger.error("[Chatbot/send_message] Error: No se pudo obtener el token de acceso.")
            return False, "Error: No se pudo obtener el token de acceso"

        url = f"https://chat.googleapis.com/v1/{space_name}/messages"
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json"
        }
        logger.debug(f"[Chatbot/send_message] Enviando a {url} con payload: {json.dumps(message)}")
        response = requests.post(url, headers=headers, json=message, timeout=300)
        logger.debug(f"[Chatbot/send_message] Respuesta de la api: {response.status_code}")
        if response.status_code == 200:
            logger.info(f"[Chatbot/send_message] Mensaje enviado con éxito a {space_name}: {response.json()}")
            return True, response.json() # Devuelve el json de la respuesta para obtener el nombre del mensaje
        else:
            error_msg = f"[Chatbot/send_message] Error al enviar el mensaje: {response.status_code}"
            logger.error(f"{error_msg}")
            return False, error_msg

    def edit_message(self, message_name, message_content):
        """
        Edita un mensaje en un espacio
        """
        access_token = self.get_access_token()
        if not access_token:
            logger.error("[Chatbot/edit_message] Error: No se pudo obtener el token de acceso.")
            return False, "Error: No se pudo obtener el token de acceso"

        url = f"https://chat.googleapis.com/v1/{message_name}"
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json"
        }

        # Determinar qué actualizar y construir el payload/mask dinámicamente
        payload = {}
        update_mask = []

        if "text" in message_content:
            payload["text"] = message_content["text"]
            update_mask.append("text")
        elif "cardsV2" in message_content: # Usar elif para priorizar texto si ambos estuvieran presentes
            payload["cardsV2"] = message_content["cardsV2"]
            update_mask.append("cardsV2")
        else:
            # Si no hay ni texto ni cards, no se puede editar (esto previene el error)
            error_msg = "[Chatbot/edit_message] Error: El contenido del mensaje para editar está vacío (no contiene 'text' ni 'cardsV2')."
            logger.error(error_msg)
            return False, error_msg

        if not update_mask:
             error_msg = "[Chatbot/edit_message] Error: No se pudo determinar qué campo actualizar."
             logger.error(error_msg)
             return False, error_msg

        params = {
            "updateMask": ",".join(update_mask) # Unir si hubiera múltiples máscaras en el futuro
        }

        # Añadir el cuerpo del mensaje (text o cardsV2) a payload si se usa PATCH
        # payload.update(message_content) # <- Esto podría sobrescribir mal, usar el payload construido

        logger.info(f"[Chatbot/edit_message] Editando mensaje {message_name} con payload: {payload} y mask: {params['updateMask']}")

        # Usar PATCH para actualizaciones parciales
        response = requests.patch(url, headers=headers, json=payload, params=params, timeout=450)

        if response.status_code == 200:
            logger.info(f"[Chatbot/edit_message] Mensaje editado con éxito: {response.json()}")
            return True, response.json()
        else:
            # Mantener el log detallado del error de la API
            error_msg = f"[Chatbot/edit_message] Error al editar el mensaje: {response.status_code} - {response.text}"
            logger.error(f"{error_msg}")
            return False, error_msg


    def get_access_token(self):
        """
        Obtiene el token de acceso para el bot
        """
        try:

            bot_scopes = [
                'https://www.googleapis.com/auth/chat.bot',
                'https://www.googleapis.com/auth/chat.messages'
            ]


            credentials = service_account.Credentials.from_service_account_file(
                self.credentials_file,
                scopes=bot_scopes # <-- Usar la lista de scopes actualizada
            )
            credentials.refresh(Request())
            logger.info("Token de acceso obtenido correctamente.")
            return credentials.token
        except Exception as e:
            logger.error(f"[Chatbot/get_access_token] Error al obtener el token de acceso: {str(e)}")
            return None

    def build_chat_service(self):
        """
        Construye y devuelve el servicio de Google Chat utilizando las credenciales.
        """
        try:
            logger.info(f"[Chatbot/build_chat_service] Construyendo servicio de chat")
            credentials = service_account.Credentials.from_service_account_file(
                self.credentials_file,
                scopes=['https://www.googleapis.com/auth/chat.messages', 'https://www.googleapis.com/auth/chat.bot']
            )
            service = build('chat', 'v1', credentials=credentials)
            logger.info("[Chatbot/build_chat_service] Servicio de chat construido correctamente")
            return service
        except Exception as e:
            logger.critical(f"[Chatbot/build_chat_service] Error catastrófico al construir el servicio: {str(e)}, consulte al administrador del sistema")
            return None

    def get_user_credentials(self, user_id):
        """Obtiene credenciales de usuario para operaciones que requieren autenticación de usuario"""
        creds = None
        # Usar un directorio para almacenar tokens de usuarios
        tokens_dir = "user_tokens"
        os.makedirs(tokens_dir, exist_ok=True)
        
        # Archivo de token específico para este usuario
        token_path = os.path.join(tokens_dir, f"token_{user_id}.json")
        client_secrets_path = 'client_secret_783922470241.json'

        # Verificar si el archivo token existe y tiene credenciales válidas
        if os.path.exists(token_path):
            try:
                with open(token_path, 'r') as token:
                    info = json.loads(token.read())
                    if all(scope in info.get('scopes', []) for scope in UPLOAD_SCOPES):
                        creds = Credentials.from_authorized_user_info(info, UPLOAD_SCOPES)
                    else:
                        logger.warning(f"Token existente para usuario {user_id} no tiene todos los scopes requeridos")
                        creds = None
            except Exception as e:
                logger.error(f"Error al leer token para usuario {user_id}: {e}")
                creds = None

        # Si no hay credenciales válidas o están expiradas
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                 try:
                    logger.info(f"Refrescando token expirado para usuario {user_id}...")
                    creds.refresh(Request())
                    logger.info("Token refrescado correctamente.")
                 except Exception as refresh_error:
                    logger.error(f"Error al refrescar token: {refresh_error}")
                    creds = None

            # Si necesitamos nueva autorización
            if not creds or not creds.valid:
                if not os.path.exists(client_secrets_path):
                    logger.error(f"El archivo {client_secrets_path} no existe.")
                    return None

                try:
                    flow = InstalledAppFlow.from_client_secrets_file(client_secrets_path, UPLOAD_SCOPES)
                    flow.redirect_uri = "https://weasel-fast-repeatedly.ngrok-free.app/oauth2callback"
                    
                    # Incluir el user_id en el state para recuperarlo en el callback
                    state = {"user_id": user_id}
                    auth_url, _ = flow.authorization_url(
                        access_type='offline', 
                        prompt='consent',
                        state=json.dumps(state)
                    )
                    
                    return {"auth_url": auth_url}
                except Exception as flow_error:
                    logger.error(f"Error en flujo de autorización: {flow_error}")
                    return None

        # Guardar credenciales válidas
        if creds:
            try:
                with open(token_path, 'w') as token:
                    token.write(creds.to_json())
                logger.info(f"[get_user_credentials/save_credentials] Credenciales guardadas para usuario {user_id}")
            except Exception as write_error:
                logger.error(f"Error al guardar token: {write_error}")

        return creds


    def upload_file_to_chat(self, id_space, file_path, id_user, mimetype='text/csv'):
        """Sube un archivo a Google Chat usando autenticación del usuario específico."""
        logger.info(f"[Chatbot/upload_file_to_chat] Intentando subir archivo para usuario {id_user}")
        try:
            creds = self.get_user_credentials(id_user)
            if not creds:
                return False, "Error: No se pudieron obtener las credenciales del usuario."

            service = build('chat', 'v1', credentials=creds)
            logger.info("[Chatbot/upload_file_to_chat] Servicio de Chat (usuario) construido.")

            media = MediaFileUpload(file_path, mimetype=mimetype)
            logger.info(f"[Chatbot/upload_file_to_chat] MediaFileUpload creado para: {file_path}")

            request_body = {
                'filename': os.path.basename(file_path)
            }
            logger.debug(f"[Chatbot/upload_file_to_chat] Request body para subida: {request_body}")
            request = service.media().upload(
                parent=f'spaces/{id_space}',
                body=request_body,
                media_body=media
            )

            logger.info(f"[Chatbot/upload_file_to_chat] Ejecutando solicitud de subida...")
            result = request.execute()

            # Verificar que la respuesta tenga la estructura mínima esperada
            if result and 'attachmentDataRef' in result:
                # Registrar el tipo de referencia que obtuvimos (token o resourceName)
                if 'resourceName' in result['attachmentDataRef']:
                    logger.info(f"[Chatbot/upload_file_to_chat] Archivo subido con éxito. resourceName: {result['attachmentDataRef']['resourceName']}")
                elif 'attachmentUploadToken' in result['attachmentDataRef']:
                    logger.info(f"[Chatbot/upload_file_to_chat] Archivo subido con éxito. Recibido attachmentUploadToken.")
                
                return True, result  # Devolver el resultado completo
            else:
                logger.error(f"[Chatbot/upload_file_to_chat] La respuesta no contiene 'attachmentDataRef': {result}")
                return False, "Error: La respuesta de la API no incluye la referencia al archivo."

        except HttpError as error:
             # Manejar errores específicos de la API de Google
             error_details = error.resp.get('content', '{}')
             try:
                 error_json = json.loads(error_details)
                 # Extraer el mensaje de error más específico si está disponible
                 violations = error_json.get('error', {}).get('details', [{}])
                 if violations and isinstance(violations, list) and len(violations) > 0:
                    # Intentar obtener el mensaje de fieldViolations o directamente de message
                    field_violations = violations[0].get('fieldViolations', [{}])
                    if field_violations and isinstance(field_violations, list) and len(field_violations) > 0 and field_violations[0].get('description'):
                        error_message = field_violations[0]['description']
                    else:
                         error_message = error_json.get('error', {}).get('message', str(error))
                 else:
                     error_message = error_json.get('error', {}).get('message', str(error))

             except (json.JSONDecodeError, IndexError, KeyError):
                 error_message = str(error) # Fallback al mensaje genérico del HttpError

             logger.error(f"[Chatbot/upload_file_to_chat] HttpError al subir el archivo: {error_message}", exc_info=True)
             # Devolver el mensaje de error específico extraído
             return False, f"Error de API al subir el archivo: {error_message}"
        except Exception as e:
            logger.error(f"[Chatbot/upload_file_to_chat] Error general al subir el archivo: {str(e)}", exc_info=True)
            return False, f"Error inesperado al subir archivo: {str(e)}"

    def get_space_members(self, space_name):
        try:
            access_token = self.get_access_token()
            if not access_token:
                logger.error("No se pudo obtener el token de acceso.")
                return None

            url = f"https://chat.googleapis.com/v1/{space_name}/members"
            headers = {
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json"
            }

            response = requests.get(url, headers=headers)
            logger.debug(f"[chatbot/get_space_members]: access_token: {access_token}")
            # ---------------------------------------- #
            if response.status_code == 200:
                members_data_temp = response.json()
                
                # Nuevo documento final para la edicion.
                name_temp=members_data_temp["memberships"][0]['name'].split("/members/")
                final_members={}
                members_temp={}
                for member in members_data_temp["memberships"]:
                    member["name"]=member["name"].replace(f"{name_temp[0]}/members/", "")
                    member["member"]["name"]=member["member"]["name"].replace("users/", "")
                    members_temp[member["member"]["name"]]=member
                name_temp=name_temp[0].replace("spaces/", "")
                final_members[name_temp]=members_temp
                
                # Funcion de actualizacion del JSON
                def update_json(original, nuevo):
                    for key, value in nuevo.items():
                        if key in original:
                            if isinstance(original[key], dict) and isinstance(value, dict):
                                update_json(original[key], value)
                        else:
                            original[key]=value
                
                # Grabado final de la informacion
                with open("./database/spaces/members_data.json", "r") as file:
                    members_data_original=json.load(file)
                update_json(members_data_original, final_members)
                with open("./database/spaces/members_data.json", "w") as file:
                    json.dump(members_data_original, file, indent=4)


                #----------------------------------------#
                members = []
                for membership in members_data_temp.get('memberships', []):
                    member_info = membership.get('member', {})
                    member = {
                        'name': member_info.get('displayName', 'Usuario desconocido'),
                        'type': member_info.get('type', 'UNKNOWN'),
                        'role': membership.get('role', 'UNKNOWN')
                    }
                    members.append(member)
                members_info = [
                    f"Nombre: {member['name']}, Tipo: {member['type']}, Rol: {member['role']}"
                    for member in members
                    if member['type'] != 'BOT'
                ]
                logger.info(f"Miembros del espacio {space_name}: {members_info}")

                return members_info
            else:
                logger.error(f"Error al obtener los miembros del espacio {response.status_code}: {response.text}")
                return None
        except Exception as e:
            logger.error(f"Error al obtener los miembros del espacio: {str(e)}")
            return None

    def get_attachment_content(self, attachment):
        """
        Descarga y devuelve el contenido del archivo adjunto
        """
        logger.info(f"[Chatbot/get_attachment_content] Procesando archivo adjunto: {attachment}")
        try:
            # Obtener el resourceName del attachmentDataRef
            attachment_data = attachment.get('attachmentDataRef', {})
            resource_name = attachment_data.get('resourceName')
            attachment_name = attachment.get('contentName', 'Sin archivo adjunto')

            if resource_name:
                logger.info(f"[Chatbot/get_attachment_content] Usando resourceName: {resource_name}")
                service = self.build_chat_service()

                if service:
                    try:
                        # Construir la solicitud para descargar el archivo
                        request = service.media().download_media(
                            resourceName=resource_name,
                            alt='media'
                        )

                        # Descargar el contenido
                        fh = io.BytesIO()
                        downloader = MediaIoBaseDownload(fh, request)

                        done = False
                        while not done:
                            status, done = downloader.next_chunk()
                            if status:
                                logger.info(f"[Chatbot/get_attachment_content/download] Descarga: {int(status.progress() * 100)}%")

                        # Procesar el contenido según el tipo
                        content_type = attachment.get('contentType', '')
                        logger.warning(f"[Chatbot/get_attachment_content/content_type] Tipo de contenido: {content_type}")
                        if content_type == 'text/plain':
                            # Guardar el contenido en un archivo para debugging
                            logger.warning(f"[Chatbot/get_attachment_content/txt] Guardando archivo TXT")
                            with open(f'attachments/{attachment_name}', 'w', encoding='utf-8') as file:
                                file.write(fh.getvalue().decode('utf-8'))
                            content = fh.getvalue().decode('utf-8')
                            logger.warning(f"[Chatbot/get_attachment_content/txt] Archivo TXT adjunto enviado al modelo")

                        elif content_type == 'text/csv':
                            logger.warning(f"[Chatbot/get_attachment_content/csv] Leyendo archivo CSV")
                            try:
                                # Verificar si el archivo tiene contenido
                                fh.seek(0)  # Asegurarse de que el puntero del archivo esté al inicio
                                content_preview = fh.read(1024)  # Leer los primeros 1024 bytes para verificar el contenido
                                if not content_preview.strip():
                                    logger.error(f"[Chatbot/get_attachment_content/csv] El archivo CSV está vacío o no tiene contenido válido.")
                                    content = f"SISTEMA: El archivo CSV adjunto {attachment_name} está vacío o no tiene contenido válido."
                                else:
                                    fh.seek(0)  # Volver al inicio del archivo para leer con pandas
                                    dfContent = pd.read_csv(fh)
                                    logger.warning(f"[Chatbot/get_attachment_content/csv] Archivo CSV adjunto leído")
                                    dfContent = dfContent.head(10)  # Limitar a las primeras 10 filas
                                    content = f"SISTEMA: Archivo CSV adjunto {attachment_name}, por el momento solo puedo procesar las primeras 10 filas del archivo, sin embargo el archivo adjunto completo se ha guardado en el servidor. \n {dfContent.to_string(index=False)}"
                                    logger.warning(f"[Chatbot/get_attachment_content/csv] Mensaje de archivo CSV adjunto enviado al modelo")
                            except pd.errors.EmptyDataError:
                                logger.error("[Chatbot/get_attachment_content/csv] No se encontraron columnas para analizar en el archivo CSV.")
                                content = "Error: No se encontraron columnas para analizar en el archivo CSV."
                            except Exception as e:
                                logger.error(f"[Chatbot/get_attachment_content/csv] Error al procesar el archivo CSV: {str(e)}")
                                content = f"Error al procesar el archivo CSV: {str(e)}"

                        elif content_type == 'application/json':
                            # Guardar el contenido en un archivo para debugging
                            logger.warning(f"[Chatbot/get_attachment_content/json] Guardando archivo JSON")
                            with open(f'attachments/{attachment_name}', 'w', encoding='utf-8') as file:
                                file.write(fh.getvalue().decode('utf-8'))
                            content = json.loads(fh.getvalue())
                            if len(content) > 20000:
                                # Limitar json a 20000 caracteres
                                content = str(content)[:20000]
                                content = f"SISTEMA: Archivo JSON adjunto {attachment_name}, por el momento solo puedes procesar las primeras 20000 caracteres del archivo, sin embargo el archivo adjunto completo se ha guardado en el servidor. \n {content}"
                            content = f"SISTEMA: Archivo JSON adjunto {attachment_name}: {content}"
                            logger.warning(f"[Chatbot/get_attachment_content/json] Archivo JSON adjunto enviado al modelo")

                        elif content_type == 'image/png':
                            # Guardar el contenido en un archivo para debugging
                            logger.warning(f"[Chatbot/get_attachment_content/image] Guardando imagen")
                            with open(f'attachments/{attachment_name}', 'wb') as file:
                                file.write(fh.getvalue())
                            content = f"SISTEMA: Imagen adjunta {attachment_name}, por el momento no puedes procesar imágenes, sin embargo el archivo adjunto se ha guardado en el servidor."
                            logger.warning(f"[Chatbot/get_attachment_content/image] Mensaje de imagen adjunto enviado al modelo")

                        elif content_type == 'application/pdf':
                            logger.warning(f"[Chatbot/get_attachment_content/pdf] Leyendo archivo PDF")
                            with open(f'attachments/{attachment_name}', 'wb') as file:
                                file.write(fh.getvalue())
                            content = f"SISTEMA: Archivo PDF adjunto {attachment_name}, por el momento no puedes procesar archivos PDF debido a las dimensiones del archivo, esta función se encuentra en desarrollo. El archivo adjunto se ha guardado en el servidor."
                            logger.warning(f"[Chatbot/get_attachment_content/pdf] Mensaje de archivo PDF adjunto enviado al modelo")

                        else:
                            content = fh.getvalue().decode('utf-8')

                        logger.info(f"[Chatbot/get_attachment_content/success] Contenido obtenido exitosamente")
                        return content, attachment_name

                    except Exception as api_error:
                        logger.error(f"[Chatbot/get_attachment_content/except] Error con API: {str(api_error)}")
                        return 'Error al obtener el contenido del archivo adjunto', 'Sin archivo adjunto'
                else:
                    logger.error("[Chatbot/get_attachment_content] No se pudo construir el servicio")
                    return 'Error al obtener el contenido del archivo adjunto', 'Sin archivo adjunto'
            else:
                logger.error("[Chatbot/get_attachment_content] No se encontró resourceName en el adjunto")
                return 'Error al obtener el contenido del archivo adjunto', 'Sin archivo adjunto'

        except Exception as e:
            logger.error(f"[Chatbot/get_attachment_content/except] Error general: {str(e)}")
            return None

    def export_to_csv_and_send(self, tipo_consulta, datos_df, id_space, prompt, id_user=None, output_text_funct=None, loading_msg_name=None):
        """Genera un archivo CSV y lo envía al espacio de chat"""
        logger.info(f"[Chatbot/export_to_csv] Iniciando exportación para consulta: '{prompt}' en espacio {id_space}")
        filename = None 
        
        try:
            # Si no se proporcionó id_user, usar un valor por defecto para compatibilidad
            if id_user is None:
                id_user = "default_user"
                logger.warning("[Chatbot/export_to_csv] No se proporcionó id_user, usando valor por defecto")
            
            # Crear directorio de archivos temporales si no existe
            temp_dir = 'temp_attachments'
            os.makedirs(temp_dir, exist_ok=True)

            # Crear el archivo CSV temporal
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_tipo_consulta = "".join(c if c.isalnum() else "_" for c in tipo_consulta) # Limpiar nombre
            filename = os.path.join(temp_dir, f"{safe_tipo_consulta}_{timestamp}.csv")
            logger.info(f"[Chatbot/export_to_csv] Guardando CSV en: {filename}")

            # Guardar DataFrame a CSV
            datos_df.to_csv(filename, index=False, encoding='utf-8-sig') # utf-8-sig para mejor compatibilidad Excel
            logger.info(f"[Chatbot/export_to_csv] DataFrame guardado en {filename}")

            # --- Paso 1: Obtener credenciales y verificar si se necesita autorización ---
            logger.info(f"[Chatbot/export_to_csv] Obteniendo credenciales de usuario...")
            creds_or_auth = self.get_user_credentials(id_user)
            
            
            if isinstance(creds_or_auth, dict) and "auth_url" in creds_or_auth:
                auth_url = creds_or_auth["auth_url"]
                logger.info(f"[Chatbot/export_to_csv] Se requiere autorización. URL: {auth_url}")
                
                # Crear un mensaje con el enlace de autorización para el usuario
                auth_message = {
                    "cardsV2": [{
                        "card": {
                            "header": {
                                "title": "🔑 Autorización requerida",
                                "subtitle": "Se necesita autorizar subida de archivos",
                                "imageUrl": self.PROFILE_IMG
                            },
                            "sections": [{
                                "widgets": [
                                    {"textParagraph": {"text": "Para poder exportar archivos CSV, necesitas autorizar el acceso a Google Chat. Por favor, haz clic en el siguiente enlace:"}},
                                    {"buttonList": {"buttons": [{"text": "Autorizar acceso", "onClick": {"openLink": {"url": auth_url}}}]}}
                                ]
                            }]
                        }
                    }]
                }
                
                # Enviar el mensaje con el enlace de autorización
                self.send_message(f"spaces/{id_space}", auth_message)
                
                # Si había un mensaje de carga, actualizarlo
                if loading_msg_name:
                    self.edit_message(loading_msg_name, {
                        "text": "⚠️ Se requiere autorización para exportar el archivo CSV. Por favor, revisa el mensaje con el enlace de autorización."
                    })
                
                # Limpiar archivo temporal
                if filename and os.path.exists(filename):
                    try:
                        os.remove(filename)
                        logger.info(f"[Chatbot/export_to_csv] Archivo temporal eliminado.")
                    except OSError as e:
                        logger.warning(f"[Chatbot/export_to_csv] No se pudo eliminar el archivo temporal: {e}")
                
                return False, "Se requiere autorización del usuario. Se ha enviado un enlace de autorización."

            # --- Paso 2: Subir archivo CON CREDENCIALES DE USUARIO ---
            logger.info(f"[Chatbot/export_to_csv] Iniciando subida de archivo como usuario...")
            # upload_file_to_chat ahora devuelve (True, diccionario_resultado) o (False, mensaje_error)
            upload_success, upload_result_dict = self.upload_file_to_chat(id_space, filename, id_user, 'text/csv')

            if not upload_success:
                logger.error(f"[Chatbot/export_to_csv] Error al subir archivo: {upload_result_dict}")
                # Intentar limpiar archivo temporal
                if filename and os.path.exists(filename):
                    try: os.remove(filename)
                    except OSError: pass
                # Editar mensaje de carga con error si existe
                if loading_msg_name:
                    self.edit_message(loading_msg_name, {
                        "text": f"❌ Error al subir el archivo CSV: {upload_result_dict}"
                    })
                return False, f"Error al subir archivo CSV: {upload_result_dict}"

            # --- Ya no necesitamos extraer resourceName, usaremos upload_result_dict directamente ---
            logger.info(f"[Chatbot/export_to_csv] Archivo subido, respuesta de API: {upload_result_dict}")


            # --- Paso 3: Editar mensaje de carga CON CREDENCIALES DE BOT ---
            if loading_msg_name:
                edit_content = {
                    "text": f"📊 Archivo CSV generado y subido para: '{prompt}'. Adjuntando al chat...",
                }
                edit_success, _ = self.edit_message(loading_msg_name, edit_content)
                if not edit_success:
                    logger.warning(f"[Chatbot/export_to_csv] No se pudo editar el mensaje de carga {loading_msg_name}, continuando...")
            else:
                logger.warning("[Chatbot/export_to_csv] No se proporcionó loading_msg_name para editar.")


            # --- Paso 4: Enviar mensaje final con adjunto CON CREDENCIALES DE USUARIO ---
            # --- Corrección: Usar la respuesta completa de la subida ---
            # El campo 'attachment' espera una LISTA de recursos Attachment.
            # El resultado de upload() ya tiene la estructura de un recurso Attachment.
            attachment_message = {
                "text": f"📊 Aquí está tu archivo CSV exportado para tu consulta: ''{prompt}''",
                "attachment": [upload_result_dict]  # Esto ya es correcto
            }
            # --- Fin Corrección ---

            logger.info(f"[Chatbot/export_to_csv] Intentando enviar mensaje final con adjunto COMO USUARIO...")
            logger.debug(f"[Chatbot/export_to_csv] Payload del mensaje final: {json.dumps(attachment_message)}")

            # ----- Usar send_message_as_user -----
            success, response = self.send_message_as_user(id_space, attachment_message, id_user)
            # -------------------------------------

            # Limpiar archivo temporal después de intentar enviar
            logger.info(f"[Chatbot/export_to_csv] Intentando eliminar archivo temporal: {filename}")
            if filename and os.path.exists(filename):
                try:
                    os.remove(filename)
                    logger.info(f"[Chatbot/export_to_csv] Archivo temporal eliminado.")
                except OSError as e:
                    logger.warning(f"[Chatbot/export_to_csv] No se pudo eliminar el archivo temporal: {e}")
            else:
                 logger.warning(f"[Chatbot/export_to_csv] No se encontró el archivo temporal {filename} para eliminar.")


            if success:
                logger.info(f"[Chatbot/export_to_csv] Mensaje con CSV enviado correctamente COMO USUARIO.")
                # Opcionalmente, eliminar/editar el mensaje de carga
                if loading_msg_name:
                   try:
                       # Podrías editarlo a "Completado" o eliminarlo
                       # self.edit_message(loading_msg_name, {"text": f"✅ Archivo CSV para '{prompt}' enviado."})
                       self.delete_message(loading_msg_name) # O eliminarlo
                       logger.info(f"[Chatbot/export_to_csv] Mensaje de carga {loading_msg_name} eliminado.")
                   except Exception as final_edit_err:
                       logger.warning(f"[Chatbot/export_to_csv] No se pudo editar/eliminar mensaje de carga final: {final_edit_err}")
                return True, "Archivo CSV enviado correctamente"
            else:
                # response contendrá el mensaje de error de send_message_as_user
                error_detail = response
                logger.error(f"[Chatbot/export_to_csv] Error al enviar mensaje con CSV COMO USUARIO: {error_detail}")
                # Editar el mensaje de carga para mostrar el error específico
                if loading_msg_name:
                     self.edit_message(loading_msg_name, {
                         "text": f"❌ Error al adjuntar el archivo CSV al chat: {error_detail}"
                     })
                return False, f"Error al enviar archivo CSV como usuario: {error_detail}"

        except Exception as e:
             logger.error(f"[Chatbot/export_to_csv] Error GRANDE al exportar a CSV: {str(e)}", exc_info=True)
             # Intentar limpiar si filename está definido
             if filename and os.path.exists(filename):
                  try: os.remove(filename)
                  except OSError: pass
             # Editar mensaje de carga con error si existe
             if loading_msg_name:
                 try:
                     self.edit_message(loading_msg_name, {
                         "text": f"❌ Error inesperado durante la exportación: {str(e)}"
                     })
                 except Exception as edit_err:
                     logger.error(f"Error al intentar editar mensaje de error: {edit_err}")
             return False, f"Error al exportar a CSV: {str(e)}"

    def delete_message(self, message_name):
        """
        Elimina un mensaje en un espacio
        """
        access_token = self.get_access_token()
        if not access_token:
            logger.error("[Chatbot/delete_message] Error: No se pudo obtener el token de acceso.")
            return False, "Error: No se pudo obtener el token de acceso"

        url = f"https://chat.googleapis.com/v1/{message_name}"
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json"
        }

        response = requests.delete(url, headers=headers, timeout=300)

        if response.status_code == 200:
            logger.info(f"[Chatbot/delete_message] Mensaje eliminado con éxito")
            return True, "Mensaje eliminado correctamente"
        else:
            error_msg = f"[Chatbot/delete_message] Error al eliminar el mensaje: {response.status_code} - {response.text}"
            logger.error(f"{error_msg}")
            return False, error_msg

    def send_message_as_user(self, space_id, message, id_user):
        """
        Envía un mensaje al espacio utilizando las credenciales del usuario específico.
        """
        logger.info(f"[Chatbot/send_message_as_user] Intentando enviar mensaje como usuario {id_user}")
        try:
            # Obtener credenciales específicas para este usuario
            creds = self.get_user_credentials(id_user)
            if not creds:
                return False, "Error: No se pudieron obtener las credenciales del usuario."

            # Construir el servicio con las credenciales de usuario
            service = build('chat', 'v1', credentials=creds)
            logger.info("[Chatbot/send_message_as_user] Servicio de Chat (usuario) construido.")

            # Google Chat API espera el 'parent' en el formato 'spaces/ID_ESPACIO'
            parent_space = f"spaces/{space_id}" # Asegurar formato correcto
            logger.debug(f"[Chatbot/send_message_as_user] Enviando a parent: {parent_space}")
            logger.debug(f"[Chatbot/send_message_as_user] Body del mensaje: {json.dumps(message)}")


            # Realizar la llamada para crear el mensaje
            request = service.spaces().messages().create(
                parent=parent_space,
                body=message # message ya debe tener la estructura correcta
            )
            response = request.execute()

            logger.info(f"[Chatbot/send_message_as_user] Mensaje enviado con éxito como usuario a {parent_space}: {response}")
            return True, response

        except HttpError as error:
             # Manejar errores específicos de la API de Google
             error_details = error.resp.get('content', '{}')
             try:
                 error_json = json.loads(error_details)
                 # Extraer el mensaje de error más específico si está disponible
                 violations = error_json.get('error', {}).get('details', [{}])
                 if violations and isinstance(violations, list) and len(violations) > 0:
                    # Intentar obtener el mensaje de fieldViolations o directamente de message
                    field_violations = violations[0].get('fieldViolations', [{}])
                    if field_violations and isinstance(field_violations, list) and len(field_violations) > 0 and field_violations[0].get('description'):
                        error_message = field_violations[0]['description']
                    else:
                         error_message = error_json.get('error', {}).get('message', str(error))
                 else:
                     error_message = error_json.get('error', {}).get('message', str(error))

             except (json.JSONDecodeError, IndexError, KeyError):
                 error_message = str(error) # Fallback al mensaje genérico del HttpError

             logger.error(f"[Chatbot/send_message_as_user] HttpError al enviar mensaje como usuario: {error_message}", exc_info=True)
             # Devolver el mensaje de error específico extraído
             return False, f"Error de API al enviar mensaje: {error_message}"
        except Exception as e:
            # Capturar otros errores
            error_msg = f"Error inesperado al enviar mensaje como usuario: {str(e)}"
            logger.error(f"[Chatbot/send_message_as_user] {error_msg}", exc_info=True) # Loggear traceback completo
            return False, error_msg



# Inicializar el chatbot
chatbot = Chatbot()