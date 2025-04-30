import warnings
import os
import sys
from contextlib import contextmanager
import io
from llama_cpp import Llama  # pip install llama-cpp-python
import datetime  # Importar para obtener la fecha actual

class JQLModel:
    def __init__(self, model_path="../models/phi3-mini-jql-unsloth.Q4_K_M.gguf", n_ctx=4096, n_threads=4):
        print("[JQLModel] Iniciando modelo JQL, por favor espere...")
        # Filtrar todos los warnings de Python
        warnings.filterwarnings("ignore")
        self.model_path = model_path
        self.n_ctx = n_ctx
        self.n_threads = n_threads
        self.llm = None
        self._cargar_modelo()

    @contextmanager
    def suppress_stdout_stderr(self):
        # Guardar los descriptores actuales
        save_stderr = sys.stderr
        save_stdout = sys.stdout

        # Redirigir stdout y stderr a null
        sys.stderr = io.StringIO()
        sys.stdout = io.StringIO()

        try:
            yield
        finally:
            # Restaurar los descriptores originales
            sys.stderr = save_stderr
            sys.stdout = save_stdout

    def _cargar_modelo(self):
        print("[JQLModel] Cargando modelo...")
        with self.suppress_stdout_stderr():
            self.llm = Llama(
                model_path=self.model_path,
                n_ctx=self.n_ctx,
                n_threads=self.n_threads
            )

    def generar_jql(self, system_prompt, user_prompt):
        # Añadir la fecha actual al system_prompt
        current_date = datetime.datetime.now().strftime("%Y-%m-%d")
        print(f"Fecha actual: {current_date}")
        enhanced_system_prompt = f"{system_prompt} Today's date is {current_date}."
        
        # Prompts
        #system_prompt = "<|system|> Tu tarea es escribir consultas JQL (Jira Query Language) válidas basadas en las instrucciones del usuario. Proporciona solo la consulta JQL sin explicaciones adicionales. <|end|>"
        #user_prompt = f"<|user|> {instruccion_usuario}<|end|>"
        prompt = f"<|user|> {user_prompt}<|end|> <|system|> {enhanced_system_prompt}<|end|> <|assistant|>"

        # Generar respuesta
        with self.suppress_stdout_stderr():
            response = self.llm(
                prompt,
                max_tokens=4000,
                temperature=0.2,
                stop=["<|end|>", "<|user|>", "###"]
            )

        # Extraer y retornar solo la respuesta
        jql_query = response["choices"][0]["text"].strip()
        return jql_query

    def mostrar_respuesta(self, system_prompt, user_prompt):
        print(f"[JQLModel] Generando consulta JQL, por favor espere...\n")
        jql_query = self.generar_jql(system_prompt, user_prompt)
        
        # Procesar la respuesta para extraer solo una consulta JQL válida
        jql_lines = jql_query.split('\n')
        valid_jql = None
        
        for line in jql_lines:
            # Ignorar líneas vacías, comentarios o líneas que no parecen JQL válido
            line = line.strip()
            if not line or line.startswith('-') or line.startswith('#') or line.startswith('*') or ':' in line:
                continue
                
            # Verificar que la línea parece una consulta JQL de Jira
            if 'project' in line.lower() and ('=' in line or '>' in line or '<' in line):
                valid_jql = line
                break
        
        # Si no se encontró ninguna línea válida, usar la primera línea de todas formas
        if valid_jql is None and jql_lines:
            valid_jql = jql_lines[0].strip()
            
        print("\nRespuesta del modelo:")
        print("-" * 100)
        print(valid_jql)
        print("-" * 100)
        
        # Devolver la consulta JQL limpia
        return valid_jql

# Ejemplo de uso:
# modelo = JQLModel()
# modelo.mostrar_respuesta("Quiero ver los issues asignados a Roger y que tengan el label release del proyecto Bajaware")
