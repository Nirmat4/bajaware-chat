import pandas as pd
import numpy as np
import os
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import spacy # python -m spacy download es_core_news_sm

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from jqlModel import JQLModel
from apiJira2 import request_jira_tickets

nltk.download('punkt_tab')

print("Importaciones realizadas correctamente.")

modelo = JQLModel()

class TfIdfProcessor:
    def __init__(self):
        # Descargar recursos necesarios de NLTK
        self._download_nltk_resources()
        
        # Cargar el archivo CSV
        self.csv_path = "funciones_jql.csv"
        self.df = self._load_csv(self.csv_path)
    
    def _download_nltk_resources(self):
        resources = ['corpora/stopwords', 'tokenizers/punkt', 'corpora/wordnet']
        for resource in resources:
            try:
                nltk.data.find(resource)
            except LookupError:
                print(f"Descargando {resource.split('/')[1]}...")
                nltk.download(resource.split('/')[1], quiet=True)
    
    def _load_csv(self, path):
        try:
            print(f"Leyendo archivo CSV desde: {path}")
            df = pd.read_csv(path)
            print("Archivo CSV cargado con éxito. Columnas disponibles:")
            for col in df.columns:
                print(f"- {col}")
            return df
        except FileNotFoundError:
            print(f"Error: No se encontró el archivo {path}")
            raise
        except Exception as e:
            print(f"Error al cargar el CSV: {str(e)}")
            raise

    def clean_text(self, text):
        print(f"Limpieza del texto: {text}")
        try:
            nlp = spacy.load("es_core_news_lg")
            stop_words = self._get_stop_words(nlp)
            tokens = word_tokenize(text.lower())
            
            filtered_tokens = self._filter_tokens(tokens, stop_words)
            
            clean = " ".join(filtered_tokens)
            print(f"Texto limpio: {clean}")
            return clean
        except Exception as e:
            print(f"Error en la limpieza del texto: {str(e)}")
            return text
    
    def _get_stop_words(self, nlp):
        stop_words = set(nlp.Defaults.stop_words) - {"mas", "más", "menos", "es"}
        stopwords_es = set(stopwords.words('spanish')) - {"mas", "más", "menos", "es"}
        return stop_words.union(stopwords_es)
    
    def _filter_tokens(self, tokens, stop_words):
        math_keywords = {
            "mas", "más", "menos", "por", "entre", "suma", "resta", "multiplicacion", "division",
            "multiplicación", "división", "calculo", "cálculo", "calcular", "cuanto", "cuánto",
            "es", "igual", "resultado", "total", "operacion", "operación", "numero", "número"
        }
        filtered_tokens = []
        for token in tokens:
            token_normalized = self._normalize_token(token)
            if token.isalpha() and token_normalized not in stop_words:
                filtered_tokens.append(token_normalized)
            elif token_normalized in math_keywords:
                filtered_tokens.append(token_normalized)
        return filtered_tokens
    
    def _normalize_token(self, token):
        token_normalized = token.lower()
        for a, b in [("á", "a"), ("é", "e"), ("í", "i"), ("ó", "o"), ("ú", "u")]:
            token_normalized = token_normalized.replace(a, b)
        return token_normalized
    
    def vectorize_functions(self, text_clean):
        print("Vectorización de funciones iniciada.")
        try:
            all_texts = (self.df['descripcion'] + " " + self.df['key_words']).tolist()
            vectorizer = TfidfVectorizer()
            vectorizer.fit(all_texts + [text_clean])
            vectorized_functions = vectorizer.transform(all_texts)
            vectorized_input = vectorizer.transform([text_clean])
            print("Vectorización completada.")
            return vectorized_input, vectorized_functions
        except Exception as e:
            print(f"Error en la vectorización: {str(e)}")
            raise
    
    def search_query(self, query, vec_usr_input, vec_funcs):
        print("Cálculo de similitud de coseno iniciado.")
        try:
            similarities = cosine_similarity(vec_usr_input, vec_funcs).flatten()
            self.df['cosine_similarity'] = similarities
            print("Similitudes calculadas. Mostrando top 10")
            return self.df.sort_values(by='cosine_similarity', ascending=False).head(10)
        except Exception as e:
            print(f"Error en la búsqueda: {str(e)}")
            raise

    def main(self):
        print("\nPrograma iniciado. Introduce tu consulta.")
        print("""\nEjemplos de consultas por categoría:
1. CONSULTAS POR ESTADO (consultaEstadoTickets):
    - ¿Cuántos tickets están en estado 'waiting for support'?
    - ¿Cuántos tickets están en estado crítico?
    - ¿Qué tickets están esperando respuesta del cliente?
2. CONSULTAS DE SLA (consultaSLA):
    - ¿Qué tickets exceden los SLAs de resolución?
    - Muestra los tickets que exceden SLA para el cliente ABC Corp
    - ¿Cuáles son los tickets vencidos esta semana?
3. CONSULTAS DE ANTIGÜEDAD (consultaTicketsAntiguos):
    - ¿Qué tickets llevan más de 30 días abiertos?
    - Muestra los tickets sin actualizar en 2 semanas
    - ¿Cuáles son los tickets más antiguos sin resolver?
4. CONSULTAS TEMPORALES (consultaTicketsTemporal):
    - ¿Cuántos tickets se crearon esta semana?
    - ¿Qué tickets se resolvieron en los últimos 7 días?
    - Muestra los tickets creados hoy
5. CONSULTAS POR EQUIPO (consultaTicketsEquipo):
    - Lista los tickets asignados al equipo de QA sin responsable
    - ¿Qué tareas tiene pendientes el equipo de desarrollo?
    - Muestra los tickets escalados al equipo de soporte
6. CONSULTAS POR CLIENTE (consultaTicketsCliente):
    - ¿Qué tickets tiene pendientes el cliente XYZ?
    - Muestra la actividad del cliente ABC Corp esta semana
    - ¿Qué clientes tienen más de 5 tickets abiertos?
7. CONSULTAS POR ETIQUETAS (consultaTicketsEtiquetas):
    - Busca tickets con la etiqueta 'urgente' en backlog
    - ¿Qué tickets están marcados como 'bloqueados'?
    - Muestra tickets con etiquetas 'alta_prioridad' y 'bug'
8. CONSULTAS DE RENDIMIENTO (consultaRendimiento):
    - ¿Cuál es el tiempo promedio de resolución esta semana?
    - Muestra el rendimiento del equipo de soporte
    - ¿Qué tickets tienen más de 3 cambios de prioridad?
9. CONSULTAS AVANZADAS (consultaTicketsAvanzada):
    - Encuentra tickets con más de dos adjuntos en estado 'Por hacer'
    - ¿Qué tickets tienen dependencias sin resolver?
    - Muestra tickets bloqueadores del sprint actual
10. CONSULTAS POR TIPO (consultaTicketsTipo):
    - ¿Cuántos tickets hay por tipo de incidencia?
    - Muestra la distribución de bugs vs features
    - ¿Qué tipos de tickets tienen mayor tiempo de resolución?
11. CONSULTAS DE COMENTARIOS (consultaTicketsComentarios):
    - ¿Qué tickets tienen comentarios añadidos en los últimos 2 días?
    - ¿Cuántos tickets tienen comentarios sin responder en los últimos 5 días?
    - ¿Qué tickets tienen comentarios vinculados con discusiones internas del equipo?
12. CONSULTAS DE ADJUNTOS (consultaTicketsAdjuntos):
    - ¿Qué tickets tienen más de dos adjuntos y siguen en estado 'Por hacer'?
    - ¿Qué tickets se asignaron durante el último sprint pero no tienen adjuntos?
Escribe tu consulta (o 'salir' para terminar):""")

        while True:
            try:
                print("\n" + "=" * 80 + "\n")
                user_input = input("Introduzca su consulta (Escribe '/bye' o 'salir' para terminar): ")
                if user_input.lower() in ['salir', '/bye']:
                    print("Saliendo del programa..." if user_input.lower() == 'salir' else "Adios!")
                    break
                
                print(f"Consulta del usuario: {user_input}")
                text_clean = self.clean_text(user_input)
                vectorized_user_input, vectorized_functions = self.vectorize_functions(text_clean)
                search_results = self.search_query(user_input, vectorized_user_input, vectorized_functions)
                response = self._display_results(user_input, search_results)
                
            except Exception as e:
                print(f"Error procesando la consulta: {str(e)}")
                continue
                #return None

    def _display_results(self, user_input, search_results):
        try:
            print("\nResultados relevantes:\n", search_results[['index', 'funcion', 'descripcion', 'cosine_similarity']])
            print(f"\n🔍 Resultado más relevante de la consulta \"{user_input}\" con similitud de {search_results['cosine_similarity'].iloc[0]:.4f}:")
            print(f"\n📌 Función: {search_results['funcion'].iloc[0]}")
            print(f"📝 Descripción: {search_results['descripcion'].iloc[0]}")
            print(f"⚙️ Acciones: {search_results['acciones'].iloc[0]}")
            print(f"🛠️ Parámetros: {search_results['parametros'].iloc[0]}")
            print("=" * 80)
            # Obtener y mostrar system_prompt
            system_prompt = search_results['system_prompt'].iloc[0]
            print(f"System prompt: {system_prompt}")
            response = self._execute_function(
                search_results['funcion'].iloc[0],
                search_results['acciones'].iloc[0],
                search_results['parametros'].iloc[0],
                search_results['descripcion'].iloc[0],
                system_prompt,
                user_input
            )
            print(f"[TfIdfProcessor/_display_results] Respuesta del modelo: {response}")
            
            return response
        except Exception as e:
            print(f"Error mostrando resultados: {str(e)}")

    def _execute_function(self, funcion, acciones, parametros, descripcion, system_prompt, user_input):
        try:
            # Llamar a la función estática pasando system_prompt
            response = getattr(JqlFunctions, funcion)(acciones, parametros, descripcion, user_input, system_prompt)
            print(f"[JqlFunctions/{funcion}/mostrar_respuesta] Respuesta del modelo: {response}")
            
            # Preguntar al usuario si desea ejecutar la consulta JQL generada
            if response:
                execute_query = input(f"\n¿Desea ejecutar esta consulta JQL y obtener los resultados? (s/n): ")
                if execute_query.lower() in ['s', 'si', 'sí', 'y', 'yes']:
                    output_file = input("Nombre del archivo de salida (deje en blanco para generar automáticamente): ")
                    print("\n🚀 Ejecutando consulta en Jira...\n")
                    tickets = request_jira_tickets(response, output_file)
                    print(f"\n✅ Consulta ejecutada y resultados guardados.")
            
            return response
        except Exception as e:
            print(f"Error ejecutando la función {funcion}: {str(e)}")
            return None

class JqlFunctions:
    @staticmethod
    def consultaEstadoTickets(acciones, parametros, descripcion, user_input, system_prompt):
        print(f"\n[JqlFunctions/consultaEstadoTickets]")
        print(f"Descripción de la consulta: {descripcion}\n\n")
        print(f"Consulta original: {user_input}")
        print(f"Acciones disponibles: {acciones}")
        print(f"Parámetros requeridos: {parametros}")
        print(f"System prompt: {system_prompt}")
        response = modelo.mostrar_respuesta(system_prompt, user_input)
        print(f"[JqlFunctions/consultaEstadoTickets] Respuesta del modelo: {response}")
        return response

    @staticmethod
    def consultaSLA(acciones, parametros, descripcion, user_input, system_prompt):
        print(f"\n[JqlFunctions/consultaSLA]")
        print(f"Descripción de la consulta: {descripcion}")
        print(f"Consulta original: {user_input}")
        print(f"Acciones disponibles: {acciones}")
        print(f"Parámetros requeridos: {parametros}")
        print(f"System prompt: {system_prompt}")
        response = modelo.mostrar_respuesta(system_prompt, user_input)
        print(f"[JqlFunctions/consultaSLA] Respuesta del modelo: {response}")
        return response
    @staticmethod
    def consultaTicketsAntiguos(acciones, parametros, descripcion, user_input, system_prompt):
        print(f"\n[JqlFunctions/consultaTicketsAntiguos]")
        print(f"Descripción de la consulta: {descripcion}")
        print(f"Consulta original: {user_input}")
        print(f"Acciones disponibles: {acciones}")
        print(f"Parámetros requeridos: {parametros}")
        print(f"System prompt: {system_prompt}")
        response = modelo.mostrar_respuesta(system_prompt, user_input)
        print(f"[JqlFunctions/consultaTicketsAntiguos] Respuesta del modelo: {response}")
        return response
    
    @staticmethod
    def consultaTicketsTemporal(acciones, parametros, descripcion, user_input, system_prompt):
        print(f"\n[JqlFunctions/consultaTicketsTemporal]")
        print(f"Descripción de la consulta: {descripcion}")
        print(f"Consulta original: {user_input}")
        print(f"Acciones disponibles: {acciones}")
        print(f"Parámetros requeridos: {parametros}")
        print(f"System prompt: {system_prompt}")
        response = modelo.mostrar_respuesta(system_prompt, user_input)
        print(f"[JqlFunctions/consultaTicketsTemporal] Respuesta del modelo: {response}")
        return response
    
    @staticmethod
    def consultaTicketsEquipo(acciones, parametros, descripcion, user_input, system_prompt):
        print(f"\n[JqlFunctions/consultaTicketsEquipo]")
        print(f"Descripción de la consulta: {descripcion}")
        print(f"Consulta original: {user_input}")
        print(f"Acciones disponibles: {acciones}")
        print(f"Parámetros requeridos: {parametros}")
        print(f"System prompt: {system_prompt}")
        response = modelo.mostrar_respuesta(system_prompt, user_input)
        print(f"[JqlFunctions/consultaTicketsEquipo] Respuesta del modelo: {response}")
        return response

    @staticmethod
    def consultaTicketsCliente(acciones, parametros, descripcion, user_input, system_prompt):
        print(f"\n[JqlFunctions/consultaTicketsCliente]")
        print(f"Descripción de la consulta: {descripcion}")
        print(f"Consulta original: {user_input}")
        print(f"Acciones disponibles: {acciones}")
        print(f"Parámetros requeridos: {parametros}")
        print(f"System prompt: {system_prompt}")
        response = modelo.mostrar_respuesta(system_prompt, user_input)
        print(f"[JqlFunctions/consultaTicketsCliente] Respuesta del modelo: {response}")
        return response

    @staticmethod
    def consultaTicketsEtiquetas(acciones, parametros, descripcion, user_input, system_prompt):
        print(f"\n[JqlFunctions/consultaTicketsEtiquetas]")
        print(f"Descripción de la consulta: {descripcion}")
        print(f"Consulta original: {user_input}")
        print(f"Acciones disponibles: {acciones}")
        print(f"Parámetros requeridos: {parametros}")
        print(f"System prompt: {system_prompt}")
        response = modelo.mostrar_respuesta(system_prompt, user_input)
        print(f"[JqlFunctions/consultaTicketsEtiquetas] Respuesta del modelo: {response}")
        return response

    @staticmethod
    def consultaRendimiento(acciones, parametros, descripcion, user_input, system_prompt):
        print(f"\n[JqlFunctions/consultaRendimiento]")
        print(f"Descripción de la consulta: {descripcion}")
        print(f"Consulta original: {user_input}")
        print(f"Acciones disponibles: {acciones}")
        print(f"Parámetros requeridos: {parametros}")
        print(f"System prompt: {system_prompt}")
        response = modelo.mostrar_respuesta(system_prompt, user_input)
        print(f"[JqlFunctions/consultaRendimiento] Respuesta del modelo: {response}")
        return response

    @staticmethod
    def consultaTicketsAvanzada(acciones, parametros, descripcion, user_input, system_prompt):
        print(f"\n[JqlFunctions/consultaTicketsAvanzada]")
        print(f"Descripción de la consulta: {descripcion}")
        print(f"Consulta original: {user_input}")
        print(f"Acciones disponibles: {acciones}")
        print(f"Parámetros requeridos: {parametros}")
        print(f"System prompt: {system_prompt}")
        response = modelo.mostrar_respuesta(system_prompt, user_input)
        print(f"[JqlFunctions/consultaTicketsAvanzada] Respuesta del modelo: {response}")
        return response

    @staticmethod
    def consultaTicketsTipo(acciones, parametros, descripcion, user_input, system_prompt):
        print(f"\n[JqlFunctions/consultaTicketsTipo]")
        print(f"Descripción de la consulta: {descripcion}")
        print(f"Consulta original: {user_input}")
        print(f"Acciones disponibles: {acciones}")
        print(f"Parámetros requeridos: {parametros}")
        print(f"System prompt: {system_prompt}")
        response = modelo.mostrar_respuesta(system_prompt, user_input)
        print(f"[JqlFunctions/consultaTicketsTipo] Respuesta del modelo: {response}")
        return response
        
    @staticmethod
    def consultaTicketsComentarios(acciones, parametros, descripcion, user_input, system_prompt):
        print(f"\n[JqlFunctions/consultaTicketsComentarios]")
        print(f"Descripción de la consulta: {descripcion}")
        print(f"Consulta original: {user_input}")
        print(f"Acciones disponibles: {acciones}")
        print(f"Parámetros requeridos: {parametros}")
        print(f"System prompt: {system_prompt}")
        response = modelo.mostrar_respuesta(system_prompt, user_input)
        print(f"[JqlFunctions/consultaTicketsComentarios] Respuesta del modelo: {response}")
        return response
        
    @staticmethod
    def consultaTicketsAdjuntos(acciones, parametros, descripcion, user_input, system_prompt):
        print(f"\n[JqlFunctions/consultaTicketsAdjuntos]")
        print(f"Descripción de la consulta: {descripcion}")
        print(f"Consulta original: {user_input}")
        print(f"Acciones disponibles: {acciones}")
        print(f"Parámetros requeridos: {parametros}")
        print(f"System prompt: {system_prompt}")
        response = modelo.mostrar_respuesta(system_prompt, user_input)
        print(f"[JqlFunctions/consultaTicketsAdjuntos] Respuesta del modelo: {response}")
        return response


if __name__ == "__main__":
    try:
        processor = TfIdfProcessor()
        response = processor.main()
        print(f"[testFunctJql.py] Respuesta del programa: {response}")
    except Exception as e:
        print(f"Error fatal en la aplicación: {str(e)}")
        