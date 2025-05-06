# Contextual Answering System with Multi-Source Integration

Este proyecto implementa un sistema inteligente de respuesta a preguntas que combina múltiples fuentes de información (consultas SQL, recuperación por embeddings, entre otras) para generar un contexto preciso y relevante. Este contexto es procesado y enriquecido por un modelo de lenguaje (LLM), permitiendo responder preguntas complejas y específicas sobre la información interna de una empresa.

---

## 🚀 Objetivo

Permitir a los usuarios realizar consultas específicas sobre los datos de una organización, utilizando información dispersa y estructurada desde distintas fuentes, y generando respuestas precisas mediante un modelo LLM (Large Language Model).

---

## 🧠 ¿Cómo funciona?

1. **Recepción de la pregunta del usuario**
2. **Clasificación y análisis de intención** para determinar qué tipo de datos son necesarios (bases SQL, documentos, embeddings, etc.)
3. **Recuperación de información** desde:
   - Bases de datos relacionales (vía SQL)
   - Vector stores (usando embeddings semánticos)
   - Otras APIs o sistemas documentales
4. **Construcción del contexto** unificado a partir de los resultados recuperados.
5. **Generación de respuesta** utilizando un LLM (como GPT, LLaMA o similar) alimentado con el contexto.
6. **Entrega de la respuesta enriquecida y precisa** al usuario.

---

## 🧱 Arquitectura

```text
[User Question]
      |
      v
[Intención + Extracción de entidades]
      |
      v
[Recuperación de contexto] <---- SQL / Embeddings / APIs
      |
      v
[Construcción de prompt]
      |
      v
[LLM (GPT, LLaMA, etc.)]
      |
      v
[Respuesta final]
