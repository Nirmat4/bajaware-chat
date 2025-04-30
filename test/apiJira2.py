import requests
import json
import re
from datetime import datetime
from jsonschema import validate, ValidationError # pip install jsonschema

# Schema to validate Jira API JSON response structure
JIRA_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "total": {"type": "integer"},
        "issues": {
            "type": "array",
            "items": {"type": "object", "required": ["fields"]}
        }
    },
    "required": ["total", "issues"]
}

def fix_common_jql_errors(jql_query):
    """
    Corrige errores comunes de sintaxis en consultas JQL.
    
    Parámetros:
    - jql_query: Query JQL a corregir
    
    Retorna:
    - Query JQL corregido
    """
    # Start with original
    corrected_query = jql_query
    
    # Replace brackets in IN clauses with parentheses (case-insensitive)
    corrected_query = re.sub(r'\bIN\s*\[([^\]]+)\]', r'IN (\1)', corrected_query, flags=re.IGNORECASE)
    
    # Fix incorrect field references for customfields
    corrected_query = re.sub(r'customfield_10500\.name', r'customfield_10500[].name', corrected_query)
    corrected_query = re.sub(r'customfield_11208 ?= ?[\'"]([^\'"]+)[\'"]', r"customfield_11208.value = '\1'", corrected_query)
    
    # Normalize quotes around values to single quotes
    corrected_query = re.sub(r'=\s*"([^"]+)"', r"= '\1'", corrected_query)
    
    # Clean up extra whitespace inside parentheses
    corrected_query = re.sub(r'\(\s+', '(', corrected_query)
    corrected_query = re.sub(r'\s+\)', ')', corrected_query)
    
    # Uppercase logical operators
    corrected_query = re.sub(r'\b(and|or|not|in|empty)\b', lambda m: m.group(1).upper(), corrected_query, flags=re.IGNORECASE)
    
    # Ensure project = 'BS' is always included
    if 'project' not in corrected_query.lower():
        corrected_query = f"project = 'BS' AND {corrected_query}"
    
    # Log corrections if any
    if corrected_query != jql_query:
        print("⚠️ Fixed common JQL syntax errors:")
        print(f"Original: {jql_query}")
        print(f"Corrected: {corrected_query}")
    
    return corrected_query

def request_jira_tickets(jql_query, output_file=None):
    """
    Función para obtener tickets de Jira mediante su API REST utilizando un query JQL personalizado.
    
    Parámetros:
    - jql_query: Query JQL para filtrar tickets
    - output_file: Nombre del archivo de salida (opcional)
    """
    # Corregir errores comunes en la consulta JQL
    jql_query = fix_common_jql_errors(jql_query)
    
    # Configuración del cliente REST con la URL base de Jira
    base_url = "https://bajaware.atlassian.net"
    
    # Credenciales de autenticación para la API de Jira (usuario y token)
    auth = ("dmoncivais@bajaware.com", "ATATT3xFfGF0j2KBKb7Pqq1T4oJtuX84Zb9YpfLsc4c6ZRBkgOiljgIswRATjWMksHPbZLXy5aJzMya1YKJKisfLYOoyW4Rb6fq1PAi7__JnLGcRr7zQeG1xRUt6NXOF6WK7eora0pX6TlFRIS3WH66baE8ja665Mn9hGpzNpn19IHZuuvCKfPU=59FC5052")
    
    print("⏳ Conectando a Jira API...")
    
    # Construir la URL de la consulta con el JQL proporcionado
    api_endpoint = "/rest/api/3/search"
    
    # Parámetros de la consulta
    params = {
        'jql': jql_query,
        'fields': 'customfield_10500,key,status,created,summary,resolved,resolution,assignee,customfield_11208,customfield_10705',
        'maxResults': 100
    }
    
    # Ejecutar la solicitud HTTP GET a la API de Jira
    try:
        print(f"📊 Executing JQL query: {jql_query}")
        response = requests.get(f"{base_url}{api_endpoint}", auth=auth, params=params)
        
        # Procesar la respuesta
        if response.status_code == 200:
            data = response.json()
            # Validate JSON structure
            try:
                validate(instance=data, schema=JIRA_RESPONSE_SCHEMA)
            except ValidationError as e:
                print(f"❌ Invalid JSON structure in Jira response: {e.message}")
                return None
            total_issues = data.get("total", 0)
            issues = data.get("issues", [])
            
            print(f"✅ Consulta exitosa!")
            print(f"📈 Total de tickets encontrados: {total_issues}")
            print(f"📋 Tickets obtenidos: {len(issues)}")
            
            # Generar nombre de archivo si no se proporcionó
            if not output_file:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_file = f"jira_tickets_{timestamp}.json"
            
            # Guardar los tickets en un archivo JSON
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(issues, f, ensure_ascii=False, indent=4)
            
            print(f"💾 Tickets guardados en {output_file}")
            return issues
        else:
            print(f"❌ Error en la consulta: {response.status_code}")
            print(response.text)
            
            # Sugerir correcciones si hay errores específicos
            if "Expecting either a value, list or function but got '['" in response.text:
                print("\n⚠️ SUGERENCIA: Use paréntesis en lugar de corchetes para el operador IN")
                print("Ejemplo correcto: status IN (\"Open\", \"Resolved\")")
                print("Ejemplo incorrecto: status IN [\"Open\", \"Resolved\"]")
            
            return None
    except Exception as e:
        print(f"❌ Error de conexión: {str(e)}")
        return None

def main():
    print("=" * 60)
    print("🔍 CONSULTA DE TICKETS JIRA")
    print("=" * 60)
    print("Ingrese su consulta JQL. Por ejemplo:")
    print("  - project = 'BS' AND status = 'Open'")
    print("  - project = 'BS' AND created >= -7d")
    print("  - project = 'BS' AND customfield_10500 = 'BAJAWARE'")
    print("  - project = 'BS' AND status IN ('Open', 'Waiting for customer')")
    print("  - project = 'BS' AND assignee = EMPTY")
    print("=" * 60)
    print("📋 SINTAXIS CORRECTA:")
    print("  - Use comillas simples o dobles para valores de texto: status = 'Open'")
    print("  - Use paréntesis para listas con IN: status IN ('Open', 'Closed')")
    print("  - Use EMPTY para valores nulos: assignee = EMPTY")
    print("=" * 60)
    
    jql_query = input("Ingrese su consulta JQL: ")
    if not jql_query:
        print("❌ La consulta JQL no puede estar vacía.")
        return
    
    output_file = input("Nombre del archivo de salida (deje en blanco para generar automáticamente): ")
    
    request_jira_tickets(jql_query, output_file)

if __name__ == "__main__":
    main() 