import sqlite3
from ollama import chat
from rich.status import Status
import subprocess
from rich import print
from tabulate import tabulate
import pandas as pd
from components.nlp import clean_prompt

def sql_search(prompt):
  prompt=clean_prompt(prompt)
  conn=sqlite3.connect("database/database.db")
  prompt_compuest=f'''
  ### Instructions:
  Your task is to convert a question into a SQL query, given a SQLite3 database schema.
  Adhere to these rules:
  - **Deliberately go through the question and database schema word by word** to appropriately answer the question
  - **Use Table Aliases** to prevent ambiguity. For example, `SELECT table1.col1, table2.col1 FROM table1 JOIN table2 ON table1.id = table2.id`.
  - When creating a ratio, always cast the numerator as float

  ### Input:
  Generate a SQL query that answers the question "{prompt}".
  This query will run on a database whose schema is represented in this string:
  CREATE TABLE REPORTES (
    CLAVE_REPORTE varchar(100) NOT NULL, -- Unique identifier for the report. Encodes metadata like country, entity, series, subseries, section, and version.
    NOMBRE varchar(100) NULL, -- Human-readable name of the report.
    PAIS varchar(10) NULL, -- Country code to which the report belongs
    ENTIDAD_REGULADA varchar(10) NULL, -- Code of the financial institution required to submit the report
    REGULADOR varchar(10) NULL, -- Code of the regulatory authority requesting the report
    SERIE varchar(15) NULL, -- Report series code that groups thematically related reports.
    SUBSERIE varchar(15) NULL, -- Subgroup within the series for further classification.
    GRUPO varchar(15) NULL, -- Additional thematic or functional grouping of the report.
    SECCION varchar(50) NULL, -- Code identifying the specific section within the report.
    VERSION varchar(3) NULL, -- Format version of the report
    PERIODO varchar(15) NULL, -- Reporting frequency or period
    DESCRIPCION varchar(500) NULL, -- Extended description of the report’s content, scope, or purpose.
    FECHA_ENTREGA varchar(100) NULL, -- Scheduled submission date for the report; distinct from the reporting period.
    CARACTERISTICAS varchar(500) NULL, -- Additional characteristics of the report such as format requirements or notes.
    REGULACION_REPORTE varchar(50) NULL, -- Reference to the regulation or legal framework that mandates the report.
    FECHA_ALTA datetime NULL, -- Date when the report record was first registered in the system.
    FECHA_ACTUALIZADA datetime NULL, -- Date of the most recent update to the report metadata.
    CLAVE_REPORTE_GENERAL varchar(100) NULL, -- General identifier for the base report, excluding version or section (parent key).
    CLASIFICACION date NULL, -- Date when the report was classified or formally activated.
    VIGENTE INTEGER NULL, -- Flag indicating whether the report is currently valid (1 = active, 0 = inactive).
    CONSTRAINT PK_CLAVE_REPORTE PRIMARY KEY (CLAVE_REPORTE) 
  );

  CREATE TABLE VALIDACIONES (
    CLAVE_VALIDACION varchar(100) NOT NULL, -- Unique identifier for the validation rule.
    PAIS varchar(10) NOT NULL, -- Country code where the validation applies.
    ENTIDAD_REGULADA varchar(10) NOT NULL, -- Code of the financial institution subject to the validation.
    REGULADOR varchar(10) NOT NULL, -- Code of the regulatory authority enforcing the validation.
    CLAVE_REPORTE varchar(100) NOT NULL, -- Identifier of the report associated with this validation.
    ID_VALIDACION_ANTERIOR varchar(100) NULL, -- Identifier of the previous version of the validation (if applicable).
    DESCRIPCION varchar(2000) NULL, -- Detailed explanation of the purpose and logic behind the validation rule.
    TIPO varchar(30) NULL, -- Type of validation 
    TIPO_CALCULO varchar(50) NULL, -- Calculation method used in the validation
    FECHA_ALTA datetime NULL, -- Date when the validation rule was first registered.
    FECHA_ACTUALIZADA datetime NULL, -- Date of the latest update to the validation rule.
    CAMPO varchar(250) NULL, -- Fields or columns within the report to which the validation applies.
    CONSTRAINT PK_CLAVE_VALIDACION PRIMARY KEY (CLAVE_VALIDACION)
  );

  -- REPORTES.CLAVE_REPORTE can be joined with VALIDACIONES.CLAVE_REPORTE

  ### Response:
  Based on your instructions, here is the SQL query I have generated to answer the question "{prompt}":
  ```sql
  '''
  stream = chat(
      model="duckdb-nsql:7b",
      messages=[{"role": "user", "content": prompt_compuest}],
      stream=True,
  )
  response=""
  with Status("[bold light_steel_blue]generating query...[/]", spinner="dots", spinner_style="light_steel_blue bold") as status:
      for chunk in stream:
          response+=(chunk['message']['content'])
  subprocess.run(['ollama', 'stop', "duckdb-nsql:7b"])
  response=response.split('```')[0].strip()
  print(f"[bold sky_blue2]{response}[/]")

  try:
    query_result=pd.read_sql_query(response, conn)    
    num_rows=len(query_result)
    context=query_result.sample(n=min(10, num_rows), random_state=42)
    context=tabulate(context, headers='keys', tablefmt='grid')
    context=f"{response}\n{context}"
    print(f"[bold dark_sea_green2]{context}[/]")
  except Exception as e:
    print(f"[bold red]error en la ejecución de la consulta: {e}[/red]")

  return context