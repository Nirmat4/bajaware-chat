import spacy
from spacy.language import Language
import hjson
import re
from rich import print

nlp=spacy.load("es_core_news_lg")
with open("assets/replacements.hjson", "r") as file:
  replacements=hjson.load(file)

@Language.component("replacer_component")
def replacer_component(doc):
  text=doc.text
  for k, v in replacements.items():
    pattern=r"\b" + re.escape(k) + r"\b"
    text=re.sub(pattern, v, text, flags=re.IGNORECASE)
  new_doc=nlp.make_doc(text)
  return new_doc
nlp.add_pipe("replacer_component", first=True)

def clean_prompt(prompt):
  text=(nlp(prompt)).text
  print(f"[bold light_cyan1]{text}[/]")
  return text