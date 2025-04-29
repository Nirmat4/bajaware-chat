from transformers import TapexTokenizer, BartForConditionalGeneration
import pandas as pd

tokenizer = TapexTokenizer.from_pretrained("../models/tapex-large-finetuned-wtq")
model = BartForConditionalGeneration.from_pretrained("../models/tapex-large-finetuned-wtq")

table = pd.read_csv("../database/inventario_reportes.csv", dtype=str)

# tapex accepts uncased input since it is pre-trained on the uncased corpus
query = "Que reportes son de CLAVE_ENTIDADREGULADA BM?"
encoding = tokenizer(table=table, query=query, return_tensors="pt")

outputs = model.generate(**encoding)

print(tokenizer.batch_decode(outputs, skip_special_tokens=True))
# [' 2008.0']