from transformers import AutoTokenizer, AutoModel
import torch
import gc
import time
from components.commons import modules, table_docs, table_desc, jina_path, device

jina=AutoModel.from_pretrained(
    jina_path, 
    trust_remote_code=True, 
    torch_dtype=torch.float16, 
    local_files_only=True
).to("cpu")
tokenizer_jina=AutoTokenizer.from_pretrained(jina_path, trust_remote_code=True, use_fast=False)

def module_ranker(prompt: str) -> str:
    pairs=[[prompt, schema] for schema in modules.values()]
    
    jina.to(device)
    with torch.no_grad():
        scores=jina.compute_score(pairs, max_length=1024, doc_type="text")

    jina.to("cpu")
    torch.cuda.empty_cache()
    best_idx=int(torch.argmax(torch.tensor(scores, device="cpu")))
    return list(modules.keys())[best_idx]

def choose_table(query: str) -> str:
    pairs=[[query, schema] for schema in table_docs.values()]
    
    jina.to(device)
    with torch.no_grad():
        scores=jina.compute_score(pairs, max_length=1024, doc_type="text")

    jina.to("cpu")
    torch.cuda.empty_cache()
    best_idx=int(torch.argmax(torch.tensor(scores, device="cpu")))
    choose=list(table_docs.keys())[best_idx]
    if choose=="GENERALES":
        return choose
    else:
        return table_desc[choose]