from transformers import AutoTokenizer, AutoModel
import torch
import gc
import time
from commons import modules, table_docs, table_desc, jina_path

def load_model():
    jina=AutoModel.from_pretrained(
        jina_path, 
        trust_remote_code=True, 
        torch_dtype=torch.float16, 
        local_files_only=True,
        device_map='auto',
        max_memory={0: "3000MB"}
    )
    tokenizer_jina=AutoTokenizer.from_pretrained(jina_path, trust_remote_code=True, use_fast=True)
    return jina, tokenizer_jina

def dep_model(model, tokenizer):
    try:
        model.cpu()
    except Exception:
        pass
    model=None
    tokenizer=None
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(1)
    return model, tokenizer

def module_ranker(prompt: str) -> str:
    jina, tokenizer_jina=load_model()
    pairs=[[prompt, schema] for schema in modules.values()]
    
    with torch.no_grad():
        scores=jina.compute_score(pairs, max_length=1024, doc_type="text")

    best_idx=int(torch.argmax(torch.tensor(scores, device="cpu")))
    jina, tokenizer_jina=dep_model(jina, tokenizer_jina)
    return list(modules.keys())[best_idx]

def choose_table(query: str) -> str:
    jina, tokenizer_jina=load_model()
    pairs=[[query, schema] for schema in table_docs.values()]
    
    with torch.no_grad():
        scores=jina.compute_score(pairs, max_length=1024, doc_type="text")

    best_idx=int(torch.argmax(torch.tensor(scores, device="cpu")))
    choose=list(table_docs.keys())[best_idx]
    jina, tokenizer_jina=dep_model(jina, tokenizer_jina)
    return table_desc[choose]