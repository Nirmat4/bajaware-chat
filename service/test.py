from ollama import chat

def test_ollama(prompt):
  stream=chat(model="qwen3:30b-a3b", messages=[{'role': 'user', 'content': prompt}], stream=True)
  response=""
  for chunk in stream:
      print(f"[cyan]{chunk['message']['content']}[/cyan]", end='', flush=True)
      response+=(chunk['message']['content'])
  print()