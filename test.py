import ollama

response = ollama.chat(
    model='qwen2.5:7b',
    messages=[{'role': 'user', 'content': 'Explain LLMs simply'}]
)

print(response['message']['content'])