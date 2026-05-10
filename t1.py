import ollama

response = ollama.chat(
    model='qwen2.5:latest',
    messages=[{'role': 'user', 'content': 'Explain LLMs simply'}]
)

print(response['message']['content'])