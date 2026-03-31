import os

from ollama import Client


auth_token = os.environ.get("OLLAMA_AUTH_TOKEN", "")
headers = {'Authorization': auth_token} if auth_token else {}

client = Client(
    host="https://ollama.com",
    headers=headers,
)

messages = [
  {
    'role': 'user',
    'content': 'Why is the sky blue?',
  },
]

for part in client.chat('gpt-oss:120b', messages=messages, stream=True):
  print(part['message']['content'], end='', flush=True)
