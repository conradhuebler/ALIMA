from ollama import Client

client = Client(
    host="https://ollama.com",
    headers={'Authorization': '3906daf19dbc42bab768c2c2ae5df634.zksO6Spr0fDqAcLM0eYr8Jaz'}
)

messages = [
  {
    'role': 'user',
    'content': 'Why is the sky blue?',
  },
]

for part in client.chat('gpt-oss:120b', messages=messages, stream=True):
  print(part['message']['content'], end='', flush=True)
