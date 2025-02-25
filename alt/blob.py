import requests
import json

url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"
api_key = "AIzaSyA1FlLQUE2TO0_ryEcWl1SvYdamscqOts4"  # Ersetze dies durch deinen API-Key

headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

payload = {
    "model": "gemini-pro",  # Oder ein anderes verfügbares Modell
    "prompt": "Schreibe einen kurzen Science-Fiction-Text.",
    "temperature": 0.7,       # Beispielwert für Temperatur
    "seed": 42,             # Beispielwert für Seed
    # Weitere Parameter hier hinzufügen...
}

response = requests.post(url, headers=headers, json=payload)

if response.status_code == 200:
    data = response.json()
    print(data["text"])  # Gib die generierte Textausgabe aus
else:
    print(f"Fehler: {response.status_code} - {response.text}")
