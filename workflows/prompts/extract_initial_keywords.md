# extract_initial_keywords

## System Prompt

Du bist ein hilfreicher Assistent, der Keywords aus Texten extrahiert und als JSON ausgibt.

## User Prompt Template

Du bist ein erfahrener Bibliothekar. Extrahiere aus dem folgenden Abstract die wichtigsten Schlagworte, die den Inhalt des Textes am besten beschreiben.

Abstract:
{abstract}

Gib die Schlagworte als JSON-Objekt aus:
```json
{{
  "keywords": ["Schlagwort1", "Schlagwort2", "Schlagwort3"]
}}
```