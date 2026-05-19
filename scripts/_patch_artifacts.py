"""Patch the two prompts that came out garbled after marker-strip."""
import json
from pathlib import Path

PATH = Path("prompts.json")
data = json.loads(PATH.read_text(encoding="utf-8"))

new_keywords_user = (
    "**Aufgabe**:\n"
    "Analysiere den folgenden Abstract und wende einen systematischen "
    "GND-Schlagwortungsprozess an. Wähle aus den vorgegebenen GND-Schlagworten "
    "die relevanten aus und bilde Schlagwortketten.\n\n"
    "**Strikte Regeln**:\n"
    "1. Nutze NUR Schlagworte aus der vorgegebenen Liste (keine Erfindungen).\n"
    "2. Bilde Schlagwortketten (10–20 Endkeywords) zur Erhöhung der Spezifität.\n"
    "3. Dokumentiere fehlende Konzepte als `missing_concepts`.\n\n"
    "**Format** (direkt als JSON, ohne Fences, ohne Marker, ohne Erläuterung):\n"
    "{{\n"
    '  "keywords": [{{"keyword": "...", "gnd_id": "..."}}],\n'
    '  "keyword_chains": [{{"chain": ["..."], "reason": "..."}}],\n'
    '  "missing_concepts": ["..."]\n'
    "}}\n\n"
    "Abstract:\n{abstract}\n\n"
    "GND-Schlagworte:\n{keywords}\n\n"
    "**Output**: Genau ein valides JSON-Objekt. Direkt mit `{` beginnen, keine "
    "Erläuterung außerhalb des JSON."
)

new_dk_user = (
    "**Aufgabe**:\n"
    "Wähle bis zu 10 DK/DDC/RVK-Klassifikationen für den Abstract aus dem "
    "Bibliotheksbestand.\n\n"
    "Regeln:\n"
    "- Nur Codes aus dem gelieferten Bibliotheksbestand.\n"
    "- Hierarchie beachten (spezifische Codes bevorzugen).\n"
    "- Keine Redundanz.\n\n"
    "Abstract:\n{abstract}\n\n"
    "Bibliotheksbestand:\n{keywords}\n\n"
    "**Format** (direkt als JSON, ohne Fences oder Marker):\n"
    "{{\n"
    '  "classifications": [{{"code": "DK ...", "type": "DK"}}],\n'
    '  "analyse": "..."\n'
    "}}\n\n"
    "**Output**: Genau ein valides JSON-Objekt. Direkt mit `{` beginnen, keine "
    "Erläuterung außerhalb des JSON."
)

data["keywords"]["prompts"][1][0] = new_keywords_user
data["dk_classification"]["prompts"][1][0] = new_dk_user

PATH.write_text(json.dumps(data, indent=4, ensure_ascii=False), encoding="utf-8")
print("patched 2 prompts")
