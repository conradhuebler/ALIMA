from dataclasses import dataclass, field
from typing import List, Dict

@dataclass
class PromptTemplate:
    """Template für einen KI-Prompt"""
    name: str
    description: str
    template: str
    required_variables: List[str]
    model: str = "gemini-1.5-flash"

@dataclass
class PromptConfig:
    """Konfiguration für Prompts"""
    templates: Dict[str, PromptTemplate] = field(default_factory=lambda: {
        "abstract_analysis": PromptTemplate(
            name="abstract_analysis",
            description="Analysiert einen Abstract und extrahiert relevante Schlagworte",
            template="""Basierend auf folgendem Abstract und Keywords, schlage passende deutsche Schlagworte vor.
            
            Abstract:
            {abstract}
            
            Vorhandene Keywords:
            {keywords}
            
            Bitte gib nur eine Liste deutscher Schlagworte zurück, die für eine bibliothekarische Erschließung geeignet sind.
            Die Schlagworte sollten möglichst präzise und spezifisch sein.""",
            required_variables=["abstract", "keywords"]
        ),
        "results_verification": PromptTemplate(
            name="results_verification",
            description="Überprüft die Qualität der gefundenen GND-Schlagworte",
            template="""Wähle aus der Liste der OGND-Schlagworte diejenigen heraus, die zur inhaltlichen Beschreibung des Abstraktes verwendet werden können. Nutze nur Schlagworte, die in der OGND-Liste korrekt auftauchen und keine Synonyme. Führe auch keine weitere Erschließung durch, außer in der letzten Diskussion.


            Abstract:
            {abstract}
            
            Zur Auswahl stehende GND-Schlagworte:
            {gnd_results}
            
            Bitte gib deine Antwort in folgendem Format:
            
                ANALYSE:
                [Deine qualitative Analyse der Verschlagwortung]

                Schlagworte:
                [Liste der passende Schlagwort aus dem Prompt - bitte kommagetrennt. ***Nutze keine Synonyme oder alternative Schreibweisen/Formulierungen***] 

                Schlagworte OGND Eintrage:
                [Liste der passende Konzepte mit der zugeörigen OGND-ID aus dem Prompt  - bitte kommagetrennt]

               Schlagwortketten:
              [Nutze Kombinationen von OGND-Schlagworten um bestimmte Themenbereiche konkret zu beschreiben oder um Konzepte, die durch ein Schlagwort nicht korrekt abgedeckt sind. Trenne die Schlagworte (mit GND-ID) in den Ketten mit Komma. Nimm für jede Schlagwortkette eine neue Zeile - Kommentiere zu jeder Schlagwortkette kurz, wieso diese passend ist]	

                FEHLENDE KONZEPTE:
                [Liste von Konzepten, die noch nicht durch GND abgedeckt sind]""",
            required_variables=["abstract", "gnd_results"]
        )
    })
