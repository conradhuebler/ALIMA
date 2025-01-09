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
            template="""Bitte analysiere die Qualität der Verschlagwortung für folgenden Abstract.

            Abstract:
            {abstract}
            
            Gefundene GND-Schlagworte:
            {gnd_results}
            
            Bitte gib deine Antwort in folgendem Format:
            
            ANALYSE:
            [Deine qualitative Analyse der Verschlagwortung]
            
            SUCHBEGRIFFE:
            [Liste der Konzepte, nach denen gesucht werden soll - bitte kommagetrennt]
            
            FEHLENDE KONZEPTE:
            [Liste von Konzepten, die noch nicht durch GND abgedeckt sind]""",
            required_variables=["abstract", "gnd_results"]
        )
    })
