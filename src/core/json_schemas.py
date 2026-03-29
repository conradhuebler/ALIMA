"""JSON-Schema-Definitionen für LLM-Antworten pro Task - Claude Generated

Definiert erwartete JSON-Strukturen und liefert Prompt-Instruktions-Texte
für die JSON-basierte Kommunikation mit LLM-Providern.
"""
import logging
from typing import Optional, Dict

logger = logging.getLogger(__name__)

# Schema-Definitionen pro Task
INITIALISATION_SCHEMA = {
    "type": "object",
    "properties": {
        "keywords": {"type": "array", "items": {"type": "string"}},
        "gnd_classes": {"type": "array", "items": {"type": "string"}},
        "title": {"type": "string"},
    },
    "required": ["keywords"],
}

KEYWORDS_SCHEMA = {
    "type": "object",
    "properties": {
        "analyse": {"type": "string"},
        "keywords": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "keyword": {"type": "string"},
                    "gnd_id": {"type": "string"},
                },
                "required": ["keyword"],
            },
        },
        "keyword_chains": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "chain": {"type": "array", "items": {"type": "string"}},
                    "reason": {"type": "string"},
                },
            },
        },
        "missing_concepts": {"type": "array", "items": {"type": "string"}},
        "missing_superordinate_terms": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["keywords"],
}

DK_CLASS_SCHEMA = {
    "type": "object",
    "properties": {
        "analyse": {"type": "string"},
        "classifications": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "code": {"type": "string"},
                    "type": {"type": "string"},
                },
                "required": ["code"],
            },
        },
    },
    "required": ["classifications"],
}

RVK_SCORING_SCHEMA = {
    "type": "object",
    "properties": {
        "scores": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "code": {"type": "string"},
                    "thematic_fit": {"type": "integer"},
                    "branch_fit": {"type": "integer"},
                    "specificity": {"type": "integer"},
                    "total_score": {"type": "integer"},
                    "reason": {"type": "string"},
                },
                "required": ["code"],
            },
        },
    },
    "required": ["scores"],
}

RVK_ANCHOR_SELECTION_SCHEMA = {
    "type": "object",
    "properties": {
        "anchors": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "keyword": {"type": "string"},
                    "gnd_id": {"type": "string"},
                    "role": {"type": "string"},
                    "reason": {"type": "string"},
                },
                "required": ["keyword"],
            },
        },
    },
    "required": ["anchors"],
}

_TASK_SCHEMAS = {
    "initialisation": INITIALISATION_SCHEMA,
    "keywords": KEYWORDS_SCHEMA,
    "keywords_chunked": KEYWORDS_SCHEMA,
    "rephrase": KEYWORDS_SCHEMA,
    "dk_class": DK_CLASS_SCHEMA,
    "dk_classification": DK_CLASS_SCHEMA,
    "rvk_scoring": RVK_SCORING_SCHEMA,
    "rvk_anchor_selection": RVK_ANCHOR_SELECTION_SCHEMA,
}

# JSON-Instruktions-Texte (deutsch, da Prompts deutsch sind)
_JSON_INSTRUCTIONS = {
    "initialisation": (
        '\n\nAntworte ausschließlich mit einem validen JSON-Objekt im folgenden Format:\n'
        '{\n'
        '  "keywords": ["Schlagwort1", "Schlagwort2", ...],\n'
        '  "gnd_classes": ["22.1 Chemie", "21.1 Physik", ...],\n'
        '  "title": "Kurzer Arbeitstitel"\n'
        '}\n'
        'Gib NUR das JSON-Objekt aus, keinen weiteren Text.'
    ),
    "keywords": (
        '\n\nGib deine gesamte Antwort als ein einziges valides JSON-Objekt aus. '
        'Strukturiere deine vollständige Analyse wie folgt:\n'
        '{\n'
        '  "analyse": "Deine qualitative Analyse der Verschlagwortung als Fließtext. '
        'Diskutiere hier, wie gut die verfügbaren Schlagworte den Text erschließen, '
        'welche Aspekte gut abgedeckt sind und welche fehlen.",\n'
        '  "keywords": [\n'
        '    {"keyword": "Schlagwort1", "gnd_id": "1234567-8"},\n'
        '    {"keyword": "Schlagwort2", "gnd_id": "2345678-9"},\n'
        '    {"keyword": "SchlagwortOhneGND"}\n'
        '  ],\n'
        '  "keyword_chains": [\n'
        '    {"chain": ["Schlagwort1 (1234567-8)", "Schlagwort2 (2345678-9)"], '
        '"reason": "Begründung, warum diese Kombination passend ist"},\n'
        '    {"chain": ["Schlagwort3 (3456789-0)", "Schlagwort4 (4567890-1)"], '
        '"reason": "Begründung"}\n'
        '  ],\n'
        '  "missing_concepts": ["Konzept ohne GND-Abdeckung", ...],\n'
        '  "missing_superordinate_terms": ["Oberbegriff1", "Oberbegriff2"]\n'
        '}\n'
        'WICHTIG: Nutze nur Schlagworte aus der gelieferten Liste. '
        'Nutze keine Synonyme oder alternative Schreibweisen. '
        'Gib NUR das JSON-Objekt aus, keinen weiteren Text.'
    ),
    "keywords_chunked": (
        '\n\nAntworte ausschließlich mit einem validen JSON-Objekt im folgenden Format:\n'
        '{\n'
        '  "keywords": [\n'
        '    {"keyword": "Schlagwort1", "gnd_id": "1234567-8"},\n'
        '    {"keyword": "Schlagwort2", "gnd_id": "2345678-9"}\n'
        '  ]\n'
        '}\n'
        'Gib NUR die relevanten Schlagworte aus der Liste als JSON aus, keinen weiteren Text.'
    ),
    "rephrase": None,  # Uses fallback to "keywords"
    "dk_classification": None,  # Uses same as "dk_class"
    "dk_class": (
        '\n\nGib deine gesamte Antwort als ein einziges valides JSON-Objekt aus:\n'
        '{\n'
        '  "analyse": "Deine Analyse und Begründung der Klassifikationszuordnung als Fließtext.",\n'
        '  "classifications": [\n'
        '    {"code": "DK 615.9", "type": "DK"},\n'
        '    {"code": "RVK QC 130", "type": "RVK"}\n'
        '  ]\n'
        '}\n'
        'Gib NUR das JSON-Objekt aus, keinen weiteren Text.'
    ),
    "rvk_scoring": (
        '\n\nGib deine gesamte Antwort als ein einziges valides JSON-Objekt aus:\n'
        '{\n'
        '  "scores": [\n'
        '    {\n'
        '      "code": "RVK XX 1234",\n'
        '      "thematic_fit": 0,\n'
        '      "branch_fit": 0,\n'
        '      "specificity": 0,\n'
        '      "total_score": 0,\n'
        '      "reason": "Kurze Begründung"\n'
        '    }\n'
        '  ]\n'
        '}\n'
        'Nutze nur die angegebenen RVK-Kandidaten. Werte alle Kandidaten auf einer Skala von 0 bis 5 je Kriterium. '
        'Gib NUR das JSON-Objekt aus, keinen weiteren Text.'
    ),
    "rvk_anchor_selection": (
        '\n\nGib deine gesamte Antwort als ein einziges valides JSON-Objekt aus:\n'
        '{\n'
        '  "anchors": [\n'
        '    {\n'
        '      "keyword": "Schlagwort1",\n'
        '      "gnd_id": "1234567-8",\n'
        '      "role": "core",\n'
        '      "reason": "Kurzbegründung"\n'
        '    }\n'
        '  ]\n'
        '}\n'
        'Nutze nur Schlagworte aus der gelieferten Liste. Waehle hoechstens 8 RVK-Anker. '
        'Bevorzuge Kernkonzepte und vermeide Kontext-, Medien- oder Institutionsterme, sofern sie nicht zentrales Thema sind. '
        'Gib NUR das JSON-Objekt aus, keinen weiteren Text.'
    ),
}


def get_schema_for_task(task_name: str) -> Optional[Dict]:
    """Liefert Schema-Dict für einen Task oder None - Claude Generated"""
    return _TASK_SCHEMAS.get(task_name)


_TASK_FALLBACKS = {
    "keywords_chunked": "keywords_chunked",  # Has own instruction
    "rephrase": "keywords",
    "dk_classification": "dk_class",
}


def get_json_instruction_text(task_name: str) -> str:
    """Liefert den JSON-Instruktions-Text für Prompt-Injection - Claude Generated

    Fällt auf task-spezifischen Fallback oder 'keywords' zurück.
    """
    text = _JSON_INSTRUCTIONS.get(task_name)
    if text is None:
        # Check task-specific fallback
        fallback = _TASK_FALLBACKS.get(task_name, "keywords")
        text = _JSON_INSTRUCTIONS.get(fallback, "")
    return text
