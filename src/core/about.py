"""Project self-description — one source of truth for "what is ALIMA?".

The chat agent must be able to answer questions about ALIMA itself: what the
acronym stands for, what it does, where it was developed, which publication
describes it, under which licence it stands. Those facts sit here rather than in
a prompt on purpose — the whole chat ruleset is built on "name nothing from
memory, take it from a tool", and a citation is exactly the kind of thing a
model will otherwise invent a plausible-but-wrong version of (a page range, a
volume, a year).

Keep in sync with ``README.md`` and ``CLAUDE.md``; ``tests/test_about.py``
pins the publication fields against the README so the three cannot drift.

Claude Generated.
"""

from __future__ import annotations

from typing import Any, Dict

NAME = "ALIMA"
EXPANSION = "Automated Library Indexing and Metadata Assignment"

SUMMARY = (
    "Pipeline für die bibliothekarische Sacherschließung: sie analysiert einen "
    "Text (oder ein über DOI/ISBN/PPN aufgelöstes Werk) mit einem LLM, sucht "
    "die Begriffe gegen GND-Quellen und Katalogdaten und schlägt GND-konforme "
    "Schlagworte, Schlagwortketten sowie DK/RVK-Klassifikationen vor."
)

INSTITUTION = (
    'Universitätsbibliothek "Georgius Agricola" der TU Bergakademie Freiberg'
)

REPOSITORY = "https://github.com/conradhuebler/ALIMA"
LICENSE = "LGPL v3"

# Bibliographic record of the paper describing ALIMA. Fields verified against
# Crossref for DOI 10.1515/bfp-2026-0014.
PUBLICATION: Dict[str, Any] = {
    "authors": ["Hübler, Conrad"],
    "title": (
        "ALIMA – Ein RAG-basiertes System zur LLM-gestützten Sacherschließung: "
        "Prototypentwicklung und erste Erfahrungen aus der Praxis"
    ),
    "journal": "Bibliothek Forschung und Praxis",
    "volume": "50",
    "issue": "2",
    "year": "2026",
    "pages": "197–216",
    "doi": "10.1515/bfp-2026-0014",
    "url": "https://doi.org/10.1515/bfp-2026-0014",
    "open_access": True,
    "license": "CC BY 4.0",
    "citation": (
        "Hübler, Conrad: ALIMA – Ein RAG-basiertes System zur LLM-gestützten "
        "Sacherschließung: Prototypentwicklung und erste Erfahrungen aus der "
        "Praxis. In: Bibliothek Forschung und Praxis 50 (2026), Heft 2, "
        "S. 197–216. DOI: 10.1515/bfp-2026-0014"
    ),
}

CONTRIBUTORS = ["Conrad Hübler", "Claude und Gemini AI"]

ACKNOWLEDGEMENTS = (
    "Fachreferats- und IT-Team der Universitätsbibliothek; Patrick Reichel für "
    "die effiziente Lobid-Abfrage."
)

# The honest maturity caveat from the README's "Anmerkung des Autors". Someone
# asking a system whether they can rely on its output deserves the same answer
# the README gives a reader — and an LLM asked to self-assess will otherwise
# reach for the confident version. - Claude Generated
STATUS = (
    "In Entwicklung. ALIMA und seine Dokumentation sind größtenteils mit "
    "LLM-Unterstützung entstanden; die Dokumentation hinkt dem Code stellenweise "
    "hinterher, und nicht jede beschriebene Funktion ist als fertig anzusehen. "
    "Vorschläge sind Vorschläge: die fachliche Prüfung bleibt beim Fachreferat."
)

# The two ways a run can be orchestrated. Both produce the same kind of result;
# they differ in who decides the order of the steps.
PIPELINE_MODES = {
    "classic": (
        "Feste 5-Schritt-Pipeline: Eingabe → freie Schlagworte → GND-Suche → "
        "Verifikation/Auswahl → DK-Klassifikation."
    ),
    "agentic": (
        "YAML-definierte Workflows (v4), in denen ein Meta-Agent Schritte "
        "auswählt, wiederholt oder überspringt; Standard-Workflow alima_v51."
    ),
}

# Questions that look like "about ALIMA" but are answered by live state, not by
# these static facts. Named here so the agent is pointed at the right tool
# instead of answering a data question from a description. - Claude Generated
SEE_ALSO = {
    "list_plugins": "welche Such- und Eingabequellen gerade aktiv sind",
    "list_workflows": "welche Workflows ausführbar sind",
    "get_db_stats": "wie viele GND-Einträge, Mappings und Klassifikationen die Datenbank hält",
}

# What ALIMA can be asked to do, in the words an operator would use. Kept short:
# the detail belongs to the tools that actually do the work, and a list that
# tries to enumerate every feature goes stale.
CAPABILITIES = [
    "GND-Schlagworte und Schlagwortketten zu einem Werk vorschlagen (RSWK/RDA)",
    "DK- und RVK-Klassifikationen aus Katalogdaten ableiten",
    "Eingaben auflösen: freier Text, PDF, Bild (OCR), DOI, ISBN, PPN, URL",
    "Klassische 5-Schritt-Pipeline und agentische YAML-Workflows",
    "Batch-Verarbeitung vieler Werke",
    "Drei Oberflächen auf derselben Konfiguration: GUI (PyQt6), CLI, Webapp",
    "K10plus/WinIBW-Export der Ergebnisse",
]


def about_payload() -> Dict[str, Any]:
    """The project facts as a JSON-serialisable dict. - Claude Generated"""
    return {
        "name": NAME,
        "expansion": EXPANSION,
        "summary": SUMMARY,
        "status": STATUS,
        "institution": INSTITUTION,
        "capabilities": list(CAPABILITIES),
        "pipeline_modes": dict(PIPELINE_MODES),
        "publication": dict(PUBLICATION),
        "repository": REPOSITORY,
        "license": LICENSE,
        "contributors": list(CONTRIBUTORS),
        "acknowledgements": ACKNOWLEDGEMENTS,
        "see_also": dict(SEE_ALSO),
    }
