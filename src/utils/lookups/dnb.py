"""DNB GND-classification lookup plugin - Claude Generated.

Wraps the DNB GND-RDF classification client (`src/core/dnb_utils.py`) as a lookup
plugin exposing one agent tool: a GND-ID → its DNB classification data (RDF types,
category, preferred name, DDC with degree of determinacy, GND subject categories).
This is the raw DNB ``about/lds`` RDF access as a plugin/tool, so it is raw-cached +
per-plugin-toggleable like the other lookups; the GUI DNB-sync routes through it.
"""

from __future__ import annotations

from typing import Any, List

from src.core.plugins.schema import INT, ConfigField, PluginDoc

from .registry import LookupToolSpec, register_lookup


@register_lookup
class DnbLookup:
    id = "dnb"
    label = "DNB-Klassifikation (GND-RDF)"

    def __init__(self, **config: Any):
        self._config = config or {}

    @classmethod
    def config_fields(cls) -> List[ConfigField]:
        return [ConfigField(key="timeout", label="Timeout (s)", kind=INT, default=10)]

    @classmethod
    def doc(cls) -> PluginDoc:
        return PluginDoc(
            description="DNB-Klassifikation über die GND-RDF-Daten der Deutschen "
            "Nationalbibliothek (d-nb.info/gnd/<id>/about/lds): RDF-Typen, Kategorie, "
            "bevorzugter Name, DDC (mit Determiniertheitsgrad) und GND-Sachgruppen.",
            input="Eine GND-ID, z.B. '4045956-1'.",
            output="Klassifikationsdaten: types, category, preferred_name, ddc, "
            "gnd_subject_categories.",
        )

    @classmethod
    def mcp_tool_specs(cls) -> List[LookupToolSpec]:
        return [LookupToolSpec(
            name="dnb_classification",
            description="Fetch the DNB GND classification for a GND-ID from the DNB "
            "RDF (d-nb.info). Returns RDF types, category, preferred name, DDC codes "
            "(with degree of determinacy) and GND subject categories.",
            parameters={
                "type": "object",
                "properties": {
                    "gnd_id": {"type": "string", "description": "GND-ID, e.g. 4045956-1"},
                },
                "required": ["gnd_id"],
            },
            method="classify",
            cache_key_param="gnd_id",
        )]

    def classify(self, gnd_id: str) -> dict:
        from src.core.dnb_utils import get_dnb_classification

        return get_dnb_classification(str(gnd_id), timeout=int(self._config.get("timeout", 10) or 10))
