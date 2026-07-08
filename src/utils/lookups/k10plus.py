"""K10plus Paketsigel lookup plugin - Claude Generated.

Wraps the K10plus package-seal harvester (`src/utils/k10plus_resolver.py`) as a
lookup plugin exposing one agent tool: a Paketsigel (e.g. ``ZDB-2-CMS``) → its
bibliographic records via the K10plus SRU/PICA-XML API. Output is capped
(``max_records``) since a package can hold thousands of records. The direct batch
usages (`pipeline_cmd.fetch_dois_for_siegel`, the batch dialog) stay untouched.
"""

from __future__ import annotations

from typing import Any, List

from src.core.plugins.schema import INT, ConfigField, PluginDoc

from .registry import LookupToolSpec, register_lookup


@register_lookup
class K10PlusLookup:
    id = "k10plus"
    label = "K10plus (Paketsigel)"

    def __init__(self, **config: Any):
        self._config = config or {}

    @classmethod
    def config_fields(cls) -> List[ConfigField]:
        return [ConfigField(key="max_records", label="Max. Datensätze", kind=INT, default=50)]

    @classmethod
    def doc(cls) -> PluginDoc:
        return PluginDoc(
            description="K10plus-Paketsigel-Harvester: die bibliografischen "
            "Datensätze eines Pakets (Siegel) via K10plus-SRU/PICA-XML.",
            input="Ein Paketsigel, z.B. 'ZDB-2-CMS'.",
            output="Bibliografische Datensätze (ppn, doi, title, authors, year, "
            "ddc, subjects, url …), gedeckelt durch max_records.",
        )

    @classmethod
    def mcp_tool_specs(cls) -> List[LookupToolSpec]:
        return [LookupToolSpec(
            name="k10plus_package",
            description="Fetch the bibliographic records of a K10plus Paketsigel "
            "(package seal), e.g. 'ZDB-2-CMS'. Returns records (ppn, doi, title, "
            "authors, year, ddc, subjects, url). Capped by max_records.",
            parameters={
                "type": "object",
                "properties": {
                    "siegel": {"type": "string", "description": "K10plus Paketsigel, e.g. ZDB-2-CMS"},
                    "max_records": {"type": "integer", "default": 50, "description": "Max records to return"},
                },
                "required": ["siegel"],
            },
            method="fetch_package",
            cache_key_param="siegel",
        )]

    def fetch_package(self, siegel: str, max_records: int = None) -> dict:
        from dataclasses import asdict

        from src.utils.k10plus_resolver import fetch_records_for_siegel

        cap = int(max_records if max_records is not None else self._config.get("max_records", 50) or 50)
        records = fetch_records_for_siegel(str(siegel))
        recs = [asdict(r) for r in records[:cap]]
        return {"siegel": siegel, "total": len(records), "returned": len(recs), "records": recs}
