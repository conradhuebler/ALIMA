"""K10plus Paketsigel lookup plugin - Claude Generated.

Wraps the K10plus package-seal harvester (`src/utils/k10plus_resolver.py`) as a
lookup plugin exposing one agent tool: a Paketsigel (e.g. ``ZDB-2-CMS``) → its
bibliographic records via the K10plus SRU/PICA-XML API. Output is capped
(``max_records``) since a package can hold thousands of records. The batch callers
(CLI siegel harvest, GUI batch dialog) route through ``fetch_records`` as well —
one call path per source (Phase D).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, List, Optional

from src.core.plugins.schema import INT, TEXT, ConfigField, PluginDoc

from .registry import LookupToolSpec, register_lookup

if TYPE_CHECKING:  # pragma: no cover
    import logging

    from src.utils.k10plus_resolver import K10PlusRecord


@register_lookup
class K10PlusLookup:
    id = "k10plus"
    label = "K10plus (Paketsigel)"

    def __init__(self, **config: Any):
        self._config = config or {}

    @classmethod
    def config_fields(cls) -> List[ConfigField]:
        return [
            ConfigField(key="max_records", label="Max. Datensätze", kind=INT, default=50),
            # A K10plus package can hold thousands of records — too large for the DB
            # raw cache, so this plugin caches raw XML to a directory instead (its
            # "cache capability"). Empty → no dir cache. - Claude Generated
            ConfigField(key="cache_dir", label="Cache-Verzeichnis (raw XML)", kind=TEXT, default=""),
        ]

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
            name="k10plus_resolve",
            description="Identifier-Crosswalk über K10plus (WP-D1 P4): löst eine "
            "DOI, ISBN oder PPN zum bibliografischen K10plus-Datensatz auf und "
            "liefert die jeweils ANDEREN Identifier mit (ppn/doi/isbn) plus "
            "Titel/Autoren/DDC/Schlagwörter. Nackte 10-stellige Nummern werden "
            "als PPN gedeutet — für ISBN-10 ohne Bindestriche kind='isbn' setzen.",
            parameters={
                "type": "object",
                "properties": {
                    "identifier": {"type": "string", "description": "DOI (auch doi.org-URL), ISBN oder PPN"},
                    "kind": {"type": "string", "enum": ["doi", "isbn", "ppn"],
                             "description": "Optional: Identifier-Typ erzwingen"},
                },
                "required": ["identifier"],
            },
            method="resolve_identifier",
            cache_key_param="identifier",
        ), LookupToolSpec(
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

    def resolve_identifier(
        self,
        identifier: str,
        kind: Optional[str] = None,
        logger: "Optional[logging.Logger]" = None,
    ) -> dict:
        """DOI/ISBN/PPN → K10plus record incl. the other identifiers - Claude Generated.

        The crosswalk tool body (WP-D1 P4). Returns the record normalised as a
        ``BibRecord`` dict, so downstream consumers (P1 analysis text, P2
        priors, P3 GND subjects) read the one canonical shape.
        """
        from dataclasses import asdict

        from src.core.bib_record import to_bibrecord
        from src.utils.k10plus_resolver import (
            detect_identifier_kind,
            fetch_record_for_identifier,
        )

        kind = (kind or detect_identifier_kind(identifier)).lower()
        record = fetch_record_for_identifier(identifier, kind=kind, logger=logger)
        if record is None:
            return {
                "success": False,
                "identifier": identifier,
                "kind": kind,
                "error": f"Kein verifizierter K10plus-Treffer für {kind} '{identifier}'",
            }
        bib = to_bibrecord(asdict(record), "k10plus")
        return {
            "success": True,
            "identifier": identifier,
            "kind": kind,
            "identifiers": bib.identifiers,
            "record": bib.to_dict(),
        }

    def fetch_records(
        self,
        siegel: str,
        cache_dir: Optional[str] = None,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        logger: "Optional[logging.Logger]" = None,
    ) -> "List[K10PlusRecord]":
        """All records of a Paketsigel as ``K10PlusRecord`` objects (uncapped) - Claude Generated.

        The single K10plus harvest entry point: the CLI/GUI batch callers and the
        JSON ``fetch_package`` tool wrapper both go through here, so there is one
        call path per source. ``cache_dir`` overrides the per-instance setting when
        given (``None`` → fall back to the plugin's configured ``cache_dir``); the
        result is *not* capped — capping is a tool-boundary concern (``fetch_package``).
        """
        from src.utils.k10plus_resolver import fetch_records_for_siegel

        cd = cache_dir if cache_dir is not None else (self._config.get("cache_dir") or None)
        return fetch_records_for_siegel(
            str(siegel), cache_dir=cd, progress_callback=progress_callback, logger=logger
        )

    def load_cached(
        self,
        siegel: str,
        cache_dir: Optional[str] = None,
        logger: "Optional[logging.Logger]" = None,
    ) -> "List[K10PlusRecord]":
        """Cached records of a Paketsigel, no API call - Claude Generated.

        Cache-only read counterpart to :meth:`fetch_records` (same ``cache_dir``
        fallback to the instance setting); empty list when nothing is cached.
        """
        from src.utils.k10plus_resolver import load_cached_records

        cd = cache_dir if cache_dir is not None else (self._config.get("cache_dir") or None)
        if not cd:
            return []
        return load_cached_records(str(cd), str(siegel), logger)

    def fetch_package(self, siegel: str, max_records: int = None) -> dict:
        from dataclasses import asdict

        cap = int(max_records if max_records is not None else self._config.get("max_records", 50) or 50)
        records = self.fetch_records(siegel)
        recs = [asdict(r) for r in records[:cap]]
        return {"siegel": siegel, "total": len(records), "returned": len(recs), "records": recs}
