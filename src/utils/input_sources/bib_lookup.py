"""ISBN/PPN input sources — catalog record → analysis text - Claude Generated.

WP-D1 P1 (record → analysis input): an identifier is looked up via SRU
(``MarcXmlClient``), the hit is normalised through :func:`to_bibrecord` and
formatted with :meth:`BibRecord.to_analysis_text` — the first real consumer of
the ``BibRecord`` seam. The batch ISBN/PPN path delegates to
:func:`lookup_bibrecord` (it additionally needs the record's title/authors for
file naming, which the plain ``extract`` tuple cannot carry).

No ``mcp_tool_spec``: agent-facing identifier lookups already exist in the
``lookup`` category (k10plus); exposing a second tool for the same job would
duplicate them.
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional, Tuple

from src.core.bib_record import BibRecord, to_bibrecord
from src.core.plugins.schema import INT, TEXT, ConfigField, PluginDoc

from .registry import register_input_source


def lookup_bibrecord(
    identifier: str,
    *,
    search_type: str = "keyword",
    preset: str = "k10plus",
    timeout: int = 30,
    logger: Any = None,
) -> BibRecord:
    """Look up one identifier via SRU and return the first hit as a BibRecord.

    Raises ``ValueError`` when the catalog has no hit — the caller decides how
    to surface that (batch wraps it per item, ``extract`` lets it propagate).
    """
    from src.utils.clients.marcxml_client import MarcXmlClient

    if logger:
        logger.info(f"Looking up {identifier} via {preset} (search_type={search_type})...")
    client = MarcXmlClient(preset=preset, timeout=timeout, max_records=1)
    results = client.search(str(identifier).strip(), search_type=search_type)
    if not results:
        raise ValueError(f"Keine Treffer für {identifier}")
    return to_bibrecord(results[0], "sru")


def crosswalk_doi_record(doi: str, logger: Any = None) -> Optional[BibRecord]:
    """DOI → K10plus-Katalog-Record als :class:`BibRecord` (WP-D1-Adoption).

    Die DOI-Anreicherung: zu einer DOI den Katalog-Datensatz holen, damit
    dessen Klassifikationen (P2-Priors) und GND-Subjects (P3-Signale) in die
    Pipeline fließen. Gate ist das **k10plus-Lookup-Plugin** (Plugins-Tab) —
    deaktiviert ⇒ keine Anreicherung, kein eigenes Config-Feld. Viele
    K10plus-Records tragen keine DOI; bei einem Miss wird die im DOI-Suffix
    eingebettete ISBN probiert (Springer-Buch-DOIs: ``10.1007/978-…``).
    Best-effort: ``None`` bei Miss, Fehler oder deaktiviertem Plugin.
    - Claude Generated
    """
    from dataclasses import asdict

    from src.utils.error_visibility import log_caught

    log = logger or logging.getLogger(__name__)
    try:
        from src.utils.lookups.resolve import build_lookup

        if build_lookup(None, "k10plus") is None:
            log.debug("DOI-Anreicherung übersprungen: k10plus-Lookup deaktiviert")
            return None

        from src.utils.k10plus_resolver import fetch_record_for_identifier

        record = fetch_record_for_identifier(doi, kind="doi", logger=log)
        if record is None:
            # eingebettete ISBN im DOI-Suffix (978/979 + 13 Ziffern)?
            compact = str(doi or "").rsplit("/", 1)[-1].replace("-", "")
            if compact.isdigit() and len(compact) == 13 and compact.startswith(("978", "979")):
                log.info(f"DOI {doi}: kein K10plus-Treffer — ISBN-Fallback {compact}")
                record = fetch_record_for_identifier(compact, kind="isbn", logger=log)
        if record is None:
            return None
        return to_bibrecord(asdict(record), "k10plus")
    except Exception as e:  # Netz/Config — Anreicherung ist strikt best-effort
        log_caught(log, e, "crosswalk_doi_record")
        return None


class _BibLookupSource:
    """Shared base: identifier → SRU record → analysis text."""

    id = ""
    label = ""
    _search_type = "keyword"
    _identifier_label = ""

    def __init__(self, **config: Any):
        self._config = config or {}

    @classmethod
    def config_fields(cls) -> List[ConfigField]:
        return [
            ConfigField(
                key="preset", label="SRU-Preset", kind=TEXT, default="k10plus",
                help="Endpoint-Preset des MarcXmlClient (z.B. k10plus, dnb, swb).",
            ),
            ConfigField(key="timeout", label="Timeout (s)", kind=INT, default=30),
        ]

    @classmethod
    def doc(cls) -> PluginDoc:
        return PluginDoc(
            description=f"Schlägt eine {cls._identifier_label} im Katalog (SRU, Standard: "
            "K10plus) nach und macht den Treffer als Analyse-Text verfügbar "
            "(Titel, Autor, Abstract sofern vorhanden, Schlagwörter).",
            input=f"Eine {cls._identifier_label}.",
            output="Formatierter Analyse-Text aus dem bibliographischen Datensatz.",
        )

    def can_handle(self, source: str, input_type: str) -> bool:
        return input_type == self.id

    def extract(
        self, source, *, llm_service=None, stream_callback=None, logger=None, **opts
    ) -> Tuple[str, str, str]:
        if stream_callback:
            stream_callback(f"📚 Katalog-Lookup ({self.label}): {source}")
        record = lookup_bibrecord(
            source,
            search_type=self._search_type,
            preset=str(self._config.get("preset") or "k10plus"),
            timeout=int(self._config.get("timeout", 30) or 30),
            logger=logger,
        )
        text = record.to_analysis_text()
        if not text.strip():
            raise ValueError("Keine Metadaten vom Katalog zurückgegeben")
        # WP-D1 P2 side channel: the 3-tuple contract cannot carry the record,
        # so a caller that wants it (classification priors) passes a dict as
        # ``record_sink`` and finds the BibRecord under "record". - Claude Generated
        sink = opts.get("record_sink")
        if isinstance(sink, dict):
            sink["record"] = record
        source_info = f"{self.label}: {source} — {record.title}" if record.title else f"{self.label}: {source}"
        return text, source_info, self.id


@register_input_source
class IsbnInputSource(_BibLookupSource):
    id = "isbn"
    label = "ISBN (Katalog)"
    _search_type = "isbn"
    _identifier_label = "ISBN"


@register_input_source
class PpnInputSource(_BibLookupSource):
    id = "ppn"
    label = "PPN (Katalog)"
    # WP-D1 P4: dedicated pica.ppn index (k10plus preset). The former default
    # keyword index searched the PPN as a SUBJECT term and could never match —
    # the batch PPN path inherited that latent defect.
    _search_type = "ppn"
    _identifier_label = "PPN (K10plus-Datensatznummer)"
