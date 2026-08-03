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

from typing import Any, List, Tuple

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
    # The SRU clients expose no dedicated PPN index; the record id is found via
    # the default keyword index (same behaviour the batch path had).
    _search_type = "keyword"
    _identifier_label = "PPN (K10plus-Datensatznummer)"
