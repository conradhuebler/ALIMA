"""DOI resolver input sources — one plugin per source - Claude Generated.

The DOI resolver is split into three *separately configurable* input sources
(``doi_crossref``, ``doi_openalex``, ``doi_datacite``), each individually
enable-able with its own config (Debt D-10).

Two distinct jobs (deliberately different):

* :meth:`mcp_execute` (the agent tool ``resolve_doi_*``) queries **that one
  source's API directly** and returns the **complete raw metadata record** — a
  source is a success whenever it has the record, *independent of whether it
  carries an abstract*. Sources therefore return different amounts of data, by
  design (operator request): "liefere die vollständigen Daten aus der API".
* :meth:`extract` (pipeline input) still returns *abstract text* via the shared
  :class:`UnifiedResolver`, which is abstract-oriented.

The merged/fallback resolver (``resolve_doi`` tool + ``resolve_input_to_text``) is
unchanged and remains the abstract-getter with the crossref→openalex→datacite
chain. (Direct API calls here duplicate the endpoints in ``doi_resolver.py`` —
noted under D-12; kept separate so the abstract logic is not entangled with the
"return everything" tool.)
"""

from __future__ import annotations

import re
from typing import Any, List, Tuple

from src.core.plugins.schema import INT, TEXT, ConfigField, PluginDoc

from .registry import register_input_source

_DOI_RE = re.compile(r"^10\.\d{4,9}/\S+$")


def _looks_like_doi(source: str) -> bool:
    s = str(source or "").strip()
    if s.lower().startswith(("http://", "https://")):
        return "doi.org/" in s.lower()
    return bool(_DOI_RE.match(s))


class _DoiInputSource:
    """Shared base: one DOI backend (crossref/openalex/datacite)."""

    id = ""
    label = ""
    _use = ""  # "crossref" | "openalex" | "datacite"
    _has_email = True
    _backend_label = ""

    def __init__(self, **config: Any):
        self._config = config or {}

    @classmethod
    def doc(cls) -> PluginDoc:
        return PluginDoc(
            description=f"Fragt die {cls._backend_label}-API zu einer DOI ab und gibt den "
            "vollständigen Metadaten-Datensatz dieser Quelle zurück (auch ohne Abstract). "
            "Einzeln aktivierbar — ideal für den Quellenvergleich.",
            input="Eine DOI (oder DOI-URL, z.B. 10.1002/cmtd.202200006 bzw. https://doi.org/…).",
            output="Vollständiger Roh-Metadaten-Datensatz aus der Quelle (Felder je Quelle "
            "unterschiedlich); Abstract nur wenn die Quelle ihn führt.",
        )

    @classmethod
    def config_fields(cls) -> List[ConfigField]:
        fields = [ConfigField(key="timeout", label="Timeout (s)", kind=INT, default=30)]
        if cls._has_email:
            fields.insert(
                0,
                ConfigField(
                    key="contact_email", label="Kontakt-E-Mail (Polite-Pool)", kind=TEXT,
                    help="Für den Polite-Pool von Crossref/OpenAlex (mailto).",
                ),
            )
        return fields

    @classmethod
    def mcp_tool_spec(cls):
        from .registry import InputToolSpec

        return InputToolSpec(
            name=f"resolve_{cls.id}",  # resolve_doi_crossref / _openalex / _datacite
            description=(
                f"Fragt die {cls._backend_label}-API zu einer DOI ab und gibt den "
                f"VOLLSTÄNDIGEN Roh-Metadaten-Datensatz NUR aus dieser Quelle zurück "
                f"(auch wenn kein Abstract vorhanden ist). Nutze die drei resolve_doi_* "
                f"Tools, um Quellen zu vergleichen; resolve_doi liefert dagegen ein "
                f"zusammengeführtes Abstract-Ergebnis mit Fallback-Kette."
            ),
            param="doi",
            param_description="Die DOI, z.B. 10.1002/cmtd.202200006.",
        )

    # -- raw API (the tool) ---------------------------------------------------
    @staticmethod
    def _normalize_doi(source: str) -> str:
        s = str(source or "").strip()
        low = s.lower()
        if low.startswith(("http://", "https://")) and "doi.org/" in low:
            s = s[low.index("doi.org/") + len("doi.org/"):]
        return s.strip().strip("/")

    def _api_url(self, doi: str, email: str) -> str:
        if self._use == "crossref":
            u = f"https://api.crossref.org/works/{doi}"
            return u + (f"?mailto={email}" if email else "")
        if self._use == "openalex":
            u = f"https://api.openalex.org/works/doi:{doi}"
            return u + (f"?mailto={email}" if email else "")
        return f"https://api.datacite.org/dois/{doi}"  # datacite

    @staticmethod
    def _invert_abstract(inv) -> str:
        try:
            positions = {}
            for word, idxs in (inv or {}).items():
                for i in idxs:
                    positions[int(i)] = word
            return " ".join(positions[i] for i in sorted(positions))
        except Exception:
            return ""

    def _extract_record(self, data: dict) -> dict:
        """The meaningful record from each source's raw JSON (kept complete)."""
        if self._use == "crossref":
            return data.get("message", data)
        if self._use == "openalex":
            rec = dict(data or {})
            inv = rec.pop("abstract_inverted_index", None)
            if inv:
                rec["abstract"] = self._invert_abstract(inv)
            return rec
        # datacite
        return (data or {}).get("data", {}).get("attributes", data)

    def mcp_execute(self, doi) -> dict:
        """Return the COMPLETE raw metadata record from this one source's API."""
        doi = self._normalize_doi(doi)
        email = self._config.get("contact_email", "") or ""
        out = {"source": self._use, "doi": doi}
        try:
            import requests

            headers = {"User-Agent": f"ALIMA/2.0 (mailto:{email})"} if email else {}
            timeout = int(self._config.get("timeout", 30) or 30)
            resp = requests.get(self._api_url(doi, email), headers=headers, timeout=timeout)
        except Exception as e:
            out.update(success=False, error=f"request failed: {e}")
            return out
        if resp.status_code != 200:
            out.update(success=False, error=f"HTTP {resp.status_code}", http_status=resp.status_code)
            return out
        try:
            data = resp.json()
        except Exception as e:
            out.update(success=False, error=f"invalid JSON: {e}")
            return out
        out.update(success=True, metadata=self._extract_record(data))
        return out

    # -- abstract text (pipeline input) ---------------------------------------
    def can_handle(self, source: str, input_type: str) -> bool:
        if input_type == "doi":
            return True
        return input_type == "auto" and _looks_like_doi(source)

    def extract(self, source, *, llm_service=None, stream_callback=None, logger=None, **opts) -> Tuple[str, str, str]:
        from src.utils.doi_resolver import UnifiedResolver

        resolver = UnifiedResolver(
            None,
            contact_email=self._config.get("contact_email", "") or "",
            use_crossref=(self._use == "crossref"),
            use_openalex=(self._use == "openalex"),
            use_datacite=(self._use == "datacite"),
        )
        success, _metadata, result = resolver.resolve(source)
        if not success:
            raise RuntimeError(result or f"{self.id} konnte '{source}' nicht auflösen")
        return result, f"DOI ({self._use}): {source}", self.id


@register_input_source
class DoiCrossrefInputSource(_DoiInputSource):
    id = "doi_crossref"
    label = "DOI: Crossref"
    _use = "crossref"
    _backend_label = "Crossref"


@register_input_source
class DoiOpenalexInputSource(_DoiInputSource):
    id = "doi_openalex"
    label = "DOI: OpenAlex"
    _use = "openalex"
    _backend_label = "OpenAlex"


@register_input_source
class DoiDataciteInputSource(_DoiInputSource):
    id = "doi_datacite"
    label = "DOI: DataCite"
    _use = "datacite"
    _has_email = False
    _backend_label = "DataCite"
