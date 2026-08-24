"""KVK (Karlsruher Virtueller Katalog) client — HTTP + pure parsing. - Claude Generated

The KVK answers ``maske=kvk-json`` with **NDJSON**: one JSON object per line, one
per queried union catalog, plus a trailing ``{"type": "error"}`` block listing the
catalogs that returned nothing or failed::

    {"type":"catalog","data":{"name":…,"homepage":…,"results":18,"items":[…],"next":…}}
    {"type":"error","data":[{"name":…,"url":…,"message":"Keine Datensätze gefunden."}]}

Parsing is split from fetching so the whole normalisation path is testable
against a stored response without network access.

Two properties of the payload drive the design here:

* **The item fields are catalog-dependent.** The DNB fills ``author``/``year``;
  K10plus leaves both empty and puts the whole imprint into ``text``
  ("Quintes, Florian. - Freiburg im Breisgau, 06.07.2026"). ``parse_item``
  therefore falls back to ``text`` instead of trusting the named fields.
* **The record link carries the catalog's own id.** K10plus/StaBi/KOBV links
  expose a PPN, the DNB link an IDN — see :func:`extract_identifiers`. That is
  what connects a KVK hit to ALIMA's existing K10plus enrichment; the KVK JSON
  itself carries no subjects and no classifications.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import parse_qs, urlparse

import requests

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://kvk.bibliothek.kit.edu/hylib-bin/kvk/nph-kvk2.cgi"

# Catalog ids as they appear in the KVK search URL (`kataloge=…`). This is the
# set the operator's own KVK link uses; the KVK front page is behind a JS bot
# challenge, so the full list is not machine-readable — the field stays free
# text on purpose.
DEFAULT_CATALOGS = (
    "K10PLUS", "BVB", "NRW", "HEBIS", "HEBIS_RETRO", "KOBV_SOLR", "DDB", "STABI_BERLIN",
)

# ALIMA search axis -> KVK query parameter. KVK indexes each axis separately;
# ALL is the free "alle Wörter" search.
SEARCH_PARAMS = {
    "kw": "ALL",
    "keyword": "ALL",
    "all": "ALL",
    "title": "TI",
    "author": "AU",
    "subject": "ST",
    "isbn": "SB",
}


def parse_item(item: Dict[str, Any], catalog: str = "", homepage: str = "") -> Dict[str, Any]:
    """One KVK item -> a flat record dict (the producer shape for ``to_bibrecord``).

    ``author``/``year`` are taken from the named fields when the catalog fills
    them and derived from ``text`` otherwise. ``text`` is an imprint line of the
    form ``"Autor. - Ort : Verlag, Jahr"``; only the leading name and a trailing
    4-digit year are read from it, because that is all the format guarantees.
    """
    item = item or {}
    url = str(item.get("url") or "").strip()
    author = str(item.get("author") or "").strip()
    year = str(item.get("year") or "").strip()
    text = str(item.get("text") or "").strip()

    if not author and text:
        head = text.split(". - ", 1)[0].strip()
        # A leading "Nachname, Vorname" is an author; an imprint that starts with
        # a place ("Berlin : Springer, 2020") is not.
        if head and ":" not in head and not head.isdigit():
            author = head
    if not year and text:
        years = re.findall(r"\b(1[5-9]\d{2}|20\d{2})\b", text)
        if years:
            year = years[-1]

    record: Dict[str, Any] = {
        "title": str(item.get("title") or "").strip(),
        "author": author,
        "year": year,
        "url": url,
        "text": text,
        "digital": bool(item.get("digital")),
        "catalog": catalog,
        "catalog_homepage": homepage,
    }
    record.update(extract_identifiers(url))
    return record


def extract_identifiers(url: str) -> Dict[str, str]:
    """Identifiers readable from a KVK record link.

    Each union catalog links its own record in its own way; the id is in the
    link, never in a JSON field:

    ===================  ==========================================  ========
    Katalog              Linkform                                    Ergebnis
    ===================  ==========================================  ========
    K10plus              ``…?bibtip_docid=1981371435``               ``ppn``
    StaBi Berlin         ``stabikat.de/Record/366303287``            ``ppn``
    KOBV                 ``portal.kobv.de/KobvIndexRecord/gbv_537…`` ``ppn``
    DNB                  ``portal.dnb.de/…?bibtip_docid=1415663890`` ``idn``
    BVB                  ``gateway-bayern.de/BV044038433``           ``bvnumber``
    ===================  ==========================================  ========

    A PPN found this way is a *candidate* for the K10plus lookup, not a promise:
    the id is valid in the catalog that issued it, and not every one of them is
    retrievable through the K10plus SRU endpoint ALIMA queries.

    Catalogs whose links carry an id ALIMA cannot resolve return **nothing** on
    purpose. KOBV is the case that makes this matter: it mixes ``gbv_<ppn>``
    with ``almahu_<mms-id>``, and hbz links an Alma id as well. Passing one of
    those on as a "ppn" would surface as a lookup miss rather than as the wrong
    identifier it is.
    """
    url = str(url or "").strip()
    if not url:
        return {}
    try:
        parsed = urlparse(url)
    except ValueError:
        return {}
    host = (parsed.hostname or "").lower()
    path = parsed.path or ""
    docid = (parse_qs(parsed.query).get("bibtip_docid") or [""])[0].strip()

    if docid:
        if "dnb.de" in host:
            return {"idn": docid}
        return {"ppn": docid}
    if "stabikat.de" in host:
        match = re.search(r"/Record/([0-9Xx]+)", path)
        if match:
            return {"ppn": match.group(1)}
    if "kobv.de" in host:
        match = re.search(r"/KobvIndexRecord/gbv_([0-9Xx]+)", path)
        if match:
            return {"ppn": match.group(1)}
    if "gateway-bayern.de" in host:
        match = re.search(r"/(BV\d+)", path)
        if match:
            return {"bvnumber": match.group(1)}
    return {}


def parse_response(payload: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, str]], List[Dict[str, Any]]]:
    """Parse an NDJSON KVK response.

    Returns ``(records, catalog_errors, catalog_stats)``. A line that is not
    valid JSON is skipped with a warning rather than failing the whole response —
    one broken catalog block must not lose the seven that parsed.
    """
    records: List[Dict[str, Any]] = []
    catalog_errors: List[Dict[str, str]] = []
    stats: List[Dict[str, Any]] = []

    for line in (payload or "").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            block = json.loads(line)
        except json.JSONDecodeError as exc:
            logger.warning("KVK: skipping unparseable NDJSON line: %s", exc)
            continue
        kind = block.get("type")
        data = block.get("data")
        if kind == "catalog" and isinstance(data, dict):
            name = str(data.get("name") or "")
            homepage = str(data.get("homepage") or "")
            items = data.get("items") or []
            stats.append({
                "catalog": name,
                "results": int(data.get("results") or 0),
                "returned": len(items),
                # KVK pages per catalog; `next` means the hit count exceeds what
                # this response carries.
                "truncated": bool(data.get("next")),
            })
            for item in items:
                records.append(parse_item(item, catalog=name, homepage=homepage))
        elif kind == "error" and isinstance(data, list):
            for entry in data:
                if not isinstance(entry, dict):
                    continue
                catalog_errors.append({
                    "catalog": str(entry.get("name") or ""),
                    "message": str(entry.get("message") or ""),
                })
    return records, catalog_errors, stats


def merge_round_robin(records: List[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
    """Cap the merged hit list by taking turns between catalogs.

    The KVK emits its catalogs one block after another, so a plain ``[:limit]``
    would answer a 5-hit request entirely from whichever catalog happened to be
    listed first — and silently hide the other seven. Since the point of a
    meta-search is breadth, the cap takes one record per catalog per round,
    keeping each catalog's own order intact. - Claude Generated
    """
    limit = max(0, int(limit))
    if limit <= 0:
        return []
    by_catalog: Dict[str, List[Dict[str, Any]]] = {}
    for record in records:
        by_catalog.setdefault(str(record.get("catalog") or ""), []).append(record)
    out: List[Dict[str, Any]] = []
    round_index = 0
    while len(out) < limit:
        added = False
        for bucket in by_catalog.values():
            if round_index < len(bucket):
                out.append(bucket[round_index])
                added = True
                if len(out) >= limit:
                    break
        if not added:
            break
        round_index += 1
    return out


def build_params(
    term: str,
    *,
    search_type: str = "kw",
    catalogs: Optional[List[str]] = None,
    language: str = "de",
) -> List[Tuple[str, str]]:
    """Query parameters for one KVK search, as an ordered pair list.

    ``kataloge`` is repeated once per catalog — a dict cannot express that, which
    is why this returns pairs.
    """
    axis = SEARCH_PARAMS.get(str(search_type or "kw").lower())
    if axis is None:
        raise ValueError(
            f"KVK: unknown search_type {search_type!r} "
            f"(known: {', '.join(sorted(set(SEARCH_PARAMS)))})"
        )
    params: List[Tuple[str, str]] = [
        ("maske", "kvk-json"),
        ("lang", str(language or "de")),
        ("ref", "direct"),
        (axis, str(term or "")),
    ]
    for catalog in catalogs or DEFAULT_CATALOGS:
        catalog = str(catalog).strip()
        if catalog:
            params.append(("kataloge", catalog))
    return params


class KvkClient:
    """Thin HTTP wrapper around the KVK JSON endpoint."""

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        *,
        catalogs: Optional[List[str]] = None,
        timeout: int = 30,
        language: str = "de",
        session: Any = None,
    ):
        from src.utils.net_guard import require_http_url

        self.base_url = require_http_url(base_url or DEFAULT_BASE_URL, what="KVK-Basis-URL")
        self.catalogs = list(catalogs or DEFAULT_CATALOGS)
        self.timeout = int(timeout or 30)
        self.language = str(language or "de")
        self._session = session or requests
        # Verbatim response per term — the raw-cache dual-write reads this.
        self.last_raw: Dict[str, str] = {}
        self.last_http_status: Dict[str, int] = {}
        self.last_errors: Dict[str, str] = {}

    def search(self, term: str, *, search_type: str = "kw") -> Tuple[List[Dict[str, Any]], List[Dict[str, str]], List[Dict[str, Any]]]:
        """Run one KVK query. Raises ``requests.RequestException`` on transport failure."""
        params = build_params(
            term, search_type=search_type, catalogs=self.catalogs, language=self.language
        )
        response = self._session.get(self.base_url, params=params, timeout=self.timeout)
        self.last_http_status[term] = int(getattr(response, "status_code", 0) or 0)
        response.raise_for_status()
        payload = response.text or ""
        self.last_raw[term] = payload
        return parse_response(payload)
