"""K10plus SRU in PICA-XML: query building, parsing, local data - Claude Generated.

The K10plus SRU service (``https://sru.k10plus.de/k10plus``, SRU 1.1) answers
``recordSchema=picaxml`` with the full PICA+ record, local data of the
holding libraries included; the MARC-XML schema leaves the local fields out.
This module builds the CQL query, sends it (through ``source_health`` and the
``net_guard`` scheme gate) and reads the records. Parsing is split from the
request so everything is testable without network. Qt-free.

Fields read per record (the mapping rests on ONE real record, a printed
dissertation with a copy of the UB Freiberg; fields not present in it are
marked "unverified"):

====== =============================================== =====================
Field  Read as                                         Seen in a real record
====== =============================================== =====================
003@   ``$0`` PPN                                      yes
021A   ``$a`` title, ``$d`` addition                   ``$a`` yes, ``$d`` no
028A   ``$A`` surname ``$D`` forename (or ``$a``/      yes
028C   ``$d``, ``$P``), ``$B`` role, ``$7 gnd/<id>``   yes
033A   ``$p`` place, ``$n`` publisher                  ``$p`` yes, ``$n`` no
011@   ``$a`` year                                     yes
004A   ``$0`` ISBN (else ``$A``)                       no (unverified)
013D   ``$a`` form (GND form heading)                  yes
037C   ``$a`` or ``$d``/``$e``/``$f`` thesis note      ``$d/$e/$f`` yes
044K   subject chains: ``$9`` PPN, ``$7 gnd/<id>``,    yes
       ``$a`` heading; a line with only ``$A<ISIL>``
       closes a chain and names its source
045F/G ``$a`` DDC                                      no (unverified)
045H
101@   start of one library's local data               yes
209A   copy: ``$B`` ELN, ``$a`` call number            yes
245Z   copy: local notation ``$a`` (``DK 541.5``),      yes
       ``$9`` PPN
====== =============================================== =====================

Copy-level fields (tags ``2…``) belong to the copy with the same
``occurrence`` inside the same ``101@`` block; a ``245Z`` is therefore the
notation of the copy whose ``209A`` carries the same occurrence. That is the
PICA level convention; in the one real record the only copy and all 245Z
share occurrence ``01``.
"""

from __future__ import annotations

import logging
import re
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional, Tuple

import requests

logger = logging.getLogger(__name__)

#: The K10plus SRU database with local data (``operation=explain`` names it
#: ``k10plus``, "K10plus-Katalogisierungsdatenbank").
K10PLUS_SRU_URL = "https://sru.k10plus.de/k10plus"
RECORD_SCHEMA = "picaxml"

#: Search axis → CQL index (names from the service's ``explain`` answer).
SEARCH_INDEXES = {
    "kw": "pica.all",      # [ALL] Alle Wörter
    "title": "pica.tit",   # [TIT] Titel (Stichwort)
    "author": "pica.per",  # [PER] Person, sent as a phrase "Surname, Forename"
    "subject": "pica.slw",  # [SLW] Schlagwörter
    "isbn": "pica.isb",    # [ISB] ISBN
    "ppn": "pica.ppn",     # [PPN] Pica-Produktionsnummer
}
DEFAULT_SEARCH_TYPE = "kw"
DEFAULT_MAX_RESULTS = 10
MAX_RESULTS_LIMIT = 50

_SRW_NS = "http://www.loc.gov/zing/srw/"
_PICA_NS = "info:srw/schema/5/picaXML-v1.0"
_DDC_TAGS = ("045F", "045G", "045H")


# ── query ─────────────────────────────────────────────────────────────────


def escape_cql(value: Any) -> str:
    """A value for a quoted CQL term: backslash and double quote escaped.

    ``a "b" \\c`` → ``a \\"b\\" \\\\c``. - Claude Generated
    """
    text = str(value or "")
    return text.replace("\\", "\\\\").replace('"', '\\"')


def normalize_author(value: Any) -> str:
    """``Vorname Nachname`` → ``Nachname, Vorname``; everything else unchanged.

    Turned only when the value has no comma and exactly two words, the form
    ``pica.per`` finds as a phrase ("Hübler, Conrad"). One word, three or more
    words, or a comma: sent as given. - Claude Generated
    """
    text = " ".join(str(value or "").split())
    if "," in text:
        return text
    words = text.split(" ")
    if len(words) == 2 and all(words):
        return f"{words[1]}, {words[0]}"
    return text


def normalize_search_type(search_type: Any) -> str:
    """A known search axis; ``None``/empty → ``kw``; unknown raises ``ValueError``."""
    key = str(search_type or DEFAULT_SEARCH_TYPE).strip().lower()
    if key in ("keyword", "all"):
        key = "kw"
    if key not in SEARCH_INDEXES:
        raise ValueError(
            f"K10plus: unknown search_type {search_type!r} (known: {', '.join(SEARCH_INDEXES)})"
        )
    return key


def clamp_max_results(value: Any, default: int = DEFAULT_MAX_RESULTS,
                      limit: int = MAX_RESULTS_LIMIT) -> int:
    """``max_results`` in 1..limit; ``None``/invalid → ``default`` - Claude Generated"""
    try:
        number = int(value) if value is not None else int(default)
    except (TypeError, ValueError):
        number = int(default)
    return max(1, min(int(limit), number))


def build_cql(term: Any, search_type: Any = DEFAULT_SEARCH_TYPE) -> str:
    """CQL query ``<index>="<escaped term>"``; ``author`` is turned first.

    - Claude Generated
    """
    key = normalize_search_type(search_type)
    value = str(term or "").strip()
    if key == "author":
        value = normalize_author(value)
    return f'{SEARCH_INDEXES[key]}="{escape_cql(value)}"'


def build_params(term: Any, search_type: Any = DEFAULT_SEARCH_TYPE,
                 max_results: Any = DEFAULT_MAX_RESULTS) -> Dict[str, str]:
    """SRU 1.1 ``searchRetrieve`` parameters, always ``recordSchema=picaxml``."""
    return {
        "version": "1.1",
        "operation": "searchRetrieve",
        "query": build_cql(term, search_type),
        "maximumRecords": str(clamp_max_results(max_results, limit=10_000)),
        "recordSchema": RECORD_SCHEMA,
    }


# ── parsing ───────────────────────────────────────────────────────────────


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag


def _clean(text: Any) -> str:
    """Whitespace collapsed; the PICA sorting mark ``@`` removed."""
    return " ".join(str(text or "").replace("@", "").split())


def _subfields(field: ET.Element) -> List[Tuple[str, str]]:
    out = []
    for sub in field:
        if _local(sub.tag) == "subfield":
            out.append((sub.get("code") or "", _clean(sub.text)))
    return out


def _first(subs: List[Tuple[str, str]], *codes: str) -> str:
    for code in codes:
        for c, value in subs:
            if c == code and value:
                return value
    return ""


def _gnd_id(subs: List[Tuple[str, str]]) -> str:
    for code, value in subs:
        if code == "7" and value.lower().startswith("gnd/"):
            return value[4:]
    return ""


def _person(subs: List[Tuple[str, str]]) -> Dict[str, str]:
    surname = _first(subs, "A", "a")
    forename = _first(subs, "D", "d")
    name = ", ".join(part for part in (surname, forename) if part) or _first(subs, "P", "8")
    return {"name": name, "role": _first(subs, "B"), "gnd_id": _gnd_id(subs)}


def parse_record(record: ET.Element) -> Dict[str, Any]:
    """One ``<record>`` → the stored record shape.

    ``{ppn, title, authors, year, publisher, isbn, form, thesis, ddc,
    subjects, copies}``; ``subjects`` are the 044K chains ``{terms: [{term,
    gnd_id, ppn}], source}``, ``copies`` the copy-level data ``{eln,
    signature, notations: [{text, ppn}]}`` of every holding library.
    - Claude Generated
    """
    out: Dict[str, Any] = {
        "ppn": "", "title": "", "authors": [], "year": "", "publisher": "",
        "isbn": [], "form": [], "thesis": "", "ddc": [], "subjects": [], "copies": [],
    }
    chains: List[Dict[str, Any]] = []
    current_chain: Optional[Dict[str, Any]] = None
    current_occ: Optional[str] = None
    blocks: List[Dict[str, Dict[str, Any]]] = []
    block: Optional[Dict[str, Dict[str, Any]]] = None

    for field in record:
        if _local(field.tag) != "datafield":
            continue
        tag = field.get("tag") or ""
        occ = field.get("occurrence") or ""
        subs = _subfields(field)

        if tag == "003@":
            out["ppn"] = out["ppn"] or _first(subs, "0")
        elif tag == "021A" and not out["title"]:
            main, addition = _first(subs, "a"), _first(subs, "d")
            out["title"] = f"{main} : {addition}" if main and addition else (main or addition)
        elif tag in ("028A", "028C"):
            person = _person(subs)
            if person["name"]:
                out["authors"].append(person)
        elif tag == "033A" and not out["publisher"]:
            place, name = _first(subs, "p"), _first(subs, "n")
            out["publisher"] = f"{place} : {name}" if place and name else (place or name)
        elif tag == "011@" and not out["year"]:
            out["year"] = _first(subs, "a")
        elif tag == "004A":
            isbn = _first(subs, "0", "A")
            if isbn and isbn not in out["isbn"]:
                out["isbn"].append(isbn)
        elif tag == "013D":
            form = _first(subs, "a")
            if form and form not in out["form"]:
                out["form"].append(form)
        elif tag == "037C" and not out["thesis"]:
            out["thesis"] = _first(subs, "a") or ", ".join(
                v for v in (_first(subs, "d"), _first(subs, "e"), _first(subs, "f")) if v)
        elif tag in _DDC_TAGS:
            ddc = _first(subs, "a")
            if ddc and ddc not in out["ddc"]:
                out["ddc"].append(ddc)
        elif tag == "044K":
            heading = _first(subs, "a")
            source = _first(subs, "A")
            if current_chain is not None and occ != current_occ:
                chains.append(current_chain)
                current_chain = None
            current_occ = occ
            if heading:
                if current_chain is None:
                    current_chain = {"terms": [], "source": ""}
                current_chain["terms"].append(
                    {"term": heading, "gnd_id": _gnd_id(subs), "ppn": _first(subs, "9")})
            elif source and current_chain is not None:
                # "044K $A<ISIL>" closes the chain and names its source.
                current_chain["source"] = source
                chains.append(current_chain)
                current_chain = None
        elif tag == "101@":
            block = {}
            blocks.append(block)
        elif tag.startswith("2"):
            if block is None:
                block = {}
                blocks.append(block)
            copy = block.setdefault(occ, {"eln": "", "signature": "", "notations": []})
            if tag == "209A":
                copy["eln"] = copy["eln"] or _first(subs, "B")
                copy["signature"] = copy["signature"] or _first(subs, "a")
            elif tag == "245Z":
                text = _first(subs, "a")
                if text:
                    copy["notations"].append({"text": text, "ppn": _first(subs, "9")})
    if current_chain is not None:
        chains.append(current_chain)
    out["subjects"] = [c for c in chains if c["terms"]]
    out["copies"] = [copy for blk in blocks for copy in blk.values()
                     if copy["eln"] or copy["signature"] or copy["notations"]]
    return out


def parse_response(payload: Any) -> Tuple[List[Dict[str, Any]], int, List[str]]:
    """An SRU answer → ``(records, numberOfRecords, diagnostics)``.

    Unparseable XML raises ``ValueError`` (a broken answer is a failure of the
    source, not an empty hit). - Claude Generated
    """
    data = payload if isinstance(payload, bytes) else str(payload or "").encode("utf-8")
    try:
        root = ET.fromstring(data)
    except ET.ParseError as exc:
        raise ValueError(f"K10plus: unreadable SRU answer ({exc})") from exc
    total_text = root.findtext(f"{{{_SRW_NS}}}numberOfRecords") or "0"
    try:
        total = int(total_text.strip() or 0)
    except ValueError:
        total = 0
    diagnostics = []
    for diag in root.iter():
        if _local(diag.tag) != "diagnostic":
            continue
        for child in diag:
            if _local(child.tag) == "message":
                diagnostics.append(_clean(child.text))
    records = []
    for data_elem in root.iter(f"{{{_SRW_NS}}}recordData"):
        for child in data_elem:
            if _local(child.tag) == "record":
                records.append(parse_record(child))
    return records, total, [d for d in diagnostics if d]


# ── local data of the library ─────────────────────────────────────────────


ELN_NOT_SET_HINT = "ELN nicht gesetzt (Einstellungen ▸ Datenquellen ▸ Einrichtung)"
ISIL_NOT_SET_HINT = "ISIL nicht gesetzt (Einstellungen ▸ Datenquellen ▸ Einrichtung)"


def no_copy_hint(eln: str) -> str:
    return f"kein Exemplar der Bibliothek (ELN {eln}) im Satz"


def copies_of(record: Dict[str, Any], eln: Optional[str]) -> List[Dict[str, Any]]:
    """The copies of ``record`` whose ``209A $B`` equals ``eln`` - Claude Generated"""
    if not eln:
        return []
    return [c for c in record.get("copies") or [] if str(c.get("eln") or "").strip() == eln]


def local_block(record: Dict[str, Any], eln: Optional[str],
                isil: Optional[str]) -> Tuple[Optional[Dict[str, Any]], str]:
    """``(local, hint)``: the library's data of one record, or why there is none.

    ``local`` = ``{signature, notations: [{text, ppn}], subject_chains}``
    (``signatures`` too when the copies carry more than one); the notations
    come from the 245Z of the library's copies, the subject chains are the
    044K chains closed by ``$A<ISIL>``. - Claude Generated
    """
    if not eln:
        return None, ELN_NOT_SET_HINT
    copies = copies_of(record, eln)
    if not copies:
        return None, no_copy_hint(eln)
    signatures: List[str] = []
    notations: List[Dict[str, str]] = []
    for copy in copies:
        sig = str(copy.get("signature") or "")
        if sig and sig not in signatures:
            signatures.append(sig)
        for item in copy.get("notations") or []:
            entry = {"text": str(item.get("text") or ""), "ppn": str(item.get("ppn") or "")}
            if entry["text"] and entry not in notations:
                notations.append(entry)
    local: Dict[str, Any] = {"signature": signatures[0] if signatures else "",
                             "notations": notations}
    if len(signatures) > 1:
        local["signatures"] = signatures
    if isil:
        local["subject_chains"] = [
            {"terms": list(chain.get("terms") or [])}
            for chain in record.get("subjects") or [] if chain.get("source") == isil
        ]
    else:
        local["subject_chains"] = []
        local["subject_chains_hint"] = ISIL_NOT_SET_HINT
    return local, ""


def shape_record(record: Dict[str, Any], eln: Optional[str],
                 isil: Optional[str]) -> Dict[str, Any]:
    """A stored record as the tool returns it: no raw copies, ``local`` or
    ``local_hint`` instead. - Claude Generated"""
    out = {k: v for k, v in record.items() if k != "copies"}
    local, hint = local_block(record, eln, isil)
    if local is not None:
        out["local"] = local
    else:
        out["local_hint"] = hint
    return out


def split_notation(text: str) -> Tuple[str, str]:
    """``"DK 541.5"`` → ``("DK", "541.5")``, ``"RVK VE 5650"`` → ``("RVK", …)``;
    anything else (``FGAUT``) → ``("", text)``. - Claude Generated"""
    value = " ".join(str(text or "").split())
    match = re.match(r"^(DK|RVK)\s+(.+)$", value, re.IGNORECASE)
    if not match:
        return "", value
    return match.group(1).upper(), match.group(2).strip()


# ── transport ─────────────────────────────────────────────────────────────


class PicaXmlClient:
    """SRU 1.1 ``searchRetrieve`` against the K10plus in PICA-XML - Claude Generated"""

    def __init__(self, base_url: str = K10PLUS_SRU_URL, *, timeout: int = 30,
                 session: Any = None, source_id: Optional[str] = None):
        from src.utils.net_guard import require_http_url

        self.base_url = require_http_url(base_url or K10PLUS_SRU_URL, what="K10plus-SRU-URL")
        self.timeout = int(timeout or 30)
        self._session = session or requests.Session()
        # Instance id for source_health; None leaves requests untracked.
        self.source_id = source_id

    def search(self, term: Any, *, search_type: Any = DEFAULT_SEARCH_TYPE,
               max_results: Any = DEFAULT_MAX_RESULTS) -> Dict[str, Any]:
        """One query → ``{records, total, query, diagnostics}``.

        Transport failures and HTTP error statuses raise
        (``requests.RequestException``); an unreadable answer raises
        ``ValueError``. - Claude Generated
        """
        from src.core.search import source_health

        params = build_params(term, search_type, max_results)
        response = source_health.call(
            self.source_id, self._session.get, self.base_url, params=params, timeout=self.timeout
        )
        response.raise_for_status()
        records, total, diagnostics = parse_response(response.content)
        if diagnostics:
            logger.warning("K10plus SRU diagnostic for %r: %s", params["query"], "; ".join(diagnostics))
            if not records:
                # The server refused the query: a failure, not an empty hit.
                raise ValueError(f"K10plus SRU: {'; '.join(diagnostics)}")
        return {"records": records, "total": total, "query": params["query"],
                "diagnostics": diagnostics}


# ── DK extractor over the library's local notations ───────────────────────


class PicaDkExtractor:
    """DK/RVK per keyword from the local notations (245Z) of the library's copies.

    Same contract as ``MarcXmlClient.extract_dk_classifications_for_keywords``:
    one entry per keyword with classifications ``{dk, classification_type,
    count, titles, keywords}``, ``count`` = number of records whose copies of
    the library carry the notation. A record counts only when it has a copy
    with ``209A $B<ELN>``; the filter is applied here, not in the query (the
    indexes ``pica.bib``/``pica.sge`` gave no hits). Searched per keyword with
    ``pica.slw``; only the first ``max_results`` records of the answer are
    looked at.

    ``cache_read(term, search_type)`` / ``cache_write(term, search_type,
    answer)`` are the raw-cache seam of the provider; with ``cache_only`` no
    request is made and a keyword without entry is listed in
    ``last_cache_misses``. - Claude Generated
    """

    SEARCH_TYPE = "subject"

    def __init__(self, client: PicaXmlClient, eln: str, *, cache_only: bool = False,
                 cache_read: Any = None, cache_write: Any = None):
        self.client = client
        self.eln = str(eln or "").strip()
        self.cache_only = bool(cache_only)
        self._cache_read = cache_read
        self._cache_write = cache_write
        self.source_id = client.source_id
        self.timeout = client.timeout
        self.last_cache_misses: List[str] = []
        self.last_source_down: List[str] = []
        self.last_errors: Dict[str, str] = {}
        # Origin of the keyword being processed, for source_usage. - Claude Generated
        self._usage_origin: Optional[tuple] = None

    def _answer(self, term: str, max_results: int) -> Optional[Dict[str, Any]]:
        from src.core.search import source_health

        from src.core.search import source_usage as _usage

        if self.cache_only:
            entry = self._cache_read(term, self.SEARCH_TYPE) if self._cache_read else None
            if not isinstance(entry, dict) or not isinstance(entry.get("records"), list):
                self.last_cache_misses.append(term)
                self._usage_origin = (_usage.CACHE_MISS, False)
                return None
            self._usage_origin = (_usage.CACHE, False)
            return entry
        if self.source_id and source_health.is_down(self.source_id):
            self.last_source_down.append(term)
            self._usage_origin = (_usage.SKIPPED, False)
            return None
        try:
            answer = self.client.search(term, search_type=self.SEARCH_TYPE, max_results=max_results)
        except source_health.SourceDownError:
            self.last_source_down.append(term)
            self._usage_origin = (_usage.SKIPPED, False)
            return None
        except (requests.RequestException, ValueError) as exc:
            logger.warning("K10plus DK search failed for %r: %s", term, exc)
            self.last_errors[term] = str(exc)
            self._usage_origin = (_usage.NETWORK, True)
            return None
        self._usage_origin = (_usage.NETWORK, False)
        if self._cache_write:
            self._cache_write(term, self.SEARCH_TYPE, answer)
        return answer

    def extract_dk_classifications_for_keywords(self, keywords: List[str],
                                                max_results: int = 50) -> List[Dict[str, Any]]:
        self.last_cache_misses = []
        self.last_source_down = []
        self.last_errors = {}
        results: List[Dict[str, Any]] = []
        limit = clamp_max_results(max_results, default=50, limit=10_000)
        for keyword in keywords or []:
            term = re.sub(r"\s*\(GND-ID:\s*[^)]+\)\s*$", "", str(keyword or "")).strip()
            if not term:
                continue
            from src.core.search import source_usage as _usage

            self._usage_origin = None
            answer = self._answer(term, limit)
            origin, failed = self._usage_origin or (_usage.NETWORK, False)
            if answer is None:
                _usage.record(self.source_id, origin, error=failed)
                continue
            counts: Dict[Tuple[str, str], Dict[str, Any]] = {}
            for record in (answer.get("records") or [])[:limit]:
                seen = set()
                for copy in copies_of(record, self.eln):
                    for item in copy.get("notations") or []:
                        system, code = split_notation(item.get("text", ""))
                        if not system or (system, code) in seen:
                            continue
                        seen.add((system, code))
                        entry = counts.setdefault((system, code), {
                            "dk": code, "classification_type": system, "count": 0,
                            "titles": [], "keywords": [keyword],
                        })
                        entry["count"] += 1
                        title = record.get("title") or ""
                        if title and title not in entry["titles"]:
                            entry["titles"].append(title)
            _usage.record(self.source_id, origin, hit=bool(counts), error=failed)
            if counts:
                classifications = sorted(counts.values(), key=lambda c: c["count"], reverse=True)
                results.append({"keyword": keyword, "source": "sru", "search_time_ms": 0.0,
                                "classifications": classifications})
        return results
