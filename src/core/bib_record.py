"""bib_record.py — the shared bibliographic record shape (WP-D1).

Every bibliographic source emits the *same kind* of thing — identifiers, title,
authors, year, abstract, subjects, its own classifications — yet ALIMA grew a
narrow extractor per source, each with its own field names: ``id``/``rsn``/
``ppn`` for the record id, ``authors`` (nested dict) / ``authors`` (list) /
``author`` (singular list) for the same concept, classifications as parallel
``dk_codes``/``rvk_codes``/``ddc_codes`` lists here and prefixed strings
(``"DK 530.145"``) there.

``BibRecord`` is that one shape; :func:`to_bibrecord` normalises a producer's
raw dict into it. Producers keep their own shapes — normalisation happens at the
boundary, so nothing downstream has to know which source a record came from.

Deliberately NOT included: ``count``/``display_count``. Those are GND-pool
concepts (how often a *keyword* occurs across a result set); a title record has
no such number, and a field no producer fills is dead weight.

Claude Generated (WP-D1, records as first class).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional

from src.utils.classification_systems import (
    ORIGIN_AUTHORITY,
    build_classifications,
    normalize_classifications,
    split_classification_code,
)

# Role order for deriving the canonical single ``url`` from ``urls``. A bare
# ``url`` in this codebase overwhelmingly means "where a human should click",
# which is the record page — not the full text. This is a decision, not an
# obvious default: F-7 in the field audit is exactly the damage done by treating
# these roles as interchangeable. - Claude Generated
URL_ROLE_PRIORITY = ("landing", "catalog", "fulltext", "authority")


@dataclass
class BibRecord:
    """One bibliographic record, source-independent.

    Container conventions are pinned (docs/wp_records_as_first_class.md):

    * ``authors`` is ALWAYS ``List[str]`` — every producer normalizer converts.
    * ``classifications`` is ``{system: [{code, origin}]}`` with the canonical
      uppercase system keys — the same vocabulary AND entry shape as the GND
      pool, so a record's own classifications feed the classification step
      without a translation layer. A record states its own classification, so
      every entry is ``origin="authority"`` and carries no ``count``.
    * ``urls`` is a typed role map; ``url`` is DERIVED from it
      (:data:`URL_ROLE_PRIORITY`), never set independently.
    * ``identifiers`` carries only non-empty ids (``doi``/``ppn``/``isbn``/
      ``rsn``/``id``/``gnd``, plus a source's own scheme where it has one —
      ``idn`` for the DNB, ``bvnumber`` for the BVB) — the shared envelope for F-8.
    """

    source: str = ""
    identifiers: Dict[str, str] = field(default_factory=dict)
    title: str = ""
    authors: List[str] = field(default_factory=list)
    year: str = ""
    language: str = ""
    abstract: str = ""
    subjects: List[str] = field(default_factory=list)
    # WP-D1 P3: subjects that carry a GND id, as ``{term, gnd_id}`` dicts —
    # these are verified keyword candidates (the catalog links them), so they
    # can seed the GND pool without a live search. The plain terms are ALSO in
    # ``subjects`` (P1 analysis text); this field only adds the ids.
    gnd_subjects: List[Dict[str, str]] = field(default_factory=list)
    classifications: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    urls: Dict[str, str] = field(default_factory=dict)
    publisher: str = ""
    raw: Optional[Dict[str, Any]] = None

    @property
    def url(self) -> str:
        """The canonical single link, derived from :attr:`urls` by role priority."""
        for role in URL_ROLE_PRIORITY:
            if self.urls.get(role):
                return self.urls[role]
        return ""

    def to_analysis_text(self, *, max_subjects: int = 10) -> str:
        """Format this record as pipeline analysis input (WP-D1 P1).

        Title + abstract + subjects, degrading gracefully when a catalog record
        carries no abstract (then title + subjects still make a usable input).
        The labels follow the format the batch ISBN/PPN path established — this
        method replaces its two hand-rolled copies. ``classifications`` are
        deliberately NOT included: they feed the classification step as priors
        (P2), not the keyword-extraction text. - Claude Generated
        """
        parts: List[str] = []
        if self.title:
            parts.append(f"Titel: {self.title}")
        if self.authors:
            parts.append(f"Autor: {'; '.join(self.authors)}")
        published = self.publisher or self.year
        if published:
            parts.append(f"Erschienen: {published}")
        if self.abstract:
            parts.append(f"Abstract:\n{self.abstract}")
        if self.subjects:
            parts.append(f"Schlagwörter: {'; '.join(self.subjects[:max_subjects])}")
        return "\n\n".join(parts)

    def to_dict(self, *, include_raw: bool = False) -> Dict[str, Any]:
        """Serialise, omitting empty fields.

        Omission rather than ``None``/``[]``/``{}`` keeps the JSON small and
        makes "absent" unambiguous — the same convention the GND-pool payload
        uses for its optional ``display_count``.
        """
        out = {k: v for k, v in asdict(self).items() if v and k != "raw"}
        if self.url:
            out["url"] = self.url
        if include_raw and self.raw:
            out["raw"] = self.raw
        return out


def _clean_strings(values: Any) -> List[str]:
    """Flatten to a list of non-empty stripped strings.

    Handles the nesting finc uses for ``subjects`` (a list of lists).
    """
    out: List[str] = []
    if values is None:
        return out
    if isinstance(values, (str, bytes)):
        values = [values]
    for value in values:
        if isinstance(value, (list, tuple, set)):
            out.extend(_clean_strings(value))
        elif value:
            text = str(value).strip()
            if text and text not in out:
                out.append(text)
    return out


def _classifications(pairs: Iterable[tuple]) -> Dict[str, List[Dict[str, Any]]]:
    """``(system, codes)`` pairs → canonical dict, with string cleaning.

    Thin wrapper over the shared :func:`build_classifications` (WP-D2 owner);
    only the ``_clean_strings`` flattening (finc nests lists) lives here.
    Everything is ``origin="authority"``: a record STATES its classification.
    """
    return build_classifications(
        [(system, _clean_strings(codes)) for system, codes in pairs]
    )


def _from_prefixed(values: Any) -> List[tuple]:
    """``["DK 530.145", "RVK WI 5000"]`` → ``[("DK", "530.145"), ...]``.

    Unprefixed entries yield an empty system and are dropped by
    :func:`_classifications` — better a missing classification than one filed
    under a guessed system.
    """
    return [split_classification_code(value) for value in _clean_strings(values)]


def _finc_authors(authors: Any) -> List[str]:
    """Flatten VuFind's nested author structure to plain names.

    Shape is ``{"primary": {"Name": [roles]}, "secondary": {...}, "corporate":
    [...], ...}`` — the NAMES ARE THE INNER KEYS, so ``list(authors.values())``
    yields role lists, not people. The ``*_orig`` buckets repeat the same names
    unromanised and are skipped to avoid duplicates. - Claude Generated
    """
    if not isinstance(authors, dict):
        return _clean_strings(authors)
    names: List[str] = []
    for bucket, value in authors.items():
        if bucket.endswith("_orig"):
            continue
        names.extend(_clean_strings(list(value.keys()) if isinstance(value, dict) else value))
    # dedupe, preserve order
    return list(dict.fromkeys(names))


def _identifiers(**candidates: Any) -> Dict[str, str]:
    return {k: str(v).strip() for k, v in candidates.items() if v and str(v).strip()}


def _urls(**candidates: Any) -> Dict[str, str]:
    return {k: str(v).strip() for k, v in candidates.items() if v and str(v).strip()}


def _from_finc(rec: Dict[str, Any]) -> BibRecord:
    return BibRecord(
        source="finc",
        identifiers=_identifiers(id=rec.get("id"), isbn=rec.get("isbn")),
        title=str(rec.get("title") or ""),
        authors=_finc_authors(rec.get("authors")),
        year=str(rec.get("year") or ""),
        subjects=_clean_strings(rec.get("subjects")),
        # finc's facets are a separate SUBJECT_FACETS path; a record carries no
        # classifications of its own, so none are invented here.
        classifications={},
        urls=_urls(catalog=rec.get("web_url"), fulltext=rec.get("resource_url")),
        publisher=str(rec.get("publisher") or ""),
        raw=rec.get("raw"),
    )


def _from_catalog(rec: Dict[str, Any]) -> BibRecord:
    """Catalog title records carry the canonical ``classifications`` dict since
    WP-D2 (formerly parallel ``dk_codes``/``rvk_codes``/``ddc_codes`` lists) —
    normalised through the choke point rather than trusted verbatim."""
    return BibRecord(
        source="catalog",
        identifiers=_identifiers(rsn=rec.get("rsn"), isbn=rec.get("isbn")),
        title=str(rec.get("title") or ""),
        authors=_clean_strings(rec.get("authors")),
        year=str(rec.get("year") or ""),
        subjects=_clean_strings([rec.get("subjects"), rec.get("mab_subjects")]),
        classifications=normalize_classifications(
            rec.get("classifications"), origin=ORIGIN_AUTHORITY
        ),
        urls=_urls(catalog=rec.get("web_url")),
        publisher=str(rec.get("publication") or ""),
        raw=rec,
    )


def _from_sru(rec: Dict[str, Any]) -> BibRecord:
    """SRU/MARC-XML — the only producer with a native abstract.

    Classifications come from the PREFIXED ``classifications`` strings, not from
    ``decimal_classifications``: that helper strips the prefix with a
    ``(?:DDC|DK)`` regex and returns bare numbers, so it cannot say which of the
    two systems a code belongs to. ``rvk_classifications`` is unambiguous and is
    merged in. - Claude Generated
    """
    return BibRecord(
        source="sru",
        identifiers=_identifiers(rsn=rec.get("rsn"), isbn=rec.get("isbn")),
        title=str(rec.get("title") or ""),
        authors=_clean_strings(rec.get("author") or rec.get("authors")),
        year=str(rec.get("year") or ""),
        abstract=str(rec.get("abstract") or ""),
        # gnd_subjects entries are {term, gnd_id} dicts (marcxml_client.py:484);
        # a bare _clean_strings would stringify the dicts. String entries (older
        # shapes, hand-built fixtures) pass through unchanged.
        subjects=_clean_strings([
            rec.get("subjects"),
            [s.get("term") if isinstance(s, dict) else s
             for s in (rec.get("gnd_subjects") or [])],
        ]),
        gnd_subjects=[
            {"term": str(s.get("term") or "").strip(), "gnd_id": str(s.get("gnd_id") or "").strip()}
            for s in (rec.get("gnd_subjects") or [])
            if isinstance(s, dict) and str(s.get("term") or "").strip() and str(s.get("gnd_id") or "").strip()
        ],
        classifications=_classifications(
            _from_prefixed(rec.get("classifications"))
            + [("RVK", rec.get("rvk_classifications"))]
        ),
        publisher=str(rec.get("publication") or ""),
        raw=rec,
    )


def _from_k10plus(rec: Dict[str, Any]) -> BibRecord:
    """K10plus/PICA — ``ddc`` is a bare string here, and there is no abstract.

    The record's ``url`` is the 017C/209R link, i.e. the resource itself, so it
    is filed under ``fulltext``. Filing it under ``catalog`` would be the F-7
    mistake (a link role guessed from a bare field name).
    """
    isbns = _clean_strings([rec.get("isbn"), rec.get("additional_isbns")])
    return BibRecord(
        source="k10plus",
        identifiers=_identifiers(
            ppn=rec.get("ppn"), doi=rec.get("doi"), isbn=isbns[0] if isbns else "",
        ),
        title=str(rec.get("title") or ""),
        authors=_clean_strings(rec.get("authors")),
        year=str(rec.get("year") or ""),
        subjects=_clean_strings(rec.get("subjects")),
        classifications=_classifications([("DDC", rec.get("ddc"))]),
        urls=_urls(fulltext=rec.get("url")),
        publisher=str(rec.get("publisher") or ""),
        raw=rec,
    )


def _from_kvk(rec: Dict[str, Any]) -> BibRecord:
    """KVK meta-search hit — the thinnest record shape ALIMA takes in.

    The KVK JSON has no subjects and no notations, so both containers stay
    empty; inventing them from the imprint line would be a guess dressed up as
    catalogue data. ``identifiers`` come from the record *link*, not from a
    field (see ``providers/kvk/client.extract_identifiers``), and the link is
    the union catalog's record page → role ``catalog``.
    """
    return BibRecord(
        source="kvk",
        identifiers=_identifiers(
            ppn=rec.get("ppn"), idn=rec.get("idn"), bvnumber=rec.get("bvnumber"),
        ),
        title=str(rec.get("title") or ""),
        authors=_clean_strings(rec.get("author") or rec.get("authors")),
        year=str(rec.get("year") or ""),
        urls=_urls(catalog=rec.get("url")),
        # The imprint line is the only publisher-ish text the KVK gives, and only
        # for the catalogs that leave `author`/`year` empty. Kept verbatim rather
        # than split into publisher/place, which the format does not guarantee.
        publisher=str(rec.get("text") or ""),
        raw=rec,
    )


_NORMALIZERS = {
    "finc": _from_finc,
    "catalog": _from_catalog,
    "sru": _from_sru,
    "k10plus": _from_k10plus,
    "kvk": _from_kvk,
}


def to_bibrecord(record: Dict[str, Any], source: str) -> BibRecord:
    """Normalise one producer record dict into a :class:`BibRecord`.

    Takes a plain dict rather than a producer object on purpose: no imports of
    producer modules, hence no import cycles, and a dataclass producer
    (``K10PlusRecord``) converts with ``dataclasses.asdict`` at the call site.

    Raises ``ValueError`` for an unknown source — silently returning an empty
    record would hide a typo behind plausible-looking output.
    """
    normalizer = _NORMALIZERS.get(str(source or "").strip().lower())
    if normalizer is None:
        raise ValueError(
            f"to_bibrecord: unknown source {source!r} "
            f"(known: {', '.join(sorted(_NORMALIZERS))})"
        )
    return normalizer(record or {})
