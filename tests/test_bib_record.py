"""Tests for the shared BibRecord shape and its producer normalizers - Claude Generated.

The fixtures are the REAL producer shapes, lifted from the existing client
tests / dataclass definitions rather than invented, so a producer changing its
output makes these fail instead of quietly drifting:

* finc   → tests/test_finc_client.py (VuFind ``authors`` is a NESTED dict)
* catalog→ BiblioClient.search_titles (src/utils/clients/biblio_client.py)
* sru    → MarcXmlClient._parse_record (src/utils/clients/marcxml_client.py)
* k10plus→ K10PlusRecord (src/utils/k10plus_resolver.py), via asdict
"""

from __future__ import annotations

import unittest
from dataclasses import asdict

from src.core.bib_record import BibRecord, to_bibrecord
from src.utils.classification_systems import (
    ORIGIN_AUTHORITY,
    SYSTEM_KEYS,
    codes_for_system,
)
from src.utils.k10plus_resolver import K10PlusRecord

FINC_RECORD = {
    "id": "0-1025700295",
    "title": "Python: der Grundkurs",
    # The names are the INNER keys; *_orig repeats them unromanised.
    "authors": {
        "primary": {"Kofler, Michael": ["aut"]},
        "primary_orig": {"Kofler, Michael": []},
        "corporate": [], "corporate_orig": [],
        "corporate_secondary": [], "corporate_secondary_orig": [],
        "secondary": [], "secondary_orig": [],
    },
    "subjects": [["Python"], ["Programmierung"]],  # list of LISTS
    "year": "2021",
    "publisher": "Rheinwerk",
    "isbn": "9783836278454",
    "web_url": "https://katalog.example/Record/0-1025700295",
    "resource_url": "https://doi.org/10.0000/example",
    "raw": {"id": "0-1025700295"},
}

CATALOG_RECORD = {
    "rsn": 12345,
    "web_url": "https://katalog.example/rsn/12345",
    "title": "Cadmium in Böden",
    "authors": ["Jha, Ashok", "Kumar, Vinod"],
    "isbn": "9783031473906",
    "publication": "Springer",
    "year": "2023",
    "dk_codes": ["504.53"],
    "rvk_codes": ["AR 12000"],
    "ddc_codes": ["631.4"],
    "subjects": ["Schwermetall"],
    "mab_subjects": ["Bodenkunde"],
}

SRU_RECORD = {
    "rsn": "998877",
    "title": "Limnologie der Alpenseen",
    "author": ["Müller, Anna"],  # SINGULAR key
    "publication": "Verlag X",
    "isbn": "9780000000001",
    "classifications": ["DK 556.55", "DDC 551.48"],  # PREFIXED strings
    "decimal_classifications": ["556.55", "551.48"],  # lossy: system stripped
    "rvk_classifications": ["WI 5000"],
    "subjects": ["Seenkunde"],
    # Real MarcXmlClient shape: dict entries {term, gnd_id} (marcxml_client.py:484).
    # The original string-only fixture hid that _from_sru stringified these dicts.
    "gnd_subjects": [{"term": "Limnologie", "gnd_id": "4074296-3"}],
    "abstract": "Studien zur Seenkunde im Alpenraum.",
}


class TestPinnedContainerConventions(unittest.TestCase):
    """The three formerly under-specified concepts (F-3, F-7, frequency)."""

    def _all_records(self):
        return [
            to_bibrecord(FINC_RECORD, "finc"),
            to_bibrecord(CATALOG_RECORD, "catalog"),
            to_bibrecord(SRU_RECORD, "sru"),
            to_bibrecord(asdict(K10PlusRecord(ppn="123", title="T", ddc="540")), "k10plus"),
        ]

    def test_authors_is_always_a_list_of_strings(self):
        for rec in self._all_records():
            with self.subTest(source=rec.source):
                self.assertIsInstance(rec.authors, list)
                for author in rec.authors:
                    self.assertIsInstance(author, str)

    def test_classification_keys_are_canonical_systems(self):
        for rec in self._all_records():
            with self.subTest(source=rec.source):
                for system, entries in rec.classifications.items():
                    self.assertIn(system, SYSTEM_KEYS)
                    self.assertIsInstance(entries, list)
                    for e in entries:
                        self.assertIsInstance(e, dict)
                        self.assertTrue(e.get("code"))
                        self.assertEqual(e.get("origin"), ORIGIN_AUTHORITY)

    def test_url_is_derived_by_role_priority_never_set_alone(self):
        """A record page (catalog) wins over the full text — the F-7 decision."""
        rec = BibRecord(urls={"fulltext": "F", "catalog": "C", "landing": "L"})
        self.assertEqual(rec.url, "L")
        self.assertEqual(BibRecord(urls={"fulltext": "F", "catalog": "C"}).url, "C")
        self.assertEqual(BibRecord(urls={"fulltext": "F"}).url, "F")
        self.assertEqual(BibRecord().url, "")

    def test_to_dict_omits_empty_fields(self):
        rec = to_bibrecord({"ppn": "123", "title": "T"}, "k10plus")
        out = rec.to_dict()
        self.assertEqual(out["identifiers"], {"ppn": "123"})
        for absent in ("abstract", "subjects", "classifications", "urls", "url", "raw"):
            self.assertNotIn(absent, out, f"{absent} should be omitted when empty")


class TestFincNormalizer(unittest.TestCase):
    def test_nested_vufind_authors_flatten_to_names(self):
        """list(authors.values()) would yield ROLE LISTS, not people."""
        rec = to_bibrecord(FINC_RECORD, "finc")
        self.assertEqual(rec.authors, ["Kofler, Michael"])

    def test_nested_subject_lists_are_flattened(self):
        self.assertEqual(
            to_bibrecord(FINC_RECORD, "finc").subjects, ["Python", "Programmierung"]
        )

    def test_url_roles_are_distinguished(self):
        rec = to_bibrecord(FINC_RECORD, "finc")
        self.assertEqual(rec.urls["catalog"], FINC_RECORD["web_url"])
        self.assertEqual(rec.urls["fulltext"], FINC_RECORD["resource_url"])
        self.assertEqual(rec.url, FINC_RECORD["web_url"])  # catalog wins

    def test_no_classifications_are_invented(self):
        self.assertEqual(to_bibrecord(FINC_RECORD, "finc").classifications, {})


class TestCatalogNormalizer(unittest.TestCase):
    def test_parallel_code_lists_become_one_dict(self):
        rec = to_bibrecord(CATALOG_RECORD, "catalog")
        self.assertEqual(
            {s: codes_for_system(rec.classifications, s) for s in rec.classifications},
            {"DK": ["504.53"], "RVK": ["AR 12000"], "DDC": ["631.4"]},
        )
        # A record STATES its classification — it is authority, not evidence,
        # so no count is invented for it.
        for entries in rec.classifications.values():
            for e in entries:
                self.assertEqual(e["origin"], ORIGIN_AUTHORITY)
                self.assertNotIn("count", e)

    def test_rsn_lands_in_the_identifier_envelope(self):
        rec = to_bibrecord(CATALOG_RECORD, "catalog")
        self.assertEqual(rec.identifiers["rsn"], "12345")

    def test_both_subject_fields_are_merged(self):
        self.assertEqual(
            to_bibrecord(CATALOG_RECORD, "catalog").subjects,
            ["Schwermetall", "Bodenkunde"],
        )


class TestSruNormalizer(unittest.TestCase):
    def test_prefixed_strings_keep_their_system(self):
        """decimal_classifications cannot: its regex strips DDC/DK alike."""
        rec = to_bibrecord(SRU_RECORD, "sru")
        self.assertEqual(
            {s: codes_for_system(rec.classifications, s) for s in rec.classifications},
            {"DK": ["556.55"], "DDC": ["551.48"], "RVK": ["WI 5000"]},
        )

    def test_singular_author_key_is_read(self):
        self.assertEqual(to_bibrecord(SRU_RECORD, "sru").authors, ["Müller, Anna"])

    def test_abstract_is_carried(self):
        """The only producer with a native abstract — this is what unlocks P1."""
        self.assertTrue(to_bibrecord(SRU_RECORD, "sru").abstract)

    def test_gnd_subject_dicts_yield_terms_not_stringified_dicts(self):
        """gnd_subjects entries are {term, gnd_id} dicts in the real client;
        they must contribute the TERM, never str(dict)."""
        rec = to_bibrecord(SRU_RECORD, "sru")
        self.assertEqual(rec.subjects, ["Seenkunde", "Limnologie"])

    def test_gnd_subjects_carry_term_and_id(self):
        """WP-D1 P3: the GND link itself is preserved (not only the term);
        string entries (no id) stay out of gnd_subjects."""
        rec = to_bibrecord(dict(SRU_RECORD, gnd_subjects=[
            {"term": "Limnologie", "gnd_id": "4074296-3"},
            "Nur-Term-Altform",
        ]), "sru")
        self.assertEqual(rec.gnd_subjects, [{"term": "Limnologie", "gnd_id": "4074296-3"}])
        self.assertIn("Nur-Term-Altform", rec.subjects)

    def test_unprefixed_classification_is_dropped_not_guessed(self):
        rec = to_bibrecord({"classifications": ["530.145"]}, "sru")
        self.assertEqual(rec.classifications, {})


class TestToAnalysisText(unittest.TestCase):
    """The one Record→analysis-text formatter (WP-D1 P1) — replaces the two
    hand-rolled copies the batch ISBN/PPN path carried."""

    def test_full_record_keeps_the_batch_format(self):
        rec = to_bibrecord(SRU_RECORD, "sru")
        self.assertEqual(
            rec.to_analysis_text(),
            "Titel: Limnologie der Alpenseen\n\n"
            "Autor: Müller, Anna\n\n"
            "Erschienen: Verlag X\n\n"
            "Abstract:\nStudien zur Seenkunde im Alpenraum.\n\n"
            "Schlagwörter: Seenkunde; Limnologie",
        )

    def test_degrades_to_title_plus_subjects_without_abstract(self):
        """P1 requirement: not every catalog record carries an abstract."""
        rec = to_bibrecord(dict(SRU_RECORD, abstract=""), "sru")
        text = rec.to_analysis_text()
        self.assertNotIn("Abstract", text)
        self.assertIn("Titel: Limnologie der Alpenseen", text)
        self.assertIn("Schlagwörter: Seenkunde; Limnologie", text)

    def test_subjects_are_capped(self):
        rec = to_bibrecord(
            dict(SRU_RECORD, subjects=[f"S{i}" for i in range(15)], gnd_subjects=[]),
            "sru",
        )
        text = rec.to_analysis_text()
        self.assertIn("S9", text)
        self.assertNotIn("S10", text)

    def test_empty_record_yields_empty_text(self):
        self.assertEqual(BibRecord().to_analysis_text(), "")

    def test_year_stands_in_for_missing_publisher(self):
        rec = BibRecord(title="T", year="2020")
        self.assertIn("Erschienen: 2020", rec.to_analysis_text())


class TestK10PlusNormalizer(unittest.TestCase):
    def test_bare_ddc_string_becomes_a_system_dict(self):
        rec = to_bibrecord(asdict(K10PlusRecord(ddc="540")), "k10plus")
        self.assertEqual(codes_for_system(rec.classifications, "DDC"), ["540"])
        self.assertEqual(rec.classifications["DDC"][0]["origin"], ORIGIN_AUTHORITY)

    def test_record_url_is_fulltext_not_catalog(self):
        rec = to_bibrecord(asdict(K10PlusRecord(url="https://doi.org/10.1/x")), "k10plus")
        self.assertEqual(rec.urls, {"fulltext": "https://doi.org/10.1/x"})

    def test_identifiers_carry_ppn_and_doi(self):
        rec = to_bibrecord(
            asdict(K10PlusRecord(ppn="1750", doi="10.1007/x", isbn="978")), "k10plus"
        )
        self.assertEqual(rec.identifiers, {"ppn": "1750", "doi": "10.1007/x", "isbn": "978"})

    def test_no_abstract_degrades_cleanly(self):
        """K10plus has no abstract field — P1 must degrade to title+subjects."""
        self.assertEqual(to_bibrecord(asdict(K10PlusRecord()), "k10plus").abstract, "")


class TestUnknownSource(unittest.TestCase):
    def test_unknown_source_raises_rather_than_returning_empty(self):
        with self.assertRaises(ValueError):
            to_bibrecord({"title": "X"}, "nope")


if __name__ == "__main__":
    unittest.main()
