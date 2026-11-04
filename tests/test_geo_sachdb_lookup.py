"""Lookup-Plugin geo_sachdb gegen eine kleine synthetische SQLite-Datenbank - Claude Generated.

Alle Daten sind erfunden. Geprüft werden: Verfügbarkeit (fehlende Datei, fehlende
Abbildung, fehlende Spalte), Suche mit Faltung (ä/ae), Verweisungsformen,
Notationsauflösung (offen, mehrdeutig), Satzabruf, Notationsabgleich, das
Verfügbarkeits-Gate der Werkzeugerzeugung und das Laden als Tier-2-Plugin.
"""

from __future__ import annotations

import importlib
import json
import os
import shutil
import sqlite3
import sys
import tempfile
import types
import unittest
from pathlib import Path

from src.utils.lookups.registry import LOOKUP_REGISTRY, LookupToolSpec
import src.utils.lookups  # noqa: F401 — registers the category + built-ins
from src.utils.lookups.geo_sachdb import provider as provider_mod
from src.utils.lookups.geo_sachdb.provider import GeoSachdbLookup
from src.utils.lookups.geo_sachdb.store import fold, normalize_code, parse_heading
from src.core.plugins.schema import PluginDoc
from src.utils.config_models import AlimaConfig, PluginInstanceConfig

PLUGIN_DIR = Path(__file__).resolve().parent.parent / "src" / "utils" / "lookups" / "geo_sachdb"
_HOME = {}


def setUpModule():
    _HOME["old"] = os.environ.get("HOME")
    _HOME["tmp"] = tempfile.mkdtemp()
    os.environ["HOME"] = _HOME["tmp"]


def tearDownModule():
    if _HOME.get("old") is None:
        os.environ.pop("HOME", None)
    else:
        os.environ["HOME"] = _HOME["old"]
    shutil.rmtree(_HOME["tmp"], ignore_errors=True)


def build_synthetic_db(path: Path) -> None:
    """Zwei Rollen-Tabellen + Herkunft, erfundene Werte."""
    conn = sqlite3.connect(path)
    conn.executescript("""
        CREATE TABLE sw (lauf_nr INTEGER, ansetzung TEXT, ober TEXT, verw TEXT, hinweis TEXT,
                         ref1 REAL, ref2 REAL);
        CREATE TABLE nt (nid REAL, code TEXT, erl TEXT);
        CREATE TABLE alima_import_run (id INTEGER PRIMARY KEY, source_path TEXT, source_sha256 TEXT,
                         source_size INTEGER, source_mtime TEXT, reader TEXT, reader_version TEXT,
                         jet_version TEXT, tool_version TEXT, started_at TEXT, finished_at TEXT);
        CREATE TABLE alima_import_table (source_table TEXT, target_table TEXT, rows_source INTEGER,
                         rows_written INTEGER, rows_counted INTEGER, checksum_source TEXT,
                         checksum_written TEXT, coercion_warnings INTEGER, index_notes TEXT);
    """)
    conn.executemany("INSERT INTO sw VALUES (?,?,?,?,?,?,?)", [
        (1, "|s|Testgestein !111111111! !a222222222! 12345 ‡ |s|Probegestein |s|Prüfstein",
         "s.Gestein", "|s|Mustererz", "Hinweis A", 10.0, None),
        (2, "|s|Lagerstätte !333333333!", "kein", None, None, 11.0, 99.0),
        (3, "|g|Musterland !444444444! ‡ |g|Beispielland", "kein", None, "XA-ZZ", 12.0, None),
        (4, "|s|Gestein", None, None, None, None, None),
        (5, None, None, None, None, None, None),
    ])
    conn.executemany("INSERT INTO nt VALUES (?,?,?)", [
        (10.0, "DK 999.1 !555555555!", "Testgesteine"),
        (11.0, "DK 999.21", "Lagerstaetten (Test)"),
        (12.0, "DK 999.3", "Musterland"),
        (12.0, "DK 999.31", "Doppelte ID"),
        (13.0, "DK 999", "Testklasse"),
    ])
    conn.execute("INSERT INTO alima_import_run VALUES (1,'/x/TEST.mdb','abcdef0123456789ff',1,"
                 "'2026-01-01T00:00:00+00:00','fake','0','JET4','1','t0','t1')")
    conn.executemany("INSERT INTO alima_import_table VALUES (?,?,?,?,?,?,?,?,?)", [
        ("SW", "sw", 5, 5, 5, "a", "a", 0, ""), ("NT", "nt", 5, 5, 5, "b", "b", 0, ""),
    ])
    conn.commit()
    conn.close()


def mapping(db_path) -> dict:
    return dict(
        db_path=str(db_path), term_table="sw", term_id_column="lauf_nr", term_column="ansetzung",
        term_format="swb_packed", broader_column="ober", related_column="verw", note_columns="hinweis",
        term_notation_columns="ref1,ref2", notation_table="nt", notation_id_column="nid",
        notation_code_column="code", notation_caption_column="erl", max_results=10,
    )


class _DbCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Path(self.tmp.name) / "geo.db"
        build_synthetic_db(self.db)

    def tearDown(self):
        self.tmp.cleanup()


class ParsingHelpers(unittest.TestCase):
    def test_packed_heading(self):
        head = parse_heading("|s|Testgestein !111111111! !a222222222! 12345 ‡ |s|Probegestein |s|Prüfstein",
                             "swb_packed")
        self.assertEqual((head.type, head.text), ("s", "Testgestein"))
        self.assertEqual(head.identifiers, ["111111111", "a222222222"])
        self.assertEqual(head.local_number, "12345")
        self.assertEqual([v for _, v in head.variants], ["Probegestein", "Prüfstein"])

    def test_old_style_and_non_sorting_marks(self):
        head = parse_heading("g.Musterland <Nord> 54321 ‡ g.Nordmusterland", "swb_packed")
        self.assertEqual((head.type, head.text, head.local_number), ("g", "Musterland <Nord>", "54321"))
        self.assertEqual(head.variants, [("g", "Nordmusterland")])
        person = parse_heading("|p|Muster, Hans ¬von¬ !123456789!4321 ‡ |p|Muster, H. ¬von¬", "swb_packed")
        self.assertEqual((person.text, person.local_number), ("Muster, Hans von", "4321"))

    def test_year_at_end_is_not_a_local_number(self):
        head = parse_heading("|s|Testjahr 2000 ‡ |s|Anderes", "swb_packed")
        self.assertEqual((head.text, head.local_number), ("Testjahr 2000", ""))

    def test_plain_format_keeps_text(self):
        self.assertEqual(parse_heading("  |s|Roh  ", "plain").text, "|s|Roh")

    def test_fold_and_normalize_code(self):
        self.assertEqual(fold("Lagerstätte"), fold("Lagerstaette"))
        self.assertEqual(fold("Bénédict ¬de¬"), "benedict de")
        self.assertEqual(normalize_code("DK 999.1 !555555555! !a666666666!"), ("999.1", ["555555555", "a666666666"]))
        self.assertEqual(normalize_code("999.1"), ("999.1", []))


class Availability(_DbCase):
    def test_missing_path_file_mapping_column(self):
        self.assertFalse(GeoSachdbLookup().is_available())
        self.assertIn("db_path", GeoSachdbLookup().availability()["reason"])
        gone = GeoSachdbLookup(**{**mapping(self.db), "db_path": str(self.db) + ".fehlt"})
        self.assertIn("nicht gefunden", gone.availability()["reason"])
        bare = GeoSachdbLookup(db_path=str(self.db))
        self.assertIn("Rollen-Abbildung", bare.availability()["reason"])
        wrong = GeoSachdbLookup(**{**mapping(self.db), "term_column": "gibtsnicht"})
        self.assertIn("gibtsnicht", wrong.availability()["reason"])
        out = wrong.search_terms("Testgestein")
        self.assertFalse(out["available"])
        self.assertIn("error", out)

    def test_available_with_provenance(self):
        lk = GeoSachdbLookup(**mapping(self.db))
        self.assertTrue(lk.is_available())
        prov = lk.availability()["provenance"]
        self.assertEqual(prov["source_file"], "TEST.mdb")
        self.assertEqual(prov["rows"], {"sw": 5, "nt": 5})

    def test_database_is_opened_read_only(self):
        lk = GeoSachdbLookup(**mapping(self.db))
        conn = lk.store._connect()
        try:
            with self.assertRaises(sqlite3.OperationalError):
                conn.execute("INSERT INTO nt VALUES (1, 'x', 'y')")
        finally:
            conn.close()

    def test_plugin_contract(self):
        keys = [f.key for f in GeoSachdbLookup.config_fields()]
        self.assertIn("db_path", keys)
        self.assertIn("term_notation_columns", keys)
        self.assertTrue(GeoSachdbLookup.doc().is_complete())
        self.assertEqual([s.name for s in GeoSachdbLookup.mcp_tool_specs()],
                         ["geo_sachdb_search", "geo_sachdb_record", "geo_sachdb_notation"])


class Search(_DbCase):
    def setUp(self):
        super().setUp()
        self.lk = GeoSachdbLookup(**mapping(self.db))

    def test_exact_heading_with_parsed_fields_and_notations(self):
        hit = self.lk.search_terms("Testgestein")["results"][0]
        self.assertEqual((hit["id"], hit["matched"], hit["score"]), (1, "ansetzung_exakt", 100.0))
        self.assertEqual(hit["type_label"], "Sachschlagwort")
        self.assertEqual(hit["broader"], "Gestein")
        self.assertEqual(hit["related"], ["Mustererz"])
        self.assertEqual(hit["notations"][0]["code"], "999.1")
        self.assertEqual(hit["notations"][0]["caption"], "Testgesteine")

    def test_variant_and_umlaut_folding(self):
        self.assertEqual(self.lk.search_terms("Probegestein")["results"][0]["matched"], "verweisung_exakt")
        self.assertEqual(self.lk.search_terms("Pruefstein")["results"][0]["id"], 1)
        for query in ("Lagerstätte", "Lagerstaette"):
            self.assertEqual(self.lk.search_terms(query)["results"][0]["id"], 2)

    def test_open_and_ambiguous_references_are_reported(self):
        two = self.lk.search_terms("Lagerstätte")["results"][0]["notations"]
        self.assertEqual([n["resolved"] for n in two], [True, False])
        three = self.lk.search_terms("Musterland")["results"][0]["notations"]
        self.assertEqual(len(three), 2)
        self.assertTrue(all(n["ambiguous"] for n in three))

    def test_min_score_keeps_only_identical_matches(self):
        loose = self.lk.search_terms("Gestein")
        self.assertGreater(loose["total"], 1)  # "Testgestein" als Teilwort
        strict = self.lk.search_terms("Gestein", min_score=90)
        self.assertEqual([r["id"] for r in strict["results"]], [4])

    def test_get_record_and_match_notation(self):
        rec = self.lk.get_record("1")
        self.assertEqual(rec["record"]["term"], "Testgestein")
        self.assertTrue(rec["stored"]["ansetzung"].startswith("|s|Testgestein"))
        self.assertIn("error", self.lk.get_record("777"))
        exact = self.lk.match_notation("DK 999.1")
        self.assertEqual([n["code"] for n in exact["exact"]], ["999.1"])
        self.assertEqual(exact["linked_terms_total"], 1)
        near = self.lk.match_notation("DK 999.15")
        self.assertEqual(near["exact"], [])
        self.assertEqual([n["code"] for n in near["broader"]], ["999.1", "999"])

    def test_plain_format_with_separate_variant_column(self):
        lk = GeoSachdbLookup(db_path=str(self.db), term_table="nt", term_column="erl", variant_column="code")
        self.assertEqual(lk.search_terms("Testgesteine")["results"][0]["term"], "Testgesteine")
        self.assertEqual(lk.search_terms("dk 999 21")["results"][0]["matched"], "verweisung_exakt")


class _PlainLookup:
    """Lookup ohne is_available(): das Gate darf es nicht berühren."""

    id = "geo_plain_lk"
    label = "Plain"

    def __init__(self, **config):
        self.built = True

    @classmethod
    def config_fields(cls):
        return []

    @classmethod
    def doc(cls):
        return PluginDoc(description="x", input="y", output="z")

    @classmethod
    def mcp_tool_specs(cls):
        return [LookupToolSpec(name="geo_plain_tool", description="d",
                               parameters={"type": "object", "properties": {}}, method="run",
                               cacheable=False)]

    def run(self):
        return {"ok": True}


class ToolGenerationGate(_DbCase):
    def setUp(self):
        super().setUp()
        LOOKUP_REGISTRY["geo_sachdb"] = GeoSachdbLookup
        LOOKUP_REGISTRY["geo_plain_lk"] = _PlainLookup

    def tearDown(self):
        LOOKUP_REGISTRY.pop("geo_sachdb", None)
        LOOKUP_REGISTRY.pop("geo_plain_lk", None)
        super().tearDown()

    def _registry(self, settings):
        from src.mcp.tool_registry import ToolRegistry

        cfg = AlimaConfig()
        cfg.plugins = [
            PluginInstanceConfig("geo_sachdb", "lookup", "geo_sachdb", enabled=True, settings=settings),
            PluginInstanceConfig("geo_plain_lk", "lookup", "geo_plain_lk", enabled=True),
        ]
        reg = ToolRegistry.__new__(ToolRegistry)
        reg._config_manager = types.SimpleNamespace(load_config=lambda **k: cfg)
        reg._tools, reg._handlers, reg._lookup_cache = {}, {}, {}
        return reg

    def test_unavailable_instance_gets_no_tools_others_unchanged(self):
        names = {td.name for td, _ in self._registry({"db_path": ""})._generated_lookup_tools()}
        self.assertNotIn("geo_sachdb_search", names)
        self.assertIn("geo_plain_tool", names)

    def test_available_instance_gets_tools_and_handler_answers(self):
        tools = {td.name: h for td, h in self._registry(mapping(self.db))._generated_lookup_tools()}
        self.assertTrue({"geo_sachdb_search", "geo_sachdb_record", "geo_sachdb_notation"} <= set(tools))
        out = json.loads(tools["geo_sachdb_search"](query="Testgestein"))
        self.assertEqual(out["results"][0]["id"], 1)


class Packaging(unittest.TestCase):
    def test_not_registered_on_import(self):
        LOOKUP_REGISTRY.pop("geo_sachdb", None)
        importlib.reload(provider_mod)
        importlib.import_module("src.utils.lookups.geo_sachdb")
        self.assertNotIn("geo_sachdb", LOOKUP_REGISTRY)
        init_src = (PLUGIN_DIR.parent / "__init__.py").read_text(encoding="utf-8")
        self.assertNotIn("geo_sachdb", init_src)

    def test_loads_as_tier2_code_plugin(self):
        from src.core.plugins import loader as loader_mod

        root = Path(tempfile.mkdtemp())
        try:
            shutil.copytree(PLUGIN_DIR, root / "geo_sachdb", ignore=shutil.ignore_patterns("__pycache__"))
            result = loader_mod.discover(root, approved_plugins={}, approve_cb=lambda *a: True,
                                         enable_code_plugins=True)
            self.assertEqual(result.plugins[0].status, "loaded", result.plugins[0].detail)
            self.assertEqual(result.plugins[0].type_id, "geo_sachdb")
            self.assertEqual([i.provider_id for i in result.instances], ["geo_sachdb"])
            cls = LOOKUP_REGISTRY["geo_sachdb"]
            self.assertFalse(cls().is_available())  # ohne db_path: aus
        finally:
            LOOKUP_REGISTRY.pop("geo_sachdb", None)
            for name in [m for m in sys.modules if m.startswith("alima_plugin_")]:
                sys.modules.pop(name, None)
            shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
