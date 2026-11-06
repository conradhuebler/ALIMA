"""Lookup-Plugin geo_sachdb gegen eine kleine synthetische Datenbank im Schema v1 - Claude Generated.

Alle Daten sind erfunden. Geprüft werden: die Schema-Helfer, Verfügbarkeit
(fehlende Datei, fremdes Schema, falsche Version, leere Tabellen), Suche mit
Faltung (ä/ae), Verweisungsformen und Filter nach Art, die fertigen WinIBW-Zeilen
(5550 mit PPN, Rückfall auf chain/label, 6700 mit PPN, ``6700 DK``, keine Zeile
für andere Systeme ohne PPN), gefundene und nicht gefundene Begriffe getrennt,
Notationsabgleich, das Verfügbarkeits-Gate der Werkzeugerzeugung und das Laden
als Tier-2-Plugin.
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
from src.utils.lookups.geo_sachdb import schema as schema_mod
from src.utils.lookups.geo_sachdb.provider import GeoSachdbLookup
from src.utils.lookups.geo_sachdb.store import fold
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


HEADINGS = [
    (1, "Testgestein", "Sachschlagwort", "111111111", "|s|Testgestein !111111111! !a222222222! 12345",
     "|s|Probegestein |s|Prüfstein"),
    (2, "Lagerstätte", "Sachschlagwort", "333333333", "|s|Lagerstätte !333333333!", None),
    (3, "Musterland", "Geographikum", "444444444", "|g|Musterland !444444444!", "|g|Beispielland"),
    (4, "Gestein", "Sachschlagwort", None, "|s|Gestein", None),
    (5, "Muster, Hans von {Balken ueber}", "Person", None, "|p|Muster, Hans ¬von¬ {Balken ueber}", None),
    (6, "Reinlabel", "sonstiges", None, None, None),
    (7, "Musterkongress", "sonstiges", "777777777", "|c|Musterkongress !777777777!", None),
]
CLASSIFICATIONS = [
    (1, "DK", "999.1", "Testgesteine", "555555555"),
    (2, "DK", "999.21", "Lagerstaetten (Test)", None),
    (3, "DK", "999.3", "Musterland", "666666666"),
    (4, "DK", "999.31", "Doppelte ID", None),
    (5, "DK", "999", "Testklasse", None),
    (6, "RVK", "QX 100", "Testwirtschaft", None),
    (7, "RVK", "QX 200", "Andere Wirtschaft", "888888888"),
    (8, "DK", "999.1", "Testgesteine (zweiter Eintrag)", "999999999"),
]
LINKS = [(1, 1, 0, 0), (2, 2, 0, 0), (2, 6, 1, 0), (3, 3, 0, 1), (3, 4, 0, 1), (7, 7, 0, 0)]


def build_synthetic_db(path: Path) -> None:
    """Schema v1 mit erfundenen Werten."""
    conn = sqlite3.connect(path)
    schema_mod.create_schema(conn)
    conn.executemany("INSERT INTO meta VALUES (?, ?)", [
        ("source", "TEST.mdb"), ("source_date", "2026-01-01T00:00:00+00:00"),
        ("source_sha256", "abcdef0123456789ff"), ("importer", "test"),
    ])
    conn.executemany("INSERT INTO heading VALUES (?,?,?,?,?,?)", HEADINGS)
    conn.executemany("INSERT INTO classification VALUES (?,?,?,?,?)", CLASSIFICATIONS)
    conn.executemany("INSERT INTO heading_classification VALUES (?,?,?,?)", LINKS)
    conn.commit()
    conn.close()


def settings(db_path) -> dict:
    return {"db_path": str(db_path), "max_results": 10}


class _DbCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Path(self.tmp.name) / "geo.db"
        build_synthetic_db(self.db)

    def tearDown(self):
        self.tmp.cleanup()


class SchemaHelpers(unittest.TestCase):
    def test_normalize_notation(self):
        norm = schema_mod.normalize_notation
        self.assertEqual(norm("DK 553.62"), ("DK", "553.62"))
        self.assertEqual(norm("UDK  553.62 "), ("DK", "553.62"))
        self.assertEqual(norm("553.62"), ("DK", "553.62"))
        self.assertEqual(norm("RVK QP 925"), ("RVK", "QP 925"))
        self.assertEqual(norm("QP925"), ("RVK", "QP 925"))
        self.assertEqual(norm(""), ("", ""))

    def test_split_variants_and_clean_label(self):
        self.assertEqual(schema_mod.split_variants("|s|Probegestein |s|Prüfstein"), ["Probegestein", "Prüfstein"])
        self.assertEqual(schema_mod.split_variants("Form A; Form B\nForm C ‡ Form A"), ["Form A", "Form B", "Form C"])
        self.assertEqual(schema_mod.split_variants(None), [])
        self.assertEqual(schema_mod.clean_label("|p|Muster, Hans ¬von¬ !123456789!"), "Muster, Hans von")
        self.assertEqual(schema_mod.clean_label("Strich {Balken ueber}"), "Strich {Balken ueber}")

    def test_fold(self):
        self.assertEqual(fold("Lagerstätte"), fold("Lagerstaette"))
        self.assertEqual(fold("Bénédict ¬de¬"), "benedict de")

    def test_fold_meets_decomposed_umlauts(self):
        """GND-Titel kommen teils mit zerlegten Umlauten (a + U+0308)."""
        import unicodedata

        for word in ("Lagerstätte", "Größe", "Überhang"):
            decomposed = unicodedata.normalize("NFD", word)
            self.assertNotEqual(decomposed, word)
            self.assertEqual(fold(decomposed), fold(word))
        self.assertEqual(fold(unicodedata.normalize("NFD", "Silberlagerstätte")),
                         fold("Silberlagerstaette"))

    def test_check_schema(self):
        conn = sqlite3.connect(":memory:")
        self.assertIn("fehlen", schema_mod.check_schema(conn))
        schema_mod.create_schema(conn)
        self.assertEqual(schema_mod.check_schema(conn), "")
        conn.execute("UPDATE meta SET value='2' WHERE key='schema_version'")
        self.assertIn("Schemaversion", schema_mod.check_schema(conn))
        self.assertTrue(schema_mod.is_ppn("21000729X"))
        self.assertFalse(schema_mod.is_ppn("a123456789"))


class Availability(_DbCase):
    def test_missing_path_and_file(self):
        self.assertFalse(GeoSachdbLookup().is_available())
        self.assertIn("db_path", GeoSachdbLookup().availability()["reason"])
        gone = GeoSachdbLookup(db_path=str(self.db) + ".fehlt")
        self.assertIn("nicht gefunden", gone.availability()["reason"])
        out = gone.search_terms("Testgestein")
        self.assertFalse(out["available"])
        self.assertIn("error", out)

    def test_foreign_schema_wrong_version_and_empty_file(self):
        staging = Path(self.tmp.name) / "staging.db"
        conn = sqlite3.connect(staging)
        conn.execute("CREATE TABLE swb (lauf_nr INTEGER, swb_aequiv TEXT)")
        conn.execute("INSERT INTO swb VALUES (1, '|s|Testgestein')")
        conn.commit()
        conn.close()
        self.assertIn("Schema", GeoSachdbLookup(db_path=str(staging)).availability()["reason"])

        conn = sqlite3.connect(self.db)
        conn.execute("UPDATE meta SET value='2' WHERE key='schema_version'")
        conn.commit()
        conn.close()
        self.assertIn("Schemaversion", GeoSachdbLookup(**settings(self.db)).availability()["reason"])

        empty = Path(self.tmp.name) / "empty.db"
        conn = sqlite3.connect(empty)
        schema_mod.create_schema(conn)
        conn.commit()
        conn.close()
        self.assertIn("leer", GeoSachdbLookup(db_path=str(empty)).availability()["reason"])

    def test_available_with_provenance(self):
        lk = GeoSachdbLookup(**settings(self.db))
        self.assertTrue(lk.is_available())
        prov = lk.availability()["provenance"]
        self.assertEqual(prov["source"], "TEST.mdb")
        self.assertEqual(prov["source_sha256"], "abcdef0123456789")
        self.assertEqual(prov["rows"], {"heading": 7, "classification": 8, "heading_classification": 6})

    def test_database_is_opened_read_only(self):
        lk = GeoSachdbLookup(**settings(self.db))
        conn = lk.store._connect()
        try:
            with self.assertRaises(sqlite3.OperationalError):
                conn.execute("INSERT INTO meta VALUES ('x', 'y')")
        finally:
            conn.close()

    def test_plugin_contract(self):
        self.assertEqual([f.key for f in GeoSachdbLookup.config_fields()], ["db_path", "max_results"])
        self.assertTrue(GeoSachdbLookup.doc().is_complete())
        self.assertEqual([s.name for s in GeoSachdbLookup.mcp_tool_specs()],
                         ["geo_sachdb_search", "geo_sachdb_lines", "geo_sachdb_notation"])


class Search(_DbCase):
    def setUp(self):
        super().setUp()
        self.lk = GeoSachdbLookup(**settings(self.db))

    def test_exact_heading_with_kind_ppn_and_lines(self):
        hit = self.lk.search_terms("Testgestein")["results"][0]
        self.assertEqual((hit["id"], hit["matched"], hit["score"]), (1, "ansetzung_exakt", 100.0))
        self.assertEqual((hit["kind"], hit["ppn"]), ("Sachschlagwort", "111111111"))
        self.assertEqual((hit["line"], hit["line_form"]), ("5550 !111111111!", "ppn"))
        self.assertEqual(hit["classifications"], [{"system": "DK", "notation": "999.1", "label": "Testgesteine",
                                                   "ppn": "555555555", "line": "6700 !555555555!"}])

    def test_variant_and_umlaut_folding(self):
        self.assertEqual(self.lk.search_terms("Probegestein")["results"][0]["matched"], "verweisung_exakt")
        self.assertEqual(self.lk.search_terms("Pruefstein")["results"][0]["id"], 1)
        for query in ("Lagerstätte", "Lagerstaette"):
            self.assertEqual(self.lk.search_terms(query)["results"][0]["id"], 2)

    def test_kind_filter(self):
        self.assertEqual(self.lk.search_terms("Musterland", kinds=["Sachschlagwort"])["total"], 0)
        self.assertEqual([r["id"] for r in self.lk.search_terms("Musterland", kinds="Geographikum")["results"]], [3])
        self.assertIn("error", self.lk.search_terms("Musterland", kinds=["Ort"]))

    def test_min_score_keeps_only_identical_matches(self):
        self.assertGreater(self.lk.search_terms("Gestein")["total"], 1)  # "Testgestein" als Teilwort
        strict = self.lk.search_terms("Gestein", min_score=90)
        self.assertEqual([r["id"] for r in strict["results"]], [4])


class Lines(_DbCase):
    def setUp(self):
        super().setUp()
        self.lk = GeoSachdbLookup(**settings(self.db))

    def test_found_and_not_found_are_reported_apart(self):
        out = self.lk.lookup_terms(["Testgestein", "Lagerstätte", "Unbekanntwort", "Lagerstaette", "Musterland"])
        self.assertEqual([f["term"] for f in out["found"]], ["Testgestein", "Lagerstätte", "Musterland"])
        self.assertEqual(out["not_found"], ["Unbekanntwort"])
        self.assertEqual(out["lines"], [
            "5550 !111111111!", "6700 !555555555!",
            "5550 !333333333!", "6700 DK 999.21",
            "5550 !444444444!", "6700 !666666666!", "6700 DK 999.31",
        ])

    def test_rvk_without_ppn_has_no_line(self):
        cls = self.lk.lookup_terms(["Lagerstätte"])["found"][0]["headings"][0]["classifications"]
        rvk = [c for c in cls if c["system"] == "RVK"]
        self.assertEqual((rvk[0]["notation"], rvk[0]["line"]), ("QX 100", None))
        self.assertEqual(self.lk.lookup_terms(["Musterkongress"])["lines"], ["5550 !777777777!", "6700 !888888888!"])

    def test_without_ppn_the_line_falls_back_to_chain_then_label(self):
        out = self.lk.lookup_terms(["Gestein", "Muster, Hans von {Balken ueber}", "Reinlabel"])
        forms = [(h["line"], h["line_form"]) for f in out["found"] for h in f["headings"]]
        self.assertEqual(forms, [
            ("5550 |s|Gestein", "chain"),
            ("5550 |p|Muster, Hans ¬von¬ {Balken ueber}", "chain"),
            ("5550 Reinlabel", "label"),
        ])

    def test_ambiguous_links_are_marked_and_kinds_filter(self):
        head = self.lk.lookup_terms(["Musterland"])["found"][0]["headings"][0]
        self.assertEqual([c.get("ambiguous") for c in head["classifications"]], [True, True])
        filtered = self.lk.lookup_terms(["Musterland", "Testgestein"], kinds=["Geographikum"])
        self.assertEqual(([f["term"] for f in filtered["found"]], filtered["not_found"]),
                         (["Musterland"], ["Testgestein"]))


class Notation(_DbCase):
    def setUp(self):
        super().setUp()
        self.lk = GeoSachdbLookup(**settings(self.db))

    def test_exact_entries_line_and_linked_headings(self):
        exact = self.lk.match_notation("DK 999.1")
        self.assertEqual([n["label"] for n in exact["exact"]], ["Testgesteine", "Testgesteine (zweiter Eintrag)"])
        self.assertEqual((exact["line"], exact["ambiguous"]), ("6700 !555555555!", True))
        self.assertEqual(exact["linked_terms_total"], 1)
        plain = self.lk.match_notation("999.21")
        self.assertEqual((plain["line"], plain["ambiguous"]), ("6700 DK 999.21", False))

    def test_broader_and_rvk(self):
        near = self.lk.match_notation("DK 999.15")
        self.assertEqual(near["exact"], [])
        self.assertEqual([n["notation"] for n in near["broader"]], ["999.1", "999.1", "999"])
        rvk = self.lk.match_notation("QX 100")
        self.assertEqual((rvk["system"], rvk["line"], rvk["linked_terms"][0]["id"]), ("RVK", None, 2))


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
        self._prior = LOOKUP_REGISTRY.get("geo_sachdb")
        LOOKUP_REGISTRY["geo_sachdb"] = GeoSachdbLookup
        LOOKUP_REGISTRY["geo_plain_lk"] = _PlainLookup

    def tearDown(self):
        # Der Eintrag ist ein Built-in; er darf nach dem Test nicht fehlen.
        if self._prior is not None:
            LOOKUP_REGISTRY["geo_sachdb"] = self._prior
        else:
            LOOKUP_REGISTRY.pop("geo_sachdb", None)
        LOOKUP_REGISTRY.pop("geo_plain_lk", None)
        super().tearDown()

    def _registry(self, instance_settings):
        from src.mcp.tool_registry import ToolRegistry

        cfg = AlimaConfig()
        cfg.plugins = [
            PluginInstanceConfig("geo_sachdb", "lookup", "geo_sachdb", enabled=True, settings=instance_settings),
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
        tools = {td.name: h for td, h in self._registry(settings(self.db))._generated_lookup_tools()}
        self.assertTrue({"geo_sachdb_search", "geo_sachdb_lines", "geo_sachdb_notation"} <= set(tools))
        out = json.loads(tools["geo_sachdb_lines"](terms=["Testgestein", "Unbekanntwort"]))
        self.assertEqual((out["lines"][0], out["not_found"]), ("5550 !111111111!", ["Unbekanntwort"]))


class Packaging(unittest.TestCase):
    def test_registered_as_builtin(self):
        import src.utils.lookups as lookups

        self.assertIn("geo_sachdb", lookups.list_lookups())
        self.assertIs(lookups.get_lookup("geo_sachdb"), provider_mod.GeoSachdbLookup)

    def test_importing_the_lookup_package_alone_registers_it(self):
        """In einem frischen Prozess, ohne dass jemand das Paket selbst importiert."""
        import subprocess

        repo = Path(__file__).resolve().parent.parent
        out = subprocess.run(
            [sys.executable, "-c",
             "import src.utils.lookups as l; print('geo_sachdb' in l.list_lookups())"],
            cwd=repo, capture_output=True, text=True, timeout=60,
        )
        self.assertEqual(out.stdout.strip().splitlines()[-1], "True", out.stderr[-500:])

    def test_builtin_without_database_is_unavailable_and_offers_no_tools(self):
        cls = provider_mod.GeoSachdbLookup
        self.assertFalse(cls().is_available())
        self.assertFalse(cls(db_path="/nonexistent/geo.db").is_available())

    def test_loads_as_tier2_code_plugin(self):
        from src.core.plugins import loader as loader_mod

        root = Path(tempfile.mkdtemp())
        builtin = LOOKUP_REGISTRY.pop("geo_sachdb", None)
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
            if builtin is not None:
                LOOKUP_REGISTRY["geo_sachdb"] = builtin
            for name in [m for m in sys.modules if m.startswith("alima_plugin_")]:
                sys.modules.pop(name, None)
            shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
