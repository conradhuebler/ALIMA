"""scripts/access_to_sql.py — Umwandlungslogik ohne .mdb und ohne Netz - Claude Generated.

Alle Daten sind erfunden. Die Quelle ist ein ``FakeSource`` mit dem
``AccessSource``-Vertrag; der mdbtools-Pfad wird über seinen CSV-Leser und die
Schema-Parser geprüft, nicht über echte Programme.
"""

from __future__ import annotations

import importlib.util
import io
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "access_to_sql.py"
_spec = importlib.util.spec_from_file_location("access_to_sql", SCRIPT)
a2s = importlib.util.module_from_spec(_spec)
sys.modules["access_to_sql"] = a2s
_spec.loader.exec_module(a2s)


class FakeSource:
    label = "fake"

    def __init__(self, schema, data, counts=None):
        self._schema = schema
        self._data = data
        self._counts = counts or {}

    def describe(self):
        return {"reader": "fake", "reader_version": "0", "jet_version": "JET4"}

    def schema(self):
        return self._schema

    def rows(self, table):
        yield from (list(r) for r in self._data[table])

    def count(self, table):
        return self._counts.get(table)


def _fake():
    schema = {
        "Schlagwörter": a2s.TableSpec("Schlagwörter", [
            a2s.ColumnSpec("LAUF_NR", "Long Integer"),
            a2s.ColumnSpec("ERLÄUTER", "Text", 200),
            a2s.ColumnSpec("NOTIZ", "Memo/Hyperlink", 255),
            a2s.ColumnSpec("REF_ID", "Double"),
            a2s.ColumnSpec("AKTIV", "Boolean"),
        ], [a2s.IndexSpec("pk", ["LAUF_NR"], unique=True, primary=True)]),
        "DK-Liste": a2s.TableSpec("DK-Liste", [
            a2s.ColumnSpec("DK_ID", "Double"),
            a2s.ColumnSpec("DK", "Text", 80),
            a2s.ColumnSpec("BILD", "OLE"),
        ]),
    }
    data = {
        "Schlagwörter": [
            ["1", "Testgestein „Ä\"", "Zeile 1\nZeile 2", "10", "1"],
            ["2", "", None, None, "0"],
            ["3", "Probe ß", "x", "11.0", None],
        ],
        "DK-Liste": [["10", "DK 999.1", "0a0b"], ["11", "DK 999.2", None]],
    }
    return FakeSource(schema, data)


class PureHelpers(unittest.TestCase):
    def test_sanitize_identifier(self):
        used = set()
        self.assertEqual(a2s.sanitize_identifier("ERLÄUTER"), "erlaeuter")
        self.assertEqual(a2s.sanitize_identifier("DK-Geographica"), "dk_geographica")
        self.assertEqual(a2s.sanitize_identifier("Einfügefehler"), "einfuegefehler")
        self.assertEqual(a2s.sanitize_identifier("1Tabelle"), "t_1tabelle")
        self.assertEqual(a2s.sanitize_identifier("Order"), "order_")
        self.assertEqual(a2s.sanitize_identifier("ERL_UTER", used), "erl_uter")
        self.assertEqual(a2s.sanitize_identifier("ERL UTER", used), "erl_uter_2")
        self.assertEqual(a2s.sanitize_identifier("???"), "unnamed")

    def test_type_map(self):
        self.assertEqual(a2s.sqlite_type_for("Long Integer"), "INTEGER")
        self.assertEqual(a2s.sqlite_type_for("Double"), "REAL")
        self.assertEqual(a2s.sqlite_type_for("Memo/Hyperlink"), "TEXT")
        self.assertEqual(a2s.sqlite_type_for("Currency"), "TEXT")
        self.assertEqual(a2s.sqlite_type_for("OLE"), "BLOB")
        self.assertEqual(a2s.sqlite_type_for("Unbekannt"), "TEXT")

    def test_coerce_value(self):
        self.assertEqual(a2s.coerce_value("12", "INTEGER"), (12, True))
        self.assertEqual(a2s.coerce_value("12.0", "INTEGER"), (12, True))
        self.assertEqual(a2s.coerce_value(7.0, "INTEGER"), (7, True))
        self.assertEqual(a2s.coerce_value("1.5", "INTEGER"), ("1.5", False))
        self.assertEqual(a2s.coerce_value("abc", "REAL"), ("abc", False))
        self.assertEqual(a2s.coerce_value("0a0b", "BLOB"), (b"\x0a\x0b", True))
        self.assertEqual(a2s.coerce_value(None, "TEXT"), (None, True))
        self.assertEqual(a2s.coerce_value("", "TEXT"), ("", True))

    def test_parse_access_schema(self):
        text = (
            "CREATE TABLE [DK_alt]\n (\n\t[LAUF_NR]\t\t\tLong Integer, \n\t[ERLÄUTER]\t\t\tText (200), \n"
            "\t[DK_ID]\t\t\tDouble\n);\nCREATE TABLE [Leer]\n (\n);\n"
            "CREATE TABLE [Einfügefehler]\n (\n\t[Feld0]\t\t\tMemo/Hyperlink (255)\n);\n"
        )
        tables = a2s.parse_access_schema(text)
        self.assertEqual(list(tables), ["DK_alt", "Leer", "Einfügefehler"])
        cols = [(c.name, c.access_type, c.size) for c in tables["DK_alt"].columns]
        self.assertEqual(cols, [("LAUF_NR", "Long Integer", None), ("ERLÄUTER", "Text", 200), ("DK_ID", "Double", None)])
        self.assertEqual(tables["Leer"].columns, [])
        self.assertEqual(tables["Einfügefehler"].columns[0].access_type, "Memo/Hyperlink")

    def test_parse_sqlite_keys(self):
        text = (
            "CREATE TABLE `T`\n (\n\t`ID`\t\t\tINTEGER\n\t, PRIMARY KEY (`ID`)\n);\n"
            "CREATE INDEX `T_Nummer_idx` ON `T` (`Nummer`);\n"
            "CREATE UNIQUE INDEX `U_LAUF_NR_idx` ON `U` (`LAUF_NR`);\n"
        )
        keys = a2s.parse_sqlite_keys(text)
        self.assertEqual(keys["T"][0].columns, ["ID"])
        self.assertTrue(keys["T"][0].primary)
        self.assertFalse(keys["T"][1].unique)
        self.assertTrue(keys["U"][0].unique)

    def test_iter_mdb_csv_quoting_newlines_null(self):
        sentinel = a2s.NULL_SENTINEL
        raw = f'A,B,C\n1,"zwei\nZeilen","mit ""Zitat"""\n2,"",{sentinel}\n'
        rows = list(a2s.iter_mdb_csv(io.StringIO(raw, newline=""), ["A", "B", "C"]))
        self.assertEqual(rows, [["1", "zwei\nZeilen", 'mit "Zitat"'], ["2", "", None]])

    def test_iter_mdb_csv_rejects_wrong_header_and_width(self):
        with self.assertRaises(a2s.ConversionError):
            list(a2s.iter_mdb_csv(io.StringIO("A,B\n1,2\n", newline=""), ["A", "X"]))
        with self.assertRaises(a2s.ConversionError):
            list(a2s.iter_mdb_csv(io.StringIO("A,B\n1\n", newline=""), ["A", "B"]))

    def test_digest_distinguishes_null_empty_and_types(self):
        def h(row):
            d = a2s.TableDigest()
            d.update(row)
            return d.hexdigest()

        self.assertNotEqual(h([None]), h([""]))
        self.assertNotEqual(h([1]), h([1.0]))
        self.assertNotEqual(h(["1"]), h([1]))


class Readers(unittest.TestCase):
    def test_missing_mdbtools_gives_clear_message(self):
        with patch.object(a2s.shutil, "which", return_value=None):
            with self.assertRaises(a2s.ReaderUnavailable) as ctx:
                a2s.MdbToolsSource(Path("x.mdb"))
        self.assertIn("mdbtools nicht gefunden", str(ctx.exception))

    def test_missing_access_parser_gives_clear_message(self):
        with patch.dict(sys.modules, {"access_parser": None}):
            with self.assertRaises(a2s.ReaderUnavailable) as ctx:
                a2s.AccessParserSource(Path("x.mdb"))
        self.assertIn("access_parser", str(ctx.exception))

    def test_main_reports_unavailable_reader(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "q.mdb"
            src.write_bytes(b"not really")

            def factory(path, reader):
                raise a2s.ReaderUnavailable("fehlt")

            self.assertEqual(a2s.main(["inspect", str(src)], source_factory=factory), 1)


class Convert(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.src = self.dir / "quelle.mdb"
        self.src.write_bytes(b"synthetic")
        self.target = self.dir / "ziel.db"

    def tearDown(self):
        self.tmp.cleanup()

    def _rows(self, sql):
        conn = sqlite3.connect(self.target)
        try:
            return conn.execute(sql).fetchall()
        finally:
            conn.close()

    def test_convert_writes_types_nulls_and_provenance(self):
        results = a2s.convert(_fake(), self.src, self.target)
        self.assertTrue(all(r.verified for r in results))
        rows = self._rows("SELECT lauf_nr, erlaeuter, notiz, ref_id, aktiv FROM schlagwoerter ORDER BY rowid")
        self.assertEqual(rows[0], (1, 'Testgestein „Ä"', "Zeile 1\nZeile 2", 10.0, 1))
        self.assertEqual(rows[1], (2, "", None, None, 0))
        self.assertEqual(self._rows("SELECT bild FROM dk_liste ORDER BY rowid"), [(b"\x0a\x0b",), (None,)])
        meta = dict(self._rows("SELECT source_table, rows_written FROM alima_import_table"))
        self.assertEqual(meta, {"Schlagwörter": 3, "DK-Liste": 2})
        cols = self._rows("SELECT source_column, target_column, sqlite_type FROM alima_import_column "
                          "WHERE source_table='Schlagwörter' ORDER BY ordinal")
        self.assertEqual(cols[1], ("ERLÄUTER", "erlaeuter", "TEXT"))
        run = self._rows("SELECT reader, source_size, tool_version FROM alima_import_run")
        self.assertEqual(run, [("fake", len(b"synthetic"), a2s.TOOL_VERSION)])
        idx = self._rows("SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='schlagwoerter'")
        self.assertTrue(any("lauf_nr" in name for (name,) in idx))

    def test_convert_is_idempotent(self):
        a2s.convert(_fake(), self.src, self.target)
        first = self._rows("SELECT source_table, checksum_written FROM alima_import_table ORDER BY 1")
        data1 = self._rows("SELECT * FROM schlagwoerter ORDER BY rowid")
        a2s.convert(_fake(), self.src, self.target)
        self.assertEqual(first, self._rows("SELECT source_table, checksum_written FROM alima_import_table ORDER BY 1"))
        self.assertEqual(data1, self._rows("SELECT * FROM schlagwoerter ORDER BY rowid"))
        self.assertEqual(len(self._rows("SELECT * FROM alima_import_run")), 1)

    def test_count_mismatch_keeps_existing_target(self):
        a2s.convert(_fake(), self.src, self.target)
        before = self.target.read_bytes()
        bad = _fake()
        bad._counts = {"DK-Liste": 99}
        with self.assertRaises(a2s.ConversionError):
            a2s.convert(bad, self.src, self.target)
        self.assertEqual(self.target.read_bytes(), before)
        results = a2s.convert(bad, self.src, self.target, allow_count_mismatch=True)
        self.assertEqual({r.source: r.rows_counted for r in results}["DK-Liste"], 99)
        self.assertEqual(self._rows("SELECT rows_counted FROM alima_import_table WHERE source_table='DK-Liste'"), [(99,)])

    def test_failed_coercion_is_kept_and_counted(self):
        src = _fake()
        src._data["Schlagwörter"][2][0] = "3,5"
        results = a2s.convert(src, self.src, self.target)
        warn = {r.source: r.coercion_warnings for r in results}
        self.assertEqual(warn["Schlagwörter"], 1)
        self.assertEqual(self._rows("SELECT lauf_nr FROM schlagwoerter WHERE rowid=3"), [("3,5",)])

    def test_select_and_exclude(self):
        results = a2s.convert(_fake(), self.src, self.target, exclude=["DK-Liste"])
        self.assertEqual([r.source for r in results], ["Schlagwörter"])
        with self.assertRaises(a2s.ConversionError):
            a2s.plan_tables(_fake().schema(), tables=["Gibtsnicht"])

    def test_inspect_writes_nothing_and_caps_samples(self):
        report = a2s.inspect(_fake(), samples=9, max_chars=5)
        self.assertFalse(self.target.exists())
        table = report["tables"][0]
        self.assertEqual(table["rows"], 3)
        self.assertEqual(len(table["samples"]), 3)
        self.assertEqual(table["samples"][0]["ERLÄUTER"], "Testg")


MARKERS = ("GEHEIMNOTIZ", "SWBNOTIZ", "OBERMARKE", "VERWANDTMARKE", "ALTMARKE", "BESTELLMARKE", "GEOGRMARKE")


def build_geo_staging(path: Path) -> None:
    """Zwischenstand wie aus ``convert``: erfundene Werte, Markierungen in
    Spalten und Tabellen, die der Importer nicht übernehmen darf. - Claude Generated"""
    conn = sqlite3.connect(path)
    conn.executescript("""
        CREATE TABLE swb (lauf_nr INTEGER, ober_begr TEXT, swb_aequiv TEXT, swb_notes TEXT, notes TEXT,
                          dk_id REAL, dk_id1 REAL, dk_id2 REAL, dk_id3 REAL, verwandte TEXT);
        CREATE TABLE dk (dk_id REAL, erl_uter TEXT, dk TEXT);
        CREATE TABLE dk_bio (dk_mittler TEXT, erlaeuteru TEXT);
        CREATE TABLE dk_berg (dk_bergbau TEXT, erlaeuteru TEXT);
        CREATE TABLE dk_geographica (dk TEXT, erl_uter TEXT, dk_id REAL);
        CREATE TABLE dk_alt (lauf_nr INTEGER, erlaeuter TEXT, dk_id INTEGER, dk TEXT);
        CREATE TABLE eas (autor TEXT, titel TEXT, preis TEXT);
        CREATE TABLE geographica (swb_aequiv TEXT, dk_id REAL);
        CREATE TABLE alima_import_run (id INTEGER PRIMARY KEY, source_path TEXT, source_sha256 TEXT,
                         source_size INTEGER, source_mtime TEXT, reader TEXT, reader_version TEXT,
                         jet_version TEXT, tool_version TEXT, started_at TEXT, finished_at TEXT);
    """)
    conn.executemany("INSERT INTO swb VALUES (?,?,?,?,?,?,?,?,?,?)", [
        (1, "OBERMARKE", "|s|Testgestein !111111111! !a222222222! !333333333! 12345 ‡ |s|Probegestein |s|Prüfstein",
         "SWBNOTIZ", "GEHEIMNOTIZ", 10.0, 9999.0, 10.0, None, "|s|VERWANDTMARKE"),
        (2, "kein", "|g|Musterland !a999999999! !444444444! |g|Beispielland", None, None, 12.0, None, None, None, None),
        (3, None, "|p|Muster, Hans ¬von¬ {Balken ueber} 54321 ‡ |p|Muster, H.", None, None, 9999.0, None, None, None, None),
        (4, None, "|k|Testverein\n!555555555!", None, None, 77.0, None, None, None, None),
        (5, None, "|g|Testort / |s|Testthema !666666666! 480947", None, None, 13.0, None, None, None, None),
        (6, None, "|b|Testamt", None, None, None, None, None, None, None),
        (7, None, "|c|Testkongress 2000", None, None, None, None, None, None, None),
        (8, None, "Ohne Kennzeichen", None, None, None, None, None, None, None),
        (9, None, "|s| !!", None, None, None, None, None, None, None),
    ])
    conn.executemany("INSERT INTO dk VALUES (?,?,?)", [
        (10.0, "Testgesteine", "DK 999.1 !777777777! !a777777777!"),
        (12.0, "Musterland", "DK 999.3"),
        (12.0, "Doppelte\nID", "DK 999.31 !888888888!"),
        (13.0, "Testwirtschaft", "QX 100 !121212121!"),
        (9999.0, "keine DK eingetragen", "leer"),
        (None, None, None),
    ])
    conn.execute("INSERT INTO dk_bio VALUES ('999.5', 'Testbiologie')")
    conn.execute("INSERT INTO dk_berg VALUES ('DK 999.6', 'Testbergbau')")
    conn.execute("INSERT INTO dk_geographica VALUES ('DK 999.7', 'Testgeografie', 1.0)")
    conn.execute("INSERT INTO dk_alt VALUES (1, 'ALTMARKE', 10, 'DK 111.1')")
    conn.execute("INSERT INTO eas VALUES ('Autor', 'BESTELLMARKE', '9,99')")
    conn.execute("INSERT INTO geographica VALUES ('g.GEOGRMARKE', 1.0)")
    conn.execute("INSERT INTO alima_import_run VALUES (1,'/x/TEST-GEO.mdb','abcdef0123456789ff',1,"
                 "'2026-01-01T00:00:00+00:00','fake','0','JET4','1','t0','t1')")
    conn.commit()
    conn.close()


class GeoHeadingParsing(unittest.TestCase):
    def setUp(self):
        self.clean = a2s.load_geo_schema().clean_label

    def test_packed_heading(self):
        head = a2s.parse_geo_heading("|s|Testgestein !111111111! !a222222222! 12345 ‡ |s|Probegestein |s|Prüfstein",
                                     self.clean)
        self.assertEqual((head.kind, head.label, head.ppn), ("Sachschlagwort", "Testgestein", "111111111"))
        self.assertEqual(head.chain, "|s|Testgestein !111111111! !a222222222! 12345")
        self.assertEqual(head.variants, "|s|Probegestein |s|Prüfstein")

    def test_variants_inside_the_head_and_chains(self):
        head = a2s.parse_geo_heading("|g|Musterland !444444444! |g|Beispielland", self.clean)
        self.assertEqual((head.chain, head.variants), ("|g|Musterland !444444444!", "|g|Beispielland"))
        chain = a2s.parse_geo_heading("|g|Testort / |s|Testthema !666666666! 480947", self.clean)
        self.assertEqual((chain.label, chain.ppn, chain.kind), ("Testort / Testthema", "666666666", "Geographikum"))

    def test_year_at_end_is_not_a_local_number(self):
        head = a2s.parse_geo_heading("|s|Testjahr 2000 ‡ |s|Anderes", self.clean)
        self.assertEqual(head.label, "Testjahr 2000")
        number = a2s.parse_geo_heading("|s|Testjahr 123456 ‡ |s|Anderes", self.clean)
        self.assertEqual(number.label, "Testjahr")

    def test_one_line(self):
        self.assertEqual(a2s.one_line("Zeile 1 \r\n  Zeile 2\n"), ("Zeile 1 Zeile 2", True))
        self.assertEqual(a2s.one_line("  ohne  "), ("ohne", False))
        self.assertEqual(a2s.one_line(None), (None, False))


class GeoImport(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.staging = self.dir / "zwischenstand.sqlite"
        build_geo_staging(self.staging)
        self.target = self.dir / "geo_v1.db"
        self.report = a2s.import_geo(self.staging, self.target)

    def tearDown(self):
        self.tmp.cleanup()

    def _rows(self, sql, *args):
        conn = sqlite3.connect(self.target)
        try:
            return conn.execute(sql, args).fetchall()
        finally:
            conn.close()

    def _heading(self, hid):
        return self._rows("SELECT label, kind, ppn, chain, variants FROM heading WHERE id = ?", hid)[0]

    def _notations_of(self, hid):
        return self._rows("SELECT c.system, c.notation, hc.ambiguous FROM heading_classification hc "
                          "JOIN classification c ON c.id = hc.classification_id WHERE hc.heading_id = ? "
                          "ORDER BY hc.position, c.id", hid)

    def test_target_has_the_plugin_schema_and_provenance(self):
        schema = a2s.load_geo_schema()
        conn = sqlite3.connect(self.target)
        try:
            self.assertEqual(schema.check_schema(conn), "")
        finally:
            conn.close()
        meta = dict(self._rows("SELECT key, value FROM meta"))
        self.assertEqual((meta["schema_version"], meta["source"]), ("1", "TEST-GEO.mdb"))
        self.assertEqual(self._rows("SELECT source_path FROM alima_import_run"), [("/x/TEST-GEO.mdb",)])
        rules = dict(self._rows("SELECT rule, count FROM alima_import_rule"))
        self.assertEqual(rules, self.report.rules)

    def test_first_ppn_only_and_kinds(self):
        self.assertEqual(self._heading(1)[2], "111111111")
        self.assertEqual(self._heading(2)[2], "444444444")  # !a…! ist keine PPN
        self.assertEqual(self.report.rules["heading_further_ppn_ignored"], 1)
        kinds = dict(self._rows("SELECT id, kind FROM heading"))
        self.assertEqual(kinds, {1: "Sachschlagwort", 2: "Geographikum", 3: "Person", 4: "Körperschaft",
                                 5: "Geographikum", 6: "Körperschaft", 7: "sonstiges", 8: "sonstiges"})
        self.assertEqual(self.report.rules["heading_without_label"], 1)  # Satz 9
        self.assertEqual(self.report.with_ppn, 4)

    def test_placeholder_9999_is_hidden(self):
        self.assertEqual(self._notations_of(3), [])
        self.assertEqual([n[1] for n in self._notations_of(1)], ["999.1"])  # 9999 in dk_id1 fehlt
        self.assertNotIn(("leer",), self._rows("SELECT notation FROM classification"))
        self.assertEqual((self.report.rules["dk_ref_9999_hidden"], self.report.rules["dk_id_9999_entry"]), (2, 1))
        self.assertEqual(self.report.rules["dk_ref_repeated"], 1)  # dk_id2 = dk_id

    def test_duplicate_dk_id_is_marked_and_unresolved_counted(self):
        self.assertEqual(self._notations_of(2), [("DK", "999.3", 1), ("DK", "999.31", 1)])
        self.assertEqual(self.report.rules["dk_id_duplicate"], 1)
        self.assertEqual(self._notations_of(4), [])
        self.assertEqual(self.report.rules["dk_ref_unresolved"], 1)

    def test_notation_systems_ppn_and_extra_tables(self):
        rows = {(s, n): p for s, n, p in self._rows("SELECT system, notation, ppn FROM classification")}
        self.assertEqual(rows[("DK", "999.1")], "777777777")
        self.assertEqual(rows[("RVK", "QX 100")], "121212121")
        self.assertIsNone(rows[("DK", "999.3")])
        self.assertTrue({("DK", "999.5"), ("DK", "999.6"), ("DK", "999.7")} <= set(rows))
        self.assertNotIn(("DK", "111.1"), rows)  # dk_alt wird ignoriert

    def test_text_rules_line_breaks_non_sorting_marks_placeholders(self):
        label, _, _, chain, variants = self._heading(3)
        self.assertEqual(chain, "|p|Muster, Hans ¬von¬ {Balken ueber} 54321")
        self.assertEqual(label, "Muster, Hans von {Balken ueber}")
        self.assertEqual(variants, "|p|Muster, H.")
        self.assertEqual(self._heading(4)[3], "|k|Testverein !555555555!")
        self.assertEqual(self._rows("SELECT label FROM classification WHERE notation='999.31'"), [("Doppelte ID",)])
        self.assertEqual(self.report.rules["newline_to_space"], 2)

    def test_excluded_tables_and_columns_leave_no_trace(self):
        conn = sqlite3.connect(self.target)
        try:
            dump = "\n".join(conn.iterdump())
        finally:
            conn.close()
        for marker in MARKERS:
            self.assertNotIn(marker, dump)

    def test_plugin_reads_the_import(self):
        from src.utils.lookups.geo_sachdb.provider import GeoSachdbLookup

        lk = GeoSachdbLookup(db_path=str(self.target))
        self.assertTrue(lk.is_available(), lk.availability()["reason"])
        out = lk.lookup_terms(["Prüfstein", "Musterland", "Muster, Hans von {Balken ueber}", "Fehlt"])
        self.assertEqual(out["not_found"], ["Fehlt"])
        self.assertEqual(out["lines"], ["5550 !111111111!", "6700 !777777777!", "5550 !444444444!",
                                        "6700 DK 999.3", "6700 !888888888!",
                                        "5550 |p|Muster, Hans ¬von¬ {Balken ueber} 54321"])
        self.assertEqual(lk.availability()["provenance"]["source"], "TEST-GEO.mdb")

    def test_missing_tables_keep_existing_target(self):
        before = self.target.read_bytes()
        bad = self.dir / "leer.sqlite"
        sqlite3.connect(bad).close()
        with self.assertRaises(a2s.ConversionError):
            a2s.import_geo(bad, self.target)
        self.assertEqual(self.target.read_bytes(), before)

    def test_cli_needs_no_access_reader(self):
        def factory(path, reader):
            raise AssertionError("import-geo darf keinen Access-Leser öffnen")

        target = self.dir / "cli.db"
        with patch("sys.stdout", new_callable=io.StringIO) as out:
            self.assertEqual(a2s.main(["import-geo", str(self.staging), str(target)], source_factory=factory), 0)
        self.assertIn("heading 8 (mit PPN 4)", out.getvalue())
        self.assertEqual(a2s.main(["import-geo", str(self.staging), str(self.staging)], source_factory=factory), 1)


if __name__ == "__main__":
    unittest.main()
