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


if __name__ == "__main__":
    unittest.main()
