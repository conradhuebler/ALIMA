"""Cache parts of an institution package: export, plan, merge - Claude Generated.

Pinned here (Nov 2026):

1. Export: no cache by default; ``dk`` only ``success`` rows with hits, failure
   fields not shipped; ``mappings`` all rows; raw answers per source, only
   valid ones, never ``kvk``; no export without ISIL; checksums match.
2. Format: gzip, header ``{"table", "columns", "rows"}``, one JSON list per row.
3. Merge: per key the newer ``last_updated`` wins, local rows are never
   deleted, time stamps come from the package, a second merge changes nothing.
4. ISIL guard: equal (case and white space ignored) or the cache is skipped;
   the ISIL the preset sets counts before the configured one.
5. A bad cache part skips only itself; whitelists of tables and columns, types,
   bounds against decompression bombs.
6. ``--dry-run`` writes nothing, ``revert`` leaves the cache, an old preset
   without ``cache`` is unchanged, the running knowledge manager is the one
   written through, UI and CLI.

Real ``UnifiedKnowledgeManager`` on SQLite files in a temporary directory, the
singleton reset (connections closed) before and after each test. All data is
invented; HOME is temporary; no network. Not covered: MariaDB/MySQL (no server
here), a running GUI process, Windows.
"""

from __future__ import annotations

import gzip
import io
import json
import logging
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from src.core import _ukm_cache_merge as cm_mod
from src.utils import preset_apply as pa
from src.utils import preset_cache as pc
from tests.test_institution_presets import REPO, _KEEP, _Home, _app, _need_widgets, reset_knowledge_manager

T_OLD = "2026-01-10 10:00:00"
T_MID = "2026-02-10 10:00:00"
T_NEW = "2026-03-10 10:00:00"


def _ukm_class():
    from src.core.unified_knowledge_manager import UnifiedKnowledgeManager

    return UnifiedKnowledgeManager


class _CacheHome(_Home):
    """One installation whose cache database can be switched between files."""

    def setUp(self):
        _app()  # a QApplication before QtSql creates a bare QCoreApplication
        super().setUp()
        self.use_db("a", isil="ZZ-A")

    # -- databases
    def db_path(self, name: str) -> Path:
        return self.tmp / name / "alima_knowledge.db"

    def use_db(self, name: str, *, isil=None) -> None:
        """Point the saved configuration at ``<tmp>/<name>/``; reset the manager."""
        reset_knowledge_manager()
        cfg = self.config()
        self.db_path(name).parent.mkdir(parents=True, exist_ok=True)
        cfg.database_config.db_type = "sqlite"
        cfg.database_config.sqlite_path = str(self.db_path(name))
        if isil is not None:
            cfg.system_config.institution_isil = isil
        self.cm.save_config(cfg)
        self.assertTrue(self.config().database_config.sqlite_path.startswith(str(self.tmp)))

    def km(self):
        return _ukm_class()(database_config=self.config().database_config)

    def rows(self, table: str):
        spec = cm_mod.CACHE_TABLES[table]
        return self.km().db_manager.fetch_all(
            f"SELECT * FROM {table} ORDER BY {', '.join(spec.key)}")

    def stamp(self, table: str, when: str, **where) -> None:
        cond = " AND ".join(f"{k} = ?" for k in where)
        self.km().db_manager.execute_query(
            f"UPDATE {table} SET last_updated = ?, created_at = ? WHERE {cond}",
            [when, "2025-12-01 08:00:00", *where.values()])

    def seed(self) -> None:
        """Invented cache rows of every kind in the current database."""
        km = self.km()
        titles = [{"title": "Gletscher im Wandel", "classifications": ["DK 551.32"]},
                  {"title": "Eis und Fels", "classifications": ["DK 551.32", "RVK RB 10103"]}]
        km.store_catalog_dk_cache("Gletscherkunde", titles)
        km.store_catalog_dk_cache("Erzbergbau", [{"title": "Erz", "classifications": ["DK 622"]}])
        km.store_catalog_dk_cache("Leerstelle", [], status="no_results")
        km.store_catalog_dk_cache("Ausfallzeit", [], status="error", error_message="Zeitüberschreitung")
        km.update_search_mapping("Gletscherkunde", "lobid", found_gnd_ids=["4000000-1"],
                                 titles={"4000000-1": "Gletscher"})
        km.update_search_mapping("Erzbergbau", "swb", found_gnd_ids=["4000000-2"])
        km.update_search_mapping("Nichts", "lobid", found_gnd_ids=[])
        lobid = {"search_type": "kw"}
        km.store_raw_response("lobid", "Gletscherkunde", lobid,
                              json.dumps({"totalItems": 2, "member": [{"id": 1}, {"id": 2}]}),
                              http_status=200)
        km.store_raw_response("lobid", "Leerstelle", lobid,
                              json.dumps({"totalItems": 0, "member": []}), http_status=200)
        km.store_raw_response("lobid", "Serverfehler", lobid,
                              json.dumps({"totalItems": 3, "member": [{}]}), http_status=500)
        km.store_raw_response("swb", "Erzbergbau", lobid,
                              json.dumps({"subjects": {"Erz": {"count": 1}}, "totalItems": 1}))
        km.store_raw_response("kvk", "Gletscherkunde", {},
                              json.dumps({"records": [{"t": "x"}]}), result_count=1)
        for table in ("catalog_dk_cache", "search_mappings", "search_response_cache"):
            km.db_manager.execute_query(
                f"UPDATE {table} SET last_updated = ?, created_at = ?", [T_MID, "2025-12-01 08:00:00"])

    # -- package
    def export_pkg(self, parts=("dk", "mappings"), name="pkg", **kw):
        from src.utils.preset_export import export_preset

        return export_preset(self.cm, self.tmp / name, [], preset_id="cache-test",
                             cache_tables=list(parts), **kw)

    def preset_data(self, pkg: Path) -> dict:
        return json.loads((pkg / pa.PRESET_FILENAME).read_text(encoding="utf-8"))

    def edit_preset(self, pkg: Path, fn) -> None:
        data = self.preset_data(pkg)
        fn(data)
        (pkg / pa.PRESET_FILENAME).write_text(json.dumps(data, indent=2), encoding="utf-8")

    def made(self, parts=("dk", "mappings"), **kw) -> Path:
        """Seed database a, export, switch to a fresh database b."""
        self.seed()
        pkg = self.export_pkg(parts, **kw).path
        self.use_db("b", isil="ZZ-A")
        return pkg

    def cache_actions(self, plan):
        return {a.id: a for a in plan.cache.actions}


def write_file(path: Path, header: dict, lines) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wb") as gz:
        gz.write((json.dumps(header) + "\n").encode("utf-8"))
        for line in lines:
            gz.write(((line if isinstance(line, str) else json.dumps(line)) + "\n").encode("utf-8"))
    return pa.sha256_file(path)


DK_COLS = list(cm_mod.CACHE_TABLES["catalog_dk_cache"].export_columns)


def dk_row(term, when=T_NEW, status="success", count=1):
    return [term, term.lower(), json.dumps([{"title": "x", "classifications": ["DK 1"]}]),
            count, when, "2025-12-01 08:00:00", status]


# ===========================================================================
# 1./2. Export and format
# ===========================================================================
class TestCacheExport(_CacheHome):
    def test_default_export_has_no_cache(self):
        from src.utils.preset_export import export_preset

        self.seed()
        result = export_preset(self.cm, self.tmp / "plain", [], preset_id="p")
        self.assertNotIn("cache", self.preset_data(result.path))
        self.assertFalse((result.path / "files" / "cache").exists())
        self.assertEqual(result.cache, [])

    def test_dk_only_success_with_hits_and_no_failure_fields(self):
        self.seed()
        self.km().db_manager.execute_query(
            "UPDATE catalog_dk_cache SET error_message = ?, consecutive_failures = 3 "
            "WHERE search_term = ?", ["alter Fehler", "Erzbergbau"])
        self.km().store_catalog_dk_cache("Leertreffer", [], status="success")
        result = self.export_pkg(["dk"])
        block = self.preset_data(result.path)["cache"]
        self.assertEqual(block["isil"], "ZZ-A")
        self.assertTrue(block["created"])
        entry = block["tables"][0]
        self.assertEqual(set(entry), {"id", "table", "source", "sha256", "rows", "merge"})
        self.assertEqual((entry["id"], entry["table"], entry["source"], entry["merge"], entry["rows"]),
                         ("dk_cache", "catalog_dk_cache", "files/cache/catalog_dk_cache.jsonl.gz",
                          "newer_wins", 2))
        path = result.path / entry["source"]
        self.assertEqual(entry["sha256"], pa.sha256_file(path))
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            lines = fh.read().splitlines()
        header = json.loads(lines[0])
        self.assertEqual(header, {"table": "catalog_dk_cache", "columns": DK_COLS, "rows": 2})
        for col in ("error_message", "retry_after", "consecutive_failures"):
            self.assertNotIn(col, header["columns"])
        terms = [json.loads(line)[0] for line in lines[1:]]
        self.assertEqual(terms, ["Erzbergbau", "Gletscherkunde"])
        self.assertNotIn(b"alter Fehler", gzip.decompress(path.read_bytes()))
        self.assertEqual(result.cache[0]["rows"], 2)
        self.assertGreater(result.cache[0]["size"], 0)
        self.assertTrue(any("nur intern verteilen" in w for w in result.warnings))
        self.assertTrue(any("Cache: dk_cache → files/cache/" in line and "2 Einträge" in line
                            for line in result.lines()))

    def test_mappings_all_rows(self):
        self.seed()
        result = self.export_pkg(["mappings"])
        rows = list(pc.iter_part_file(result.path / "files/cache/search_mappings.jsonl.gz",
                                      "search_mappings"))
        self.assertEqual([(r["search_term"], r["suggester_type"]) for r in rows],
                         [("Erzbergbau", "swb"), ("Gletscherkunde", "lobid"), ("Nichts", "lobid")])

    def test_raw_only_valid_answers_and_never_kvk(self):
        self.seed()
        result = self.export_pkg(["raw:lobid", "raw:swb"])
        tables = {t["id"]: t for t in self.preset_data(result.path)["cache"]["tables"]}
        self.assertEqual(set(tables), {"raw_lobid", "raw_swb"})
        self.assertEqual(tables["raw_lobid"]["source_filter"], "lobid")
        lobid = list(pc.iter_part_file(result.path / tables["raw_lobid"]["source"],
                                       "search_response_cache", source_filter="lobid"))
        self.assertEqual([r["query"] for r in lobid], ["Gletscherkunde"])  # not empty, not 500
        self.assertEqual(tables["raw_swb"]["rows"], 1)
        self.assertTrue(any("raw_lobid: 1 Einträge nicht exportiert (leere Antwort)" in w
                            for w in result.warnings), result.warnings)
        with self.assertRaises(pa.PresetError):
            self.export_pkg(["raw:kvk"], name="kvk")
        self.assertFalse((self.tmp / "kvk").exists())
        offers = [o.part for o in pc.cache_offers(self.config().database_config)]
        self.assertEqual(offers, ["dk", "mappings", "raw:lobid", "raw:swb"])

    def test_no_isil_no_export(self):
        self.seed()
        cfg = self.config()
        cfg.system_config.institution_isil = "  "
        self.cm.save_config(cfg)
        with self.assertRaises(pa.PresetError) as ctx:
            self.export_pkg(["dk"])
        self.assertIn("Ohne ISIL", str(ctx.exception))
        self.assertFalse((self.tmp / "pkg").exists())

    def test_unknown_part_is_refused(self):
        with self.assertRaises(pa.PresetError):
            self.export_pkg(["gnd_entries"])

    def test_read_without_a_running_manager_closes_its_connection(self):
        from PyQt6.QtSql import QSqlDatabase

        self.seed()
        reset_knowledge_manager()
        self.assertIsNone(_ukm_class()._instance)
        result = self.export_pkg(["dk"])
        self.assertEqual(self.preset_data(result.path)["cache"]["tables"][0]["rows"], 2)
        self.assertIsNone(_ukm_class()._instance)  # export does not create the manager
        self.assertFalse(any(n.startswith("preset_cache_") for n in QSqlDatabase.connectionNames()))

    def test_same_rows_same_bytes(self):
        self.seed()
        first = self.export_pkg(["dk"], name="one")
        second = self.export_pkg(["dk"], name="two")
        name = "files/cache/catalog_dk_cache.jsonl.gz"
        self.assertEqual(pa.sha256_file(first.path / name), pa.sha256_file(second.path / name))

    def test_export_pages_by_key_without_repeats(self):
        from itertools import islice

        self.seed()
        rows = list(islice(cm_mod.iter_export_rows(self.km().db_manager, "search_mappings",
                                                   page_size=1), 10))
        self.assertEqual([(r["search_term"], r["suggester_type"]) for r in rows],
                         [("Erzbergbau", "swb"), ("Gletscherkunde", "lobid"), ("Nichts", "lobid")])

    def test_whitelist_matches_the_created_schema(self):
        db = self.km().db_manager
        for table, spec in cm_mod.CACHE_TABLES.items():
            info = db.fetch_all(f"PRAGMA table_info({table})")
            self.assertEqual(tuple(r["name"] for r in info), spec.column_names, table)
            self.assertEqual({r["name"] for r in info if r["pk"]}, set(spec.key), table)
            not_null = {r["name"] for r in info if r["notnull"] or r["pk"]}
            self.assertEqual(not_null, set(spec.not_null), table)


# ===========================================================================
# 3. Merge
# ===========================================================================
class TestCacheMerge(_CacheHome):
    def test_round_trip_gives_the_same_rows(self):
        self.seed()
        exported = {t: [{c: r[c] for c in cm_mod.CACHE_TABLES[t].export_columns}
                        for r in self.rows(t)] for t in ("catalog_dk_cache", "search_mappings")}
        exported["catalog_dk_cache"] = [r for r in exported["catalog_dk_cache"]
                                        if r["search_status"] == "success"]
        pkg = self.export_pkg(["dk", "mappings"]).path
        self.use_db("b", isil="ZZ-A")
        self.assertFalse(self.db_path("b").exists())
        plan = self.plan(pkg)
        acts = self.cache_actions(plan)
        self.assertEqual((acts["dk_cache"].new, acts["mappings"].new), (2, 3))
        entry = pa.apply_plan(plan, self.cm)
        for table, rows in exported.items():
            got = [{c: r[c] for c in cm_mod.CACHE_TABLES[table].export_columns}
                   for r in self.rows(table)]
            self.assertEqual(got, rows, table)
        self.assertEqual([(r["id"], r["status"], r["new"], r["updated"], r["unchanged"])
                          for r in entry["cache"]],
                         [("dk_cache", "merged", 2, 0, 0), ("mappings", "merged", 3, 0, 0)])
        self.assertEqual(self.config().applied_presets[-1]["cache"], entry["cache"])

    def test_newer_wins_local_rows_stay_time_stamps_from_the_package(self):
        pkg = self.made(["dk"])
        km = self.km()
        # local, older than the package: updated; newer: kept; only local: kept
        km.store_catalog_dk_cache("Gletscherkunde", [{"title": "lokal alt", "classifications": []}],
                                  status="timeout", error_message="lokal")
        km.store_catalog_dk_cache("Erzbergbau", [{"title": "lokal neu", "classifications": []}])
        km.store_catalog_dk_cache("Nur hier", [{"title": "lokal", "classifications": []}])
        self.stamp("catalog_dk_cache", T_OLD, search_term="Gletscherkunde")
        self.stamp("catalog_dk_cache", T_NEW, search_term="Erzbergbau")
        plan = self.plan(pkg)
        act = self.cache_actions(plan)["dk_cache"]
        self.assertEqual((act.new, act.updated, act.unchanged), (0, 1, 1))
        self.assertIn("dk_cache (catalog_dk_cache): neu: 0, aktualisiert: 1, "
                      "unverändert (lokal neuer oder gleich): 1", "\n".join(plan.lines()))
        pa.apply_plan(plan, self.cm)
        rows = {r["search_term"]: r for r in self.rows("catalog_dk_cache")}
        self.assertEqual(set(rows), {"Gletscherkunde", "Erzbergbau", "Nur hier"})
        glet = rows["Gletscherkunde"]
        self.assertEqual((glet["search_status"], glet["consecutive_failures"]), ("success", 0))
        nulls = self.km().db_manager.fetch_one(
            "SELECT COUNT(*) AS n FROM catalog_dk_cache WHERE search_term = ? "
            "AND error_message IS NULL AND retry_after IS NULL", ["Gletscherkunde"])
        self.assertEqual(nulls["n"], 1)  # fetch_* reads NULL as "", so ask SQL
        self.assertIn("Gletscher im Wandel", glet["found_titles"])
        self.assertEqual((glet["last_updated"], glet["created_at"]), (T_MID, "2025-12-01 08:00:00"))
        self.assertIn("lokal neu", rows["Erzbergbau"]["found_titles"])
        self.assertEqual(rows["Erzbergbau"]["last_updated"], T_NEW)

    def test_a_newer_local_failure_is_counted_and_stays(self):
        pkg = self.made(["dk"])
        self.km().store_catalog_dk_cache("Gletscherkunde", [], status="error", error_message="x")
        self.stamp("catalog_dk_cache", T_NEW, search_term="Gletscherkunde")
        act = self.cache_actions(self.plan(pkg))["dk_cache"]
        self.assertEqual((act.new, act.updated, act.unchanged, act.local_failures), (1, 0, 1, 1))

    def test_second_application_changes_nothing(self):
        pkg = self.made(["dk", "mappings"])
        self.assertIsNotNone(pa.apply_plan(self.plan(pkg), self.cm))
        before = (self.rows("catalog_dk_cache"), self.rows("search_mappings"), self.disk())
        again = self.plan(pkg)
        self.assertEqual({(a.new, a.updated) for a in again.cache.actions}, {(0, 0)})
        self.assertTrue(again.is_empty, again.lines())
        self.assertIsNone(pa.apply_plan(again, self.cm))
        self.assertEqual((self.rows("catalog_dk_cache"), self.rows("search_mappings"), self.disk()),
                         before)

    def test_blocks_commit_and_a_failing_block_rolls_back(self):
        km = self.km()
        spec = cm_mod.CACHE_TABLES["catalog_dk_cache"]
        good = [cm_mod.check_row(spec, DK_COLS, dk_row(f"Begriff {n}")) for n in range(3)]
        counts = km.merge_cache_rows("catalog_dk_cache", good, block_size=2)
        self.assertEqual(counts["new"], 3)
        rows = [cm_mod.check_row(spec, DK_COLS, dk_row(t)) for t in ("Neu A", "Neu B", "Neu C", "Neu C")]
        with self.assertRaises(ValueError):
            km.merge_cache_rows("catalog_dk_cache", rows, block_size=2)
        terms = {r["search_term"] for r in self.rows("catalog_dk_cache")}
        self.assertEqual(terms, {"Begriff 0", "Begriff 1", "Begriff 2", "Neu A", "Neu B"})

    def test_dry_run_counts_only(self):
        km = self.km()
        spec = cm_mod.CACHE_TABLES["catalog_dk_cache"]
        rows = [cm_mod.check_row(spec, DK_COLS, dk_row("Nur gezählt"))]
        self.assertEqual(km.merge_cache_rows("catalog_dk_cache", rows, dry_run=True)["new"], 1)
        self.assertEqual(self.rows("catalog_dk_cache"), [])

    def plan(self, pkg):
        return pa.build_plan(pa.load_preset(pkg), self.cm)


# ===========================================================================
# 4. ISIL guard
# ===========================================================================
class TestIsilGuard(_CacheHome):
    def plan(self, pkg):
        return pa.build_plan(pa.load_preset(pkg), self.cm)

    def statuses(self, plan):
        return {(a.status, a.reason) for a in plan.cache.actions}

    def test_equal_ignoring_case_and_white_space(self):
        pkg = self.made(["dk"])
        self.edit_preset(pkg, lambda d: d["cache"].__setitem__("isil", " zz - a "))
        self.assertEqual(self.statuses(self.plan(pkg)), {("merge", "")})

    def test_an_exported_package_sets_its_isil_so_the_cache_matches(self):
        pkg = self.made(["dk"])
        self.use_db("b", isil="ZZ-ANDERS")
        self.assertEqual(self.preset_data(pkg)["institution"]["isil"], "ZZ-A")
        plan = self.plan(pkg)
        self.assertIn("Einrichtung: ISIL: ZZ-ANDERS → ZZ-A", "\n".join(plan.lines()))
        self.assertEqual(self.statuses(plan), {("merge", "")})

    def test_mismatch_skips_with_a_warning_and_writes_nothing(self):
        pkg = self.made(["dk"])
        self.edit_preset(pkg, lambda d: d.pop("institution"))
        self.use_db("b", isil="ZZ-ANDERS")
        plan = self.plan(pkg)
        self.assertEqual(self.statuses(plan),
                         {("skipped", "ISIL passt nicht (Paket ZZ-A, Ziel ZZ-ANDERS)")})
        self.assertTrue(any("Cache nicht eingespielt: ISIL passt nicht" in w for w in plan.warnings))
        self.assertIn("übersprungen: ISIL passt nicht", "\n".join(plan.lines()))
        self.assertTrue(plan.is_empty)
        self.assertFalse(self.db_path("b").exists())

    def test_missing_in_package(self):
        pkg = self.made(["dk"])
        self.edit_preset(pkg, lambda d: d["cache"].pop("isil"))
        self.assertEqual(self.statuses(self.plan(pkg)), {("skipped", "ISIL fehlt im Paket")})

    def test_missing_in_target(self):
        pkg = self.made(["dk"])
        self.edit_preset(pkg, lambda d: d.pop("institution"))
        self.use_db("b", isil="")
        (status, reason), = self.statuses(self.plan(pkg))
        self.assertEqual(status, "skipped")
        self.assertTrue(reason.startswith("ISIL fehlt im Ziel"), reason)

    def test_the_isil_the_preset_sets_counts_first(self):
        pkg = self.made(["dk"])
        self.edit_preset(pkg, lambda d: d.__setitem__("institution", {"isil": "ZZ-A", "eln": "1"}))
        for configured in ("", "ZZ-ALT"):  # a fresh installation, and another ISIL
            self.use_db("b", isil=configured)
            plan = self.plan(pkg)
            self.assertEqual(self.statuses(plan), {("merge", "")}, configured)
            self.assertEqual(plan.cache.target_isil, "ZZ-A")
        pa.apply_plan(plan, self.cm)
        self.assertEqual(self.config().system_config.institution_isil, "ZZ-A")
        self.assertEqual(len(self.rows("catalog_dk_cache")), 2)

    def test_an_isil_changed_after_the_preview_is_refused(self):
        pkg = self.made(["dk"])
        plan = self.plan(pkg)
        cfg = self.config()
        cfg.system_config.institution_isil = "ZZ-NEU"
        self.cm.save_config(cfg)
        with self.assertRaises(pa.PresetError):
            pa.apply_plan(plan, self.cm)
        self.assertEqual(self.rows("catalog_dk_cache"), [])


# ===========================================================================
# 5. Bad cache parts skip only themselves
# ===========================================================================
class TestBadCacheParts(_CacheHome):
    def plan(self, pkg):
        return pa.build_plan(pa.load_preset(pkg), self.cm)

    def reason(self, pkg, entry_id="dk_cache"):
        plan = self.plan(pkg)
        self.assertEqual(plan.errors, [])
        act = self.cache_actions(plan)[entry_id]
        self.assertEqual(act.status, "skipped")
        return act.reason

    def replace_file(self, pkg, header, lines, entry_id="dk_cache"):
        path = pkg / "files" / "cache" / "catalog_dk_cache.jsonl.gz"
        digest = write_file(path, header, lines)

        def fix(d):
            for t in d["cache"]["tables"]:
                if t["id"] == entry_id:
                    t["sha256"] = digest
                    t["rows"] = header.get("rows", 0)
        self.edit_preset(pkg, fix)

    def test_wrong_checksum_skips_only_the_cache(self):
        pkg = self.made(["dk", "mappings"])
        self.edit_preset(pkg, lambda d: (d["cache"]["tables"][0].__setitem__("sha256", "0" * 64),
                                         d.__setitem__("institution", {"isil": "ZZ-A", "eln": "77"})))
        plan = self.plan(pkg)
        acts = self.cache_actions(plan)
        self.assertTrue(acts["dk_cache"].reason.startswith("Prüfsumme stimmt nicht"))
        self.assertEqual(acts["mappings"].status, "merge")
        self.assertEqual(plan.errors, [])
        pa.apply_plan(plan, self.cm)
        self.assertEqual(self.config().system_config.institution_eln, "77")
        self.assertEqual(self.rows("catalog_dk_cache"), [])
        self.assertEqual(len(self.rows("search_mappings")), 3)

    def test_missing_file(self):
        pkg = self.made(["dk"])
        (pkg / "files" / "cache" / "catalog_dk_cache.jsonl.gz").unlink()
        self.assertTrue(self.reason(pkg).startswith("Datei fehlt"))

    def test_file_changed_after_the_preview_is_not_merged(self):
        pkg = self.made(["dk"])
        plan = self.plan(pkg)
        write_file(pkg / "files" / "cache" / "catalog_dk_cache.jsonl.gz",
                   {"table": "catalog_dk_cache", "columns": DK_COLS, "rows": 1}, [dk_row("Fremd")])
        entry = pa.apply_plan(plan, self.cm)
        self.assertEqual(entry["cache"][0]["status"], "skipped")
        self.assertIn("seit der Vorschau geändert", entry["cache"][0]["reason"])
        self.assertEqual(self.rows("catalog_dk_cache"), [])

    def test_table_not_in_the_whitelist(self):
        pkg = self.made(["dk"])
        self.edit_preset(pkg, lambda d: d["cache"]["tables"][0].__setitem__("table", "gnd_entries"))
        self.assertTrue(self.reason(pkg).startswith("Tabelle 'gnd_entries' ist nicht erlaubt"))

    def test_header_with_another_table(self):
        pkg = self.made(["dk"])
        self.replace_file(pkg, {"table": "gnd_entries", "columns": DK_COLS, "rows": 0}, [])
        self.assertIn("Tabelle 'gnd_entries'", self.reason(pkg))

    def test_unknown_column(self):
        pkg = self.made(["dk"])
        self.replace_file(pkg, {"table": "catalog_dk_cache", "columns": DK_COLS + ["evil"], "rows": 1},
                          [dk_row("Gletscherkunde") + ["x"]])
        self.assertIn("unbekannte Spalte 'evil'", self.reason(pkg))

    def test_wrong_type(self):
        pkg = self.made(["dk"])
        row = dk_row("Gletscherkunde")
        row[3] = "1"
        self.replace_file(pkg, {"table": "catalog_dk_cache", "columns": DK_COLS, "rows": 1}, [row])
        self.assertIn("'result_count' ist keine ganze Zahl", self.reason(pkg))

    def test_a_failure_row_in_the_file_is_refused(self):
        pkg = self.made(["dk"])
        self.replace_file(pkg, {"table": "catalog_dk_cache", "columns": DK_COLS, "rows": 1},
                          [dk_row("Leerstelle", status="no_results", count=0)])
        self.assertIn("nur erfolgreiche Einträge mit Treffern", self.reason(pkg))

    def test_a_kvk_raw_file_is_refused(self):
        pkg = self.made(["dk"])
        self.edit_preset(pkg, lambda d: d["cache"]["tables"][0].update(
            table="search_response_cache", source_filter="kvk"))
        self.assertIn("'kvk' werden nie übernommen", self.reason(pkg))

    def test_a_key_twice(self):
        pkg = self.made(["dk"])
        self.replace_file(pkg, {"table": "catalog_dk_cache", "columns": DK_COLS, "rows": 2},
                          [dk_row("Doppelt"), dk_row("Doppelt")])
        self.assertIn("doppelt", self.reason(pkg))

    def test_row_count_must_match_the_header(self):
        pkg = self.made(["dk"])
        self.replace_file(pkg, {"table": "catalog_dk_cache", "columns": DK_COLS, "rows": 3},
                          [dk_row("Eins")])
        self.assertIn("1 Zeilen, die Kopfzeile nennt 3", self.reason(pkg))

    def test_decompression_bounds(self):
        pkg = self.made(["dk"])
        # one line of 20 MB packs into a few KB: refused without reading it whole
        bomb = json.dumps(dk_row("Bombe")[:2] + ["[\"" + "a" * (20 * 1024 * 1024) + "\"]"] +
                          dk_row("Bombe")[3:])
        self.replace_file(pkg, {"table": "catalog_dk_cache", "columns": DK_COLS, "rows": 1}, [bomb])
        self.assertLess((pkg / "files/cache/catalog_dk_cache.jsonl.gz").stat().st_size, 200_000)
        self.assertIn("Zeile länger als", self.reason(pkg))
        self.replace_file(pkg, {"table": "catalog_dk_cache", "columns": DK_COLS, "rows": 50},
                          [dk_row(f"Begriff {n}") for n in range(50)])
        with patch.object(pc, "MAX_UNPACKED_BYTES", 4000):
            self.assertIn("entpackt größer als", self.reason(pkg))
        with patch.object(pc, "MAX_ROWS_PER_FILE", 10):
            self.assertIn("mehr als 10 Einträge", self.reason(pkg))

    def test_not_gzip(self):
        pkg = self.made(["dk"])
        path = pkg / "files" / "cache" / "catalog_dk_cache.jsonl.gz"
        path.write_bytes(b"kein gzip")
        self.edit_preset(pkg, lambda d: d["cache"]["tables"][0].__setitem__("sha256", pa.sha256_file(path)))
        self.assertIn("keine lesbare gzip-Datei", self.reason(pkg))

    def test_source_must_stay_in_the_package(self):
        pkg = self.made(["dk"])
        self.edit_preset(pkg, lambda d: d["cache"]["tables"][0].__setitem__(
            "source", "../außerhalb.jsonl.gz"))
        self.assertIn("darf den Paketordner nicht verlassen", self.reason(pkg))


# ===========================================================================
# 6. dry run, revert, old presets, the running manager, first start
# ===========================================================================
class TestCacheLifecycle(_CacheHome):
    def plan(self, pkg):
        return pa.build_plan(pa.load_preset(pkg), self.cm)

    def run_cli(self, *argv, answer=None):
        from src.cli.commands import preset_cmd
        from src.cli.main import create_argument_parser

        args = create_argument_parser().parse_args(["preset", *argv])
        out = io.StringIO()
        with redirect_stdout(out), patch("builtins.input", return_value=answer or "n"):
            rc = preset_cmd.handle_preset(args, logging.getLogger("test"))
        return rc, out.getvalue()

    def test_dry_run_writes_nothing(self):
        pkg = self.made(["dk", "mappings"])
        before = self.disk()
        rc, out = self.run_cli("apply", str(pkg), "--dry-run")
        self.assertEqual(rc, 0)
        self.assertIn("dk_cache (catalog_dk_cache): neu: 2", out)
        self.assertEqual(self.disk(), before)
        self.assertFalse(self.db_path("b").exists())  # no database created either
        self.km()  # an existing, empty database: still nothing written
        reset_knowledge_manager()
        rc, out = self.run_cli("apply", str(pkg), "--dry-run")
        self.assertIn("neu: 3", out)
        self.assertEqual(self.rows("catalog_dk_cache"), [])

    def test_cli_apply_shows_the_counts(self):
        pkg = self.made(["dk"])
        rc, out = self.run_cli("apply", str(pkg), "--yes")
        self.assertEqual(rc, 0)
        self.assertIn("Cache dk_cache: neu: 2, aktualisiert: 0, unverändert: 0", out)

    def test_revert_leaves_the_cache(self):
        pkg = self.made(["dk"])
        self.edit_preset(pkg, lambda d: d.__setitem__("institution", {"isil": "ZZ-A", "eln": "55"}))
        pa.apply_plan(self.plan(pkg), self.cm)
        rows = self.rows("catalog_dk_cache")
        result = pa.revert_last(self.cm)
        self.assertEqual(self.config().system_config.institution_eln, "")
        self.assertEqual(self.rows("catalog_dk_cache"), rows)
        self.assertEqual(len(rows), 2)
        self.assertTrue(any(line.startswith("Cache: bleibt") for line in result.lines()))

    def test_old_preset_without_cache_is_unchanged(self):
        from tests.test_institution_presets import make_package

        plan = self.plan(make_package(self.tmp / "alt"))
        self.assertIsNone(plan.cache)
        self.assertFalse(any(line.startswith("Cache") for line in plan.lines()), plan.lines())
        self.assertIsNone(plan.as_dict()["cache"])

    def test_the_running_manager_is_written_through(self):
        pkg = self.made(["dk"])
        running = self.km()
        seen = []
        real = type(running).merge_cache_rows

        def spy(manager, table, rows, **kw):
            seen.append(manager)
            return real(manager, table, rows, **kw)

        with patch.object(type(running), "merge_cache_rows", spy):
            pa.apply_plan(self.plan(pkg), self.cm)
        self.assertEqual(seen, [running])
        self.assertIs(_ukm_class()._instance, running)
        self.assertEqual(len(running.db_manager.fetch_all("SELECT * FROM catalog_dk_cache")), 2)

    def test_a_running_manager_on_another_database_is_not_used(self):
        pkg = self.made(["dk"])
        reset_knowledge_manager()
        (self.tmp / "c").mkdir(exist_ok=True)
        other = _ukm_class()(database_config=type(self.config().database_config)(
            db_type="sqlite", sqlite_path=str(self.tmp / "c" / "x.db")))
        plan = self.plan(pkg)
        (status, reason), = {(a.status, a.reason) for a in plan.cache.actions}
        self.assertEqual(status, "skipped")
        self.assertIn("die laufende Anwendung nutzt SQLite", reason)
        self.assertEqual(other.db_manager.fetch_all("SELECT * FROM catalog_dk_cache"), [])

    def test_a_running_manager_on_the_sqlite_fallback_is_not_used(self):
        from types import SimpleNamespace

        cfg = self.config().database_config
        running = SimpleNamespace(db_manager=SimpleNamespace(_fallback_triggered=True, config=cfg))
        with self.assertRaises(pc.CacheDbUnavailable) as ctx:
            pc._check_running(running, cfg)
        self.assertIn("SQLite-Rückfall", str(ctx.exception))
        pc._check_running(SimpleNamespace(db_manager=SimpleNamespace(
            _fallback_triggered=False, config=cfg)), cfg)

    def test_first_start_merges_into_the_saved_database(self):
        pkg = self.made(["dk"])
        lines = pa.apply_after_setup(pkg, self.cm)
        self.assertIn("angewendet", lines[0])
        self.assertTrue(any("Cache dk_cache: neu: 2" in line for line in lines), lines)
        self.assertEqual(len(self.rows("catalog_dk_cache")), 2)
        self.assertTrue(self.db_path("b").exists())

    def test_first_start_never_raises_on_a_broken_cache(self):
        pkg = self.made(["dk"])
        (pkg / "files" / "cache" / "catalog_dk_cache.jsonl.gz").write_bytes(b"kaputt")
        lines = pa.apply_after_setup(pkg, self.cm)
        self.assertIn("nichts zu ändern", lines[0])
        self.assertEqual(self.rows("catalog_dk_cache"), [])

    def test_cli_export_with_cache(self):
        self.seed()
        rc, out = self.run_cli("export", str(self.tmp / "e1"), "--with-cache")
        self.assertEqual(rc, 0, out)
        ids = [t["id"] for t in self.preset_data(self.tmp / "e1")["cache"]["tables"]]
        self.assertEqual(ids, ["dk_cache", "mappings"])
        rc, out = self.run_cli("export", str(self.tmp / "e2"), "--with-cache=raw:lobid")
        self.assertEqual([t["id"] for t in self.preset_data(self.tmp / "e2")["cache"]["tables"]],
                         ["raw_lobid"])
        rc, out = self.run_cli("export", str(self.tmp / "e3"))
        self.assertNotIn("cache", self.preset_data(self.tmp / "e3"))
        rc, out = self.run_cli("export", str(self.tmp / "e4"), "--with-cache=raw:kvk")
        self.assertEqual(rc, 1)
        self.assertIn("nie exportiert", out)

    def test_example_package_cache(self):
        example = REPO / "examples" / "presets" / "demo_institution"
        data = self.preset_data(example)
        self.assertEqual(data["cache"]["isil"], "ZZ-DEMO")
        self.use_db("demo", isil="")
        plan = self.plan(example)
        self.assertEqual(plan.warnings, [])
        act = self.cache_actions(plan)["dk_cache"]
        self.assertEqual((act.status, act.new), ("merge", act.rows))
        self.assertGreater(act.rows, 0)


# ===========================================================================
# Values and rules
# ===========================================================================
class TestRules(_CacheHome):
    def test_raw_answer_problem(self):
        p = cm_mod.raw_answer_problem
        self.assertEqual(p({"totalItems": 2, "member": [{}]}, 0), "")
        self.assertEqual(p({"totalItems": 0, "member": []}, 0), "leere Antwort")
        self.assertEqual(p({"records": [], "totalItems": 0}, 0), "leere Antwort")
        self.assertEqual(p({"records": [], "total": 0}, 0), "leere Antwort")
        self.assertEqual(p({"subjects": {}, "url": "u", "totalItems": 0}, 0), "leere Antwort")
        self.assertEqual(p({"records": [{}], "errors": ["x"]}, 0), "Antwort mit Fehler")
        self.assertEqual(p({"text": "Abstract", "method": "api"}, 0), "")
        self.assertEqual(p({"records": []}, 3), "")  # result_count of kvk/sru counts

    def test_time_stamps(self):
        from datetime import datetime

        self.assertEqual(cm_mod.format_timestamp("2026-11-10T14:02:03.5"), "2026-11-10 14:02:03")
        self.assertEqual(cm_mod.format_timestamp(datetime(2026, 1, 2, 3, 4, 5)), "2026-01-02 03:04:05")
        self.assertEqual(cm_mod.format_timestamp("2026-01-02 04:04:05+01:00"), "2026-01-02 03:04:05")
        with self.assertRaises(ValueError):
            cm_mod.format_timestamp("gestern")

    def test_parse_parts(self):
        self.assertEqual(pc.parse_parts("dk, mappings,raw:lobid,dk"), ["dk", "mappings", "raw:lobid"])
        for bad in ("gnd", "raw:", "raw:KVK"):
            with self.assertRaises(ValueError):
                pc.parse_parts(bad)


# ===========================================================================
# UI (offscreen)
# ===========================================================================
class TestCacheUi(_CacheHome):
    def setUp(self):
        super().setUp()
        _need_widgets(self)

    def dialog(self, isil):
        from src.ui.plugin_settings_tab import _ExportPresetDialog

        offers = [pc.CacheOffer("dk", "DK-Cache (Libero/finc)", 3, 1260, True),
                  pc.CacheOffer("mappings", "GND-Zuordnungen", 5, 400, True),
                  pc.CacheOffer("raw:lobid", "Rohantworten lobid", 2, 3 * 1024 * 1024, False)]
        d = _ExportPresetDialog([("search_provider", "swb", "SWB", True)],
                                cache_offers=offers, isil=isil)
        _KEEP.append(d)
        return d

    def test_boxes_with_sizes_and_defaults(self):
        d = self.dialog("ZZ-A")
        texts = {part: box.text() for part, box in d.cache_boxes.items()}
        self.assertEqual(texts["dk"], "DK-Cache (Libero/finc): 3 Einträge, ca. 1,2 KB")
        self.assertEqual(texts["mappings"], "GND-Zuordnungen: 5 Einträge, ca. 400 B")
        self.assertTrue(texts["raw:lobid"].startswith("Rohantworten lobid: 2 Einträge, ca. 3,0 MB"))
        self.assertEqual(d.selected_cache_parts(), ["dk", "mappings"])
        d.cache_boxes["raw:lobid"].setChecked(True)
        self.assertEqual(d.selected_cache_parts(), ["dk", "mappings", "raw:lobid"])
        self.assertIn("nur intern verteilen", d.cache_note.text())
        self.assertNotIn("&", "".join(texts.values()) + d.cache_note.text())

    def test_locked_without_isil(self):
        d = self.dialog("")
        self.assertTrue(all(not b.isEnabled() and not b.isChecked() for b in d.cache_boxes.values()))
        self.assertEqual(d.selected_cache_parts(), [])
        self.assertIn("Ohne gesetzte ISIL", d.cache_note.text())

    def test_no_offers(self):
        from PyQt6.QtWidgets import QLabel
        from src.ui.plugin_settings_tab import _ExportPresetDialog

        d = _ExportPresetDialog([], cache_offers=[], isil="ZZ-A")
        _KEEP.append(d)
        self.assertEqual(d.cache_boxes, {})
        self.assertIn("Keine Cache-Einträge vorhanden.", [w.text() for w in d.findChildren(QLabel)])

    def test_unreadable_cache_is_named(self):
        from PyQt6.QtWidgets import QLabel
        from src.ui.plugin_settings_tab import _ExportPresetDialog

        d = _ExportPresetDialog([], cache_offers=[], isil="ZZ-A", cache_error="Datei gesperrt")
        _KEEP.append(d)
        self.assertIn("Cache nicht lesbar: Datei gesperrt", [w.text() for w in d.findChildren(QLabel)])

    def test_preview_shows_cache_lines(self):
        from src.ui.plugin_settings_tab import _PresetPlanDialog

        pkg = self.made(["dk"])
        good = _PresetPlanDialog(pa.build_plan(pa.load_preset(pkg), self.cm))
        _KEEP.append(good)
        self.assertIn("dk_cache (catalog_dk_cache): neu: 2", good.text.toPlainText())
        self.assertTrue(good.apply_button.isEnabled())
        self.assertIn("nimmt das nicht zurück", good.note.text())
        self.edit_preset(pkg, lambda d: d.pop("institution"))
        self.use_db("b", isil="ZZ-X")
        bad = _PresetPlanDialog(pa.build_plan(pa.load_preset(pkg), self.cm))
        _KEEP.append(bad)
        self.assertIn("übersprungen: ISIL passt nicht (Paket ZZ-A, Ziel ZZ-X)", bad.text.toPlainText())
        self.assertFalse(bad.apply_button.isEnabled())  # nothing else to change

    def test_export_through_the_tab_counts_in_the_worker(self):
        from PyQt6.QtWidgets import QDialog, QFileDialog

        from src.ui.plugin_settings_tab import PluginSettingsTab, _ExportPresetDialog
        from tests.test_institution_presets import TestUi

        self.seed()
        tab = PluginSettingsTab()
        _KEEP.append(tab)
        tab.load(self.cm.load_config())
        with patch.object(_ExportPresetDialog, "exec", return_value=QDialog.DialogCode.Accepted), \
                patch.object(QFileDialog, "getExistingDirectory", return_value=str(self.tmp / "gui")), \
                patch("src.ui.plugin_settings_tab.QMessageBox"):
            tab.start_preset_export()
            TestUi.drain(tab)
            TestUi.drain(tab)
        data = self.preset_data(self.tmp / "gui" / "alima_preset")
        self.assertEqual([t["id"] for t in data["cache"]["tables"]], ["dk_cache", "mappings"])
        self.assertIn("raw:lobid", tab._preset_dialog.cache_boxes)
