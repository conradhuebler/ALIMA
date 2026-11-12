"""Institution presets: find, plan, apply, revert, export - Claude Generated.

Pinned here (Nov 2026):

1. Finding: ``ALIMA_PRESET`` (file or folder) before ``<config_dir>`` before the
   project directory; the source is named in the log. Without the first two the
   project directory is used as before.
2. An old preset (``institution_name``, ``llm``, ``catalog``, ``database``) loads
   unchanged, except that ``api_key``/``token`` are dropped with one warning.
3. The plan: ``before``/``after`` per change, state → ``enabled``/``cache_only``,
   list order inside the category, an instance is set (never a sibling) or
   created, SECRET fields never, unknown settings warn, ``@file:`` resolves.
4. Files fail closed (checksum, schema of the geo database), install
   atomically and keep one ``.bak``.
5. Idempotent: a second application is an empty plan and writes nothing.
6. Revert from the ledger (values and files), keys changed since stay.
7. Export: no secret, no absolute path of the user, checksums match, and an
   export applied elsewhere gives the same setup.
8. First start, CLI (``--dry-run`` changes nothing), the UI group, bundles gone,
   round trip without diff.

All data is invented; HOME and the config directory are temporary; no network.
"""

from __future__ import annotations

import importlib.util
import io
import json
import logging
import os
import shutil
import sqlite3
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from src.core.database_manager import server_driver_loads as REAL_DRIVER_PROBE
from src.utils.config_manager import ConfigManager
from src.utils import preset_apply as pa
from tests.test_database_fallback_feedback import _without_server_drivers
from tests.test_geo_sachdb_lookup import build_synthetic_db

REPO = Path(__file__).resolve().parent.parent
_KEEP: list = []

VOCAB = """version: 1
scheme:
  id: test_vocab
  label: "Testsystematik"
  source: "erfunden"
entries:
  - notation: "55"
    label: "Geowissenschaften"
  - notation: "622"
    label: "Bergbau"
"""


def _app():
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication(["alima-tests"])
    _KEEP.append(app)
    return app


def _need_widgets(case: unittest.TestCase) -> None:
    from PyQt6.QtWidgets import QApplication

    if not isinstance(_app(), QApplication):
        case.skipTest("a bare QCoreApplication exists; widgets cannot be built")


def reset_knowledge_manager() -> None:
    """Close the connections of the knowledge manager singleton, then reset it
    (``reset`` alone does not close them) - Claude Generated"""
    from src.core.unified_knowledge_manager import UnifiedKnowledgeManager

    ukm = UnifiedKnowledgeManager._instance
    if ukm is not None:
        for db in (getattr(ukm, "db_manager", None),
                   getattr(getattr(ukm, "local_gnd", None), "db_manager", None)):
            if db is not None:
                try:
                    db.close_connection()
                except Exception:  # an already closed connection
                    pass
    UnifiedKnowledgeManager.reset()


def make_package(root: Path, mutate=None) -> Path:
    """A package folder: invented geo database, vocabulary, preset."""
    (root / "files").mkdir(parents=True, exist_ok=True)
    geo = root / "files" / "geo.db"
    if not geo.exists():
        build_synthetic_db(geo)
    vocab = root / "files" / "vocab.yaml"
    vocab.write_text(VOCAB, encoding="utf-8")
    data = {
        "format": 2,
        "preset": {"id": "test-ub", "version": "1", "label": "Test-UB"},
        "institution": {"isil": "ZZ-TEST", "eln": "4711"},
        "workflow": "alima_v51_105",
        "plugins": [
            {"instance": "geo_sachdb", "state": "on", "settings": {"db_path": "@file:geo"}},
            {"instance": "local_scheme", "settings": {"vocabulary": "@file:vocab"}},
            {"instance": "swb", "state": "off"},
            {"instance": "finc", "state": "cache_only"},
            {"instance": "k10_test", "type": "sru", "label": "K10 Test", "state": "on",
             "settings": {"preset": "k10plus"}},
        ],
        "files": [
            {"id": "geo", "kind": "geo_sachdb_db", "source": "files/geo.db",
             "target": "geo_sachdb/geo.db", "sha256": pa.sha256_file(geo)},
            {"id": "vocab", "kind": "vocabulary", "source": "files/vocab.yaml",
             "target": "local_scheme/vocab.yaml", "sha256": pa.sha256_file(vocab)},
        ],
        "secrets": [{"plugin": "catalog", "key": "token", "hint": "Token der UB"}],
    }
    if mutate is not None:
        mutate(data)
    (root / pa.PRESET_FILENAME).write_text(json.dumps(data, indent=2), encoding="utf-8")
    return root


class _Home(unittest.TestCase):
    """Temporary HOME, config directory and project directory."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)
        (self.tmp / "home").mkdir()
        env = patch.dict(os.environ, {"HOME": str(self.tmp / "home")})
        env.start()
        self.addCleanup(env.stop)
        os.environ.pop(pa.ENV_VAR, None)
        self.project = self.tmp / "project"
        self.project.mkdir()
        root = patch("src.utils.path_utils.get_project_root", return_value=self.project)
        root.start()
        self.addCleanup(root.stop)
        # Workflows only from the repository: the default search paths were
        # computed with the real HOME at import time.
        workflows = patch("src.core.agents.workflow_loader.DEFAULT_SEARCH_PATHS", [REPO / "workflows"])
        workflows.start()
        self.addCleanup(workflows.stop)
        ConfigManager.reset()
        self.addCleanup(ConfigManager.reset)
        reset_knowledge_manager()
        self.addCleanup(reset_knowledge_manager)
        self.cm = ConfigManager()
        self.config_dir = Path(self.cm.config_file).parent
        self.assertTrue(str(self.config_dir).startswith(str(self.tmp)))
        # Two saves: the very first one of a new file also writes keys the
        # later ones drop (unified_config.task_preferences), unrelated here.
        self.cm.save_config(self.cm.load_config())
        self.cm.save_config(self.cm.load_config())
        self.data_dir = self.config_dir / "data"

    # helpers
    def config(self):
        return self.cm.load_config(force_reload=True)

    def inst(self, iid, config=None):
        cfg = config or self.config()
        return next((p for p in cfg.plugins if p.instance_id == iid), None)

    def ids(self, category, config=None):
        cfg = config or self.config()
        return [p.instance_id for p in cfg.plugins if p.category == category]

    def plan(self, path):
        return pa.build_plan(pa.load_preset(path), self.cm)

    def disk(self) -> bytes:
        return Path(self.cm.config_file).read_bytes()

    def package(self, name="pkg", mutate=None) -> Path:
        return make_package(self.tmp / name, mutate)


# ===========================================================================
# 1. Finding the preset
# ===========================================================================
class TestFindPreset(_Home):
    def _write(self, folder: Path, name: str) -> Path:
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / pa.PRESET_FILENAME
        path.write_text(json.dumps({"institution_name": name}), encoding="utf-8")
        return path

    def test_nothing_found(self):
        self.assertEqual(pa.find_preset(), (None, ""))

    def test_project_directory_as_before(self):
        path = self._write(self.project, "Projekt-UB")
        self.assertEqual(pa.find_preset(), (path, pa.SOURCE_PROJECT))
        from src.utils.preset_loader import PresetLoader

        presets = PresetLoader.load()
        self.assertEqual((presets.institution_name, presets.source), ("Projekt-UB", pa.SOURCE_PROJECT))

    def test_config_directory_before_project(self):
        self._write(self.project, "Projekt-UB")
        path = self._write(self.config_dir, "Konfig-UB")
        self.assertEqual(pa.find_preset(), (path, pa.SOURCE_CONFIG_DIR))

    def test_environment_variable_first_file_or_folder(self):
        self._write(self.project, "Projekt-UB")
        self._write(self.config_dir, "Konfig-UB")
        env = self._write(self.tmp / "envpkg", "Env-UB")
        for value in (str(env), str(env.parent)):
            with patch.dict(os.environ, {pa.ENV_VAR: value}):
                self.assertEqual(pa.find_preset(), (env, pa.SOURCE_ENV))
                from src.utils.preset_loader import PresetLoader

                self.assertEqual(PresetLoader.load().institution_name, "Env-UB")

    def test_environment_variable_pointing_nowhere_falls_through(self):
        path = self._write(self.config_dir, "Konfig-UB")
        with patch.dict(os.environ, {pa.ENV_VAR: str(self.tmp / "missing")}):
            with self.assertLogs("src.utils.preset_apply", level=logging.WARNING) as logs:
                self.assertEqual(pa.find_preset(), (path, pa.SOURCE_CONFIG_DIR))
        self.assertIn(pa.ENV_VAR, "\n".join(logs.output))

    def test_log_names_the_preset_used(self):
        path = self._write(self.config_dir, "Konfig-UB")
        with self.assertLogs("src.utils.preset_apply", level=logging.INFO) as logs:
            pa.find_preset()
        self.assertIn(str(path), "\n".join(logs.output))
        self.assertIn("Konfigurationsverzeichnis", "\n".join(logs.output))

    def test_relative_sources_are_relative_to_the_preset_folder(self):
        pkg = self.package()
        with patch.dict(os.environ, {pa.ENV_VAR: str(pkg)}):
            cwd = os.getcwd()
            os.chdir(self.tmp)
            try:
                plan = self.plan(pa.find_preset()[0])
            finally:
                os.chdir(cwd)
        self.assertEqual(plan.errors, [])
        self.assertEqual({Path(f.source) for f in plan.files},
                         {(pkg / "files" / "geo.db").resolve(), (pkg / "files" / "vocab.yaml").resolve()})


# ===========================================================================
# 2. Old presets, dropped keys
# ===========================================================================
OLD_PRESET = {
    "institution_name": "Alte UB",
    "llm": {"provider_type": "openai_compatible", "provider_name": "Uni-LLM",
            "base_url": "https://llm.alte-ub.example/v1", "api_key": "SECRET-KEY",
            "task_models": {"pipeline": "m-pipe", "vision": "m-vis"}},
    "catalog": {"soap_search_url": "https://lib.alte-ub.example/s",
                "soap_details_url": "https://lib.alte-ub.example/d", "token": "SECRET-TOKEN",
                "web_search_url": "https://kat.alte-ub.example/Search/Results"},
    "database": {"type": "mariadb", "host": "db.alte-ub.example", "database": "alima",
                 "username": "alima"},
}


class TestOldPreset(_Home):
    def setUp(self):
        super().setUp()
        (self.project / pa.PRESET_FILENAME).write_text(json.dumps(OLD_PRESET), encoding="utf-8")

    def test_loads_unchanged_without_key_and_token(self):
        from src.utils.preset_loader import PresetLoader

        p = PresetLoader.load()
        self.assertEqual(p.institution_name, "Alte UB")
        self.assertEqual((p.llm_provider_type, p.llm_provider_name, p.llm_base_url),
                         ("openai_compatible", "Uni-LLM", "https://llm.alte-ub.example/v1"))
        self.assertEqual((p.pipeline_model(), p.vision_model()), ("m-pipe", "m-vis"))
        self.assertEqual((p.catalog_soap_search_url, p.catalog_soap_details_url,
                          p.catalog_web_search_url),
                         ("https://lib.alte-ub.example/s", "https://lib.alte-ub.example/d",
                          "https://kat.alte-ub.example/Search/Results"))
        self.assertEqual((p.db_type, p.db_host, p.db_database, p.db_username),
                         ("mariadb", "db.alte-ub.example", "alima", "alima"))
        self.assertFalse(hasattr(p, "llm_api_key"))
        self.assertFalse(hasattr(p, "catalog_token"))
        self.assertNotIn("SECRET", repr(p))

    def test_dropped_keys_are_warned_once(self):
        from src.utils.preset_loader import PresetLoader

        with self.assertLogs("src.utils.preset_apply", level=logging.WARNING) as logs:
            PresetLoader.load()
            PresetLoader.load()
            logging.getLogger("src.utils.preset_apply").warning("Ende")
        hits = [r for r in logs.output if "werden nicht verwendet" in r]
        self.assertEqual(len(hits), 1, logs.output)
        self.assertIn("llm.api_key", hits[0])
        self.assertIn("catalog.token", hits[0])

    def test_old_preset_changes_nothing_in_the_plan(self):
        plan = self.plan(self.project)
        self.assertEqual((plan.changes, plan.files, plan.errors), ([], [], []))
        self.assertTrue(plan.is_empty)
        self.assertTrue(any("nicht verwendet" in w for w in plan.warnings))

    def test_wizard_pages_get_no_key_and_no_token(self):
        _need_widgets(self)
        from src.ui.first_start_wizard import CatalogSetupPage, LLMSetupPage
        from src.utils.preset_loader import PresetLoader

        presets = PresetLoader.load()
        llm = LLMSetupPage(presets=presets)
        catalog = CatalogSetupPage(presets=presets)
        _KEEP.extend([llm, catalog])
        self.assertEqual(llm.api_key_input.text(), "")
        self.assertEqual(catalog.catalog_token.text(), "")
        self.assertEqual(catalog.soap_search_url.text(), "https://lib.alte-ub.example/s")


# ===========================================================================
# 3. The plan
# ===========================================================================
class TestPlan(_Home):
    def test_preview_has_before_and_after_and_writes_nothing(self):
        before = self.disk()
        plan = self.plan(self.package())
        self.assertEqual(plan.errors, [])
        isil = next(c for c in plan.changes if c.path == ["system_config", "institution_isil"])
        self.assertEqual((isil.before, isil.after, isil.before_absent), ("", "ZZ-TEST", False))
        text = "\n".join(plan.lines())
        self.assertIn("Einrichtung: ISIL: (leer) → ZZ-TEST", text)
        self.assertIn("Standard-Workflow: alima_v51 → alima_v51_105", text)
        self.assertIn("neu: geo", text)
        self.assertEqual(self.disk(), before)
        self.assertFalse(self.data_dir.exists())

    def test_state_maps_to_enabled_and_cache_only(self):
        pa.apply_plan(self.plan(self.package()), self.cm)
        self.assertFalse(self.inst("swb").enabled)
        finc = self.inst("finc")
        self.assertTrue(finc.enabled)
        self.assertIs(finc.settings.get("cache_only"), True)
        # "on" afterwards writes cache_only False where the key exists
        pa.apply_plan(self.plan(self.package("pkg2", lambda d: d["plugins"].__setitem__(
            3, {"instance": "finc", "state": "on"}))), self.cm)
        self.assertIs(self.inst("finc").settings.get("cache_only"), False)

    def test_cache_only_for_a_type_without_it_warns(self):
        def mutate(d):
            d["plugins"] = [{"instance": "local_scheme", "state": "cache_only"}]
        plan = self.plan(self.package(mutate=mutate))
        self.assertTrue(any("Nur Cache" in w for w in plan.warnings), plan.warnings)
        self.assertFalse([c for c in plan.changes if c.path[-1] == "enabled"])

    def test_order_inside_the_category(self):
        search_before = self.ids("search_provider")
        input_before = self.ids("input_source")
        pa.apply_plan(self.plan(self.package()), self.cm)
        search = self.ids("search_provider")
        self.assertEqual(search[:3], ["swb", "finc", "k10_test"])
        self.assertEqual(search[3:], [i for i in search_before if i not in ("swb", "finc")])
        self.assertEqual(self.ids("lookup")[:2], ["geo_sachdb", "local_scheme"])
        self.assertEqual(self.ids("input_source"), input_before)

    def test_existing_instance_is_set_not_duplicated(self):
        pa.apply_plan(self.plan(self.package()), self.cm)
        cfg = self.config()
        geo = [p for p in cfg.plugins if p.provider_id == "geo_sachdb"]
        self.assertEqual([p.instance_id for p in geo], ["geo_sachdb"])
        self.assertEqual(geo[0].settings["db_path"], str(self.data_dir / "geo_sachdb" / "geo.db"))
        self.assertEqual(len([p for p in cfg.plugins if p.instance_id == "swb"]), 1)

    def test_unknown_instance_with_a_type_is_created(self):
        plan = self.plan(self.package())
        create = next(c for c in plan.changes if c.path == ["instance", "search_provider", "k10_test"])
        self.assertTrue(create.before_absent)
        pa.apply_plan(plan, self.cm)
        k10 = self.inst("k10_test")
        self.assertEqual((k10.category, k10.provider_id, k10.label, k10.enabled),
                         ("search_provider", "sru", "K10 Test", True))
        self.assertEqual(k10.settings, {"preset": "k10plus"})

    def test_unknown_type_and_type_mismatch_are_skipped(self):
        def mutate(d):
            d["plugins"] = [{"instance": "gibt_es_nicht", "state": "on"},
                            {"instance": "swb", "type": "sru", "state": "off"}]
        plan = self.plan(self.package(mutate=mutate))
        self.assertEqual([c for c in plan.changes if c.path[0] == "instance"], [])
        self.assertTrue(any("kein registrierter Typ" in w for w in plan.warnings))
        self.assertTrue(any("hat den Typ 'swb'" in w for w in plan.warnings))

    def test_secret_fields_are_never_set(self):
        pkg = self.package(mutate=lambda d: d["plugins"].append(
            {"instance": "catalog", "settings": {"token": "TOK", "catalog_search_url": "https://c.example"}}))
        # the file loader drops the key ...
        self.assertNotIn("token", json.dumps(pa.load_preset(pkg).data["plugins"]))
        # ... and the schema refuses it even when it reaches the planner
        data = json.loads((pkg / pa.PRESET_FILENAME).read_text())
        plan = pa.build_plan(pa.Preset(pkg / pa.PRESET_FILENAME, data), self.cm)
        self.assertFalse([c for c in plan.changes if c.path[-1] == "token"])
        self.assertTrue(any("geheim" in w for w in plan.warnings), plan.warnings)
        pa.apply_plan(plan, self.cm)
        self.assertNotIn("token", self.inst("catalog").settings)
        self.assertEqual(self.inst("catalog").settings["catalog_search_url"], "https://c.example")

    def test_unknown_setting_and_bad_choice_warn_and_are_not_written(self):
        pkg = self.package(mutate=lambda d: d["plugins"].append(
            {"instance": "sru", "settings": {"gibt_es_nicht": 1, "preset": "erfunden"}}))
        plan = self.plan(pkg)
        self.assertTrue(any("unbekannte Einstellung 'gibt_es_nicht'" in w for w in plan.warnings))
        self.assertTrue(any("'preset' = 'erfunden'" in w for w in plan.warnings))
        self.assertFalse([c for c in plan.changes if c.path[:3] == ["instance", "search_provider", "sru"]
                          and c.path[-1] in ("gibt_es_nicht", "preset")])

    def test_file_reference_resolves_to_the_target(self):
        plan = self.plan(self.package())
        vocab = next(c for c in plan.changes if c.path[-1] == "vocabulary")
        self.assertEqual(vocab.after, str(self.data_dir / "local_scheme" / "vocab.yaml"))

    def test_unknown_file_reference_is_an_error(self):
        pkg = self.package(mutate=lambda d: d["plugins"][0]["settings"].__setitem__(
            "db_path", "@file:fehlt"))
        plan = self.plan(pkg)
        self.assertTrue(any("@file:fehlt" in e for e in plan.errors))
        with self.assertRaises(pa.PresetError):
            pa.apply_plan(plan, self.cm)

    def test_missing_workflow_is_a_warning_only(self):
        plan = self.plan(self.package(mutate=lambda d: d.__setitem__("workflow", "gibt_es_nicht")))
        self.assertEqual(plan.errors, [])
        self.assertTrue(any("'gibt_es_nicht' nicht gefunden" in w for w in plan.warnings))
        self.assertFalse([c for c in plan.changes if c.path[-1] == "default_workflow"])

    def test_secrets_are_listed_not_written(self):
        plan = self.plan(self.package())
        self.assertEqual(plan.secrets, [{"plugin": "catalog", "key": "token", "hint": "Token der UB",
                                         "env_var": "ALIMA_PLUGIN_CATALOG_TOKEN",
                                         "satisfied": False}])
        self.assertIn("[fehlt] catalog.token", "\n".join(plan.lines()))
        pa.apply_plan(plan, self.cm)
        self.assertNotIn("token", self.inst("catalog").settings)


# ===========================================================================
# 4. Files
# ===========================================================================
class TestFiles(_Home):
    def test_checksum_mismatch_fails_closed(self):
        before = self.disk()
        pkg = self.package(mutate=lambda d: d["files"][0].__setitem__("sha256", "0" * 64))
        plan = self.plan(pkg)
        self.assertTrue(any("Prüfsumme stimmt nicht" in e for e in plan.errors), plan.errors)
        self.assertNotIn("geo", [f.id for f in plan.files])
        with self.assertRaises(pa.PresetError):
            pa.apply_plan(plan, self.cm)
        self.assertFalse((self.data_dir / "geo_sachdb" / "geo.db").exists())
        self.assertFalse((self.data_dir / "local_scheme" / "vocab.yaml").exists())
        self.assertEqual(self.disk(), before)

    def test_missing_or_malformed_checksum_is_an_error(self):
        for value in ("", "abc"):
            plan = self.plan(self.package(f"p{len(value)}", lambda d, v=value: d["files"][0].__setitem__("sha256", v)))
            self.assertTrue(any("'sha256' fehlt" in e for e in plan.errors), plan.errors)

    def test_geo_database_schema_is_checked(self):
        def mutate(d):
            other = self.tmp / "pkg" / "files" / "other.db"
            conn = sqlite3.connect(other)
            conn.execute("CREATE TABLE swb (x TEXT)")
            conn.commit()
            conn.close()
            d["files"][0].update(source="files/other.db", sha256=pa.sha256_file(other))
        plan = self.plan(self.package(mutate=mutate))
        self.assertTrue(any("geo_sachdb_db" in e and "fehlen" in e for e in plan.errors), plan.errors)

    def test_vocabulary_shape_is_checked(self):
        def mutate(d):
            bad = self.tmp / "pkg" / "files" / "bad.yaml"
            bad.write_text("nur: text\n", encoding="utf-8")
            d["files"][1].update(source="files/bad.yaml", sha256=pa.sha256_file(bad))
        plan = self.plan(self.package(mutate=mutate))
        self.assertTrue(any("entries" in e for e in plan.errors), plan.errors)

    def test_target_must_stay_in_the_data_folder(self):
        for target in ("../escape.db", "/tmp/abs.db", "~/x.db"):
            plan = self.plan(self.package(f"t{abs(hash(target))}",
                                          lambda d, t=target: d["files"][0].__setitem__("target", t)))
            self.assertTrue(any("Datenordner" in e for e in plan.errors), (target, plan.errors))

    def test_install_is_atomic_and_keeps_one_bak(self):
        target = self.data_dir / "local_scheme" / "vocab.yaml"
        target.parent.mkdir(parents=True)
        target.write_text("OLD", encoding="utf-8")
        (target.parent / "vocab.yaml.bak").write_text("OLDER", encoding="utf-8")
        plan = self.plan(self.package())
        self.assertEqual({f.id: f.status for f in plan.files}, {"geo": pa.FILE_NEW, "vocab": pa.FILE_REPLACED})
        pa.apply_plan(plan, self.cm)
        self.assertEqual(target.read_text(encoding="utf-8"), VOCAB)
        self.assertEqual((target.parent / "vocab.yaml.bak").read_text(encoding="utf-8"), "OLD")
        leftovers = [p.name for p in self.data_dir.rglob("*") if p.name.endswith(".tmp")]
        self.assertEqual(leftovers, [])
        self.assertEqual(pa.sha256_file(self.data_dir / "geo_sachdb" / "geo.db"),
                         pa.sha256_file(self.tmp / "pkg" / "files" / "geo.db"))

    def test_source_changed_after_the_preview_installs_nothing(self):
        target = self.data_dir / "local_scheme" / "vocab.yaml"
        target.parent.mkdir(parents=True)
        target.write_text("OLD", encoding="utf-8")
        before = self.disk()
        pkg = self.package()
        plan = self.plan(pkg)
        (pkg / "files" / "vocab.yaml").write_text(VOCAB + "# geändert\n", encoding="utf-8")
        with self.assertRaises(pa.PresetError):
            pa.apply_plan(plan, self.cm)
        self.assertEqual(target.read_text(encoding="utf-8"), "OLD")
        self.assertFalse((self.data_dir / "geo_sachdb" / "geo.db").exists())  # undone
        self.assertEqual([p.name for p in self.data_dir.rglob("*.tmp")], [])
        self.assertEqual(self.disk(), before)


# ===========================================================================
# 5. Idempotence, stale plans
# ===========================================================================
class TestIdempotence(_Home):
    def test_second_application_is_empty_and_writes_nothing(self):
        pkg = self.package()
        self.assertIsNotNone(pa.apply_plan(self.plan(pkg), self.cm))
        after_first = self.disk()
        again = self.plan(pkg)
        self.assertTrue(again.is_empty, again.lines())
        self.assertEqual({f.status for f in again.files}, {pa.FILE_UNCHANGED})
        self.assertIsNone(pa.apply_plan(again, self.cm))
        self.assertEqual(self.disk(), after_first)
        self.assertEqual(len(self.config().applied_presets), 1)

    def test_a_plan_on_a_changed_config_is_refused(self):
        plan = self.plan(self.package())
        cfg = self.config()
        cfg.system_config.institution_isil = "ZZ-ANDERS"
        self.cm.save_config(cfg)
        with self.assertRaises(pa.PresetError) as ctx:
            pa.apply_plan(plan, self.cm)
        self.assertIn("seit der Vorschau", str(ctx.exception))
        self.assertEqual(self.config().applied_presets, [])


# ===========================================================================
# 6. Revert
# ===========================================================================
class TestRevert(_Home):
    def _snapshot(self):
        cfg = self.config()
        return {
            "system": (cfg.system_config.institution_isil, cfg.system_config.institution_eln,
                       cfg.system_config.default_workflow),
            "plugins": [(p.category, p.instance_id, p.provider_id, p.label, p.enabled,
                         dict(p.settings)) for p in cfg.plugins],
        }

    def test_values_and_files_come_back(self):
        target = self.data_dir / "local_scheme" / "vocab.yaml"
        target.parent.mkdir(parents=True)
        target.write_text("OLD-VOCAB", encoding="utf-8")
        before = self._snapshot()
        pa.apply_plan(self.plan(self.package()), self.cm)
        self.assertNotEqual(self._snapshot(), before)
        result = pa.revert_last(self.cm)
        self.assertEqual(self._snapshot(), before)
        self.assertEqual(self.config().applied_presets, [])
        self.assertFalse((self.data_dir / "geo_sachdb" / "geo.db").exists())
        self.assertEqual(target.read_text(encoding="utf-8"), "OLD-VOCAB")
        self.assertEqual(result.kept, [])
        self.assertTrue(any("k10_test" in line and "entfernt" in line for line in result.restored))

    def test_a_value_changed_since_stays(self):
        pa.apply_plan(self.plan(self.package()), self.cm)
        cfg = self.config()
        cfg.system_config.institution_isil = "ZZ-USER"
        self.cm.save_config(cfg)
        result = pa.revert_last(self.cm)
        cfg = self.config()
        self.assertEqual(cfg.system_config.institution_isil, "ZZ-USER")
        self.assertEqual(cfg.system_config.institution_eln, "")
        self.assertTrue(self.inst("swb", cfg).enabled)
        self.assertTrue(any("ISIL" in line for line in result.kept), result.kept)

    def test_a_file_changed_since_stays(self):
        pa.apply_plan(self.plan(self.package()), self.cm)
        geo = self.data_dir / "geo_sachdb" / "geo.db"
        geo.write_bytes(b"eigene Datei")
        result = pa.revert_last(self.cm)
        self.assertEqual(geo.read_bytes(), b"eigene Datei")
        self.assertTrue(any("seit dem Anwenden geändert" in line for line in result.files))

    def test_revert_without_ledger_raises(self):
        with self.assertRaises(pa.PresetError):
            pa.revert_last(self.cm)


# ===========================================================================
# 7. Export
# ===========================================================================
class TestExport(_Home):
    def setUp(self):
        super().setUp()
        from src.utils.config_models import UnifiedProvider

        self.geo = self.tmp / "eigene_daten" / "geo.db"
        self.geo.parent.mkdir()
        build_synthetic_db(self.geo)
        cfg = self.config()
        cfg.system_config.institution_isil = "ZZ-SRC"
        cfg.system_config.institution_eln = "1234"
        cfg.system_config.default_workflow = "alima_v51_105"
        cfg.unified_config.providers = [UnifiedProvider(
            name="srv", provider_type="openai_compatible", base_url="https://llm.example/v1",
            api_key="SEKRET-KEY", enabled=True, preferred_model="m1")]
        cfg.unified_config.preferred_provider = "srv"
        for p in cfg.plugins:
            if p.instance_id == "geo_sachdb":
                p.settings["db_path"] = str(self.geo)
            elif p.instance_id == "catalog":
                p.settings.update(token="SEKRET-TOKEN", catalog_search_url="https://c.example/s")
            elif p.instance_id == "swb":
                p.enabled = False
            elif p.instance_id == "finc":
                p.settings["cache_only"] = True
            elif p.instance_id == "k10plus":
                p.settings["cache_dir"] = str(self.tmp / "home" / "k10cache")
        kvk = next(p for p in cfg.plugins if p.instance_id == "kvk")
        cfg.plugins.remove(kvk)
        cfg.plugins.insert(0, kvk)  # kvk first in the search list
        self.cm.save_config(cfg)

    def export(self, **kw):
        from src.utils.preset_export import export_preset

        return export_preset(self.cm, self.tmp / "out" / "pkg", preset_id="x", **kw)

    def test_no_secret_no_user_path_and_checksums_match(self):
        result = self.export()
        text = result.preset_file.read_text(encoding="utf-8")
        for f in result.path.rglob("*"):
            if f.is_file():
                self.assertNotIn(b"SEKRET", f.read_bytes(), f)
        self.assertNotIn(str(self.tmp), text)
        data = json.loads(text)
        self.assertNotIn("api_key", data.get("llm", {}))
        self.assertEqual(data["llm"]["base_url"], "https://llm.example/v1")
        self.assertIn({"plugin": "catalog", "key": "token"},
                      [{k: s[k] for k in ("plugin", "key")} for s in data["secrets"]])
        for entry in data["files"]:
            self.assertFalse(Path(entry["source"]).is_absolute())
            self.assertEqual(entry["sha256"], pa.sha256_file(result.path / entry["source"]))
        geo = next(p for p in data["plugins"] if p["instance"] == "geo_sachdb")
        self.assertEqual(geo["settings"]["db_path"], "@file:geo_sachdb_db_path")
        self.assertTrue(any("k10plus" in w and "lokalen Pfad" in w for w in result.warnings))
        self.assertEqual({e["kind"] for e in data["files"]}, {"geo_sachdb_db", "vocabulary"})

    def test_roundtrip_gives_the_same_setup(self):
        from src.core.plugins import get_category
        from src.core.search.cache_only import cache_only_offer, instance_state
        from src.utils.preset_export import FILE_SETTINGS

        result = self.export()
        source = self.config()
        # a fresh installation elsewhere
        ConfigManager.reset()
        cm_b = ConfigManager()
        cm_b.config_file = self.tmp / "b" / ".config" / "alima" / "config.json"
        cm_b.config_file.parent.mkdir(parents=True)
        cm_b.save_config(cm_b.load_config())
        plan = pa.build_plan(pa.load_preset(result.path), cm_b)
        self.assertEqual(plan.errors, [])
        pa.apply_plan(plan, cm_b)
        target = cm_b.load_config(force_reload=True)
        sys_a, sys_b = source.system_config, target.system_config
        self.assertEqual((sys_b.institution_isil, sys_b.institution_eln, sys_b.default_workflow),
                         (sys_a.institution_isil, sys_a.institution_eln, sys_a.default_workflow))
        for category in ("search_provider", "lookup", "input_source"):
            self.assertEqual(self.ids(category, target), self.ids(category, source), category)
        for a in source.plugins:
            b = self.inst(a.instance_id, target)
            meta = get_category(a.category).type_meta(a.provider_id)
            offer = cache_only_offer(meta)
            self.assertEqual(instance_state(b, offer), instance_state(a, offer), a.instance_id)
            for key, value in a.settings.items():
                if key in ("token", "cache_dir"):
                    self.assertNotIn(key, b.settings)
                elif (a.provider_id, key) in FILE_SETTINGS:
                    self.assertEqual(pa.sha256_file(b.settings[key]), pa.sha256_file(value))
                    self.assertTrue(str(b.settings[key]).startswith(str(pa.data_dir_of(cm_b))))
                elif key != "cache_only":
                    self.assertEqual(b.settings.get(key), value, (a.instance_id, key))
        # the default vocabulary in use is shipped as well
        vocab = self.inst("local_scheme", target).settings["vocabulary"]
        self.assertEqual(pa.sha256_file(vocab), pa.sha256_file(REPO / "vocabularies" / "freihand_fg.yaml"))

    def test_selection_keeps_list_order(self):
        result = self.export(instance_ids=["swb", "kvk", "geo_sachdb"])
        data = json.loads(result.preset_file.read_text(encoding="utf-8"))
        self.assertEqual([p["instance"] for p in data["plugins"]], ["kvk", "swb", "geo_sachdb"])
        self.assertEqual([f["id"] for f in data["files"]], ["geo_sachdb_db_path"])

    def test_unknown_instance_and_non_empty_folder_raise(self):
        from src.utils.preset_export import export_preset

        with self.assertRaises(pa.PresetError):
            self.export(instance_ids=["gibt_es_nicht"])
        full = self.tmp / "voll"
        full.mkdir()
        (full / "x").write_text("x")
        with self.assertRaises(pa.PresetError):
            export_preset(self.cm, full)
        self.assertFalse((self.tmp / "out" / "pkg").exists())


# ===========================================================================
# 8. First start, CLI, UI, bundles gone, round trip
# ===========================================================================
class TestFirstStart(_Home):
    def test_apply_after_setup_never_raises(self):
        broken = self.package("broken", lambda d: d["files"][0].__setitem__("sha256", "1" * 64))
        lines = pa.apply_after_setup(broken, self.cm)
        self.assertIn("nicht angewendet", lines[0])
        garbage = self.tmp / "garbage"
        garbage.mkdir()
        (garbage / pa.PRESET_FILENAME).write_text("{kein json", encoding="utf-8")
        self.assertIn("nicht angewendet", pa.apply_after_setup(garbage, self.cm)[0])
        self.assertEqual(self.config().system_config.institution_isil, "")
        lines = pa.apply_after_setup(self.package(), self.cm)
        self.assertIn("angewendet", lines[0])
        self.assertEqual(self.config().system_config.institution_isil, "ZZ-TEST")

    def _wizard(self, pkg):
        _need_widgets(self)
        from src.ui.first_start_wizard import FirstStartWizard

        with patch.dict(os.environ, {pa.ENV_VAR: str(pkg)}):
            wizard = FirstStartWizard()
        _KEEP.append(wizard)
        self.assertEqual(wizard._presets.source, pa.SOURCE_ENV)
        return wizard

    def test_gui_wizard_applies_the_preset_without_a_dialog(self):
        wizard = self._wizard(self.package())
        wizard.summary_page.initializePage()
        summary = wizard.summary_page.summary_text.toPlainText()
        self.assertIn("Preset Test-UB", summary)
        self.assertIn("Einrichtung: ISIL ZZ-TEST, ELN 4711", summary)
        with patch("src.ui.first_start_wizard.QMessageBox") as box:
            wizard.accept()
        box.critical.assert_not_called()
        cfg = self.config()
        self.assertTrue(cfg.system_config.first_run_completed)
        self.assertEqual(cfg.system_config.institution_isil, "ZZ-TEST")
        self.assertEqual(self.inst("k10_test", cfg).provider_id, "sru")
        self.assertIn("angewendet", wizard.preset_result_lines[0])

    def test_a_broken_preset_does_not_stop_the_wizard(self):
        wizard = self._wizard(self.package(mutate=lambda d: d["files"][0].__setitem__("sha256", "2" * 64)))
        with patch("src.ui.first_start_wizard.QMessageBox") as box:
            wizard.accept()
        box.critical.assert_not_called()
        cfg = self.config()
        self.assertTrue(cfg.system_config.first_run_completed)
        self.assertEqual(cfg.system_config.institution_isil, "")
        self.assertIn("nicht angewendet", wizard.preset_result_lines[0])


class _ServerDbHome(_Home):
    """A home whose saved configuration has a server database with a password.

    The check before the export (driver, one connection) is a test double that
    succeeds, unless a test sets ``self.driver``/``self.connection``; no
    connection to ``db.example`` is ever attempted. - Claude Generated"""

    def setUp(self):
        super().setUp()
        cfg = self.config()
        db = cfg.database_config
        db.db_type = "mariadb"
        db.host = "db.example"
        db.database = "alima"
        db.username = "alima_user"
        db.password = "DB-GEHEIM"
        self.cm.save_config(cfg)
        self.driver = lambda kind: (True, "")
        self.connection = (True, "✅ MARIADB connection successful")
        self.probed = []
        self.driver_calls = []

        def probe(kind):
            self.driver_calls.append(kind)
            return self.driver(kind)

        driver = patch("src.core.database_manager.server_driver_loads", side_effect=probe)
        driver.start()
        self.addCleanup(driver.stop)

        def connect(_cm, database_config=None):
            self.probed.append(database_config)
            return self.connection

        conn = patch.object(ConfigManager, "test_database_connection", autospec=True,
                            side_effect=connect)
        conn.start()
        self.addCleanup(conn.stop)


class TestExportDatabasePassword(_ServerDbHome):
    """The password of a server database goes into the package only on request,
    in plain text, with an owner-only file. - Claude Generated"""

    def export(self, **kw):
        from src.utils.preset_export import export_preset

        return export_preset(self.cm, self.tmp / "out" / "pkg", preset_id="x", **kw)

    def test_not_exported_by_default(self):
        result = self.export()
        for f in result.path.rglob("*"):
            if f.is_file():
                self.assertNotIn(b"DB-GEHEIM", f.read_bytes(), f)
        data = json.loads(result.preset_file.read_text(encoding="utf-8"))
        self.assertNotIn("password", data["database"])
        self.assertTrue(any("Passwort wird nicht exportiert" in w for w in result.warnings))

    def test_written_on_request_with_owner_only_file(self):
        result = self.export(include_database_password=True)
        data = json.loads(result.preset_file.read_text(encoding="utf-8"))
        self.assertEqual(data["database"]["password"], "DB-GEHEIM")
        self.assertEqual(result.preset_file.stat().st_mode & 0o077, 0)
        self.assertTrue(any("Klartext" in w for w in result.warnings))

    def test_a_package_without_the_option_is_not_restricted(self):
        result = self.export()
        self.assertNotEqual(result.preset_file.stat().st_mode & 0o077, 0)

    def test_the_wizard_prefill_reads_it_from_the_package(self):
        from src.utils.preset_loader import PresetLoader

        result = self.export(include_database_password=True)
        with patch.dict(os.environ, {pa.ENV_VAR: str(result.path)}):
            presets = PresetLoader.load()
        self.assertEqual((presets.db_type, presets.db_host, presets.db_password),
                         ("mariadb", "db.example", "DB-GEHEIM"))

    def test_apply_does_not_touch_the_database_connection(self):
        result = self.export(include_database_password=True)
        before = self.config().database_config
        plan = pa.build_plan(pa.load_preset(result.preset_file), self.cm)
        pa.apply_plan(plan, self.cm)
        after = self.config().database_config
        self.assertEqual((after.db_type, after.host, after.password),
                         (before.db_type, before.host, before.password))


class TestCliDatabasePassword(_ServerDbHome):
    """The same through ``alima preset export``."""

    def run_export(self, *extra):
        from src.cli.commands import preset_cmd
        from src.cli.main import create_argument_parser

        args = create_argument_parser().parse_args(
            ["preset", "export", str(self.tmp / "cli_pkg"), *extra])
        with redirect_stdout(io.StringIO()):
            rc = preset_cmd.handle_preset(args, logging.getLogger("test"))
        return rc, self.tmp / "cli_pkg" / pa.PRESET_FILENAME

    def test_cli_flag_writes_it(self):
        rc, preset = self.run_export("--with-database-password")
        self.assertEqual(rc, 0)
        self.assertEqual(json.loads(preset.read_text(encoding="utf-8"))["database"]["password"],
                         "DB-GEHEIM")

    def test_cli_default_omits_it(self):
        rc, preset = self.run_export()
        self.assertEqual(rc, 0)
        self.assertNotIn("password", json.loads(preset.read_text(encoding="utf-8"))["database"])


class TestExportDatabaseCheck(_ServerDbHome):
    """A server database that does not work on the exporting machine is
    exported as ``{"type": "sqlite"}``. - Claude Generated"""

    def export(self, **kw):
        from src.utils.preset_export import export_preset

        result = export_preset(self.cm, self.tmp / "out" / "pkg", preset_id="x", **kw)
        return result, json.loads(result.preset_file.read_text(encoding="utf-8"))

    def assert_sqlite_only(self, result, data):
        self.assertEqual(data["database"], {"type": "sqlite"})
        for f in result.path.rglob("*"):
            if f.is_file():
                self.assertNotIn(b"DB-GEHEIM", f.read_bytes(), f)
                self.assertNotIn(b"db.example", f.read_bytes(), f)
        self.assertNotEqual(result.preset_file.stat().st_mode & 0o077, 0)
        self.assertTrue(any(w.startswith("Datenbank: MARIADB nicht nutzbar (")
                            and w.endswith("; exportiert wird SQLite") for w in result.warnings),
                        result.warnings)

    def test_driver_missing_gives_sqlite_without_password(self):
        from PyQt6.QtSql import QSqlDatabase

        self.driver = REAL_DRIVER_PROBE
        with patch.object(QSqlDatabase, "addDatabase", side_effect=_without_server_drivers):
            result, data = self.export(include_database_password=True)
        self.assert_sqlite_only(result, data)
        self.assertTrue(any("QMARIADB" in w for w in result.warnings), result.warnings)
        self.assertEqual(self.probed, [])  # no connection without a driver
        self.assertFalse(any(n.startswith("alima_driver_probe_")
                             for n in QSqlDatabase.connectionNames()))

    def test_failed_connection_gives_sqlite(self):
        self.connection = (False, "❌ Connection failed: Can't connect to server")
        result, data = self.export(include_database_password=True)
        self.assert_sqlite_only(result, data)
        self.assertTrue(any("Can't connect to server" in w for w in result.warnings))
        self.assertEqual(len(self.probed), 1)
        self.assertEqual(self.probed[0].host, "db.example")

    def test_working_database_keeps_the_server_block(self):
        result, data = self.export()
        self.assertEqual((data["database"]["type"], data["database"]["host"]),
                         ("mariadb", "db.example"))
        self.assertEqual((self.driver_calls, len(self.probed)), (["mariadb"], 1))
        self.assertFalse(any("nicht nutzbar" in w for w in result.warnings))

    def test_without_check_the_server_block_stays(self):
        self.connection = (False, "❌ Connection failed")
        result, data = self.export(verify_database=False, include_database_password=True)
        self.assertEqual(data["database"]["password"], "DB-GEHEIM")
        self.assertEqual((self.driver_calls, self.probed), ([], []))

    def test_cli_switch(self):
        from src.cli.commands import preset_cmd
        from src.cli.main import create_argument_parser

        self.connection = (False, "❌ Connection failed")
        for extra, expected in (([], {"type": "sqlite"}), (["--no-verify-database"], "mariadb")):
            folder = self.tmp / f"cli_{len(extra)}"
            args = create_argument_parser().parse_args(["preset", "export", str(folder), *extra])
            with redirect_stdout(io.StringIO()):
                self.assertEqual(preset_cmd.handle_preset(args, logging.getLogger("test")), 0)
            block = json.loads((folder / pa.PRESET_FILENAME).read_text(encoding="utf-8"))["database"]
            self.assertEqual(block if extra == [] else block["type"], expected)


class TestSqliteBlockPrefill(_Home):
    """``{"type": "sqlite"}`` without a path pre-fills SQLite with the default
    path. - Claude Generated"""

    def test_loader_and_gui_wizard(self):
        from src.utils.config_models import DatabaseConfig
        from src.utils.preset_loader import PresetLoader

        presets = PresetLoader._parse({"database": {"type": "sqlite"}})
        self.assertEqual((presets.db_type, presets.db_sqlite_path), ("sqlite", ""))
        self.assertTrue(presets.has_database_config())
        pkg = self.tmp / "sqlite_pkg"
        pkg.mkdir()
        (pkg / pa.PRESET_FILENAME).write_text(json.dumps(
            {"format": 2, "preset": {"id": "s"}, "database": {"type": "sqlite"}}), encoding="utf-8")
        _need_widgets(self)
        from src.ui.first_start_wizard import FirstStartWizard

        with patch.dict(os.environ, {pa.ENV_VAR: str(pkg)}):
            wizard = FirstStartWizard()
        _KEEP.append(wizard)
        page = wizard.database_page
        self.assertTrue(page.sqlite_radio.isChecked())
        self.assertEqual(page.sqlite_path_input.text(), "")
        self.assertEqual(page.get_database_config(), {"db_type": "sqlite", "sqlite_path": None})
        with patch("src.ui.first_start_wizard.QMessageBox"):
            wizard.accept()
        db = self.config().database_config
        self.assertEqual((db.db_type, db.sqlite_path), ("sqlite", DatabaseConfig().sqlite_path))
        self.assertTrue(db.sqlite_path.startswith(str(self.tmp)))


class TestExportDialogDatabasePassword(_Home):
    def setUp(self):
        super().setUp()
        _need_widgets(self)

    def dialog(self, available):
        from src.ui.plugin_settings_tab import _ExportPresetDialog

        d = _ExportPresetDialog([("search_provider", "swb", "SWB", True)],
                                database_password_available=available)
        _KEEP.append(d)
        return d

    def test_hidden_and_false_without_a_password(self):
        d = self.dialog(False)
        d.password_cb.setChecked(True)
        self.assertFalse(d.password_cb.isVisibleTo(d))
        self.assertFalse(d.include_database_password)

    def test_unchecked_by_default_and_true_when_checked(self):
        d = self.dialog(True)
        self.assertTrue(d.password_cb.isVisibleTo(d))
        self.assertFalse(d.password_cb.isChecked())
        self.assertFalse(d.include_database_password)
        d.password_cb.setChecked(True)
        self.assertTrue(d.include_database_password)

    def test_database_check_box(self):
        from src.ui.plugin_settings_tab import _ExportPresetDialog

        server = _ExportPresetDialog([("search_provider", "swb", "SWB", True)], server_database=True)
        _KEEP.append(server)
        self.assertEqual(server.verify_db_cb.text(), "Datenbankverbindung vor dem Export prüfen")
        self.assertTrue(server.verify_db_cb.isVisibleTo(server))
        self.assertTrue(server.verify_database)
        server.verify_db_cb.setChecked(False)
        self.assertFalse(server.verify_database)
        sqlite = self.dialog(False)
        self.assertFalse(sqlite.verify_db_cb.isVisibleTo(sqlite))
        self.assertTrue(sqlite.verify_database)


class TestCli(_Home):
    def run_cli(self, *argv, answer=None):
        from src.cli.commands import preset_cmd
        from src.cli.main import create_argument_parser

        args = create_argument_parser().parse_args(["preset", *argv])
        out = io.StringIO()
        with redirect_stdout(out), patch("builtins.input", return_value=answer or "n"):
            rc = preset_cmd.handle_preset(args, logging.getLogger("test"))
        return rc, out.getvalue()

    def test_dry_run_changes_nothing(self):
        before = self.disk()
        rc, out = self.run_cli("apply", str(self.package()), "--dry-run")
        self.assertEqual(rc, 0)
        self.assertIn("Einrichtung: ISIL: (leer) → ZZ-TEST", out)
        self.assertEqual(self.disk(), before)
        self.assertFalse(self.data_dir.exists())

    def test_apply_asks_and_no_means_nothing(self):
        before = self.disk()
        rc, out = self.run_cli("apply", str(self.package()))
        self.assertEqual(rc, 1)
        self.assertIn("Abgebrochen", out)
        self.assertEqual(self.disk(), before)

    def test_apply_show_revert(self):
        rc, _ = self.run_cli("apply", str(self.package()), "--yes")
        self.assertEqual(rc, 0)
        self.assertEqual(self.config().system_config.institution_isil, "ZZ-TEST")
        with patch.dict(os.environ, {pa.ENV_VAR: str(self.tmp / "pkg")}):
            rc, out = self.run_cli("show")
        self.assertIn("Umgebungsvariable ALIMA_PRESET", out)
        self.assertIn("Zuletzt angewendet: Test-UB", out)
        rc, out = self.run_cli("revert", "--yes")
        self.assertEqual(rc, 0)
        self.assertEqual(self.config().system_config.institution_isil, "")

    def test_errors_exit_non_zero(self):
        rc, out = self.run_cli("apply", str(self.package(mutate=lambda d: d["files"][1].__setitem__(
            "sha256", "3" * 64))), "--yes")
        self.assertEqual(rc, 1)
        self.assertIn("Prüfsumme stimmt nicht", out)

    def test_export(self):
        rc, out = self.run_cli("export", str(self.tmp / "exp"), "--plugin", "swb", "--id", "e1")
        self.assertEqual(rc, 0)
        data = json.loads((self.tmp / "exp" / pa.PRESET_FILENAME).read_text(encoding="utf-8"))
        self.assertEqual(data["preset"]["id"], "e1")
        self.assertEqual([p["instance"] for p in data["plugins"]], ["swb"])

    def test_preset_is_setup_exempt_and_bundle_is_gone(self):
        from src.cli.main import _SETUP_EXEMPT_COMMANDS, create_argument_parser

        self.assertIn("preset", _SETUP_EXEMPT_COMMANDS)
        self.assertNotIn("bundle", _SETUP_EXEMPT_COMMANDS)
        with redirect_stdout(io.StringIO()), patch("sys.stderr", io.StringIO()):
            with self.assertRaises(SystemExit):
                create_argument_parser().parse_args(["bundle", "list"])


class TestUi(_Home):
    def setUp(self):
        super().setUp()
        _need_widgets(self)

    def tab(self):
        from src.ui.plugin_settings_tab import PluginSettingsTab

        tab = PluginSettingsTab()
        _KEEP.append(tab)
        tab.load(self.cm.load_config())
        return tab

    @staticmethod
    def drain(tab, rounds=50):
        """Wait for the preset workers, including one started by a callback."""
        from PyQt6.QtWidgets import QApplication

        for _ in range(rounds):
            worker = tab._preset_worker
            if worker is not None:
                worker.wait(10000)
            QApplication.processEvents()
            QApplication.processEvents()
            if tab._preset_worker is worker and tab._preset_on_done is None:
                break

    def test_group_shows_source_name_and_buttons(self):
        with patch.dict(os.environ, {pa.ENV_VAR: str(self.package())}):
            tab = self.tab()
        text = tab._preset_found_label.text()
        self.assertIn("Test-UB 1", text)
        self.assertIn("Umgebungsvariable ALIMA_PRESET", text)
        self.assertEqual(tab._preset_last_label.text(), "Zuletzt angewendet: keines")
        self.assertEqual([tab._preset_apply_btn.text(), tab._preset_revert_btn.text(),
                          tab._preset_export_btn.text()],
                         ["Preset anwenden…", "Rückgängig", "Preset exportieren…"])
        self.assertFalse(tab._preset_revert_btn.isEnabled())
        actions = [a.text() for a in tab._preset_apply_btn.menu().actions()]
        self.assertEqual(actions, ["Gefundenes Preset", "Preset-Datei wählen…", "Preset-Ordner wählen…"])
        self.assertFalse(hasattr(tab, "_bundle_list"))

    def test_config_directory_source_is_shown(self):
        make_package(self.config_dir)
        tab = self.tab()
        self.assertIn("Quelle: Konfigurationsverzeichnis", tab._preset_found_label.text())

    def test_preview_dialog(self):
        from src.ui.plugin_settings_tab import _PresetPlanDialog

        good = _PresetPlanDialog(self.plan(self.package()))
        _KEEP.append(good)
        self.assertIn("Einrichtung: ISIL: (leer) → ZZ-TEST", good.text.toPlainText())
        self.assertTrue(good.apply_button.isEnabled())
        bad = _PresetPlanDialog(self.plan(self.package(
            "bad", lambda d: d["files"][0].__setitem__("sha256", "4" * 64))))
        _KEEP.append(bad)
        self.assertFalse(bad.apply_button.isEnabled())
        self.assertIn("Fehler", bad.text.toPlainText())

    def test_apply_and_revert_through_the_tab(self):
        from PyQt6.QtWidgets import QDialog, QMessageBox

        from src.ui.plugin_settings_tab import _PresetPlanDialog

        tab = self.tab()
        seen = []
        tab.preset_applied.connect(lambda steps, saved: seen.append((steps, saved)))
        with patch.object(_PresetPlanDialog, "exec", return_value=QDialog.DialogCode.Accepted):
            tab.preview_preset(self.package())
            self.drain(tab)      # plan → dialog → apply worker
            self.drain(tab)
        self.assertEqual(self.config().system_config.institution_isil, "ZZ-TEST")
        ids = [i.instance_id for i in tab._panels["search_provider"].collect()]
        self.assertEqual(ids[:3], ["swb", "finc", "k10_test"])
        self.assertEqual(len(seen), 1)
        self.assertIn(["system_config", "institution_isil"], [s["path"] for s in seen[0][0]])
        self.assertIn("angewendet", tab._preset_status.text())
        self.assertTrue(tab._preset_revert_btn.isEnabled())
        with patch.object(QMessageBox, "question", return_value=QMessageBox.StandardButton.Yes), \
                patch.object(QMessageBox, "information"):
            tab._revert_preset_clicked()
            self.drain(tab)
        self.assertEqual(self.config().system_config.institution_isil, "")
        self.assertNotIn("k10_test", [i.instance_id for i in tab._panels["search_provider"].collect()])
        self.assertEqual(len(seen), 2)

    def test_unsaved_tab_edits_survive_an_applied_preset(self):
        tab = self.tab()
        panel = tab._panels["search_provider"]
        # an unsaved edit in the visible form of the first row (lobid)
        panel._hint_edit.setPlainText("ungespeichert")
        plan = self.plan(self.package())
        pa.apply_plan(plan, self.cm)
        tab._after_preset(plan.steps(), "x")
        collected = {i.instance_id: i for i in panel.collect()}
        self.assertEqual(collected["lobid"].usage_hint, "ungespeichert")
        self.assertFalse(collected["swb"].enabled)
        self.assertEqual(self.inst("lobid").usage_hint, "")  # saved state untouched

    def test_geo_database_state_line(self):
        from PyQt6.QtWidgets import QLabel

        from src.ui.plugin_settings_tab import _CategoryPanel, _ensure_categories
        from src.utils.config_models import PluginInstanceConfig

        _ensure_categories()
        db = self.tmp / "geo.db"
        build_synthetic_db(db)
        for path, expected in ((db, True), (self.tmp / "fehlt.db", False)):
            panel = _CategoryPanel("lookup")
            _KEEP.append(panel)
            panel.load([PluginInstanceConfig("geo_sachdb", "lookup", "geo_sachdb",
                                             settings={"db_path": str(path)})])
            panel.list.setCurrentRow(0)
            texts = [w.text() for w in panel.findChildren(QLabel)]
            line = [t for t in texts if t.startswith("Quelle TEST.mdb")]
            self.assertEqual(bool(line), expected, texts)
            if expected:
                self.assertIn("Stand 2026-01-01", line[0])
                self.assertIn("Prüfsumme abcdef0123456789", line[0])

    def test_settings_dialog_follows_an_applied_preset(self):
        from src.ui.comprehensive_settings_dialog import ComprehensiveSettingsDialog

        dialog = ComprehensiveSettingsDialog()
        _KEEP.append(dialog)
        plan = self.plan(self.package())
        pa.apply_plan(plan, self.cm)
        saved = self.config()
        dialog.plugin_settings_tab._after_preset(plan.steps(), "x")
        self.assertEqual(dialog.institution_isil.text(), "ZZ-TEST")
        self.assertEqual(dialog.institution_eln.text(), "4711")
        self.assertEqual(dialog.workflow_list.default_stem(), "alima_v51_105")
        self.assertEqual(dialog.config_to_edit.applied_presets, saved.applied_presets)
        # Save writes the preset values back, not the values shown at opening
        cfg = dialog._get_config_from_ui()
        self.assertEqual(cfg.system_config.institution_isil, "ZZ-TEST")
        self.assertEqual(len(cfg.applied_presets), 1)
        self.assertEqual(dialog.export_preset_button.text(), "📤 Als Preset exportieren")


class TestBundlesGone(_Home):
    def test_no_bundle_code(self):
        self.assertIsNone(importlib.util.find_spec("src.utils.bundle"))
        self.assertIsNone(importlib.util.find_spec("src.cli.commands.bundle_cmd"))
        self.assertFalse((REPO / "docs" / "institutional_bundles.md").exists())
        self.assertFalse((REPO / "examples" / "bundles").exists())

    def test_old_config_with_the_bundle_ledger_loads_and_drops_it(self):
        path = Path(self.cm.config_file)
        data = json.loads(path.read_text(encoding="utf-8"))
        data["installed_bundles"] = {"ub-x": {"version": "2", "plugin_ids": ["p"]}}
        data["system_config"]["autosave_dir"] = "ergebnisse_ub"
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        cfg = self.config()
        self.assertFalse(hasattr(cfg, "installed_bundles"))
        self.assertEqual(cfg.system_config.autosave_dir, "ergebnisse_ub")
        self.cm.save_config(cfg)
        saved = json.loads(path.read_text(encoding="utf-8"))
        self.assertNotIn("installed_bundles", saved)
        self.assertEqual(saved["system_config"]["autosave_dir"], "ergebnisse_ub")

    def test_state_helpers_are_shared_not_copied(self):
        from src.core.search import cache_only
        from src.ui import plugin_settings_tab as tab

        for name in ("apply_state", "instance_state", "cache_only_offer", "STATE_CACHE_ONLY",
                     "OFFER_CACHE_ONLY"):
            self.assertIs(getattr(tab, name), getattr(cache_only, name), name)


class TestRoundTrip(_Home):
    def test_load_and_save_without_diff(self):
        first = self.disk()
        self.assertNotIn("applied_presets", json.loads(first))
        ConfigManager.reset()
        cm = ConfigManager()
        cm.save_config(cm.load_config())
        self.assertEqual(self.disk(), first)

    def test_ledger_survives_load_and_save(self):
        pa.apply_plan(self.plan(self.package()), self.cm)
        first = self.disk()
        ConfigManager.reset()
        cm = ConfigManager()
        cm.save_config(cm.load_config())
        self.assertEqual(self.disk(), first)
        self.assertEqual(len(json.loads(first)["applied_presets"]), 1)


class TestExamples(_Home):
    def test_example_package_plans_cleanly(self):
        plan = self.plan(REPO / "examples" / "presets" / "demo_institution")
        self.assertEqual((plan.errors, plan.warnings), ([], []))
        self.assertEqual(sorted(f.status for f in plan.files), [pa.FILE_NEW, pa.FILE_NEW])
        self.assertTrue(any(c.path == ["instance", "search_provider", "k10plus_demo"]
                            for c in plan.changes))

    def test_template_file_reads(self):
        from src.utils.preset_loader import PresetLoader

        preset = pa.load_preset(REPO / "alima_presets.json.example")
        self.assertEqual(preset.ignored_keys, [])
        self.assertNotIn("api_key", preset.data["llm"])
        self.assertNotIn("token", preset.data["catalog"])
        presets = PresetLoader._parse(preset.data)
        self.assertEqual(presets.catalog_soap_search_url,
                         "https://libero.ub.uni-example.de/libero/LiberoWebServices.CatalogueSearcher.cls")
        self.assertEqual(preset.label, "UB Musterstadt")


if __name__ == "__main__":
    unittest.main()
