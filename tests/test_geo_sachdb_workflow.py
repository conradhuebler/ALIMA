"""geo_sachdb im Freiberger Workflow alima_v51_105 - Claude Generated.

Verdrahtung (strenges Laden, depends_on, Reihenfolge vor den shelf-Schritten),
die deterministische Funktion gegen eine synthetische Datenbank im Schema v1
(Zeilen, nicht gefundene Begriffe, Filter nach Art und System) und der Lauf ohne
Plugin (leeres Ergebnis, dk_classifications unverändert). Daten erfunden.
"""

from __future__ import annotations

import copy
import importlib.util
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.core.agents.registry import TOOL_FN_REGISTRY
from src.core.agents.deterministic_functions import geo_sachdb_candidates
from src.utils.config_models import AlimaConfig
from src.utils.lookups.geo_sachdb.provider import GeoSachdbLookup

_spec = importlib.util.spec_from_file_location(
    "geo_lookup_fixture", Path(__file__).resolve().parent / "test_geo_sachdb_lookup.py"
)
_fixture = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_fixture)
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


def _workflow(name):
    from src.core.agents.workflow_loader import find_workflow_file, load_workflow

    return load_workflow(find_workflow_file(name), strict=True)


class Wiring(unittest.TestCase):
    def test_step_sits_after_dk_postprocess_and_before_the_shelf_steps(self):
        steps = _workflow("alima_v51_105").steps
        ids = [s.id for s in steps]
        self.assertEqual(ids[-4:], ["geo_candidates", "shelf_candidates", "shelf_select", "shelf_guard"])
        by_id = {s.id: s for s in steps}
        self.assertEqual(by_id["geo_candidates"].depends_on, ["dk_postprocess"])
        self.assertIn(by_id["geo_candidates"].raw.get("function"), TOOL_FN_REGISTRY)
        self.assertEqual(by_id["geo_candidates"].raw["config"]["lookup"], "geo_sachdb")

    def test_every_depends_on_names_an_existing_step(self):
        for name in ("alima_v51_105", "alima_v51"):
            steps = _workflow(name).steps
            ids = {s.id for s in steps}
            bad = [(s.id, d) for s in steps for d in s.depends_on if d not in ids]
            self.assertEqual(bad, [], name)

    def test_the_general_workflow_has_no_geo_step(self):
        self.assertNotIn("geo_candidates", [s.id for s in _workflow("alima_v51").steps])

    def test_planner_rules_name_the_step(self):
        rules = _workflow("alima_v51_105").meta_agent["planning"]["rules"]
        self.assertIn("geo_candidates", rules)


class CandidatesFunction(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Path(self.tmp.name) / "geo.db"
        _fixture.build_synthetic_db(self.db)
        self.lookup = GeoSachdbLookup(**_fixture.settings(self.db))
        self.notations = [{"code": "DK 999.1", "rank": "core"}, {"code": "RVK AB 1234", "rank": "core"},
                          {"code": "QX 200", "rank": "core"}, {"code": "DK 888.8", "rank": "additional"}]
        self.keywords = [{"keyword": "Testgestein", "gnd_id": "x"}, {"keyword": "Lagerstätte"},
                         {"keyword": "Gestein"}, {"keyword": "Unbekanntwort"}]

    def tearDown(self):
        self.tmp.cleanup()

    def test_proposals_from_synthetic_database(self):
        with patch("src.utils.lookups.build_lookup", return_value=self.lookup):
            out = geo_sachdb_candidates(keywords=self.keywords, notations=self.notations, config={})
        self.assertEqual(out["status"], "ok")
        matched = [m["keyword"] for m in out["keyword_matches"]]
        self.assertEqual(matched, ["Testgestein", "Lagerstätte", "Gestein"])
        self.assertEqual(out["not_found"], ["Unbekanntwort"])
        by_code = {c["code"]: c for c in out["candidates"]}
        self.assertEqual(set(by_code), {"999.1", "999.21"})  # RVK QX 100 bleibt bei scheme DK draußen
        self.assertTrue(by_code["999.1"]["assigned"])
        self.assertFalse(by_code["999.21"]["assigned"])
        self.assertEqual(by_code["999.21"]["notation"], "DK 999.21")
        self.assertEqual(by_code["999.21"]["line"], "6700 DK 999.21")
        self.assertEqual(by_code["999.21"]["via_keywords"], ["Lagerstätte"])
        self.assertEqual(out["lines"], ["5550 !111111111!", "5550 !333333333!", "5550 |s|Gestein",
                                        "6700 DK 999.21", "6700 !555555555!"])
        checked = {m["code"]: m for m in out["notation_matches"]}
        self.assertEqual(set(checked), {"DK 999.1", "DK 888.8"})  # RVK (mit/ohne Präfix) bleibt draußen
        self.assertEqual(checked["DK 999.1"]["local"][0]["label"], "Testgesteine")
        self.assertEqual(checked["DK 999.1"]["line"], "6700 !555555555!")
        self.assertEqual(checked["DK 888.8"]["local"], [])
        self.assertIn("Lokale Sacherschließungsdatenbank", out["prompt_text"])
        self.assertIn("5550 !111111111!", out["prompt_text"])
        self.assertIn("Unbekanntwort", out["prompt_text"])
        self.assertEqual(out["count"], len(out["candidates"]))

    def test_kinds_and_scheme_from_config(self):
        with patch("src.utils.lookups.build_lookup", return_value=self.lookup):
            only_geo = geo_sachdb_candidates(keywords=[{"keyword": "Musterland"}, {"keyword": "Testgestein"}],
                                             config={"kinds": ["Geographikum"]})
            rvk = geo_sachdb_candidates(keywords=[{"keyword": "Lagerstätte"}],
                                        notations=[{"code": "QX 100", "system": "RVK"}], config={"scheme": "RVK"})
        self.assertEqual(([m["keyword"] for m in only_geo["keyword_matches"]], only_geo["not_found"]),
                         (["Musterland"], ["Testgestein"]))
        self.assertEqual([(c["notation"], c["line"], c["assigned"]) for c in rvk["candidates"]],
                         [("RVK QX 100", None, True)])
        self.assertEqual([m["code"] for m in rvk["notation_matches"]], ["QX 100"])

    def test_reads_inputs_from_context_when_not_passed(self):
        ctx = type("Ctx", (), {})()
        ctx.extra = {"final_keywords": [{"keyword": "Testgestein"}]}
        ctx.dk_classifications = [{"code": "DK 999.1"}]
        with patch("src.utils.lookups.build_lookup", return_value=self.lookup):
            out = geo_sachdb_candidates(context=ctx, config={})
        self.assertEqual(out["keyword_matches"][0]["keyword"], "Testgestein")

    def test_without_plugin_instance_the_result_is_empty(self):
        # Echter build_lookup-Pfad: lesbare Konfiguration ohne geo_sachdb-Instanz.
        with patch("src.utils.lookups.resolve._load_config_best_effort", return_value=AlimaConfig()):
            out = geo_sachdb_candidates(keywords=self.keywords, notations=self.notations, config={})
        self.assertEqual(out["status"], "disabled")
        self.assertEqual((out["candidates"], out["prompt_text"], out["count"]), ([], "", 0))

    def test_unavailable_plugin_and_no_input(self):
        missing = GeoSachdbLookup(**{**_fixture.settings(self.db), "db_path": str(self.db) + ".fehlt"})
        with patch("src.utils.lookups.build_lookup", return_value=missing):
            out = geo_sachdb_candidates(keywords=self.keywords, config={})
        self.assertEqual(out["status"], "unavailable")
        self.assertIn("nicht gefunden", out["reason"])
        self.assertEqual(geo_sachdb_candidates(config={})["status"], "no_input")

    def test_workflow_step_runs_without_plugin_and_leaves_classification_alone(self):
        from src.core.agents.shared_context import SharedContext
        from src.core.agents.steps.deterministic_step import DeterministicStep

        cfg = {s.id: s for s in _workflow("alima_v51_105").steps}["geo_candidates"]
        ctx = SharedContext(abstract="x")
        ctx.dk_classifications = copy.deepcopy(self.notations)
        ctx.extra["final_keywords"] = list(self.keywords)
        with patch("src.utils.lookups.resolve._load_config_best_effort", return_value=AlimaConfig()):
            result = DeterministicStep(cfg).execute(ctx)
        self.assertTrue(result.success, msg=result.error)
        self.assertEqual(ctx.extra["geo_sachdb"]["status"], "disabled")
        self.assertEqual(ctx.dk_classifications, self.notations)

    def test_workflow_step_with_plugin_fills_extra(self):
        from src.core.agents.shared_context import SharedContext
        from src.core.agents.steps.deterministic_step import DeterministicStep

        cfg = {s.id: s for s in _workflow("alima_v51_105").steps}["geo_candidates"]
        ctx = SharedContext(abstract="x")
        ctx.dk_classifications = copy.deepcopy(self.notations)
        ctx.extra["final_keywords"] = list(self.keywords)
        with patch("src.utils.lookups.build_lookup", return_value=self.lookup):
            result = DeterministicStep(cfg).execute(ctx)
        self.assertTrue(result.success, msg=result.error)
        self.assertEqual(ctx.extra["geo_sachdb"]["status"], "ok")
        self.assertEqual(ctx.dk_classifications, self.notations)


if __name__ == "__main__":
    unittest.main()
