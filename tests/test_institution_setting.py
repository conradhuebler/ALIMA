"""Claude Generated - ISIL and ELN of the library as one setting (Nov 9, 2026).

"DE-105" / "105" stood fixed in the WinIBW entry, the finc DK facet, the
Libero web fallback and examples. Pinned here:

* the derivation helpers (``src/utils/institution.py``) and their ``None``
  for an empty field; the ELN is never derived from the ISIL;
* the one-time migration for a Freiberg configuration and the round trip of a
  config.json without the keys (saved unchanged);
* without ISIL: the WinIBW entry has no closing line and says so in
  ``notes``; finc and Libero report "Einrichtung nicht gesetzt" instead of an
  empty result; with another ISIL the derived values follow it (the Freiberg
  values are pinned by the existing tests, which run with DE-105/105);
* the "Einrichtung" group on the Datenquellen page loads and saves both fields;
* no string literal in ``src/`` carries the fixed values any more.

No network, no ``~/.config/alima``: configs are temporary files.
"""

from __future__ import annotations

import ast
import json
import logging
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.utils import institution
from src.utils.config_manager import ConfigManager

ROOT = Path(__file__).resolve().parent.parent


class _Unset(unittest.TestCase):
    def setUp(self):
        institution.set_current("", "")


# ── helpers ───────────────────────────────────────────────────────────────


class TestHelpers(_Unset):
    def test_derived_values(self):
        self.assertEqual(institution.chain_end_marker("DE-105"), "$ADE-105")
        self.assertEqual(institution.finc_dk_facet("DE-105"), "udk_raw_de105")
        self.assertEqual(institution.finc_dk_facet("DE-Ch1"), "udk_raw_dech1")
        self.assertEqual(institution.institution_filter(" DE-105 "), "DE-105")
        self.assertEqual(institution.institution_eln("105"), "105")

    def test_empty_is_none(self):
        for fn in (institution.chain_end_marker, institution.finc_dk_facet,
                   institution.institution_filter, institution.institution_eln):
            self.assertIsNone(fn(), fn.__name__)
            self.assertIsNone(fn(""), fn.__name__)
            self.assertIsNone(fn("  "), fn.__name__)

    def test_current_values_and_no_eln_from_the_isil(self):
        institution.set_current("DE-15", "")
        self.assertEqual(institution.chain_end_marker(), "$ADE-15")
        self.assertEqual(institution.finc_dk_facet(), "udk_raw_de15")
        self.assertIsNone(institution.institution_eln())
        institution.set_current("", "105")
        self.assertIsNone(institution.chain_end_marker())
        self.assertEqual(institution.institution_eln(), "105")

    def test_marker(self):
        self.assertTrue(institution.is_not_set(institution.NOT_SET_MESSAGE))
        self.assertIn("Einrichtung nicht gesetzt", institution.NOT_SET_MESSAGE)
        self.assertFalse(institution.is_not_set("cache_only: x"))


# ── configuration ─────────────────────────────────────────────────────────


def _plugin(pid, **settings):
    return {"instance_id": pid, "category": "search_provider", "provider_id": pid,
            "label": pid, "enabled": True, "usage_hint": "", "settings": settings}


class _ConfigFile(unittest.TestCase):
    def setUp(self):
        institution.set_current("", "")
        ConfigManager.reset()
        self.addCleanup(ConfigManager.reset)
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.path = Path(tmp.name) / "config.json"

    def _write(self, system_config=None, plugins=()):
        data = {"unified_config": {"providers": [], "gemini_api_key": "K"},
                "plugins": list(plugins)}
        if system_config is not None:
            data["system_config"] = system_config
        self.path.write_text(json.dumps(data, indent=2))

    def _cm(self):
        ConfigManager.reset()
        cm = ConfigManager()
        cm.config_file = self.path
        return cm

    def _system_on_disk(self):
        return json.loads(self.path.read_text()).get("system_config", {})


class TestMigration(_ConfigFile):
    FREIBERG = [_plugin("finc", base_url="https://finc.example/proxy",
                        web_record_url="https://katalog.ub.tu-freiberg.de/Record/")]

    def test_freiberg_config_without_keys_gets_de105(self):
        self._write({"contact_email": "x@y.z"}, self.FREIBERG)
        with self.assertLogs("src.utils.institution", level=logging.INFO) as logs:
            cfg = self._cm().load_config()
        self.assertEqual((cfg.system_config.institution_isil, cfg.system_config.institution_eln),
                         ("DE-105", "105"))
        self.assertIn("finc", "\n".join(logs.output))
        self.assertEqual(institution.current(), institution.Institution("DE-105", "105"))

    def test_catalog_and_sru_urls_count_too(self):
        for plugin in (_plugin("catalog", web_search_url="https://katalog.ub.tu-freiberg.de/x"),
                       _plugin("sru", base_url="https://sru.tu-freiberg.de/x")):
            self._write({}, [plugin])
            cfg = self._cm().load_config()
            self.assertEqual(cfg.system_config.institution_isil, "DE-105", plugin["provider_id"])

    def test_no_migration_with_a_key_present_or_without_freiberg_url(self):
        self._write({"institution_isil": ""}, self.FREIBERG)
        cfg = self._cm().load_config()
        self.assertEqual((cfg.system_config.institution_isil, cfg.system_config.institution_eln),
                         ("", ""))
        self._write({}, [_plugin("finc", base_url="https://finc.example/proxy"),
                         _plugin("lobid", base_url="https://tu-freiberg.de/x")])
        cfg = self._cm().load_config()
        self.assertEqual(cfg.system_config.institution_isil, "")


class TestRoundTrip(_ConfigFile):
    def test_config_without_the_keys_is_saved_without_them(self):
        self._write({"contact_email": "x@y.z"}, [_plugin("finc", base_url="https://f.example")])
        cm = self._cm()
        cm.save_config(cm.load_config())
        first = self.path.read_bytes()
        self.assertNotIn("institution_isil", self._system_on_disk())
        self.assertNotIn("institution_eln", self._system_on_disk())
        cm = self._cm()
        cm.save_config(cm.load_config())
        self.assertEqual(self.path.read_bytes(), first)

    def test_freiberg_config_is_written_once_then_stable(self):
        self._write({}, TestMigration.FREIBERG)
        cm = self._cm()
        cm.save_config(cm.load_config())
        self.assertEqual(self._system_on_disk()["institution_isil"], "DE-105")
        self.assertEqual(self._system_on_disk()["institution_eln"], "105")
        first = self.path.read_bytes()
        cm = self._cm()
        cm.save_config(cm.load_config())
        self.assertEqual(self.path.read_bytes(), first)

    def test_cleared_values_are_not_migrated_again(self):
        self._write({}, TestMigration.FREIBERG)
        cm = self._cm()
        cfg = cm.load_config()
        cfg.system_config.institution_isil = ""
        cfg.system_config.institution_eln = ""
        cm.save_config(cfg)
        self.assertEqual(self._system_on_disk()["institution_isil"], "")
        cfg = self._cm().load_config()
        self.assertEqual(cfg.system_config.institution_isil, "")
        self.assertEqual(institution.current(), institution.Institution("", ""))

    def test_set_values_round_trip_and_become_current(self):
        self._write({"institution_isil": "DE-15", "institution_eln": "15"})
        cm = self._cm()
        cfg = cm.load_config()
        self.assertEqual(institution.current(), institution.Institution("DE-15", "15"))
        cfg.system_config.institution_eln = "16"
        cm.save_config(cfg)
        self.assertEqual(institution.current().eln, "16")
        self.assertEqual(self._system_on_disk()["institution_eln"], "16")


# ── consumers without / with another ISIL ─────────────────────────────────


class TestWinIbw(_Unset):
    PAYLOAD = {"results": {"keyword_chains": [["Kette A", "Kette B"]],
                           "final_keywords": ["Kette A", "Einzel"]}}

    def test_without_isil_no_closing_line_and_a_note(self):
        from src.utils.winibw_record import build_winibw_record, winibw_notes

        record = build_winibw_record(self.PAYLOAD)
        self.assertEqual(record["lines"], ["5550 Kette A", "5550 Kette B", "", "5551 Einzel"])
        self.assertNotIn("chain_end", [e["origin"] for e in record["entries"]])
        self.assertEqual(len(record["notes"]), 1)
        self.assertIn("Einrichtung nicht gesetzt", record["notes"][0])
        self.assertIn(("Hinweis", record["notes"][0]), winibw_notes(record))

    def test_with_another_isil_and_without_chains(self):
        from src.utils.winibw_record import build_winibw_record

        institution.set_current("DE-15", "")
        record = build_winibw_record(self.PAYLOAD)
        self.assertIn("5550 $ADE-15", record["lines"])
        self.assertNotIn("notes", record)
        institution.set_current("", "")
        self.assertNotIn("notes", build_winibw_record({"results": {"final_keywords": ["X"]}}))

    def test_freiberg_record_has_no_notes_key(self):
        from src.utils.winibw_record import build_winibw_record

        institution.set_current("DE-105", "105")
        record = build_winibw_record(self.PAYLOAD)
        self.assertIn("5550 $ADE-105", record["lines"])
        self.assertNotIn("notes", record)


class TestFinc(_Unset):
    def test_dk_extraction_reports_not_set_without_request(self):
        from src.core.search.providers.finc.finc_catalog_client import FincCatalogClient

        client = FincCatalogClient(base_url="http://finc.invalid", use_cache=False,
                                   knowledge_manager=MagicMock())
        client._client = MagicMock()
        self.assertEqual(client.extract_dk_classifications_for_keywords(["Graph", "Baum"]), [])
        client._client.search.assert_not_called()
        self.assertEqual(client.last_not_configured, ["Graph", "Baum"])

    def test_dk_facet_follows_the_isil(self):
        from src.core.search.providers.finc.finc_catalog_client import FincCatalogClient

        institution.set_current("DE-15", "")
        client = FincCatalogClient(base_url="http://finc.invalid", use_cache=False,
                                   knowledge_manager=MagicMock())
        thread_client = MagicMock()
        thread_client.search.return_value = {"status": "OK", "facets": {
            "udk_raw_de15": [{"value": "dk 54", "count": 1}],
            "udk_raw_de105": [{"value": "dk 99", "count": 1}]}}
        client._local.client = thread_client
        self.assertEqual(client._title_classifications("id-1"), ["DK 54"])
        self.assertEqual(thread_client.search.call_args.kwargs["facets"][0], "udk_raw_de15")

    def test_dk_search_type_is_reported_not_sent(self):
        from src.core.search.providers.finc.suggester import FincSuggester

        with tempfile.TemporaryDirectory() as tmp:
            sugg = FincSuggester(base_url="http://finc.invalid", data_dir=tmp)
        sugg.client = MagicMock()
        out = sugg.search(["DK 54"], search_type="dk")
        sugg.client.search.assert_not_called()
        self.assertTrue(institution.is_not_set(out["DK 54"]["errors"][0]))
        self.assertTrue(institution.is_not_set(sugg.last_errors["DK 54"]))
        institution.set_current("DE-15", "")
        self.assertEqual(sugg._map_search_type("dk"), "udk_raw_de15")

    def test_auto_facets_without_isil(self):
        from src.core.search.provider import SearchCapability
        from src.core.search.providers.finc.provider import FincProvider

        provider = FincProvider(base_url="http://finc.invalid")
        provider._suggester = MagicMock()
        provider._suggester.search.return_value = {}
        provider._suggester.last_errors = {}
        provider._cache_raw = False
        with patch.object(type(provider), "suggester", property(lambda self: self._suggester)):
            provider.search(SearchCapability.TITLE_RECORDS, ["x"], search_type="rvk")
        self.assertEqual(provider._suggester.search.call_args.kwargs["facets"], ["rvk_facet"])


_RECORD_HTML = """<html><body><h1 property="name">Ein Titel</h1>
<a href="/Search/Results?lookfor=DK+54&amp;type=udk_raw_de105">DK 54</a>
<a href="/Search/Results?lookfor=DK+66&amp;type=udk_raw_de15">DK 66</a>
<a href="/Search/Results?lookfor=QA 12&amp;type=udk_raw_de15">QA 12</a>
</body></html>"""


class TestLibero(_Unset):
    def _client(self, html=_RECORD_HTML):
        from src.utils.clients.biblio_client import BiblioClient

        client = BiblioClient(web_search_url="https://opac.invalid/Search/Results",
                              web_record_url="https://opac.invalid/Record/")
        client.session = MagicMock()
        client.session.get.return_value = MagicMock(status_code=200, text=html)
        return client

    def test_web_details_without_isil_read_no_dk_and_note_it(self):
        client = self._client()
        details = client._get_title_details_web("1")
        self.assertEqual(details["classifications"], [])
        self.assertEqual(client.last_not_configured, ["Web-Titeldetails (DK)"])

    def test_web_details_follow_the_isil(self):
        institution.set_current("DE-15", "")
        details = self._client()._get_title_details_web("1")
        self.assertEqual(details["classifications"], ["DK 66", "Q QA 12"])
        institution.set_current("DE-105", "")
        self.assertEqual(self._client()._get_title_details_web("1")["classifications"], ["DK 54"])

    def test_web_search_without_isil_is_not_sent(self):
        client = self._client()
        self.assertEqual(client._search_web("Graph"), [])
        client.session.get.assert_not_called()
        self.assertEqual(client.last_not_configured, ["Web-Suche"])

    def test_web_search_filters_on_the_isil(self):
        institution.set_current("DE-15", "")
        client = self._client("<html></html>")
        client._search_web("Graph")
        params = client.session.get.call_args.kwargs["params"]
        self.assertIn('institution:"DE-15"', params["hiddenFilters[]"])


class TestDkStepReport(unittest.TestCase):
    def test_one_line_and_the_statistic(self):
        from src.utils.pipeline_utils import PipelineStepExecutor

        class Extractor:
            cache_only = False

            def __init__(self):
                self.last_not_configured = []

            def extract_dk_classifications_for_keywords(self, keywords, max_results=50):
                self.last_not_configured = list(keywords)
                return []

        lines = []
        executor = PipelineStepExecutor(alima_manager=None, cache_manager=None,
                                        logger=None, config_manager=None)
        keywords = [f"K{i} (GND-ID: 400000{i}-{i})" for i in range(3)]
        with patch("src.core.search.factory.resolve_dk_extractor", return_value=Extractor()), \
                patch("src.utils.config_manager.ConfigManager", MagicMock()):
            result = executor.execute_notation_search(
                list(keywords), stream_callback=lambda m, _s: lines.append(m), rvk_enabled=False)
        self.assertEqual(len(result["statistics"]["institution_not_set"]), 3)
        self.assertEqual(len([m for m in lines if "Einrichtung nicht gesetzt" in m]), 1)
        self.assertEqual([m for m in lines if "❌" in m], [])


# ── settings dialog ───────────────────────────────────────────────────────


class TestSettingsGroup(unittest.TestCase):
    def _dialog(self, isil, eln):
        from tests.test_settings_dialog_state import _DialogHarness

        class _H(_DialogHarness):
            def configure(self, cfg):
                cfg.system_config.institution_isil = isil
                cfg.system_config.institution_eln = eln

        return _H()._dialog()

    def test_loads_and_saves_both_fields(self):
        dlg = self._dialog("DE-105", "105")
        self.assertEqual(dlg.institution_isil.text(), "DE-105")
        self.assertEqual(dlg.institution_eln.text(), "105")
        dlg.institution_isil.setText(" DE-15 ")
        dlg.institution_eln.setText("")
        cfg = dlg._get_config_from_ui()
        self.assertEqual((cfg.system_config.institution_isil, cfg.system_config.institution_eln),
                         ("DE-15", ""))

    def test_the_group_sits_on_the_sources_page_and_is_searchable(self):
        dlg = self._dialog("", "")
        page_of = {e.label: e.page for e in dlg.index.entries}
        self.assertEqual(page_of["ISIL der Bibliothek"], dlg.PAGE_SOURCES)
        self.assertEqual(page_of["ELN der Bibliothek"], dlg.PAGE_SOURCES)
        self.assertEqual(dlg.institution_isil.text(), "")


# ── no fixed value left ───────────────────────────────────────────────────


class TestNoFixedValue(unittest.TestCase):
    """String literals (not docstrings, not comments) in src/ with DE-105/105."""

    #: Where a literal is the value itself: the migration target and the
    #: placeholders of the two fields that set it.
    ALLOWED = {
        "src/utils/institution.py": {"DE-105", "105"},
        "src/ui/comprehensive_settings_dialog.py": {"z. B. DE-105", "z. B. 105"},
    }

    @staticmethod
    def _docstring_nodes(tree):
        ids = set()
        for node in ast.walk(tree):
            body = getattr(node, "body", None)
            if isinstance(body, list):
                for stmt in body:
                    if (isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant)
                            and isinstance(stmt.value.value, str)):
                        ids.add(id(stmt.value))
        return ids

    def test_no_literal_outside_the_allowed_places(self):
        found = []
        for path in sorted((ROOT / "src").rglob("*.py")):
            rel = path.relative_to(ROOT).as_posix()
            tree = ast.parse(path.read_text(encoding="utf-8"))
            docs = self._docstring_nodes(tree)
            for node in ast.walk(tree):
                if not (isinstance(node, ast.Constant) and isinstance(node.value, str)):
                    continue
                if id(node) in docs:
                    continue
                text = node.value
                hit = ("DE-105" in text or "de105" in text.lower() or text.strip() == "105")
                if hit and text not in self.ALLOWED.get(rel, set()):
                    found.append(f"{rel}:{node.lineno}: {text[:60]!r}")
        self.assertEqual(found, [])


if __name__ == "__main__":
    unittest.main()
