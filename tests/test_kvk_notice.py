"""Claude Generated - search_kvk says in words when the list is not complete.

``catalog_stats[].truncated`` was in the result, but the model did not read it.
The tool now adds ``notice`` (plain text) when a union catalog delivered fewer
records than it reported, a catalog answered with an error other than "no
hits", or ``max_results`` cut the merged list; it names ``search_k10plus`` only
when a K10plus catalog was cut and that tool exists in the running
configuration. Pinned against the stored real KVK answer
(``fixtures/kvk_raw_huebler.ndjson``) and invented stats; no network.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.core.search.provider import ProviderResult, SearchCapability
from src.core.search.providers.kvk.provider import KvkProvider, catalog_notice_parts
from src.utils.config_models import AlimaConfig, PluginInstanceConfig

FIXTURE = Path(__file__).parent / "fixtures" / "kvk_raw_huebler.ndjson"


def _provider(payload: str) -> KvkProvider:
    provider = KvkProvider()
    response = MagicMock(status_code=200, text=payload)
    provider.client._session = MagicMock(get=MagicMock(return_value=response))
    return provider


def _line(name, results, items, truncated=False):
    data = {"name": name, "results": results,
            "items": [{"title": f"{name} {i}", "url": ""} for i in range(items)]}
    if truncated:
        data["next"] = "x"
    return json.dumps({"type": "catalog", "data": data})


def _inst(pid, **settings):
    return PluginInstanceConfig(instance_id=pid, category="search_provider", provider_id=pid,
                                label="", enabled=True, settings=dict(settings))


class TestNoticeText(unittest.TestCase):
    def test_real_answer_names_cut_catalogs_and_kvk_errors(self):
        provider = _provider(FIXTURE.read_text(encoding="utf-8"))
        res = provider.search(SearchCapability.TITLE_RECORDS, ["conrad hübler"])
        notice = provider.tool_notice(res, tools={"search_kvk", "search_k10plus"})
        self.assertIn("K10plus: 2 von 18 Treffern", notice)
        self.assertIn("Deutsche Nationalbibliothek: 2 von 7 Treffern", notice)
        self.assertIn("„Die Auswertung der Katalog-Antwort erbrachte kein Ergebnis“", notice)
        self.assertIn("hebis-Retro / Digitalisierte Zettelkataloge", notice)
        # "Keine Datensätze gefunden" is no hit, not a failure.
        self.assertNotIn("HBZ", notice)
        self.assertTrue(notice.endswith("Für vollständige K10plus-Treffer search_k10plus nutzen."))

    def test_the_k10plus_tool_is_named_only_when_it_exists(self):
        provider = _provider(FIXTURE.read_text(encoding="utf-8"))
        res = provider.search(SearchCapability.TITLE_RECORDS, ["x"])
        without = provider.tool_notice(res, tools={"search_kvk"})
        self.assertNotIn("search_k10plus", without)
        self.assertNotIn("Für vollständige", without)
        self.assertTrue(without.endswith("„Die Auswertung der Katalog-Antwort erbrachte kein Ergebnis“."))
        self.assertIn("search_k10plus_zwei",
                      provider.tool_notice(res, tools={"search_kvk", "search_k10plus_zwei"}))

    def test_the_k10plus_tool_is_named_only_when_k10plus_was_cut(self):
        payload = "\n".join([_line("K10plus - Verbundkatalog", 2, 2),
                             _line("BVB - Bayern", 9, 2, truncated=True)])
        provider = _provider(payload)
        res = provider.search(SearchCapability.TITLE_RECORDS, ["x"])
        notice = provider.tool_notice(res, tools={"search_k10plus"})
        self.assertEqual(notice, "BVB: 2 von 9 Treffern.")

    def test_complete_answer_has_no_notice(self):
        payload = "\n".join([
            _line("K10plus - Verbundkatalog", 2, 2),
            json.dumps({"type": "error", "data": [
                {"name": "HBZ", "message": "Keine Datensätze gefunden."}]}),
        ])
        provider = _provider(payload)
        res = provider.search(SearchCapability.TITLE_RECORDS, ["x"])
        self.assertEqual(provider.tool_notice(res, tools={"search_k10plus"}), "")

    def test_truncated_without_a_larger_count_and_the_merged_cap(self):
        parts, k10 = catalog_notice_parts({
            "catalog_stats": [{"catalog": "K10plus - X", "results": 2, "returned": 2,
                               "truncated": True}],
            "catalog_errors": [], "returned": 1,
        })
        self.assertTrue(k10)
        self.assertEqual(parts, [
            "K10plus: erste Seite (2 Treffer), weitere vorhanden",
            "zusammengeführte Liste auf 1 von 2 gelieferten Treffern gekürzt (max_results)",
        ])

    def test_several_terms_are_named(self):
        provider = KvkProvider()
        meta = {"cut": {"catalog_stats": [{"catalog": "BVB", "results": 5, "returned": 2}],
                        "returned": 2},
                "fine": {"catalog_stats": [{"catalog": "BVB", "results": 1, "returned": 1}],
                         "returned": 1}}
        res = ProviderResult(SearchCapability.TITLE_RECORDS, per_term={}, per_term_meta=meta)
        self.assertEqual(provider.tool_notice(res, tools=()), "„cut“: BVB: 2 von 5 Treffern.")

    def test_description_tells_the_agent_to_read_it(self):
        spec = KvkProvider.mcp_tool_specs()[0]
        self.assertIn("`notice`", spec.description)


class TestNoticeInTheTool(unittest.TestCase):
    def _out(self, *instances, payload):
        from src.mcp.tool_registry import ToolRegistry

        cfg = AlimaConfig()
        cfg.plugins = list(instances)
        reg = ToolRegistry(config_manager=MagicMock(load_config=lambda *a, **k: cfg))
        reg._knowledge_manager = MagicMock()
        with patch("src.core.search.factory._global_response_cache_enabled", return_value=False):
            provider = reg._provider_for(instances[0], cache=False)
        response = MagicMock(status_code=200, text=payload)
        provider.client._session = MagicMock(get=MagicMock(return_value=response))
        tools = {td.name: h for td, h in reg._generated_search_tools()}
        return json.loads(tools["search_kvk"](terms=["x"]))

    def test_notice_with_the_k10plus_tool_configured(self):
        out = self._out(_inst("kvk"), _inst("sru", preset="k10plus"),
                        payload=FIXTURE.read_text(encoding="utf-8"))
        self.assertIn("K10plus: 2 von 18 Treffern", out["notice"])
        self.assertIn("search_k10plus nutzen", out["notice"])
        self.assertIn("catalog_stats", out["meta"]["x"])   # the numbers stay

    def test_notice_without_it(self):
        out = self._out(_inst("kvk"), _inst("sru", preset="dnb"),
                        payload=FIXTURE.read_text(encoding="utf-8"))
        self.assertIn("K10plus: 2 von 18 Treffern", out["notice"])
        self.assertNotIn("search_k10plus", out["notice"])
        self.assertNotIn("Für vollständige", out["notice"])

    def test_no_notice_key_when_complete(self):
        out = self._out(_inst("kvk"), payload=_line("K10plus - Verbundkatalog", 1, 1))
        self.assertNotIn("notice", out)


if __name__ == "__main__":
    unittest.main()
