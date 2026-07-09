"""Tests for the WebIndexLookup plugin (agent tools over the store) - Claude Generated.

Netzfrei: the store is pre-populated directly; live-fetch in ``fetch_page`` is
exercised by patching ``fetch_guarded_response`` so no HTTP happens.
"""

import os
import tempfile
import types
import unittest
from unittest import mock

try:
    import src.utils.lookups  # noqa: F401 — registers webindex (+ category)
    from src.utils.lookups.registry import lookup_tool_specs
    from src.core.plugins.category import get_category
    from src.utils.config_models import AlimaConfig, DatabaseConfig, PluginInstanceConfig
    from src.utils.lookups.webindex.provider import WebIndexLookup
    from src.core.unified_knowledge_manager import UnifiedKnowledgeManager
    from src.mcp.tool_registry import ToolRegistry
    IMPORT_ERROR = None
except ModuleNotFoundError as exc:  # pragma: no cover
    IMPORT_ERROR = exc


def _populate(lookup):
    s = lookup.store
    s.upsert_page(url="https://bib/fernleihe", base_url="https://bib/", title="Fernleihe",
                  text="Die Fernleihe beschafft Literatur aus anderen Bibliotheken. "
                       "Wenden Sie sich an die Auskunft für Fernleihanfragen.",
                  http_status=200, content_type="text/html", text_truncated=False)
    s.set_page_keywords("https://bib/fernleihe",
                        [("Fernleihe", 1.0, "meta"), ("Auskunft", 1.0, "meta")])
    s.upsert_page(url="https://bib/katalog", base_url="https://bib/", title="Katalog",
                  text="Der Online-Katalog verzeichnet alle Bestände der Bibliothek.",
                  http_status=200, content_type="text/html", text_truncated=False)
    s.set_page_keywords("https://bib/katalog", [("Katalog", 1.0, "meta")])


@unittest.skipIf(IMPORT_ERROR is not None, f"stack unavailable: {IMPORT_ERROR}")
class WebIndexLookupTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.mkdtemp()
        self.lookup = WebIndexLookup(db_path=os.path.join(self._tmp, "w.db"),
                                     max_results=10, snippet_chars=80)

    def tearDown(self):
        try:
            self.lookup.store.close()
        except Exception:
            pass

    # --- tool specs / registration -------------------------------------- #
    def test_tools_registered(self):
        names = {s.name for s in lookup_tool_specs() if s.provider_id == "webindex"}
        self.assertEqual(names, {"search_webindex", "fetch_page", "list_webindex_keywords"})
        # method names match the spec so the generated handler dispatches correctly.
        methods = {s.method for s in lookup_tool_specs() if s.provider_id == "webindex"}
        self.assertEqual(methods, {"search_keyword", "fetch_page", "list_keywords"})

    def test_built_via_category(self):
        """LookupCategory.build constructs the plugin from a PluginInstanceConfig."""
        inst = PluginInstanceConfig(instance_id="wi", category="lookup",
                                     provider_id="webindex", settings={"db_path": os.path.join(self._tmp, "c.db")})
        plugin = get_category("lookup").build(inst)
        self.assertIsInstance(plugin, WebIndexLookup)
        plugin.store.close()

    # --- search_keyword ------------------------------------------------- #
    def test_search_returns_ranked_hits_with_snippet(self):
        _populate(self.lookup)
        r = self.lookup.search_keyword("Wie funktioniert die Fernleihe?")
        self.assertEqual(r["count"], 1)
        hit = r["hits"][0]
        self.assertEqual(hit["url"], "https://bib/fernleihe")
        self.assertIn("fernleihe", hit["matched_keywords"])
        self.assertIn("Fernleihe", hit["snippet"])  # snippet anchored on the match

    def test_search_multi_keyword_ranks_two_match_page_first(self):
        _populate(self.lookup)
        r = self.lookup.search_keyword("Fernleihe Auskunft")
        # fernleihe page matches both terms → ranks above katalog page.
        self.assertEqual(r["hits"][0]["url"], "https://bib/fernleihe")
        self.assertEqual(r["hits"][0]["matched_count"], 2)

    def test_search_no_match(self):
        _populate(self.lookup)
        r = self.lookup.search_keyword("Programmierung")
        self.assertEqual(r["count"], 0)

    # --- fetch_page ----------------------------------------------------- #
    def test_fetch_page_cache_hit(self):
        _populate(self.lookup)
        r = self.lookup.fetch_page("https://bib/katalog")
        self.assertEqual(r["source"], "cache")
        self.assertEqual(r["title"], "Katalog")
        self.assertIn("Online-Katalog", r["text"])
        self.assertFalse(r["truncated"])

    def test_fetch_page_max_chars_truncates(self):
        _populate(self.lookup)
        r = self.lookup.fetch_page("https://bib/katalog", max_chars=10)
        self.assertTrue(r["truncated"])
        self.assertLessEqual(len(r["text"]), 30)  # 10 + truncation marker

    def test_fetch_page_cache_miss_live_fetch(self):
        # No instance base_url set; fetch_on_miss default True.
        url = "https://live.example/x"
        html = (b"<html><head><title>Live</title></head><body><main>"
                b"<p>Live-Inhalt ueber Fernleihe der Bibliothek genuegend lang.</p>"
                b"</main></body></html>")
        resp = mock.Mock()
        resp.headers = {"Content-Type": "text/html"}
        resp.status_code = 200
        resp.content = html
        with mock.patch("src.utils.input_sources.url_fetch.fetch_guarded_response",
                        return_value=resp):
            r = self.lookup.fetch_page(url)
        self.assertEqual(r["source"], "live")
        self.assertEqual(r["title"], "Live")
        # now cached for the next call
        r2 = self.lookup.fetch_page(url)
        self.assertEqual(r2["source"], "cache")

    def test_fetch_page_miss_disabled_returns_error(self):
        self.lookup._config["fetch_on_miss"] = False
        r = self.lookup.fetch_page("https://nowhere/missing")
        self.assertIn("error", r)

    def test_fetch_page_resolves_relative_url_against_base_url(self):
        """A relative path the agent passes is joined to base_url before the
        cache lookup + live fetch (the index stores absolute URLs)."""
        self.lookup._config["base_url"] = "https://bib.example.de/"
        captured = {}

        def fake_fetch(url, *, user_agent="ua", timeout=20):
            captured["url"] = url
            resp = mock.Mock()
            resp.headers = {"Content-Type": "text/html"}
            resp.status_code = 200
            resp.content = (b"<html><head><title>Ueber uns</title></head><body><main>"
                            b"<p>Ueber die Bibliothek und ihr Team, genug Text hier.</p>"
                            b"</main></body></html>")
            return resp

        with mock.patch("src.utils.input_sources.url_fetch.fetch_guarded_response",
                        side_effect=fake_fetch):
            r = self.lookup.fetch_page("/ub/ueber-uns")
        self.assertEqual(captured["url"], "https://bib.example.de/ub/ueber-uns")
        self.assertEqual(r["source"], "live")
        self.assertEqual(r["title"], "Ueber uns")
        # now cached under the absolute URL
        self.assertIsNotNone(self.lookup.store.get_page("https://bib.example.de/ub/ueber-uns"))

    # --- list_keywords -------------------------------------------------- #
    def test_list_keywords(self):
        _populate(self.lookup)
        r = self.lookup.list_keywords(limit=50, contains="fern")
        self.assertEqual([k["keyword"] for k in r["keywords"]], ["fernleihe"])
        self.assertEqual(r["keywords"][0]["page_count"], 1)


def _sqlite_config(path):
    cfg = DatabaseConfig(db_type="sqlite")
    cfg.sqlite_path = path
    return cfg


@unittest.skipIf(IMPORT_ERROR is not None, f"stack unavailable: {IMPORT_ERROR}")
class WebIndexToolRegistryTest(unittest.TestCase):
    """The agent-facing path: the MCP tool registry generates a handler per
    enabled webindex instance that dispatches to the plugin's tool methods."""

    def setUp(self):
        UnifiedKnowledgeManager.reset()
        self.tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
        self.tmp.close()
        self.km = UnifiedKnowledgeManager(database_config=_sqlite_config(self.tmp.name))
        self._idx_tmp = tempfile.mkdtemp()
        self._db = os.path.join(self._idx_tmp, "idx.db")
        # Pre-populate the index DB the handler will read.
        pre = WebIndexLookup(db_path=self._db)
        _populate(pre)
        pre.store.close()

    def tearDown(self):
        UnifiedKnowledgeManager.reset()
        try:
            os.unlink(self.tmp.name)
        except OSError:
            pass

    def _registry(self):
        cfg = AlimaConfig()
        cfg.system_config.enable_response_cache = False
        cfg.plugins = [PluginInstanceConfig(
            "wi", "lookup", "webindex", enabled=True, is_primary=True,
            settings={"db_path": self._db, "cache_responses": "off"},
        )]
        reg = ToolRegistry.__new__(ToolRegistry)
        reg._config_manager = types.SimpleNamespace(load_config=lambda **k: cfg)
        reg._knowledge_manager = self.km
        reg._tools = {}
        reg._handlers = {}
        return reg

    def test_generated_search_handler_dispatches(self):
        import json
        reg = self._registry()
        handlers = {td.name: h for td, h in reg._generated_lookup_tools()}
        self.assertIn("search_webindex", handlers)
        out = json.loads(handlers["search_webindex"](query="Fernleihe Auskunft", max_results=5))
        self.assertEqual(out["count"], 1)
        self.assertEqual(out["hits"][0]["url"], "https://bib/fernleihe")

    def test_generated_fetch_handler_dispatches(self):
        import json
        reg = self._registry()
        handlers = {td.name: h for td, h in reg._generated_lookup_tools()}
        out = json.loads(handlers["fetch_page"](url="https://bib/katalog"))
        self.assertEqual(out["source"], "cache")
        self.assertEqual(out["title"], "Katalog")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()