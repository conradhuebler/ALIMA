"""E2E proof: run ALIMA entirely on *own* (external) plugins - Claude Generated.

Extends the single-copy blueprint proof (``test_plugin_blueprint_e2e.py``) to the
full set: generate all 6 built-in providers as ``poc_*`` user plugins, load them
through the real code-plugin path, disable the built-ins, and show that **both**
frontends resolve to the own plugins:

* agentic / MCP — the canonical search tools (``search_lobid`` …) are generated
  from the ``poc_*`` instances and their handlers build the ``poc_*`` provider
  class (not the built-in);
* classic — ``execute_gnd_search`` follows the enabled plugins via the
  empty-intersection fallback instead of searching nothing;
* offline — the loaded ``poc_gnd_local`` plugin returns keywords with no network.

Reuses the shipped generator (``examples/plugins_poc/deploy_poc.py``) so the test
exercises exactly what the operator deploys. Registry + synthetic-module state is
saved/restored per test (own-plugins are registered globally by the loader).
"""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path

import src.core.search  # noqa: F401  side-effect: registers category + built-ins
from src.core.plugins import loader as loader_mod
from src.core.search.provider import SearchCapability
from src.core.search.registry import PROVIDER_REGISTRY, get_provider

# Import the shipped generator by path (examples/ is not a package).
_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEPLOY_PY = _REPO_ROOT / "examples" / "plugins_poc" / "deploy_poc.py"
_spec = importlib.util.spec_from_file_location("_deploy_poc", _DEPLOY_PY)
deploy_poc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(deploy_poc)

BUILTIN_IDS = set(deploy_poc.PROVIDER_NAMES)
POC_IDS = [f"poc_{n}" for n in deploy_poc.PROVIDER_NAMES]
TOOL_PROVIDERS = ["poc_lobid", "poc_swb", "poc_catalog", "poc_finc"]  # declare MCP tools


class _FakeUKM:
    """Canned local GND store so the offline provider needs no DB. - Claude Generated"""

    def search_local_gnd(self, term, min_results=3):
        return [types.SimpleNamespace(title="Wasserstoff", gnd_id="4064784-5", ddcs=["546"])]


class AllExternalPluginsPocTest(unittest.TestCase):
    def setUp(self):
        loader_mod._reset_for_tests()
        self._pre_registry = dict(PROVIDER_REGISTRY)
        self.root = Path(tempfile.mkdtemp())
        deploy_poc.generate(self.root)
        self.result = loader_mod.discover(
            self.root, approved_plugins={}, approve_cb=lambda *a: True, enable_code_plugins=True
        )

    def tearDown(self):
        PROVIDER_REGISTRY.clear()
        PROVIDER_REGISTRY.update(self._pre_registry)
        loader_mod._reset_for_tests()
        for name in [m for m in sys.modules if m.startswith("alima_plugin_poc_")]:
            sys.modules.pop(name, None)
        import shutil

        shutil.rmtree(self.root, ignore_errors=True)

    # -- config helper: built-ins disabled, poc_* enabled -------------------
    def _external_only_config(self):
        from src.utils.config_models import AlimaConfig, PluginInstanceConfig

        cfg = AlimaConfig()
        cfg.plugins = [
            PluginInstanceConfig(pid, "search_provider", pid, enabled=False, is_primary=True)
            for pid in BUILTIN_IDS
        ] + [
            PluginInstanceConfig(pid, "search_provider", pid, enabled=True, is_primary=True)
            for pid in POC_IDS
        ]
        return cfg

    # -- 1. all six load ----------------------------------------------------
    def test_all_six_own_plugins_load(self):
        by_id = {p.type_id or p.manifest.id: p for p in self.result.plugins}
        self.assertEqual(set(by_id), set(POC_IDS))
        for pid, p in by_id.items():
            self.assertEqual(p.status, "loaded", f"{pid}: {p.detail}")
        # complex providers (catalog/finc/sru) resolved their intra-plugin imports
        for pid in POC_IDS:
            self.assertIn(pid, PROVIDER_REGISTRY)

    # -- 2. agentic: canonical tools bound to the own plugins ---------------
    def test_agentic_tools_served_by_own_plugins(self):
        from unittest.mock import MagicMock, patch

        from src.mcp.tool_registry import ToolRegistry

        cfg = self._external_only_config()
        reg = ToolRegistry.__new__(ToolRegistry)
        reg._config_manager = types.SimpleNamespace(load_config=lambda: cfg)
        reg._suggesters_initialized = True
        # __new__ skips __init__: the nuanced canonical handlers reach the raw
        # cache, and without these a handler dies on AttributeError and reports it
        # as {"error": …} — indistinguishable from the guarded stub below. - Claude Generated
        reg._knowledge_manager = MagicMock()
        reg._provider_cache = {}

        tools = reg._generated_search_tools()
        names = {td.name for td, _ in tools}
        # The canonical tool names survive with built-ins off (no collision/suffix).
        self.assertEqual(
            names,
            {"search_lobid", "search_swb", "search_catalog", "search_catalog_titles", "search_finc"},
        )

        handlers = {td.name: h for td, h in tools}
        built_ids = []

        def fake_build(inst, **kw):
            built_ids.append(inst.provider_id)

            class _Stub:  # unavailable → handler returns guarded JSON, no network
                def is_available(self):
                    return False

            return _Stub()

        checked = ("search_lobid", "search_swb", "search_catalog", "search_finc")
        # Two patch targets, one per import site: the canonical nuanced handlers go
        # through _provider_for (`from src.core.search.factory import …`), the
        # generic one through the re-export (`from src.core.search import …`), which
        # is bound at import and unaffected by patching the factory. - Claude Generated
        with patch("src.core.search.build_provider", side_effect=fake_build), patch(
            "src.core.search.factory.build_provider", side_effect=fake_build
        ):
            for name in checked:
                out = json.loads(handlers[name](["Wasser"]))
                # The *guarded* error, not an incidental crash.
                self.assertIn("error", out)
                self.assertIn("not available", out["error"], f"{name}: unguarded failure")

        # every canonical search tool built an own-plugin provider, never a built-in
        self.assertEqual(len(built_ids), len(checked), f"not every tool built a provider: {built_ids}")
        for pid in built_ids:
            self.assertTrue(pid.startswith("poc_"), f"tool routed to built-in '{pid}'")

    # -- 2b. WP2 nuances now reach the own plugins (WP P1) ------------------
    def test_aggregate_default_and_agent_view_follow_own_plugins(self):
        """The default source list + agent_view are derived, not id-literals.

        Scope note: this drives ``aggregate_gnd_results`` directly. The *agentic*
        path (``gnd_batch_search``) used to pass ``sources`` from a hardcoded
        ``{"swb": "search_swb", "lobid": "search_lobid"}`` map — under this POC
        those ids are disabled, so it returned an empty pool. Since WP P6a
        (July 16) that map is derived from the enabled GND providers and a
        requested built-in id resolves to the copy backing the same tool name
        (``swb`` → ``search_swb`` → ``poc_swb``), so the agentic path now
        aggregates under the POC too; the resolution is unit-tested in
        ``test_search_plugins.ResolveGndSourceToolsTest``. This test still targets
        ``aggregate_gnd_results`` for a focused check of the derived default +
        agent_view. - Claude Generated
        """
        from unittest.mock import MagicMock

        from src.core.search.provider import raw_cache_params_for
        from src.mcp.tool_registry import ToolRegistry

        cfg = self._external_only_config()

        # agent_view: poc_lobid inherits LobidSuggester.transform_agent_view.
        from src.core.search.providers.lobid.suggester import LobidSuggester

        km = MagicMock()
        blob = json.dumps({"totalItems": 12, "member": [{"title": "Z"}]})
        km.get_raw_response.return_value = {"raw_json": blob}

        reg = ToolRegistry.__new__(ToolRegistry)
        reg._config_manager = types.SimpleNamespace(load_config=lambda: cfg)
        reg._knowledge_manager = km
        reg._suggesters_initialized = True
        reg._provider_cache = {
            ("poc_lobid", False): types.SimpleNamespace(suggester=LobidSuggester)
        }

        out = {"source": "lobid", "results": {}}
        reg._attach_agent_view(out, "poc_lobid", ["wasser"], "kw")
        self.assertEqual(out["agent_view"]["wasser"]["totalItems"], 12)
        # …read under the copy's own key, i.e. the copy's own raw entries.
        km.get_raw_response.assert_called_with(
            "poc_lobid", "wasser", raw_cache_params_for("poc_lobid", search_type="kw")
        )

        # aggregate default: derived from the enabled poc_* instances.
        from src.core.search.factory import enabled_gnd_provider_ids

        ids = enabled_gnd_provider_ids(config=cfg)
        self.assertTrue(ids, "no GND ids derived from the own-plugins config")
        for pid in ids:
            self.assertTrue(pid.startswith("poc_"), f"built-in '{pid}' leaked into the default")

    # -- 3. classic: execute_gnd_search follows the enabled own plugins -----
    def test_classic_search_falls_back_to_own_plugins(self):
        from unittest.mock import Mock, patch

        from src.utils.pipeline_utils import PipelineStepExecutor

        executor = PipelineStepExecutor(
            alima_manager=Mock(), cache_manager=Mock(), logger=Mock(level=100)
        )

        recorded = {}

        class _FakeSearchCLI:
            def __init__(self, *a, **k):
                self.last_errors = {}

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def search(self, search_terms, suggester_types):
                recorded["suggester_types"] = list(suggester_types)
                return {t: {"Wasserstoff": {"count": 3, "gnd_ids": {"4064784-5"}}} for t in search_terms}

        enabled = ["poc_lobid", "poc_swb", "poc_catalog"]  # built-ins disabled
        with patch("src.utils.pipeline_utils.SearchCLI", _FakeSearchCLI), patch(
            "src.utils.pipeline_utils.enabled_gnd_provider_ids", return_value=enabled
        ):
            out = executor.execute_gnd_search(
                keywords=["Wasser"], suggesters=["lobid", "swb"], aggregate_from_raw=False
            )

        # the retired built-in ids resolved to the enabled own plugins, not empty
        self.assertEqual(recorded["suggester_types"], enabled)
        self.assertIn("Wasser", out)
        self.assertIn("Wasserstoff", out["Wasser"])

    def test_classic_still_honours_partial_disable(self):
        """Regression guard: the fallback must NOT fire when the intersection is
        non-empty — a partial disable keeps its normal (subset) behaviour."""
        from unittest.mock import Mock, patch

        from src.utils.pipeline_utils import PipelineStepExecutor

        executor = PipelineStepExecutor(
            alima_manager=Mock(), cache_manager=Mock(), logger=Mock(level=100)
        )
        recorded = {}

        class _FakeSearchCLI:
            def __init__(self, *a, **k):
                self.last_errors = {}

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def search(self, search_terms, suggester_types):
                recorded["suggester_types"] = list(suggester_types)
                return {t: {} for t in search_terms}

        with patch("src.utils.pipeline_utils.SearchCLI", _FakeSearchCLI), patch(
            "src.utils.pipeline_utils.enabled_gnd_provider_ids", return_value=["lobid"]
        ):
            executor.execute_gnd_search(
                keywords=["Wasser"], suggesters=["lobid", "swb"], aggregate_from_raw=False
            )
        self.assertEqual(recorded["suggester_types"], ["lobid"])  # swb dropped, no fallback

    # -- 4. offline: the loaded own gnd_local plugin returns keywords -------
    def test_own_gnd_local_offline_search(self):
        provider = get_provider("poc_gnd_local")()
        provider._ukm = _FakeUKM()
        out = provider.search(SearchCapability.GND_KEYWORDS, ["Wasser"])
        keywords = out.to_gnd_keywords()
        self.assertIn("Wasser", keywords)
        self.assertIn("Wasserstoff", keywords["Wasser"])


if __name__ == "__main__":
    unittest.main()
