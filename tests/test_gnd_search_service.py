"""Unified GND-keyword search service (src/core/search/service.py) - Claude Generated.

Covers the single provider-path entry point that replaces direct MetaSuggester
construction:
* instance resolution (all-enabled / per-id synth / disabled-respect / overrides);
* the live/merge path (cross-source max count, union codes, max display_count);
* the raw-first path (derives the nested view from the WP2 raw cache, count-landmine).
"""

import json
import os
import tempfile
import unittest

try:
    from src.core.unified_knowledge_manager import UnifiedKnowledgeManager
    from src.utils.config_models import DatabaseConfig, PluginInstanceConfig
    from src.core.search.provider import ProviderResult, SearchCapability, raw_cache_params_for
    from src.core.search.registry import PROVIDER_REGISTRY
    from src.core.search import service
    IMPORT_ERROR = None
except ModuleNotFoundError as exc:  # pragma: no cover
    IMPORT_ERROR = exc


def _sqlite_config(path):
    cfg = DatabaseConfig(db_type="sqlite")
    cfg.sqlite_path = path
    return cfg


class _FakeSuggester:
    """Exposes the ``transform`` the raw-first path reads off the suggester."""

    def __init__(self, transform):
        self.transform = transform


def _make_fake_provider(pid, per_term_results, transform=None):
    """Build a GND_KEYWORDS provider class returning canned results. - Claude Generated"""

    class _FakeGnd:
        id = pid
        label = pid
        capabilities = {SearchCapability.GND_KEYWORDS}

        def __init__(self, **config):
            self._config = config

        def is_available(self, cfg=None):
            return True

        def search(self, capability, query, *, progress=None, **opts):
            return ProviderResult.from_gnd_keywords(
                {t: per_term_results.get(t, {}) for t in query}
            )

        @property
        def suggester(self):
            return _FakeSuggester(transform)

    return _FakeGnd


class _StubConfig:
    """Minimal AlimaConfig surface used by resolve_gnd_instances. - Claude Generated"""

    def __init__(self, instances):
        self._instances = instances

    def instances_for(self, category):
        return [p for p in self._instances if p.category == category]

    def enabled_instances_for(self, category):
        return [p for p in self._instances if p.category == category and p.enabled]

    def primary_instance(self, category, provider_id=None):
        pool = [
            p for p in self.enabled_instances_for(category)
            if provider_id is None or p.provider_id == provider_id
        ]
        for p in pool:
            if p.is_primary:
                return p
        return pool[0] if pool else None


@unittest.skipIf(IMPORT_ERROR is not None, f"stack unavailable: {IMPORT_ERROR}")
class ResolveInstancesTest(unittest.TestCase):
    def tearDown(self):
        PROVIDER_REGISTRY.pop("fake_a", None)

    def _instance(self, pid, **kw):
        return PluginInstanceConfig(
            instance_id=kw.get("instance_id", pid), category="search_provider",
            provider_id=pid, enabled=kw.get("enabled", True),
            is_primary=kw.get("is_primary", True), settings=kw.get("settings", {}),
        )

    def test_synth_for_unknown_id(self):
        out = service.resolve_gnd_instances(["fake_a"], config=_StubConfig([]))
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].provider_id, "fake_a")
        self.assertTrue(out[0].enabled)

    def test_disabled_instance_is_respected(self):
        cfg = _StubConfig([self._instance("fake_a", enabled=False)])
        out = service.resolve_gnd_instances(["fake_a"], config=cfg)
        self.assertEqual(out, [])  # present but disabled → not searched

    def test_overrides_overlay_non_empty(self):
        cfg = _StubConfig([self._instance("fake_a", settings={"token": "old", "url": "u"})])
        out = service.resolve_gnd_instances(
            ["fake_a"], config=cfg, overrides={"fake_a": {"token": "new", "extra": ""}}
        )
        self.assertEqual(out[0].settings["token"], "new")   # overridden
        self.assertEqual(out[0].settings["url"], "u")        # preserved
        self.assertNotIn("extra", out[0].settings)           # empty override skipped

    def test_all_returns_enabled_gnd_instances(self):
        PROVIDER_REGISTRY["fake_a"] = _make_fake_provider("fake_a", {})
        cfg = _StubConfig([self._instance("fake_a")])
        out = service.resolve_gnd_instances(None, config=cfg)
        self.assertEqual([i.provider_id for i in out], ["fake_a"])


@unittest.skipIf(IMPORT_ERROR is not None, f"stack unavailable: {IMPORT_ERROR}")
class SearchServiceTest(unittest.TestCase):
    def setUp(self):
        UnifiedKnowledgeManager.reset()
        self.tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
        self.tmp.close()
        self.km = UnifiedKnowledgeManager(database_config=_sqlite_config(self.tmp.name))

    def tearDown(self):
        UnifiedKnowledgeManager.reset()
        for pid in ("fake_a", "fake_b"):
            PROVIDER_REGISTRY.pop(pid, None)
        try:
            os.unlink(self.tmp.name)
        except OSError:
            pass

    @staticmethod
    def _inst(pid):
        return PluginInstanceConfig(
            instance_id=pid, category="search_provider", provider_id=pid,
            enabled=True, is_primary=True,
        )

    def test_live_merge_across_sources(self):
        PROVIDER_REGISTRY["fake_a"] = _make_fake_provider("fake_a", {
            "wasser": {"Wasser": {"count": 47, "gndid": {"g1"}, "ddc": set(),
                                  "dk": set(), "display_count": 47}},
        })
        PROVIDER_REGISTRY["fake_b"] = _make_fake_provider("fake_b", {
            "wasser": {
                "Wasser": {"count": 1, "gndid": {"g1", "g2"}, "ddc": {"5"},
                           "dk": set(), "display_count": 10},
                "Klima": {"count": 3, "gndid": {"g3"}, "ddc": set(), "dk": set()},
            },
        })
        results, errors = service.search_gnd_keywords(
            ["wasser"], [self._inst("fake_a"), self._inst("fake_b")],
            cache=False, aggregate_from_raw=False, ukm=self.km,
        )
        self.assertEqual(errors, {})
        wasser = results["wasser"]["Wasser"]
        self.assertEqual(wasser["count"], 47)                      # max, never summed
        self.assertEqual(wasser["gnd_ids"], {"g1", "g2"})          # union
        self.assertEqual(wasser["classifications"], {"ddc": {"5"}})
        self.assertEqual(wasser["display_count"], 47)              # max across sources
        self.assertEqual(results["wasser"]["Klima"]["count"], 3)   # from B only

    def test_source_failure_recorded(self):
        class _Boom:
            id = "fake_a"
            label = "fake_a"
            capabilities = {SearchCapability.GND_KEYWORDS}

            def __init__(self, **config):
                pass

            def is_available(self, cfg=None):
                return True

            def search(self, capability, query, *, progress=None, **opts):
                raise RuntimeError("down")

        PROVIDER_REGISTRY["fake_a"] = _Boom
        results, errors = service.search_gnd_keywords(
            ["wasser"], [self._inst("fake_a")],
            cache=False, aggregate_from_raw=False, ukm=self.km,
        )
        self.assertEqual(results, {"wasser": {}})
        self.assertEqual(errors, {"fake_a:wasser": "down"})

    def test_raw_first_derives_from_cache(self):
        # Pre-store raw so the aggregate engine reads it (count-landmine → count=1,
        # real count moves to display_count).
        self.km.store_raw_response("fake_a", "wasser", raw_cache_params_for("fake_a"), "{}")
        PROVIDER_REGISTRY["fake_a"] = _make_fake_provider(
            "fake_a", {"wasser": {}},
            transform=lambda raw: {
                "Wasser": {"count": 47, "gndid": {"g1"}, "ddc": set(), "dk": set()},
            },
        )
        results, errors = service.search_gnd_keywords(
            ["wasser"], [self._inst("fake_a")],
            cache=False, aggregate_from_raw=True, ukm=self.km,
        )
        self.assertEqual(errors, {})
        wasser = results["wasser"]["Wasser"]
        self.assertEqual(wasser["count"], 1)             # landmine
        self.assertEqual(wasser["display_count"], 47)    # real count
        self.assertEqual(wasser["gnd_ids"], {"g1"})


if __name__ == "__main__":
    unittest.main()
