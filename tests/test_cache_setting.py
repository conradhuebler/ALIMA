"""Per-plugin raw-response cache setting (cache_responses tri-state) - Claude Generated.

Covers the standard cache toggle injected into every plugin (search + input):
tri-state interpretation, its presence in both category forms, and the provider
read-side honoring it (auto follows the global switch; on/off override).
"""

import unittest

try:
    from src.core.plugins.schema import cache_field, cache_pref_enabled, CACHE_RESPONSES_KEY
    from src.core.plugins.category import get_category
    from src.core.search.factory import build_provider
    from src.utils.config_models import PluginInstanceConfig
    import src.utils.input_sources  # noqa: F401 — registers the input_source category
    IMPORT_ERROR = None
except ModuleNotFoundError as exc:  # pragma: no cover
    IMPORT_ERROR = exc


@unittest.skipIf(IMPORT_ERROR is not None, f"stack unavailable: {IMPORT_ERROR}")
class CacheSettingTest(unittest.TestCase):
    def test_tristate_interpretation(self):
        self.assertFalse(cache_pref_enabled("off", global_enabled=True))
        self.assertTrue(cache_pref_enabled("on", global_enabled=False))
        self.assertFalse(cache_pref_enabled("auto", global_enabled=False))
        self.assertTrue(cache_pref_enabled("auto", global_enabled=True))
        self.assertTrue(cache_pref_enabled(None, global_enabled=True))     # absent → global
        self.assertFalse(cache_pref_enabled(False, global_enabled=True))   # legacy bool

    def test_field_shape(self):
        f = cache_field()
        self.assertEqual(f.key, CACHE_RESPONSES_KEY)
        self.assertEqual(f.default, "auto")
        self.assertEqual(set(f.choices), {"auto", "on", "off"})

    def test_field_present_in_both_categories(self):
        for cat, type_id in [("search_provider", "lobid"), ("input_source", "doi_crossref")]:
            keys = [f.key for f in get_category(cat).type_meta(type_id).config_fields]
            self.assertIn(CACHE_RESPONSES_KEY, keys, cat)

    def test_provider_read_side_honors_setting(self):
        cases = [
            ("off", True, False),   # off overrides global-on
            ("on", False, True),    # on overrides global-off
            ("auto", False, False),
            ("auto", True, True),
        ]
        for pref, glob, expected in cases:
            inst = PluginInstanceConfig(
                instance_id="lobid", category="search_provider", provider_id="lobid",
                settings={CACHE_RESPONSES_KEY: pref},
            )
            prov = build_provider(inst, cache=False, cache_raw=glob)
            inner = getattr(prov, "inner", prov)
            self.assertEqual(inner._cache_raw_enabled(), expected, f"{pref}/{glob}")

    def test_finc_store_raw_honors_off_setting(self):
        # Regression: finc's own _store_finc_raw used bool(override), so the
        # tri-state "off" (a truthy string) was ignored and raw was written
        # anyway. It now shares the base tri-state gate. - Claude Generated
        from src.core.search.providers.finc.provider import FincProvider

        raw = {"wasser": {"records": [{"id": "1"}], "result_count": 1}}
        for pref, glob, should_write in [("off", True, False), ("on", False, True)]:
            prov = FincProvider(base_url="https://x.example/proxy",
                                **{CACHE_RESPONSES_KEY: pref})
            prov._cache_raw = glob
            writes = []
            prov._ukm_ref = type("FakeUKM", (), {
                "store_raw_response": lambda self, *a, **k: writes.append(a),
            })()
            prov._store_finc_raw(["wasser"], {"search_type": "kw"}, raw)
            self.assertEqual(bool(writes), should_write, f"{pref}/{glob}")


if __name__ == "__main__":
    unittest.main()
