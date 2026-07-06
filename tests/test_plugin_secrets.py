"""Tests for the secret env-var override (ALIMA_PLUGIN_<ID>_<KEY>) - Claude Generated.

Secrets are injected at provider/source construction only — never at config
load/save — so an env-supplied secret can not be persisted.
"""

from __future__ import annotations

import json
import types
import unittest
from unittest.mock import patch

from src.core.plugins.schema import (
    SECRET,
    TEXT,
    ConfigField,
    apply_env_overrides,
    env_var_name,
)
from src.utils.config_models import PluginInstanceConfig


class EnvVarNameTest(unittest.TestCase):
    def test_naming_scheme(self):
        self.assertEqual(env_var_name("catalog", "token"), "ALIMA_PLUGIN_CATALOG_TOKEN")
        self.assertEqual(env_var_name("finc-zbw", "api.key"), "ALIMA_PLUGIN_FINC_ZBW_API_KEY")


class ApplyEnvOverridesTest(unittest.TestCase):
    FIELDS = [
        ConfigField(key="token", label="t", kind=SECRET),
        ConfigField(key="base_url", label="u", kind=TEXT),
    ]

    def test_secret_overridden_non_secret_untouched(self):
        settings = {"token": "from-config", "base_url": "http://x"}
        with patch.dict("os.environ", {
            "ALIMA_PLUGIN_CATALOG_TOKEN": "from-env",
            "ALIMA_PLUGIN_CATALOG_BASE_URL": "http://evil",
        }):
            out = apply_env_overrides("catalog", settings, self.FIELDS)
        self.assertEqual(out["token"], "from-env")
        self.assertEqual(out["base_url"], "http://x")  # non-secret: env ignored
        self.assertEqual(settings["token"], "from-config")  # input not mutated

    def test_empty_env_and_no_env_keep_config_value(self):
        with patch.dict("os.environ", {"ALIMA_PLUGIN_CATALOG_TOKEN": ""}):
            out = apply_env_overrides("catalog", {"token": "keep"}, self.FIELDS)
        self.assertEqual(out["token"], "keep")


class BuildProviderEnvOverrideTest(unittest.TestCase):
    def _instance(self, token=""):
        return PluginInstanceConfig(
            "catalog", "search_provider", "catalog",
            settings={"token": token, "catalog_search_url": "https://soap.example/x"},
        )

    def test_env_token_reaches_provider_and_gates_availability(self):
        from src.core.search.factory import build_provider

        inst = self._instance(token="")
        with patch.dict("os.environ", {"ALIMA_PLUGIN_CATALOG_TOKEN": "env-tok"}):
            provider = build_provider(inst, cache_raw=False)
        self.assertEqual(provider._config["token"], "env-tok")
        self.assertTrue(provider.is_available())
        # instance settings stay unpersisted-clean
        self.assertEqual(inst.settings["token"], "")

    def test_without_env_config_value_wins(self):
        from src.core.search.factory import build_provider

        inst = self._instance(token="cfg-tok")
        provider = build_provider(inst, cache_raw=False)
        self.assertEqual(provider._config["token"], "cfg-tok")


class ListPluginsSecretMaskTest(unittest.TestCase):
    def test_secret_keys_and_values_never_leak(self):
        from src.mcp.tool_registry import ToolRegistry
        from src.utils.config_models import AlimaConfig

        cfg = AlimaConfig()
        cfg.plugins = [
            PluginInstanceConfig(
                "catalog", "search_provider", "catalog", enabled=True, is_primary=True,
                settings={"token": "s3cret-value", "catalog_search_url": "https://soap.example/x"},
            ),
        ]
        tr = ToolRegistry.__new__(ToolRegistry)
        tr._config_manager = types.SimpleNamespace(load_config=lambda: cfg)
        raw = tr._handle_list_plugins()
        out = json.loads(raw)

        entry = {p["instance_id"]: p for p in out["plugins"]["search_provider"]}["catalog"]
        self.assertIn("catalog_search_url", entry["configured_settings"])
        self.assertNotIn("token", entry["configured_settings"])
        self.assertNotIn("s3cret-value", raw)


if __name__ == "__main__":
    unittest.main()
