"""A provider-parsing failure used to overwrite the real config with an empty
one, the moment anything else was saved.

Mechanism, confirmed by reading the code (not just a subagent's claim) and
reproduced here against real temp config files:

1. ``_parse_config`` raised ``ValueError`` for two legitimate, representable
   states — no providers at all, or every configured provider disabled.
2. That raise was caught by the broad ``except Exception`` wrapping the whole
   parse, whose fallback is ``return AlimaConfig()`` — a brand-new, totally
   empty config, in memory, standing in for whatever was actually on disk.
3. ``save_config(preserve_unified=True)`` reads the *disk* copy and overwrites
   only a fixed list of keys from the *incoming* (now empty) config —
   ``providers`` was unconditionally on that list. The next save of
   *anything* (a font size, a star, a checkbox) wrote ``providers: []`` over
   the real ones.

Two more fields shared the same merge list and were simply missing from it:
``vision_default_provider``/``model`` (a choice in the resolution table never
reached disk at all) and ``disabled_providers`` (a veto cleared in memory by
``apply_disabled_providers()`` could never be cleared on disk — the key
wasn't copied either way). - Claude Generated
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.utils.config_manager import ConfigManager


def _provider(name="claude", provider_type="anthropic", enabled=True):
    return {
        "name": name,
        "provider_type": provider_type,
        "enabled": enabled,
        "preferred_model": "m",
    }


class _TempConfig(unittest.TestCase):
    def setUp(self):
        ConfigManager.reset()
        self.tmp = Path(tempfile.mkdtemp()) / "config.json"

    def tearDown(self):
        ConfigManager.reset()

    def _write(self, payload: dict) -> None:
        self.tmp.write_text(json.dumps(payload), encoding="utf-8")

    def _manager(self) -> ConfigManager:
        ConfigManager.reset()
        cm = ConfigManager()
        cm.config_file = self.tmp
        return cm


class ParsingNeverWipesRealProvidersTest(_TempConfig):
    """The two states that used to trigger the AlimaConfig() fallback."""

    def test_an_empty_providers_list_loads_as_empty_not_as_a_wiped_config(self):
        """Faithfully empty, not "parsing failed" - the two must be
        distinguishable, or a real empty list looks identical to a crash."""
        self._write({"unified_config": {"providers": []}})
        cm = self._manager()
        config = cm.load_config()
        self.assertEqual(config.unified_config.providers, [])
        # The crash fallback also resets unrelated sections; a faithful parse
        # must not.
        self.assertEqual(config.system_config.default_workflow, "alima_v51")

    def test_all_providers_disabled_still_loads_the_rest_of_the_config(self):
        self._write({
            "unified_config": {"providers": [_provider(enabled=False)]},
            "system_config": {"cache_dir": "/custom/cache"},
        })
        cm = self._manager()
        config = cm.load_config()
        self.assertEqual(len(config.unified_config.providers), 1)
        self.assertFalse(config.unified_config.providers[0].enabled)
        # Proof it's a real parse, not the empty-fallback object.
        self.assertEqual(config.system_config.cache_dir, "/custom/cache")

    def test_a_save_after_disabling_the_last_provider_does_not_erase_it(self):
        """The actual reported mechanism, end to end: disable the last
        provider, then trigger any unrelated save - the provider must survive
        on disk."""
        self._write({"unified_config": {"providers": [_provider(name="claude")]}})
        cm = self._manager()
        config = cm.load_config()

        config.unified_config.providers[0].enabled = False
        config.system_config.cache_dir = "/new/cache"  # the "unrelated" save
        cm.save_config(config)

        on_disk = json.loads(self.tmp.read_text(encoding="utf-8"))
        names = [p["name"] for p in on_disk["unified_config"]["providers"]]
        self.assertEqual(names, ["claude"])

    def test_an_unknown_field_on_one_provider_does_not_doom_the_rest(self):
        """A future UnifiedProvider field removal must not cascade into the
        same reset via a different door."""
        self._write({"unified_config": {"providers": [
            _provider(name="old-shape"),
        ]}})
        # Inject a key UnifiedProvider does not have, simulating a removed
        # field still present in an old config file.
        payload = json.loads(self.tmp.read_text(encoding="utf-8"))
        payload["unified_config"]["providers"][0]["totally_removed_field"] = "x"
        self._write(payload)

        cm = self._manager()
        config = cm.load_config()
        self.assertEqual(len(config.unified_config.providers), 1)
        self.assertEqual(config.unified_config.providers[0].name, "old-shape")


class PreserveMergeCoversTheRightFieldsTest(_TempConfig):
    """save_config(preserve_unified=True)'s field-by-field merge."""

    def test_vision_default_round_trips(self):
        self._write({"unified_config": {
            "providers": [_provider()],
            "vision_default_provider": "claude",
            "vision_default_model": "claude-opus-5-5",
        }})
        cm = self._manager()
        config = cm.load_config()
        self.assertEqual(config.unified_config.vision_default_provider, "claude")
        self.assertEqual(config.unified_config.vision_default_model, "claude-opus-5-5")

        # Change it and save - an unrelated field change must not lose it,
        # and the new value must actually land on disk.
        config.unified_config.vision_default_provider = "other"
        config.unified_config.vision_default_model = "other-model"
        cm.save_config(config)

        on_disk = json.loads(self.tmp.read_text(encoding="utf-8"))
        self.assertEqual(on_disk["unified_config"]["vision_default_provider"], "other")
        self.assertEqual(on_disk["unified_config"]["vision_default_model"], "other-model")

    def test_clearing_vision_default_back_to_inherit_persists(self):
        """Vision is nullable ("(erbt den allgemeinen)" in ResolutionView) -
        clearing it must win, the same as pipeline/agentic defaults."""
        self._write({"unified_config": {
            "providers": [_provider()],
            "vision_default_provider": "claude",
            "vision_default_model": "claude-opus-5-5",
        }})
        cm = self._manager()
        config = cm.load_config()
        config.unified_config.vision_default_provider = ""
        config.unified_config.vision_default_model = ""
        cm.save_config(config)

        on_disk = json.loads(self.tmp.read_text(encoding="utf-8"))
        self.assertEqual(on_disk["unified_config"]["vision_default_provider"], "")

    def test_a_cleared_disabled_providers_veto_reaches_disk(self):
        """apply_disabled_providers() clears the list in memory when a vetoed
        provider is switched back on; the merge must let that clear through,
        not keep copying the stale on-disk list forever."""
        self._write({"unified_config": {
            "providers": [_provider(name="claude", enabled=True)],
            "disabled_providers": ["claude"],
        }})
        cm = self._manager()
        config = cm.load_config()
        # apply_disabled_providers ran during parsing; the provider is back on
        # and the veto list is cleared in memory.
        self.assertEqual(config.unified_config.disabled_providers, [])

        cm.save_config(config)
        on_disk = json.loads(self.tmp.read_text(encoding="utf-8"))
        self.assertEqual(on_disk["unified_config"]["disabled_providers"], [])

    def test_save_config_alone_refuses_to_blank_a_populated_provider_list(self):
        """Isolates the save-side guard from the parse-side fix: even if some
        other path someday hands save_config() an in-memory config whose
        providers list is empty (not just disabled - genuinely empty, the
        exact shape of the old AlimaConfig() fallback), the on-disk list must
        survive. Built by hand, not via load_config(), so fix A1 (the parser
        no longer producing this shape) cannot be the reason this passes."""
        self._write({"unified_config": {"providers": [_provider(name="claude")]}})
        cm = self._manager()

        from src.utils.config_models import AlimaConfig
        blank = AlimaConfig()
        self.assertEqual(blank.unified_config.providers, [])
        cm.save_config(blank)

        on_disk = json.loads(self.tmp.read_text(encoding="utf-8"))
        names = [p["name"] for p in on_disk["unified_config"]["providers"]]
        self.assertEqual(names, ["claude"])

    def test_providers_field_still_updates_normally(self):
        """The empty-list guard must not freeze providers altogether - a
        real, non-empty change still has to win."""
        self._write({"unified_config": {"providers": [_provider(name="old")]}})
        cm = self._manager()
        config = cm.load_config()
        config.unified_config.providers[0].name = "renamed"
        cm.save_config(config)

        on_disk = json.loads(self.tmp.read_text(encoding="utf-8"))
        names = [p["name"] for p in on_disk["unified_config"]["providers"]]
        self.assertEqual(names, ["renamed"])


if __name__ == "__main__":
    unittest.main()
