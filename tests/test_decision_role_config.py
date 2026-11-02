"""Die Rolle "Entscheidungsmodell" in der Konfiguration. Claude Generated.

Anders als Allgemein/Pipeline/Bild ist sie opt-in und ohne Rückfall: ein
Chat-Modell bedient ``/v1/systemone`` nicht, also darf ein nicht gesetzter oder
nicht nutzbarer Eintrag nie auf den allgemeinen Standard oder den ersten
Anbieter durchfallen. Außerdem muss sie Speichern und Laden überleben (die
Merge-Liste von ``save_config`` hat schon einmal ein Standardfeld vergessen).
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.utils.config_manager import ConfigManager
from src.utils.config_models import (
    ProviderScope,
    UnifiedProvider,
    UnifiedProviderConfig,
)


def _config(**kw) -> UnifiedProviderConfig:
    cfg = UnifiedProviderConfig(providers=[
        UnifiedProvider(name="A", provider_type="ollama", enabled=True, preferred_model="a1"),
        UnifiedProvider(name="B", provider_type="ollama", enabled=True, preferred_model="b1"),
    ])
    for k, v in kw.items():
        setattr(cfg, k, v)
    return cfg


class DecisionScopeResolutionTest(unittest.TestCase):
    def test_unset_means_off_not_the_general_default(self):
        cfg = _config(preferred_provider="A", preferred_model="a1")
        self.assertEqual(
            cfg.explain_default_provider_model(scope=ProviderScope.DECISION),
            ("", "", "none"),
        )

    def test_a_chosen_pair_wins(self):
        cfg = _config(decision_default_provider="B", decision_default_model="d1")
        self.assertEqual(
            cfg.explain_default_provider_model(scope=ProviderScope.DECISION),
            ("B", "d1", "decision_default"),
        )

    def test_a_provider_without_model_is_not_filled_with_its_first_model(self):
        """Das erste Modell eines Anbieters wäre ein Chat-Modell."""
        cfg = _config(decision_default_provider="B", decision_default_model="")
        self.assertEqual(
            cfg.explain_default_provider_model(scope=ProviderScope.DECISION),
            ("", "", "none"),
        )

    def test_a_disabled_or_deleted_provider_does_not_fall_through(self):
        cfg = _config(decision_default_provider="B", decision_default_model="d1",
                      preferred_provider="A", preferred_model="a1")
        cfg.get_provider_by_name("B").enabled = False
        self.assertEqual(
            cfg.explain_default_provider_model(scope=ProviderScope.DECISION),
            ("", "", "none"),
        )
        cfg.decision_default_provider = "gone"
        self.assertEqual(
            cfg.explain_default_provider_model(scope=ProviderScope.DECISION)[2], "none"
        )

    def test_the_other_scopes_ignore_it(self):
        cfg = _config(decision_default_provider="B", decision_default_model="d1",
                      preferred_provider="A", preferred_model="a1")
        for scope in (ProviderScope.GENERAL, ProviderScope.PIPELINE,
                      ProviderScope.VISION, ProviderScope.AGENTIC):
            self.assertEqual(
                cfg.explain_default_provider_model(scope=scope)[:2], ("A", "a1"), scope
            )

    def test_fill_empty_default_models_leaves_it_alone(self):
        cfg = _config(decision_default_provider="B", decision_default_model="")
        cfg.fill_empty_default_models()
        self.assertEqual(cfg.decision_default_model, "")


class DecisionRolePersistenceTest(unittest.TestCase):
    def setUp(self):
        ConfigManager.reset()
        self.tmp = Path(tempfile.mkdtemp()) / "config.json"

    def tearDown(self):
        ConfigManager.reset()

    def _manager(self) -> ConfigManager:
        ConfigManager.reset()
        cm = ConfigManager()
        cm.config_file = self.tmp
        return cm

    def _write(self, unified: dict) -> None:
        self.tmp.write_text(json.dumps({"unified_config": unified}), encoding="utf-8")

    @staticmethod
    def _provider(name="srv") -> dict:
        return {"name": name, "provider_type": "ollama", "enabled": True,
                "host": "localhost", "port": 11434, "preferred_model": "m"}

    def test_round_trip_and_clearing(self):
        self._write({"providers": [self._provider()],
                     "decision_default_provider": "srv",
                     "decision_default_model": "d1"})
        cm = self._manager()
        config = cm.load_config()
        self.assertEqual(config.unified_config.decision_default_provider, "srv")
        self.assertEqual(config.unified_config.decision_default_model, "d1")

        config.unified_config.decision_default_model = "d3"
        cm.save_config(config)
        on_disk = json.loads(self.tmp.read_text(encoding="utf-8"))["unified_config"]
        self.assertEqual(on_disk["decision_default_provider"], "srv")
        self.assertEqual(on_disk["decision_default_model"], "d3")

        config.unified_config.decision_default_provider = ""
        config.unified_config.decision_default_model = ""
        cm.save_config(config)
        on_disk = json.loads(self.tmp.read_text(encoding="utf-8"))["unified_config"]
        self.assertEqual(on_disk["decision_default_provider"], "")

    def test_an_old_config_without_the_keys_loads_as_off(self):
        self._write({"providers": [self._provider()]})
        config = self._manager().load_config()
        self.assertEqual(config.unified_config.decision_default_provider, "")
        self.assertEqual(config.unified_config.decision_default_model, "")


if __name__ == "__main__":
    unittest.main()
