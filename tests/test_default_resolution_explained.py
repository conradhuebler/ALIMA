"""Which of the competing settings decided the model that runs.

Six settings can determine it: the three scope defaults
(``agentic_default_*`` → ``pipeline_default_*`` → ``preferred_*``), the per-step
the vision default, and — until Sep 2026 — per-task preferences,
``provider_priority`` and ``disabled_providers``, none of which had a visible
control that matched what they did. Resolution lives in
``resolve_default_provider_model``; no surface used to show which setting applied,
so a model set in the dialog could be replaced by another with nothing saying why.

``explain_default_provider_model`` returns the winning setting alongside the
result, and ``resolve_default_provider_model`` delegates to it — one
implementation, so the explanation cannot drift from the behaviour it explains.
- Claude Generated
"""

from __future__ import annotations

import unittest

from src.utils.config_models import (
    ProviderScope,
    UnifiedProvider,
    UnifiedProviderConfig,
)


def _config(**kw) -> UnifiedProviderConfig:
    cfg = UnifiedProviderConfig(providers=[
        UnifiedProvider(name="A", provider_type="ollama", enabled=True,
                        preferred_model="a1"),
        UnifiedProvider(name="B", provider_type="ollama", enabled=True,
                        preferred_model="b1"),
    ])
    for k, v in kw.items():
        setattr(cfg, k, v)
    return cfg


class TheWinningSettingIsNamedTest(unittest.TestCase):
    def test_agentic_default_wins_for_the_agentic_scope(self):
        cfg = _config(agentic_default_provider="A", pipeline_default_provider="B",
                      preferred_provider="B")
        self.assertEqual(
            cfg.explain_default_provider_model(scope=ProviderScope.AGENTIC)[2],
            "agentic_default",
        )

    def test_agentic_falls_through_to_pipeline(self):
        cfg = _config(pipeline_default_provider="B", preferred_provider="A")
        p, _m, src = cfg.explain_default_provider_model(scope=ProviderScope.AGENTIC)
        self.assertEqual((p, src), ("B", "pipeline_default"))

    def test_pipeline_ignores_the_agentic_default(self):
        """The hierarchy is not symmetric; the display must not suggest it is."""
        cfg = _config(agentic_default_provider="A", preferred_provider="B")
        p, _m, src = cfg.explain_default_provider_model(scope=ProviderScope.PIPELINE)
        self.assertEqual((p, src), ("B", "preferred"))

    def test_general_ignores_both_scope_defaults(self):
        cfg = _config(agentic_default_provider="A", pipeline_default_provider="A",
                      preferred_provider="B")
        p, _m, src = cfg.explain_default_provider_model(scope=ProviderScope.GENERAL)
        self.assertEqual((p, src), ("B", "preferred"))

    def test_nothing_configured_names_the_fallback(self):
        cfg = _config()
        p, _m, src = cfg.explain_default_provider_model()
        self.assertEqual((p, src), ("A", "first_enabled"))

    def test_no_usable_provider_says_so(self):
        cfg = UnifiedProviderConfig(providers=[])
        self.assertEqual(cfg.explain_default_provider_model(), ("", "", "none"))

    def test_every_source_key_has_a_label(self):
        """The UI maps these; a rename must not silently show a bare key."""
        cfg = _config()
        for key in ("agentic_default", "pipeline_default", "preferred",
                    "first_enabled", "none"):
            self.assertIn(key, cfg.DEFAULT_SOURCES)


class ExplanationAndBehaviourCannotDriftTest(unittest.TestCase):
    """The reason is only worth showing if it describes the actual resolution."""

    def test_resolve_agrees_with_explain_across_the_hierarchy(self):
        cases = [
            {},
            {"preferred_provider": "A"},
            {"preferred_provider": "A", "pipeline_default_provider": "B"},
            {"preferred_provider": "A", "pipeline_default_provider": "B",
             "agentic_default_provider": "A"},
        ]
        for kw in cases:
            cfg = _config(**kw)
            for scope in ProviderScope:
                with self.subTest(kw=kw, scope=scope):
                    resolved = cfg.resolve_default_provider_model(scope=scope)
                    explained = cfg.explain_default_provider_model(scope=scope)
                    self.assertEqual(resolved, explained[:2])


class SwitchedOffProvidersAreSkippedByTheFallbackTest(unittest.TestCase):
    """The fallback honours the one remaining switch, ``enabled``.

    It used to honour ``disabled_providers`` as well; that list is folded into
    ``enabled`` at load now (see DisabledProvidersIsFoldedIntoEnabledTest).
    - Claude Generated
    """

    def test_a_disabled_provider_is_skipped(self):
        cfg = _config()
        cfg.providers[0].enabled = False
        p, _m, src = cfg.explain_default_provider_model()
        self.assertEqual((p, src), ("B", "first_enabled"))

    def test_the_legacy_list_reaches_the_fallback_via_migration(self):
        cfg = _config(disabled_providers=["A"])
        cfg.apply_disabled_providers()
        p, _m, src = cfg.explain_default_provider_model()
        self.assertEqual((p, src), ("B", "first_enabled"))


class TaskPreferencesActuallyReachExecutionTest(unittest.TestCase):
    """``PipelineStepExecutor._resolve_provider_smart`` PRIORITY 2 was dead.

    Its documented job is "task preferences from config — auto-selection based on
    task". It built a mapping dict containing ``TaskType.TEXT``, a member that
    does not exist, so the dict literal raised ``AttributeError`` on every
    evaluation; ``select_provider`` was never reached, and the surrounding broad
    ``except Exception`` logged the whole thing as a warning. Exactly the failure
    shape CLAUDE.md's Implementation Standards describe.

    Task preferences did still shape a run, but only through the pipeline config
    dialog's prefill (which lands in the per-step values that PRIORITY 1 honours).
    The execution-time fallback did not work at all. - Claude Generated
    """

    def _executor(self):
        import logging
        from unittest.mock import MagicMock

        from src.utils.pipeline_utils import PipelineStepExecutor

        ex = PipelineStepExecutor.__new__(PipelineStepExecutor)
        ex.logger = logging.getLogger("test")
        ex.config_manager = None
        ex.detection_service = None
        ex.smart_selector = MagicMock()
        ex.smart_selector.select_provider.return_value = MagicMock(
            provider="GWDG", model="apertus", fallback_used=False
        )
        return ex

    def test_the_selector_is_reached_for_every_task_type_used_in_the_code(self):
        """"text" and "classification" are what the three call sites pass."""
        for task_type in ("text", "classification", "vision", "unbekannt"):
            with self.subTest(task_type=task_type):
                ex = self._executor()
                provider, model = ex._resolve_provider_smart(
                    "", "", task_type, task_name="keywords"
                )
                self.assertTrue(
                    ex.smart_selector.select_provider.called,
                    f"PRIORITY 2 nicht erreicht für task_type={task_type!r}",
                )
                self.assertEqual((provider, model), ("GWDG", "apertus"))

    def test_an_explicit_selection_still_wins(self):
        """PRIORITY 1 must keep precedence — the user's manual pick."""
        ex = self._executor()
        self.assertEqual(
            ex._resolve_provider_smart("Mistral", "codestral", "text"),
            ("Mistral", "codestral"),
        )
        self.assertFalse(ex.smart_selector.select_provider.called)

    def test_every_mapped_task_type_exists(self):
        """The defect in one line: a mapping to a member that is not there."""
        from src.utils.config_models import TaskType

        for name in ("GENERAL", "CLASSIFICATION", "VISION"):
            self.assertTrue(hasattr(TaskType, name))
        self.assertFalse(hasattr(TaskType, "TEXT"))


class DisabledProvidersIsFoldedIntoEnabledTest(unittest.TestCase):
    """One switch for "this provider is off", not two.

    ``disabled_providers`` was a name list that overrode
    ``UnifiedProvider.enabled``. It had no control anywhere and nothing in the
    code populated it — only the parser reading an older config — but three
    places honoured it, so a provider shown as enabled could be silently
    inoperative. It is folded into ``enabled`` once at load. - Claude Generated
    """

    def test_a_listed_provider_ends_up_disabled(self):
        cfg = _config(disabled_providers=["A"])
        self.assertTrue(cfg.apply_disabled_providers())
        by_name = {p.name: p for p in cfg.providers}
        self.assertFalse(by_name["A"].enabled)
        self.assertTrue(by_name["B"].enabled)

    def test_the_legacy_list_is_cleared(self):
        cfg = _config(disabled_providers=["A"])
        cfg.apply_disabled_providers()
        self.assertEqual(cfg.disabled_providers, [])

    def test_it_is_idempotent(self):
        cfg = _config(disabled_providers=["A"])
        self.assertTrue(cfg.apply_disabled_providers())
        self.assertFalse(cfg.apply_disabled_providers())

    def test_an_unknown_name_changes_nothing(self):
        cfg = _config(disabled_providers=["gibtsnicht"])
        cfg.apply_disabled_providers()
        self.assertTrue(all(p.enabled for p in cfg.providers))

    def test_enabled_is_the_only_switch_afterwards(self):
        """The veto must not survive in get_enabled_providers."""
        cfg = _config()
        cfg.disabled_providers = ["A"]          # set but not migrated
        names = [p.name for p in cfg.get_enabled_providers()]
        self.assertEqual(names, ["A", "B"], "get_enabled_providers darf nicht mehr vetoen")

    def test_a_config_carrying_the_legacy_list_loads_migrated(self):
        """End to end through the parser, the way a real config.json arrives."""
        import json
        import tempfile
        from pathlib import Path

        from src.utils.config_manager import ConfigManager

        tmp = Path(tempfile.mkdtemp()) / "config.json"
        tmp.write_text(json.dumps({"unified_config": {
            "providers": [
                {"name": "A", "provider_type": "ollama", "enabled": True},
                {"name": "B", "provider_type": "ollama", "enabled": True},
            ],
            "disabled_providers": ["A"],
        }}))
        ConfigManager.reset()
        cm = ConfigManager()
        cm.config_file = tmp
        try:
            unified = cm.load_config().unified_config
            by_name = {p.name: p for p in unified.providers}
            self.assertFalse(by_name["A"].enabled)
            self.assertEqual(unified.disabled_providers, [])
            self.assertEqual(
                [p.name for p in unified.get_enabled_providers()], ["B"]
            )
        finally:
            ConfigManager.reset()


if __name__ == "__main__":
    unittest.main()
