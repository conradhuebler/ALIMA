"""Der Lauf-Schalter "mit Decision" für A/B-Vergleiche. Claude Generated.

Ein Schalter je Lauf (``PipelineConfig.use_decision_prefilter``), kein Einstellungswert:
derselbe Eingabetext lässt sich mit und ohne Vorfilter laufen. Wege: GUI-Kontrollkästchen
und CLI-Flag setzen die Lauf-Konfiguration; agentisch liest der Schritt sie aus dem
Kontext, klassisch geht ``decision_prefilter_top_k=0`` an den Executor.
"""

from __future__ import annotations

import argparse
import re
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

from src.core.agents.shared_context import SharedContext
from src.core.pipeline_manager import PipelineConfig
from src.ui._pipeline_tab_control import PipelineTabControlMixin
from tests.test_decision_prefilter import _Fake, _ServerCase


class AgenticSwitchTest(_ServerCase):
    """Agentischer Anschluss mit dem Lauf-Schalter."""

    def _run_with_flag(self, flag):
        # run_step baut den Kontext selbst; der Schalter wird über eine Unterklasse gesetzt
        from unittest.mock import patch
        from src.core.agent_loop import AgentResult
        from tests.test_decision_prefilter_integration import POOL, _step

        seen = []
        loop = MagicMock()
        loop.run.side_effect = lambda **kw: (seen.append(kw["user_prompt"]) or
                                             AgentResult(content='{"keywords": []}', tool_log=[], iterations=1))
        ctx = SharedContext(abstract="Schwermetall im Boden", provider="p", model="m")
        ctx.use_decision_prefilter = flag
        step = _step({"enabled": True, "top_k": 30}, 100)
        notes = []
        step.stream_callback = notes.append
        cm = MagicMock()
        cm.return_value.get_unified_config.return_value = self.config()
        with patch("src.core.agents.steps.llm_agent_step.AgentLoop", return_value=loop), \
                patch("src.utils.config_manager.ConfigManager", cm):
            step._run_chunked(step.config.raw, {"abstract": ctx.abstract, "keywords": POOL},
                              step.config.raw["chunking"], ctx)
        return seen, ctx, notes

    def test_off_means_the_full_pool_and_no_request(self):
        seen, ctx, notes = self._run_with_flag(False)
        self.assertEqual(_Fake.calls, [])
        self.assertNotIn("decision_prefilter", ctx.extra)
        self.assertEqual(len(seen), 3)                       # 300 Items in 3 Chunks zu 100
        self.assertTrue(any("für diesen Lauf aus" in n for n in notes))

    def test_on_is_the_default_behaviour(self):
        seen, ctx, _ = self._run_with_flag(True)
        self.assertEqual(len(seen), 1)
        self.assertEqual(ctx.extra["decision_prefilter"]["n_out"], 30)


class ConfigAndContextTest(unittest.TestCase):
    def test_default_is_on(self):
        self.assertTrue(PipelineConfig().use_decision_prefilter)
        self.assertTrue(SharedContext(abstract="a").use_decision_prefilter)

    def test_the_flag_survives_a_warm_start_and_old_files_mean_on(self):
        ctx = SharedContext(abstract="a")
        ctx.use_decision_prefilter = False
        self.assertFalse(SharedContext.from_dict(ctx.to_dict()).use_decision_prefilter)
        old = ctx.to_dict()
        old.pop("use_decision_prefilter")
        self.assertTrue(SharedContext.from_dict(old).use_decision_prefilter)

    def test_cli_flag_switches_it_off_and_absence_leaves_it_on(self):
        from src.utils.pipeline_config_builder import PipelineConfigBuilder

        for flag, expected in ((True, False), (False, True)):
            builder = MagicMock()
            builder.baseline = PipelineConfig()
            builder.apply_overrides.side_effect = lambda _o: builder.baseline
            cfg = PipelineConfigBuilder.parse_and_apply_cli_args(
                builder, argparse.Namespace(no_decision_prefilter=flag))
            self.assertEqual(cfg.use_decision_prefilter, expected, flag)

    def test_the_cli_parser_knows_the_flag(self):
        src = Path("src/cli/main.py").read_text(encoding="utf-8")
        self.assertRegex(src, r'pipeline_parser\.add_argument\("--no-decision-prefilter"')

    def test_the_classic_keywords_step_hands_zero_to_the_executor_when_off(self):
        """Quelltext-Wächter: die Verdrahtung im Schlüsselwort-Schritt (der Schritt selbst
        ist zu schwer für einen Stub-Test)."""
        src = Path("src/core/_pipeline_classic_steps.py").read_text(encoding="utf-8")
        self.assertRegex(src, r'if not getattr\(self\.config, "use_decision_prefilter", True\):\s+'
                              r'filtered_config\["decision_prefilter_top_k"\] = 0')


class _Check:
    def __init__(self, checked=True, enabled=True):
        self.checked, self.enabled, self.tip = checked, enabled, ""

    isChecked = lambda self: self.checked          # noqa: E731
    isEnabled = lambda self: self.enabled          # noqa: E731

    def setEnabled(self, v):
        self.enabled = v

    def setToolTip(self, t):
        self.tip = t


class GuiTickTest(unittest.TestCase):
    def _tab(self, check, role=True):
        tab = SimpleNamespace(decision_prefilter_check=check, pipeline_manager=SimpleNamespace(config=PipelineConfig()))
        tab._decision_role_configured = lambda: role
        for name in ("_apply_decision_prefilter_from_gui", "_refresh_decision_check"):
            setattr(tab, name, getattr(PipelineTabControlMixin, name).__get__(tab))
        return tab

    def test_an_unticked_box_switches_it_off_for_the_run(self):
        tab = self._tab(_Check(checked=False))
        tab._apply_decision_prefilter_from_gui()
        self.assertFalse(tab.pipeline_manager.config.use_decision_prefilter)

    def test_a_ticked_box_leaves_it_on(self):
        tab = self._tab(_Check(checked=True))
        tab._apply_decision_prefilter_from_gui()
        self.assertTrue(tab.pipeline_manager.config.use_decision_prefilter)

    def test_a_disabled_box_has_no_effect_whatever_it_shows(self):
        tab = self._tab(_Check(checked=False, enabled=False))
        tab._apply_decision_prefilter_from_gui()
        self.assertTrue(tab.pipeline_manager.config.use_decision_prefilter)

    def test_the_box_is_enabled_only_with_a_configured_decision_model(self):
        check = _Check(enabled=False)
        tab = self._tab(check, role=True)
        tab._refresh_decision_check()
        self.assertTrue(check.enabled)
        self.assertIn("Vergleich", check.tip)
        tab = self._tab(check, role=False)
        tab._refresh_decision_check()
        self.assertFalse(check.enabled)
        self.assertIn("Entscheidungsmodell eingerichtet", check.tip)

    def test_the_tab_builds_the_box_and_applies_it_before_a_run(self):
        src = Path("src/ui/pipeline_tab.py").read_text(encoding="utf-8")
        self.assertIn('self.decision_prefilter_check = QCheckBox("mit Decision")', src)
        ctl = Path("src/ui/_pipeline_tab_control.py").read_text(encoding="utf-8")
        self.assertRegex(ctl, r"_apply_global_override_from_gui\(\)\s+self\._apply_decision_prefilter_from_gui\(\)")


if __name__ == "__main__":
    unittest.main()
