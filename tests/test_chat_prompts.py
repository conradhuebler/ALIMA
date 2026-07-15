"""The chat system prompt must name only the search tools that exist - Claude Generated.

Regression (July 15, operator report): with every search plugin disabled, the agent
called ``search_finc`` and ``search_catalog_titles`` — neither was registered — and
then answered "Es wurden keine Treffer für Quantenchemie gefunden". Two defects in
one run:

* the prompt hardcoded the tool names regardless of the plugin config, so the model
  dutifully called a tool that was not there (the compact ruleset drove it — the
  reporting model was ``mistral-small``, which the tier heuristic maps to compact);
* nothing told the agent that *no source existed*, so a config state was reported as
  a search result — a plausible but false answer about a term that is plainly in the
  GND.
"""

from __future__ import annotations

import unittest

from src.core.chat_prompts import (
    SHARED_RULES,
    SHARED_RULES_COMPACT,
    _CATALOG_RULES_SLOT,
    build_system_prompt,
    is_compact_model,
)

# What the operator's registry held at 14:55 (all six search providers disabled).
# search_gnd survives: it reads the *local* GND store, not a provider.
TOOLS_ALL_SEARCH_OFF = {"search_gnd", "search_webindex", "rvk_search", "resolve_doi"}
TOOLS_LOBID_SWB = TOOLS_ALL_SEARCH_OFF | {"search_lobid", "search_swb"}
TOOLS_FULL = TOOLS_LOBID_SWB | {"search_finc", "search_catalog", "search_catalog_titles"}


class SlotMechanismTest(unittest.TestCase):
    """The placeholder must exist and must never survive into a prompt."""

    def test_both_rulesets_carry_the_slot(self):
        # A typo'd/removed slot would silently drop the tool rules entirely.
        self.assertIn(_CATALOG_RULES_SLOT, SHARED_RULES)
        self.assertIn(_CATALOG_RULES_SLOT, SHARED_RULES_COMPACT)

    def test_slot_never_leaks_into_a_rendered_prompt(self):
        for compact in (True, False):
            for mode in ("suche", "verschlagwortung", "general"):
                for tools in (None, TOOLS_ALL_SEARCH_OFF, TOOLS_FULL):
                    p = build_system_prompt(mode, compact=compact, available_tools=tools)
                    self.assertNotIn(_CATALOG_RULES_SLOT, p, f"{mode}/{compact}/{tools}")


class ToolNamesFollowTheConfigTest(unittest.TestCase):
    def test_disabled_tools_are_not_recommended(self):
        """The regression: don't tell the model to call a tool that isn't there."""
        for compact in (True, False):
            p = build_system_prompt("suche", compact=compact, available_tools=TOOLS_ALL_SEARCH_OFF)
            rules = p.split("Autorensuche")[0]  # the tool-choice block
            for absent in ("search_finc", "search_catalog_titles", "search_lobid", "search_swb"):
                self.assertNotIn(
                    f"`{absent}`", rules, f"compact={compact}: recommends absent {absent}"
                )

    def test_enabled_tools_are_recommended(self):
        for compact in (True, False):
            p = build_system_prompt("suche", compact=compact, available_tools=TOOLS_LOBID_SWB)
            self.assertIn("`search_lobid`", p)
            self.assertIn("`search_swb`", p)

    def test_finc_keeps_its_preference_when_present(self):
        for compact in (True, False):
            p = build_system_prompt("suche", compact=compact, available_tools=TOOLS_FULL)
            self.assertIn("`search_finc`", p)
            self.assertIn("`search_catalog_titles`", p)

    def test_none_keeps_the_static_list(self):
        """Callers without a registry (e.g. DEFAULT_SYSTEM_PROMPT) keep today's text."""
        for compact in (True, False):
            p = build_system_prompt("suche", compact=compact, available_tools=None)
            self.assertIn("`search_finc`", p)


class NoLiveSourceRuleTest(unittest.TestCase):
    """A config state must not be reported as a zero-hit search result."""

    def test_rule_fires_when_every_provider_is_off(self):
        for compact in (True, False):
            p = build_system_prompt("suche", compact=compact, available_tools=TOOLS_ALL_SEARCH_OFF)
            self.assertIn("KEINE Live-Katalog-/GND-Suchquelle", p)
            self.assertIn("NIEMALS 'keine Treffer gefunden'", p)

    def test_local_store_is_named_but_not_sold_as_a_source(self):
        p = build_system_prompt("suche", available_tools=TOOLS_ALL_SEARCH_OFF)
        self.assertIn("`search_gnd`", p)
        self.assertIn("lokalen", p)
        self.assertIn("NICHT\n  'existiert nicht'", p)

    def test_local_line_omitted_without_the_local_store(self):
        p = build_system_prompt("suche", available_tools={"search_webindex"})
        self.assertIn("KEINE Live-Katalog-/GND-Suchquelle", p)
        self.assertNotIn("`search_gnd`", p.split("Autorensuche")[0])

    def test_rule_absent_once_any_live_source_exists(self):
        for tools in (TOOLS_LOBID_SWB, TOOLS_FULL):
            for compact in (True, False):
                p = build_system_prompt("suche", compact=compact, available_tools=tools)
                self.assertNotIn("KEINE Live-Katalog-/GND-Suchquelle", p)


class ReportedRunTest(unittest.TestCase):
    """Reconstructs the operator's run end-to-end (prompt side)."""

    def test_mistral_small_gets_the_compact_prompt(self):
        # Why the compact ruleset was the one that mattered.
        self.assertTrue(is_compact_model("mistral-small-2603"))

    def test_that_run_would_no_longer_recommend_search_finc(self):
        p = build_system_prompt(
            "suche",
            compact=is_compact_model("mistral-small-2603"),
            available_tools=TOOLS_ALL_SEARCH_OFF,
        )
        self.assertNotIn("Katalogsuche: `search_finc`", p)
        self.assertIn("KEINE Live-Katalog-/GND-Suchquelle", p)


if __name__ == "__main__":
    unittest.main()
