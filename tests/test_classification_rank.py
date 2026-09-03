"""Core vs. additional notation, per classification system.

Both models filled the notation list to the prompt's ceiling of ten and gave no
indication which notation actually carries the work. The classification step now
assigns a ``rank`` per entry, separately for DK, DDC and RVK, and that rank
survives from the LLM response through the state into every rendering path.

A model that omits the field leaves entries unranked; no core is invented.
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from src.core.agents.shared_context import SharedContext
from src.core.data_models import KeywordAnalysisState
from src.utils.classification_systems import (
    RANK_ADDITIONAL,
    RANK_CORE,
    normalize_rank,
    rank_sort_key,
)
from src.utils.pipeline_formatters import PipelineResultFormatter
from src.webapp.result_serialization import (
    build_structured_classifications,
    extract_results_from_analysis_state,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ["alima_v51.yaml", "alima_v51_105.yaml"]


class TestRankVocabulary(unittest.TestCase):
    def test_german_and_english_wording_both_map(self):
        for value in ("core", "Kern", "KERNNOTATION", "primary", "haupt"):
            self.assertEqual(normalize_rank(value), RANK_CORE, value)
        for value in ("additional", "Zusatz", "zusatznotation", "secondary"):
            self.assertEqual(normalize_rank(value), RANK_ADDITIONAL, value)

    def test_unknown_wording_stays_unranked(self):
        """Never guess a bucket — an unrecognised rank means no rank."""
        for value in ("", None, "vielleicht", 7):
            self.assertIsNone(normalize_rank(value), value)

    def test_sort_key_puts_core_first_and_unranked_last(self):
        entries = [{"rank": None}, {"rank": RANK_ADDITIONAL}, {"rank": RANK_CORE}]
        self.assertEqual(
            [e["rank"] for e in sorted(entries, key=rank_sort_key)],
            [RANK_CORE, RANK_ADDITIONAL, None],
        )


class TestRankSurvivesTheSerializer(unittest.TestCase):
    def test_rank_is_normalized_on_structured_entries(self):
        entries = build_structured_classifications([
            {"system": "DK", "code": "620.1", "display": "DK 620.1", "rank": "Kern"},
            {"system": "DK", "code": "666.1", "display": "DK 666.1", "rank": "zusatz"},
            {"system": "DK", "code": "54", "display": "DK 54", "rank": "unfug"},
        ])
        self.assertEqual(entries[0]["rank"], RANK_CORE)
        self.assertEqual(entries[1]["rank"], RANK_ADDITIONAL)
        self.assertNotIn("rank", entries[2])

    def test_payload_prefers_the_ranked_entries(self):
        state = KeywordAnalysisState(
            original_abstract="x", initial_keywords=[], search_suggesters_used=[]
        )
        state.dk_classifications = ["DK 620.1", "DK 666.1"]
        state.classification_entries = [
            {"system": "DK", "code": "620.1", "display": "DK 620.1", "rank": "core"},
            {"system": "DK", "code": "666.1", "display": "DK 666.1", "rank": "additional"},
        ]
        results = extract_results_from_analysis_state(state)
        self.assertEqual(
            [c.get("rank") for c in results["classifications"]], ["core", "additional"]
        )

    def test_payload_falls_back_to_the_flat_list(self):
        """Classic runs and older sessions have no ranked entries."""
        state = KeywordAnalysisState(
            original_abstract="x", initial_keywords=[], search_suggesters_used=[]
        )
        state.dk_classifications = ["DK 620.1"]
        results = extract_results_from_analysis_state(state)
        self.assertEqual(results["classifications"][0]["display"], "DK 620.1")
        self.assertNotIn("rank", results["classifications"][0])


class TestSharedContextBuildsRankedEntries(unittest.TestCase):
    def _context(self, classifications):
        ctx = SharedContext(abstract="x")
        ctx.dk_classifications = classifications
        return ctx

    def test_prefix_and_rank_are_split_out(self):
        ctx = self._context([
            {"code": "DK 620.1", "type": "DK", "rank": "core"},
            {"code": "RVK ZM 3000", "type": "RVK", "rank": "Kern"},
        ])
        entries = ctx.to_keyword_analysis_state().classification_entries
        self.assertEqual(
            entries,
            [
                {"system": "DK", "code": "620.1", "display": "DK 620.1", "rank": "core"},
                {"system": "RVK", "code": "ZM 3000", "display": "RVK ZM 3000", "rank": "core"},
            ],
        )

    def test_type_field_fills_in_for_an_unprefixed_code(self):
        ctx = self._context([{"code": "620.1", "type": "DK"}])
        entry = ctx.to_keyword_analysis_state().classification_entries[0]
        self.assertEqual(entry["system"], "DK")
        self.assertEqual(entry["display"], "DK 620.1")
        self.assertNotIn("rank", entry)

    def test_flat_code_list_is_unaffected(self):
        ctx = self._context([{"code": "DK 620.1", "type": "DK", "rank": "core"}])
        self.assertEqual(ctx.to_keyword_analysis_state().dk_classifications, ["DK 620.1"])


class TestRankRendering(unittest.TestCase):
    ENTRIES = [
        {"system": "DK", "code": "666.1", "display": "DK 666.1", "rank": "additional"},
        {"system": "DK", "code": "620.1", "display": "DK 620.1", "rank": "core"},
        {"system": "RVK", "code": "ZM 3000", "display": "RVK ZM 3000", "rank": "core"},
        {"system": "DK", "code": "54", "display": "DK 54"},
    ]

    def test_badge_card_sorts_core_first_per_system(self):
        entries = PipelineResultFormatter.normalize_classifications(self.ENTRIES)
        html = PipelineResultFormatter.format_classification_badge_card_html(entries)
        codes = re.findall(r'classification-entry__code">([^<]*)</span>', html)
        # DK first (it appears first), core before additional before unranked;
        # RVK follows with its own core.
        self.assertEqual(codes, ["DK 620.1", "DK 666.1", "DK 54", "RVK ZM 3000"])
        self.assertEqual(re.findall(r">(Kern|Zusatz)<", html), ["Kern", "Zusatz", "Kern"])

    def test_plain_text_marks_the_core(self):
        state = KeywordAnalysisState(
            original_abstract="x", initial_keywords=[], search_suggesters_used=[]
        )
        state.classification_entries = self.ENTRIES
        _html, plain = PipelineResultFormatter.format_dk_classifications_card_html(state)
        self.assertEqual(plain, "DK 620.1 (Kern), DK 666.1, DK 54, RVK ZM 3000 (Kern)")

    def test_confidence_card_shows_the_rank_and_still_finds_titles(self):
        flat = [{"dk": "620.1", "classification_type": "DK",
                 "titles": ["Einführung in die Werkstoffkunde"], "count": 64}]
        html = PipelineResultFormatter.format_dk_classifications_html(
            [{"system": "DK", "code": "620.1", "display": "DK 620.1", "rank": "core"}], flat
        )
        self.assertIn("DK 620.1 · Kern", html)
        self.assertIn("Einführung in die Werkstoffkunde", html)

    def test_confidence_card_still_accepts_plain_strings(self):
        flat = [{"dk": "620.1", "classification_type": "DK", "titles": ["T"], "count": 1}]
        html = PipelineResultFormatter.format_dk_classifications_html(["DK 620.1"], flat)
        self.assertIn("DK 620.1", html)
        self.assertNotIn("·", html)


class TestPromptAndToolLogging(unittest.TestCase):
    def test_both_classification_prompts_ask_for_the_rank(self):
        for name in WORKFLOWS:
            text = (REPO_ROOT / "workflows" / name).read_text(encoding="utf-8")
            self.assertIn("Kern und Zusatz trennen", text, name)
            self.assertIn('"rank": "core"', text, name)
            self.assertIn('"rank": "additional"', text, name)
            self.assertIn("je System getrennt", text, name)

    def test_core_count_follows_the_work_not_a_number(self):
        """A ceiling gets read as a target; a hard "one" fights house practice.

        In the Freiberg stock DK 620.22 and DK 620.1 sit on 32 shared titles,
        so a work legitimately has both as its core. The rule names a criterion
        and a relative guard instead of a count.
        """
        for name in WORKFLOWS:
            text = (REPO_ROOT / "workflows" / name).read_text(encoding="utf-8")
            self.assertIn("gemeinsam an denselben Titeln", text, name)
            self.assertIn("Ist mehr als die Hälfte deiner Notationen `core`", text, name)
            self.assertNotIn("Pro System höchstens zwei", text, name)
            self.assertNotIn("Genau eine je System", text, name)

    def test_core_has_a_checkable_criterion(self):
        for name in WORKFLOWS:
            text = (REPO_ROOT / "workflows" / name).read_text(encoding="utf-8")
            self.assertIn("Prüfe die Kernnotation an ihren Titeln", text, name)
            self.assertIn("Zu weit ist genauso falsch wie zu eng", text, name)

    def test_rvk_shortlist_is_a_proposal_not_a_takeover_list(self):
        """One model took the top entry, another copied all five."""
        for name in WORKFLOWS:
            text = (REPO_ROOT / "workflows" / name).read_text(encoding="utf-8")
            self.assertIn("Vorschlagsliste, keine Übernahmeliste", text, name)
            self.assertIn("Allgemeine Lehrbücher", text, name)

    def test_tool_results_are_logged(self):
        """A run's log recorded that a tool ran but never what it returned."""
        source = (REPO_ROOT / "src" / "core" / "agent_loop.py").read_text(encoding="utf-8")
        self.assertIn('f"  ↩️ {tc.name} → {result_str[:500]}"', source)


if __name__ == "__main__":
    unittest.main()


class TestRvkLookupAcceptsKeywordObjects(unittest.TestCase):
    """The classification prompt lists the keywords as {keyword, gnd_id} objects.

    A model passing those back is the normal case. ``str(dict)`` turned them
    into "{'keyword': 'Werkstoffkunde', 'gnd_id': '4079184-1'}", the catalog
    search found nothing, and the tool returned an empty shortlist without an
    error — visible only once tool results reached the log.
    """

    def _drive(self, keywords):
        from unittest.mock import patch

        from src.mcp.tool_registry import ToolRegistry

        seen = {}

        class _FakeExecutor:
            def __init__(self, **_):
                pass

            def _derive_rvk_anchor_keywords(self, kws, original_abstract=""):
                seen["terms"] = list(kws)
                return []

            def execute_notation_search(self, keywords=None, **_):
                return {"classifications": []}

            def prepare_notation_classification_context(self, *_a, **_kw):
                return {"results_with_titles": []}

            def _build_rvk_scoring_shortlist(self, *_a, **_kw):
                return []

        registry = ToolRegistry.__new__(ToolRegistry)
        registry._config_manager = None
        registry._get_knowledge_manager = lambda: None
        with patch("src.utils.pipeline_utils.PipelineStepExecutor", _FakeExecutor):
            registry._handle_rvk_lookup(keywords=keywords, abstract="x")
        return seen.get("terms", [])

    def test_dicts_become_searchable_terms(self):
        terms = self._drive([
            {"keyword": "Werkstoffkunde", "gnd_id": "4079184-1"},
            {"keyword": "Recycling", "gnd_id": ""},
        ])
        self.assertEqual(terms, ["Werkstoffkunde (GND-ID: 4079184-1)", "Recycling"])

    def test_plain_strings_are_unchanged(self):
        self.assertEqual(self._drive(["Kristallstruktur"]), ["Kristallstruktur"])

    def test_mixed_input_and_empty_entries(self):
        terms = self._drive([
            "Kristallstruktur",
            {"keyword": "", "gnd_id": "x"},
            {"title": "Keramik"},
            "",
            None,
        ])
        self.assertEqual(terms, ["Kristallstruktur", "Keramik"])

    def test_no_usable_keyword_returns_an_empty_shortlist(self):
        import json

        from src.mcp.tool_registry import ToolRegistry

        registry = ToolRegistry.__new__(ToolRegistry)
        registry._config_manager = None
        out = json.loads(registry._handle_rvk_lookup(keywords=[{"gnd_id": "x"}]))
        self.assertEqual(out, {"rvk": [], "count": 0})


class TestRvkShortlistCarriesLabels(unittest.TestCase):
    """A bare notation gives the classification LLM nothing to judge fit with.

    The catalog-derived candidates carry no label, so ``rvk_lookup`` returned
    entries like ``{"notation": "RVK UQ 8000", "label": null}``. The labels come
    from the same cached ``rvk_validate`` path the pipeline already uses.
    """

    def _drive(self, plugin=None, labels_raise=False, api_labels=None):
        import json
        from unittest.mock import patch

        from src.mcp.tool_registry import ToolRegistry

        class _FakeExecutor:
            cache_manager = None

            def __init__(self, **_):
                pass

            def _alima_config_for_cache(self):
                return object()

            def _derive_rvk_anchor_keywords(self, kws, original_abstract=""):
                return ["Werkstoffkunde"]

            def execute_notation_search(self, **_):
                return {"classifications": []}

            def prepare_notation_classification_context(self, *_a, **_kw):
                return {"results_with_titles": []}

            def _build_rvk_scoring_shortlist(self, *_a, **_kw):
                return [{"dk": "ZM 3000", "count": 13}, {"dk": "UQ 8000", "count": 6}]

        def _cached_call(_km, _on, _key, code, _extra, fn):
            return fn()

        def _build_lookup(_config, _id):
            if labels_raise:
                raise RuntimeError("plugin exploded")
            return plugin

        def _api(code):
            return {"label": (api_labels or {}).get(code, "")}

        registry = ToolRegistry.__new__(ToolRegistry)
        registry._config_manager = None
        registry._get_knowledge_manager = lambda: None
        with patch("src.utils.pipeline_utils.PipelineStepExecutor", _FakeExecutor), \
                patch("src.utils.lookups.cache.cached_call", _cached_call), \
                patch("src.utils.lookups.cache.lookup_cache_enabled", lambda *_a: False), \
                patch("src.utils.lookups.resolve.build_lookup", _build_lookup), \
                patch("src.webapp.result_serialization.validate_rvk_notation", _api):
            return json.loads(registry._handle_rvk_lookup(keywords=["Werkstoffkunde"]))

    def test_labels_are_filled_from_the_validation_plugin(self):
        class _Plugin:
            @staticmethod
            def validate_notation(code):
                return {"result": {
                    "label": {"ZM 3000": "Allgemeine Darstellungen zur Werkstoffwissenschaft",
                              "UQ 8000": "Allgemeine Lehrbücher"}[code],
                    "ancestor_path": "Technik > Werkstoffwissenschaft",
                }}

        out = self._drive(plugin=_Plugin())
        by_notation = {e["notation"]: e for e in out["rvk"]}
        self.assertEqual(
            by_notation["RVK ZM 3000"]["label"],
            "Allgemeine Darstellungen zur Werkstoffwissenschaft",
        )
        self.assertEqual(by_notation["RVK UQ 8000"]["label"], "Allgemeine Lehrbücher")
        self.assertEqual(
            by_notation["RVK ZM 3000"]["ancestor_path"], "Technik > Werkstoffwissenschaft"
        )

    def test_the_api_fills_in_when_the_plugin_is_disabled(self):
        """The `rvk_api` lookup plugin is optional and off in some installs."""
        out = self._drive(plugin=None, api_labels={
            "ZM 3000": "Allgemeine Darstellungen zur Werkstoffwissenschaft",
            "UQ 8000": "Allgemeine Lehrbücher",
        })
        self.assertEqual(
            [e["label"] for e in out["rvk"]],
            ["Allgemeine Darstellungen zur Werkstoffwissenschaft", "Allgemeine Lehrbücher"],
        )

    def test_no_label_source_still_returns_the_shortlist(self):
        out = self._drive(plugin=None, api_labels={})
        self.assertEqual([e["notation"] for e in out["rvk"]], ["RVK ZM 3000", "RVK UQ 8000"])
        self.assertEqual(out["rvk"][0]["label"], "")

    def test_a_failing_label_lookup_does_not_fail_the_tool(self):
        out = self._drive(labels_raise=True, api_labels={})
        self.assertEqual(out["count"], 2)
        self.assertEqual(out["rvk"][0]["label"], "")


class TestChainExamplesTeachFacetsNotHierarchy(unittest.TestCase):
    """The chain rule and its examples contradicted each other.

    "Kein Oberbegriff in derselben Kette" stood three lines below "Kombiniere
    Schlagworte zu Ketten, um Spezifität zu erhöhen (z. B. 'KI → Machine
    Learning')" — which is exactly an Oberbegriff/Unterbegriff pair. A classic
    gemma4 run on 2026-09-03 produced "Festkörperchemie → Kristallstruktur →
    Gitterbaufehler" and "Werkstoffkunde → Metall → Legierung": taxonomy paths
    in the shape the examples showed.

    Classic mode reads prompts.yaml/prompts.json, agentic mode the workflow
    YAMLs, so all four carry the rule and all four are pinned here.
    """

    SOURCES = [
        REPO_ROOT / "prompts.yaml",
        REPO_ROOT / "prompts.json",
        REPO_ROOT / "workflows" / "alima_v51.yaml",
        REPO_ROOT / "workflows" / "alima_v51_105.yaml",
    ]

    def test_the_hierarchy_example_is_gone_everywhere(self):
        for path in self.SOURCES:
            text = path.read_text(encoding="utf-8")
            self.assertNotIn("KI → Machine Learning", text, path.name)
            self.assertNotIn("KI (GND-ID) → Machine Learning", text, path.name)

    def test_chains_are_described_as_facet_combinations(self):
        for path in self.SOURCES:
            text = path.read_text(encoding="utf-8")
            self.assertIn("Ober-/Unterbegriffsfolge", text, path.name)

    def test_the_no_broader_term_rule_still_stands(self):
        """The rule the examples contradicted must survive the fix."""
        for path in self.SOURCES:
            text = path.read_text(encoding="utf-8")
            self.assertIn("Kein Oberbegriff in derselben Kette", text, path.name)

    def test_prompt_files_stay_loadable(self):
        import json

        import yaml

        yaml.safe_load((REPO_ROOT / "prompts.yaml").read_text(encoding="utf-8"))
        json.loads((REPO_ROOT / "prompts.json").read_text(encoding="utf-8"))


class TestAgenticChainsReachTheResult(unittest.TestCase):
    """Schlagwortketten were lost on every agentic run.

    ``to_keyword_analysis_state`` rendered them into ``response_full_text`` as
    prose and never onto ``final_llm_analysis.keyword_chains`` or the state, so
    ``results["keyword_chains"]`` came out empty — on all ten agentic runs of
    2026-09-03, while the classic run of the same day carried four. The GUI hid
    it: ``render_pipeline_result`` scrapes the response text for "→" lines.
    """

    CHAINS = [
        {"chain": ["Werkstoffkunde", "Stoffeigenschaft"], "reason": "Struktur-Eigenschaft"},
        {"chain": ["Recycling", "Grüne Chemie"], "reason": "Nachhaltige Produktion"},
    ]

    def _state(self):
        from src.core.agents.shared_context import SharedContext

        ctx = SharedContext(abstract="x")
        ctx.keyword_chains = list(self.CHAINS)
        ctx.extra = {"final_keywords": [{"keyword": "Werkstoffkunde", "gnd_id": "4079184-1"}]}
        return ctx.to_keyword_analysis_state()

    def test_chains_land_on_the_state_and_the_llm_analysis(self):
        state = self._state()
        self.assertEqual(state.keyword_chains, self.CHAINS)
        self.assertEqual(state.final_llm_analysis.keyword_chains, self.CHAINS)

    def test_chains_reach_the_result_payload(self):
        from src.webapp.result_serialization import extract_results_from_analysis_state

        results = extract_results_from_analysis_state(self._state())
        self.assertEqual(results["keyword_chains"], self.CHAINS)

    def test_the_prose_rendering_is_kept(self):
        """render_pipeline_result scrapes the text; do not break it."""
        text = self._state().final_llm_analysis.response_full_text
        self.assertIn("Schlagwortketten (2)", text)
        self.assertIn("Werkstoffkunde → Stoffeigenschaft", text)

    def test_no_chains_stays_empty(self):
        from src.core.agents.shared_context import SharedContext

        state = SharedContext(abstract="x").to_keyword_analysis_state()
        self.assertEqual(state.keyword_chains, [])
        self.assertNotIn("Schlagwortketten", state.final_llm_analysis.response_full_text)


class TestDegradedSelectionIsVisible(unittest.TestCase):
    """A selection step whose output could not be parsed left no trace.

    glm-5.3-flash prefixed its selection JSON with prose on 2026-09-03 11:05.
    ``extra.final_keywords`` stayed empty, the state fell back to the whole
    chunk selection, and the run finished with 56 keywords instead of ~20 —
    looking, from the outside, exactly like a good run.
    """

    def _context(self, curated):
        from src.core.agents.shared_context import SharedContext

        ctx = SharedContext(abstract="x")
        ctx.selected_keywords = [
            {"title": f"Schlagwort {i}", "gnd_id": f"{i}"} for i in range(56)
        ]
        if curated:
            ctx.extra = {"final_keywords": [{"keyword": "Werkstoffkunde", "gnd_id": "4079184-1"}]}
        return ctx

    def setUp(self):
        # Several test modules call logging.disable(CRITICAL) at import; lift it
        # for this test and restore, as test_lookup_plugins.py does.
        import logging

        self._prev_disable = logging.root.manager.disable
        logging.disable(logging.NOTSET)

    def tearDown(self):
        import logging

        logging.disable(self._prev_disable)

    def test_the_fallback_warns(self):
        with self.assertLogs("src.core.agents.shared_context", level="WARNING") as logs:
            state = self._context(curated=False).to_keyword_analysis_state()
        self.assertIn("ungefilterte Auswahl", "\n".join(logs.output))
        self.assertIn("56", "\n".join(logs.output))
        self.assertEqual(len(state.final_llm_analysis.extracted_gnd_keywords), 56)

    def test_a_curated_list_warns_about_nothing(self):
        import logging

        logger = logging.getLogger("src.core.agents.shared_context")
        with self.assertNoLogs(logger, level="WARNING"):
            state = self._context(curated=True).to_keyword_analysis_state()
        self.assertEqual(len(state.final_llm_analysis.extracted_gnd_keywords), 1)
