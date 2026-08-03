"""Tests for classic↔agentic core convergence - Claude Generated

Covers the WP-K1..K4 changes:
  * gnd_batch_search: source_count ranking + source-error propagation
  * verify_final_keywords: GND-pool verification of selection output
  * prepare_notation_classification_context: shared DK filtering/formatting
"""

import json
import logging
import unittest
from unittest.mock import MagicMock, patch

from src.core.agents.shared_context import SharedContext
from src.core.agents.deterministic_functions import (
    gnd_batch_search,
    verify_final_keywords,
)
from src.utils.pipeline_utils import PipelineStepExecutor


class _FakeRegistry:
    """Tool registry stub returning canned suggester JSON per source."""

    def __init__(self, swb=None, lobid=None, swb_error=None, lobid_error=None):
        self._swb = swb or {}
        self._lobid = lobid or {}
        self._swb_error = swb_error
        self._lobid_error = lobid_error

    def execute(self, tool, args):
        if tool == "search_swb":
            if self._swb_error:
                return json.dumps({"error": self._swb_error})
            return json.dumps({"results": self._swb, "errors": {}})
        if tool == "search_lobid":
            if self._lobid_error:
                return json.dumps({"error": self._lobid_error})
            return json.dumps({"results": self._lobid, "errors": {}})
        if tool == "aggregate_gnd_results":
            # Mirror the real aggregate engine over the canned reduced results
            # (WP2 P4.4: gnd_batch_search now builds the pool from raw). The tools
            # above "populate raw"; here we reproduce the engine deterministically.
            return json.dumps(self._aggregate(args.get("terms", []), args.get("sources", [])))
        if tool == "get_gnd_batch":
            return json.dumps({"entries": {}})
        raise RuntimeError(f"unexpected tool: {tool}")

    def _aggregate(self, terms, sources):
        from src.core.gnd_search_core import merge_into_pool, parse_batch_response, rank_pool
        from src.core.search.aggregate import _to_cache_hit_shape

        data_by_source = {"swb": self._swb, "lobid": self._lobid}
        pool = {}
        src_index = {}
        terms_map = {}
        for source in sources:
            per_term = data_by_source.get(source, {})
            for term in terms:
                reduced = per_term.get(term, {})
                if not reduced:
                    continue
                nd = parse_batch_response({"results": {term: _to_cache_hit_shape(reduced)}})
                for title in nd:
                    src_index.setdefault(title.lower(), set()).add(source)
                    terms_map.setdefault(title, set()).add(term)
                merge_into_pool(pool, nd)
        return {
            "pool": rank_pool(pool, src_index),
            "sources": list(sources),
            "missing": {},
            "terms_map": {t: sorted(v) for t, v in terms_map.items()},
        }


def _hit(gnd_id, count):
    return {"gnd_ids": [gnd_id], "count": count, "classifications": {}}


class TestGndBatchSearchConvergence(unittest.TestCase):

    def setUp(self):
        # Deterministic: don't depend on the user's SystemConfig.aggregate_from_raw.
        self._agg_patch = patch(
            "src.core.search.aggregate.default_aggregate_from_raw", return_value=True
        )
        self._agg_patch.start()
        # Likewise for the source list: since WP P6a gnd_batch_search derives its
        # default sources from the *enabled* GND providers, so enabling e.g. catalog
        # in the Plugins tab would make _FakeRegistry raise "unexpected tool". Pin
        # the two sources this fake serves. - Claude Generated
        self._src_patch = patch(
            "src.core.search.factory.enabled_gnd_provider_ids",
            return_value=["swb", "lobid"],
        )
        self._src_patch.start()

    def tearDown(self):
        self._agg_patch.stop()
        self._src_patch.stop()

    def test_source_count_ranking(self):
        """Entries confirmed by multiple sources rank first."""
        reg = _FakeRegistry(
            swb={"Cadmium": {"Cadmium": _hit("1", 5), "Schwermetall": _hit("2", 9)}},
            lobid={"Cadmium": {"Cadmium": _hit("1", 2)}},
        )
        out = gnd_batch_search(["Cadmium"], tool_registry=reg)
        self.assertEqual(out["entries"][0]["title"], "Cadmium")
        self.assertEqual(out["entries"][0]["source_count"], 2)
        self.assertEqual(out["entries"][0]["sources"], ["lobid", "swb"])
        self.assertEqual(out["entries"][1]["source_count"], 1)
        self.assertEqual(out["source_errors"], {})

    def test_partial_source_failure_is_surfaced_not_silent(self):
        """One failing source → warning + source_errors, other results kept."""
        reg = _FakeRegistry(
            swb={"Cadmium": {"Cadmium": _hit("1", 5)}},
            lobid_error="timeout",
        )
        msgs = []
        out = gnd_batch_search(
            ["Cadmium"], tool_registry=reg, stream_callback=msgs.append
        )
        self.assertEqual(len(out["entries"]), 1)
        self.assertEqual(out["source_errors"], {"lobid": "timeout"})
        self.assertTrue(any("lobid" in m and "fehlgeschlagen" in m for m in msgs))

    def test_all_sources_failed_raises(self):
        reg = _FakeRegistry(swb_error="boom", lobid_error="boom")
        with self.assertRaises(RuntimeError):
            gnd_batch_search(["Cadmium"], tool_registry=reg)

    def test_empty_pool_with_partial_failure_raises(self):
        """No hits + at least one source error → result is unreliable → fail."""
        reg = _FakeRegistry(swb={}, lobid_error="timeout")
        with self.assertRaises(RuntimeError):
            gnd_batch_search(["Cadmium"], tool_registry=reg)

    def test_empty_pool_without_errors_is_legitimate(self):
        """Genuine zero hits (no source errors) must not raise."""
        reg = _FakeRegistry(swb={}, lobid={})
        msgs = []
        out = gnd_batch_search(
            ["Xyzzy123"], tool_registry=reg, stream_callback=msgs.append
        )
        self.assertEqual(out["entries"], [])
        self.assertTrue(any("0 Treffer" in m for m in msgs))


class TestVerifyFinalKeywords(unittest.TestCase):

    def _context(self):
        ctx = SharedContext(abstract="x")
        ctx.gnd_entries = [
            {"title": "Blockchain", "gnd_id": "111", "gnd_ids": ["111"]},
            {"title": "Logistik", "gnd_id": "222", "gnd_ids": ["222"]},
        ]
        return ctx

    def _run(self, ctx, db_results=None):
        km = MagicMock()
        km.search_gnd_by_title.return_value = db_results or []
        with patch(
            "src.core.unified_knowledge_manager.UnifiedKnowledgeManager",
            return_value=km,
        ):
            return verify_final_keywords(context=ctx)

    def test_pool_match_and_gnd_id_correction(self):
        """Wrong LLM gnd_id is corrected via title match against the pool."""
        ctx = self._context()
        ctx.extra["final_keywords"] = [
            {"keyword": "Blockchain", "gnd_id": "111"},
            {"keyword": "Logistik", "gnd_id": "999"},  # falsche ID
        ]
        out = self._run(ctx)
        self.assertEqual(
            out["verified_keywords"],
            [
                {"keyword": "Blockchain", "gnd_id": "111"},
                {"keyword": "Logistik", "gnd_id": "222"},
            ],
        )
        self.assertEqual(ctx.extra["final_keywords"], out["verified_keywords"])

    def test_unknown_keyword_rejected(self):
        ctx = self._context()
        ctx.extra["final_keywords"] = [{"keyword": "Quantenphysik", "gnd_id": ""}]
        out = self._run(ctx)
        self.assertEqual(out["verified_keywords"], [])
        self.assertEqual(out["rejected"], ["Quantenphysik"])

    def test_db_fallback_attaches_authoritative_id(self):
        ctx = self._context()
        ctx.extra["final_keywords"] = [{"keyword": "Photochemie", "gnd_id": ""}]
        out = self._run(
            ctx, db_results=[{"gnd_id": "333", "title": "Photochemie"}]
        )
        self.assertEqual(
            out["verified_keywords"], [{"keyword": "Photochemie", "gnd_id": "333"}]
        )

    def test_falls_back_to_selected_keywords(self):
        """No extra.final_keywords → selected_keywords are verified instead."""
        ctx = self._context()
        ctx.selected_keywords = [{"keyword": "Blockchain", "gnd_id": ""}]
        out = self._run(ctx)
        self.assertEqual(
            out["verified_keywords"], [{"keyword": "Blockchain", "gnd_id": "111"}]
        )


class TestPrepareDkClassificationContext(unittest.TestCase):

    def setUp(self):
        self.executor = PipelineStepExecutor(
            alima_manager=None,
            cache_manager=None,
            logger=logging.getLogger("test_core_convergence"),
        )

    def test_frequency_and_title_filter(self):
        classifications = [
            {"dk": "541.14", "classification_type": "DK", "count": 5,
             "titles": ["Photochemie Grundlagen"], "matched_keywords": ["Photochemie"]},
            {"dk": "530.145", "classification_type": "DK", "count": 1,
             "titles": [], "matched_keywords": []},      # titellos → raus
            {"dk": "999.9", "classification_type": "DK", "count": 0,
             "titles": ["x"], "matched_keywords": []},   # unter Schwellwert → raus
        ]
        prep = self.executor.prepare_notation_classification_context(
            classifications, "Abstract", dk_frequency_threshold=1
        )
        codes = [r["dk"] for r in prep["results_with_titles"]]
        self.assertEqual(codes, ["541.14"])
        self.assertIn("DK: 541.14", prep["catalog_text"])
        self.assertNotIn("530.145", prep["catalog_text"])

    def test_rvk_exempt_from_frequency_filter_and_guardrail(self):
        classifications = [
            {"dk": "WC 4150", "classification_type": "RVK", "count": 0,
             "titles": ["RVK Titel"], "matched_keywords": [],
             "source": "rvk_api", "label": "Biochemie",
             "rvk_validation_status": "standard"},
        ]
        prep = self.executor.prepare_notation_classification_context(
            classifications, "Abstract", dk_frequency_threshold=5
        )
        self.assertEqual(len(prep["results_with_titles"]), 1)
        self.assertTrue(prep["allowed_standard_rvk_map"])
        self.assertTrue(prep["catalog_text"].startswith("WICHTIG FÜR RVK:"))

    def test_empty_input_yields_empty_context(self):
        prep = self.executor.prepare_notation_classification_context(
            [], "Abstract", dk_frequency_threshold=1
        )
        self.assertEqual(prep["results_with_titles"], [])
        self.assertEqual(prep["catalog_text"], "")

    # ── record_priors (WP-D1 P2) ──────────────────────────────────────────

    _DK_CANDIDATE = [
        {"dk": "541.14", "classification_type": "DK", "count": 5,
         "titles": ["Photochemie Grundlagen"], "matched_keywords": ["Photochemie"]},
    ]

    _PRIORS = {
        "DK": [{"code": "546.43", "origin": "authority"}],
        "DDC": [{"code": "551.48", "origin": "authority"}],
        "RVK": [{"code": "WI 5000", "origin": "authority"}],
    }

    def test_no_priors_is_byte_identical(self):
        base = self.executor.prepare_notation_classification_context(
            self._DK_CANDIDATE, "Abstract", dk_frequency_threshold=1
        )
        for empty in (None, {}):
            prep = self.executor.prepare_notation_classification_context(
                self._DK_CANDIDATE, "Abstract", dk_frequency_threshold=1,
                record_priors=empty,
            )
            self.assertEqual(prep, base)

    def test_priors_render_authority_block_and_allow_rvk(self):
        prep = self.executor.prepare_notation_classification_context(
            self._DK_CANDIDATE, "Abstract", dk_frequency_threshold=1,
            record_priors=self._PRIORS,
        )
        text = prep["catalog_text"]
        self.assertIn("Eingabe-Datensatz", text)
        self.assertIn("- DK: 546.43", text)
        self.assertIn("- DDC: 551.48", text)
        self.assertIn("- RVK: WI 5000", text)
        # informs, never replaces: the catalog candidates are still there
        self.assertIn("541.14", text)
        # the prior RVK is selectable (guardrails gate on this map)
        self.assertIn("WI 5000", prep["allowed_standard_rvk_map"])
        self.assertEqual(prep["rvk_source_map"]["WI 5000"]["source"], "input_record")
        # guardrail appears because an allowed standard RVK now exists
        self.assertTrue(text.startswith("WICHTIG FÜR RVK:"))

    def test_prior_does_not_promote_known_nonstandard_rvk(self):
        """Status describes the notation, not document relevance."""
        catalog = self._DK_CANDIDATE + [
            {"dk": "WI 5000", "classification_type": "RVK", "count": 1,
             "titles": ["Titel"], "matched_keywords": [], "source": "catalog",
             "rvk_validation_status": "non_standard"},
        ]
        prep = self.executor.prepare_notation_classification_context(
            catalog, "Abstract", dk_frequency_threshold=1,
            record_priors={"RVK": [{"code": "WI 5000", "origin": "authority"}]},
        )
        self.assertIn("WI 5000", prep["allowed_nonstandard_rvk_map"])
        self.assertNotIn("WI 5000", prep["allowed_standard_rvk_map"])
        # the catalog's source entry is kept, not overwritten by the prior
        self.assertEqual(prep["rvk_source_map"]["WI 5000"]["source"], "catalog")

    def test_include_rvk_false_drops_rvk_prior_but_keeps_dk(self):
        prep = self.executor.prepare_notation_classification_context(
            self._DK_CANDIDATE, "Abstract", dk_frequency_threshold=1,
            include_rvk=False, record_priors=self._PRIORS,
        )
        self.assertIn("- DK: 546.43", prep["catalog_text"])
        self.assertNotIn("WI 5000", prep["catalog_text"])
        self.assertEqual(prep["allowed_standard_rvk_map"], {})

    def test_bare_code_priors_are_tolerated(self):
        """Plugins emit bare codes; the choke-point normalisation applies."""
        prep = self.executor.prepare_notation_classification_context(
            self._DK_CANDIDATE, "Abstract", dk_frequency_threshold=1,
            record_priors={"DK": ["546.43"]},
        )
        self.assertIn("- DK: 546.43", prep["catalog_text"])


class TestDkDisplayFormatTolerance(unittest.TestCase):
    """Katalog-Recherche display: keyword-centric input must not vanish."""

    KW_CENTRIC = [
        {"keyword": "Titandioxid", "source": "catalog", "classifications": [
            {"dk": "546.824", "count": 7,
             "titles": ["TiO2-Schichten", "Photokatalyse"],
             "classification_type": "DK"},
        ]},
        {"keyword": "Adsorption", "source": "catalog", "classifications": [
            {"dk": "546.824", "count": 3, "titles": ["Oberflächenchemie"],
             "classification_type": "DK"},
        ]},
    ]

    def test_format_text_accepts_keyword_centric(self):
        from src.utils.pipeline_utils import PipelineResultFormatter
        text = PipelineResultFormatter.format_dk_search_results_text(self.KW_CENTRIC)
        self.assertIn("DK: 546.824", text)
        self.assertIn("TiO2-Schichten", text)
        self.assertIn("Titandioxid", text)  # matched keywords shown

    def test_get_titles_accepts_keyword_centric(self):
        from src.utils.pipeline_utils import PipelineResultFormatter
        titles, count = PipelineResultFormatter.get_titles_for_notation_code(
            "DK 546.824", self.KW_CENTRIC
        )
        self.assertEqual(count, 3)
        self.assertIn("Oberflächenchemie", titles)

    def test_truncation_sentinel_is_ignored(self):
        from src.utils.pipeline_utils import PipelineResultFormatter
        results = [
            {"dk": "546.824", "count": 7, "titles": ["TiO2"],
             "classification_type": "DK"},
            {"_truncated": 342},
        ]
        text = PipelineResultFormatter.format_dk_search_results_text(results)
        self.assertIn("DK: 546.824", text)


class TestDkClassificationTitleFallback(unittest.TestCase):
    """End-of-pipeline sync: DK notation title lookup must degrade gracefully
    when ``dk_search_results`` (the rich DK-centric source) has no titles —
    e.g. when ``dk_postprocess`` was skipped (v5.1 ``when``-condition) or the
    rich list was built from a thin source. Without a fallback the final
    Klassifikations-Tab shows bare code headings. - Claude Generated
    """

    def test_format_classifications_html_finds_titles_from_rich_source(self):
        """Happy path: rich source has catalog titles, classifications get them."""
        from src.utils.pipeline_utils import PipelineResultFormatter

        rich = [
            {"dk": "666.76", "classification_type": "DK", "titles": ["Halbleiter I", "Halbleiter II"]},
        ]
        html = PipelineResultFormatter.format_dk_classifications_html(
            ["666.76"], rich
        )
        self.assertIn("Halbleiter I", html)
        self.assertIn("Halbleiter II", html)

    def test_format_classifications_html_empty_rich_produces_code_headings(self):
        """No rich data: formatter shows codes but no titles (acceptable empty state)."""
        from src.utils.pipeline_utils import PipelineResultFormatter

        html = PipelineResultFormatter.format_dk_classifications_html(
            ["666.76"], []
        )
        self.assertIn("666.76", html)  # code heading is present
        self.assertNotIn("<li>", html)  # no titles — no <ol> rendered

    def test_select_dk_title_source_picks_rich_over_thin(self):
        """select_dk_title_source must return the rich list when both have titles."""
        from src.utils.pipeline_utils import PipelineResultFormatter

        rich = [{"dk": "666.76", "titles": ["A", "B", "C"]}]
        thin = [{"dk": "666.76", "titles": ["Label"]}]  # only LLM label, 1 title
        chosen = PipelineResultFormatter.select_dk_title_source(rich, thin)
        self.assertIs(chosen, rich)

    def test_fallback_when_rich_has_no_titles(self):
        """Reproduces the user-reported bug: rich list is empty/keyword-centric
        and thin list has catalog titles → sync must use the thin list for
        title lookup. - Claude Generated
        """
        from src.utils.pipeline_utils import PipelineResultFormatter

        # Simulate state.dk_search_results (keyword-centric, no usable titles —
        # the "rich has no titles" case the docstring describes). AFTER dk_collect
        # but BEFORE dk_postprocess. (select_dk_title_source now flattens the
        # keyword-centric shape first; with empty titles it scores 0.)
        keyword_centric = [
            {"keyword": "Halbleiter", "classifications": [
                {"dk": "666.76", "titles": []}
            ]}
        ]
        # state.dk_search_results_flattened from shared_context (thin, has 1 title)
        thin = [
            {"dk": "666.76", "classification_type": "DK", "titles": ["Halbleitertechnologie"]},
        ]
        chosen = PipelineResultFormatter.select_dk_title_source(keyword_centric, thin)
        # keyword-centric flattens to a dk entry with no titles → score 0;
        # thin has 1 title → score 1 > 0 → chosen.
        self.assertIs(chosen, thin)
        # The classifier code can now find a title
        titles, count = PipelineResultFormatter.get_titles_for_notation_code("666.76", chosen)
        self.assertIn("Halbleitertechnologie", titles)
        self.assertEqual(count, 1)


class TestGndTierFilterResilience(unittest.TestCase):
    """The 3-tier GND-Hit table (pool → chunk-selected → final-verified) must
    preserve both the user-set filter (combo + free text) and the per-tier
    counts when tier marks are added incrementally. Tested at the formatter
    level — the GUI layer (pipeline_tab._render_gnd_hits_table) consumes these
    helpers. - Claude Generated
    """

    def test_extract_selected_gnd_keys_for_dicts_and_strings(self):
        from src.utils.pipeline_utils import PipelineResultFormatter

        selected = [
            {"keyword": "Halbleiter", "gnd_id": "4129772-7"},
            "Quantenchemie (GND-ID: 4047610-0)",
            "Molekül",  # no GND ID — label-only fallback
        ]
        ids, labels = PipelineResultFormatter.extract_selected_gnd_keys(selected)
        self.assertIn("4129772-7", ids)
        self.assertIn("4047610-0", ids)
        self.assertIn("halbleiter", labels)
        self.assertIn("quantenchemie", labels)
        self.assertIn("molekül", labels)

    def test_flatten_gnd_hits_merges_by_id_and_keeps_max_count(self):
        """flatten_gnd_hits: same GND-ID from two search terms → one row, max count."""
        from src.utils.pipeline_utils import PipelineResultFormatter

        # Agentic gnd_entries shape — uses "search_term" (not "keyword") for the
        # source term; the function maps both labels under the hood.
        entries = [
            {"gnd_id": "4129772-7", "search_term": "Halbleiter", "title": "Halbleiter", "count": 3},
            {"gnd_id": "4129772-7", "search_term": "Chip", "title": "Halbleiter", "count": 7},
        ]
        rows = PipelineResultFormatter.flatten_gnd_hits(entries)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["gnd_id"], "4129772-7")
        self.assertEqual(rows[0]["count"], 7)  # max
        # Both search terms captured for the tooltip
        self.assertIn("Halbleiter", rows[0]["search_terms"])
        self.assertIn("Chip", rows[0]["search_terms"])


class TestClassicChunkSplitting(unittest.TestCase):
    """Chunking parity: classic threshold semantics in LLMAgentStep."""

    def test_split_semantics_match_classic(self):
        from src.core.agents.steps.llm_agent_step import _split_chunks_classic

        # ≤ threshold → single call
        self.assertEqual(len(_split_chunks_classic(list(range(500)), 500)), 1)
        # ≤ 1.5×threshold → 2 equal chunks
        self.assertEqual(
            [len(c) for c in _split_chunks_classic(list(range(600)), 500)],
            [300, 300],
        )
        # > 1.5×threshold → ceil(total/threshold) equal chunks
        self.assertEqual(
            [len(c) for c in _split_chunks_classic(list(range(1700)), 500)],
            [425, 425, 425, 425],
        )
        # No items lost or reordered
        flat = [x for c in _split_chunks_classic(list(range(1700)), 500) for x in c]
        self.assertEqual(flat, list(range(1700)))
        self.assertEqual(_split_chunks_classic([], 500), [])

    def test_auto_chunk_size_uses_model_capabilities(self):
        from src.core.agents.steps.llm_agent_step import LLMAgentStep
        from src.core.agents.steps.base_step import StepConfig

        step = LLMAgentStep(
            StepConfig(id="t", type="llm_agent", raw={}),
            llm_service=MagicMock(),
            tool_registry=None,
        )
        with patch(
            "src.utils.model_capabilities.get_chunking_threshold",
            return_value=1000,
        ) as gct:
            size = step._auto_chunk_size({"provider": "ollama", "model": "cogito:32b"})
        self.assertEqual(size, 1000)
        self.assertEqual(gct.call_args.args[:2], ("ollama", "cogito:32b"))

    def test_auto_chunk_size_fallback_500(self):
        from src.core.agents.steps.llm_agent_step import LLMAgentStep
        from src.core.agents.steps.base_step import StepConfig

        step = LLMAgentStep(
            StepConfig(id="t", type="llm_agent", raw={}),
            llm_service=MagicMock(),
            tool_registry=None,
        )
        with patch(
            "src.utils.model_capabilities.get_chunking_threshold",
            side_effect=RuntimeError("boom"),
        ):
            self.assertEqual(
                step._auto_chunk_size({"provider": "x", "model": "y"}), 500
            )


if __name__ == "__main__":
    unittest.main()
