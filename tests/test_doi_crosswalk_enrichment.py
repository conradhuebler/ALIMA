"""Tests for the DOI→K10plus catalog enrichment (WP-D1 adoption) - Claude Generated.

A DOI input resolves to analysis text as before; additionally the K10plus
record is fetched so its classifications (P2 priors) and GND subjects (P3
signals) flow into the pipeline. Gate: the k10plus lookup plugin — disabled
means no enrichment, no extra request.
"""

from __future__ import annotations

import logging
import unittest
from unittest.mock import MagicMock, patch

from src.core.bib_record import BibRecord
from src.utils.input_sources.bib_lookup import crosswalk_doi_record
from src.utils.k10plus_resolver import K10PlusRecord

_DDC = {"DDC": [{"code": "571.954662", "origin": "authority"}]}


def _k10_record():
    return K10PlusRecord(
        ppn="1921841427", isbn="9783031473906", title="Cadmium Toxicity Mitigation",
        ddc="571.954662", subjects=["Cadmium"],
    )


class TestCrosswalkDoiRecord(unittest.TestCase):
    def test_disabled_plugin_skips_without_fetching(self):
        with patch("src.utils.lookups.resolve.build_lookup", return_value=None), \
             patch("src.utils.k10plus_resolver.fetch_record_for_identifier") as fetch:
            self.assertIsNone(crosswalk_doi_record("10.1007/978-3-031-47390-6"))
        fetch.assert_not_called()

    def test_doi_hit_returns_canonical_bibrecord(self):
        with patch("src.utils.lookups.resolve.build_lookup", return_value=object()), \
             patch("src.utils.k10plus_resolver.fetch_record_for_identifier",
                   return_value=_k10_record()) as fetch:
            rec = crosswalk_doi_record("10.5040/9781350067417")
        self.assertEqual(fetch.call_args.kwargs.get("kind"), "doi")
        self.assertEqual(rec.classifications, _DDC)
        self.assertEqual(rec.identifiers["ppn"], "1921841427")

    def test_doi_miss_falls_back_to_embedded_isbn(self):
        """Springer book DOIs embed the ISBN; many K10plus records carry no
        DOI field — the fallback is what makes them reachable."""
        with patch("src.utils.lookups.resolve.build_lookup", return_value=object()), \
             patch("src.utils.k10plus_resolver.fetch_record_for_identifier",
                   side_effect=[None, _k10_record()]) as fetch:
            rec = crosswalk_doi_record("10.1007/978-3-031-47390-6")
        self.assertIsNotNone(rec)
        self.assertEqual(fetch.call_count, 2)
        second = fetch.call_args_list[1]
        self.assertEqual(second.args[0], "9783031473906")
        self.assertEqual(second.kwargs.get("kind"), "isbn")

    def test_non_isbn_suffix_does_not_fall_back(self):
        with patch("src.utils.lookups.resolve.build_lookup", return_value=object()), \
             patch("src.utils.k10plus_resolver.fetch_record_for_identifier",
                   return_value=None) as fetch:
            self.assertIsNone(crosswalk_doi_record("10.1002/cmtd.202200006"))
        self.assertEqual(fetch.call_count, 1)

    def test_errors_yield_none(self):
        with patch("src.utils.lookups.resolve.build_lookup", side_effect=OSError("down")):
            self.assertIsNone(crosswalk_doi_record("10.1/x"))


class TestClassicInputStepEnrichment(unittest.TestCase):
    """GUI/CLI resolve the DOI to text before start; input_type/source_value
    carry the identity — the classic input step runs the crosswalk."""

    def _pm(self, input_type="doi", source_value="10.5040/9781350067417"):
        from src.core.data_models import KeywordAnalysisState
        from src.core.pipeline_manager import PipelineManager

        pm = PipelineManager(
            alima_manager=MagicMock(), cache_manager=MagicMock(),
            logger=logging.getLogger("test_doi_enrichment"),
        )
        pm.current_analysis_state = KeywordAnalysisState(
            original_abstract="Aufgelöster Abstract-Text.",
            initial_keywords=[], search_suggesters_used=[],
            input_type=input_type, source_value=source_value,
        )
        return pm

    def _step(self):
        from src.core.pipeline_manager import PipelineStep

        return PipelineStep(step_id="input", name="Input", input_data={"type": "text"})

    def test_doi_input_fills_prior_channels(self):
        pm = self._pm()
        enriched = BibRecord(
            source="k10plus", classifications=dict(_DDC),
            gnd_subjects=[{"term": "Cadmium", "gnd_id": "4009274-4"}],
        )
        with patch(
            "src.utils.input_sources.bib_lookup.crosswalk_doi_record",
            return_value=enriched,
        ) as cw:
            ok = pm._execute_input_step(self._step())
        self.assertTrue(ok)
        cw.assert_called_once_with("10.5040/9781350067417", logger=pm.logger)
        self.assertEqual(pm.current_analysis_state.input_record_classifications, _DDC)
        self.assertEqual(
            pm.current_analysis_state.input_record_gnd_subjects,
            [{"term": "Cadmium", "gnd_id": "4009274-4"}],
        )

    def test_non_doi_input_does_not_crosswalk(self):
        pm = self._pm(input_type="text", source_value=None)
        with patch(
            "src.utils.input_sources.bib_lookup.crosswalk_doi_record"
        ) as cw:
            ok = pm._execute_input_step(self._step())
        self.assertTrue(ok)
        cw.assert_not_called()

    def test_miss_leaves_channels_empty_and_step_green(self):
        pm = self._pm()
        with patch(
            "src.utils.input_sources.bib_lookup.crosswalk_doi_record",
            return_value=None,
        ):
            ok = pm._execute_input_step(self._step())
        self.assertTrue(ok)
        self.assertEqual(pm.current_analysis_state.input_record_classifications, {})


class TestBatchDoiEnrichment(unittest.TestCase):
    def test_metadata_gains_prior_channels(self):
        from src.utils.batch_processor import BatchProcessor, BatchSource, SourceType

        proc = object.__new__(BatchProcessor)
        proc.logger = logging.getLogger("test_batch_doi")

        class _Resolver:
            def __init__(self, *a, **k):
                pass

            def resolve(self, s):
                return True, {"title": "T", "abstract": "Langer Abstract."}, "Langer Abstract."

        enriched = BibRecord(source="k10plus", classifications=dict(_DDC))
        with patch("src.utils.doi_resolver.UnifiedResolver", _Resolver), \
             patch("src.utils.input_sources.bib_lookup.crosswalk_doi_record",
                   return_value=enriched):
            text, metadata = proc._resolve_source_to_text(
                BatchSource(SourceType.DOI, "10.5040/9781350067417")
            )
        self.assertIn("Langer Abstract.", text)
        self.assertEqual(metadata["classifications"], _DDC)
        self.assertNotIn("gnd_subjects", metadata)  # leer → weggelassen


class TestDoiInputSourceSink(unittest.TestCase):
    def _resolver(self):
        class _Resolver:
            def __init__(self, *a, **k):
                pass

            def resolve(self, s):
                return True, {}, "ABSTRACT"

        return _Resolver

    def test_sink_receives_record(self):
        from src.utils.input_sources import get_input_source

        sink = {}
        enriched = BibRecord(source="k10plus", classifications=dict(_DDC))
        with patch("src.utils.doi_resolver.UnifiedResolver", self._resolver()), \
             patch("src.utils.input_sources.bib_lookup.crosswalk_doi_record",
                   return_value=enriched):
            text, _info, _method = get_input_source("doi_crossref")().extract(
                "10.5040/9781350067417", record_sink=sink
            )
        self.assertEqual(text, "ABSTRACT")
        self.assertIs(sink["record"], enriched)

    def test_without_sink_no_crosswalk_request(self):
        from src.utils.input_sources import get_input_source

        with patch("src.utils.doi_resolver.UnifiedResolver", self._resolver()), \
             patch("src.utils.input_sources.bib_lookup.crosswalk_doi_record") as cw:
            get_input_source("doi_crossref")().extract("10.5040/9781350067417")
        cw.assert_not_called()


if __name__ == "__main__":
    unittest.main()
