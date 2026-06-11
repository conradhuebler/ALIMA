#!/usr/bin/env python3
"""Claude Generated - Tests for the finc_subject_harvest deterministic function.

Covers the opt-in gate, the finc Subject harvest + context storage, local-GND
reconciliation of record subjects, and the merge into context.gnd_entries
(new entries appended; existing titles gain 'finc' as a confirming source).
"""

import json
import types
import unittest

from src.core.agents.deterministic_functions import finc_subject_harvest


class _FakeRegistry:
    """Minimal tool registry dispatching search_finc / search_gnd."""

    def __init__(self, finc_results=None, gnd_map=None, finc_error=None):
        self.finc_results = finc_results or {}
        self.gnd_map = gnd_map or {}
        self.finc_error = finc_error
        self.calls = []

    def execute(self, name, args):
        self.calls.append((name, args))
        if name == "search_finc":
            if self.finc_error:
                return json.dumps({"error": self.finc_error})
            term = args["terms"][0]
            block = self.finc_results.get(term, {"records": [], "result_count": 0, "facets": {}})
            return json.dumps({"source": "finc", "results": {term: block}, "errors": {}})
        if name == "search_gnd":
            term = args["term"]
            entries = self.gnd_map.get(term, [])
            return json.dumps({"term": term, "count": len(entries), "entries": entries})
        raise ValueError(f"unexpected tool {name}")


def _ctx(gnd_entries=None):
    return types.SimpleNamespace(gnd_entries=list(gnd_entries or []), extra={})


_FINC_RESULTS = {
    "Quantenmechanik": {
        "records": [
            {"id": "a", "title": "QM Buch", "subjects": [["Quantenmechanik", "Physik"]]},
            {"id": "b", "title": "QM 2", "subjects": [["Quantenmechanik"]]},
        ],
        "result_count": 2,
        "facets": {"udk_raw_de105": [{"value": "dk 530.145", "count": 2}], "rvk_facet": []},
    },
}
_GND_MAP = {
    "Quantenmechanik": [{"gnd_id": "G1", "title": "Quantenmechanik", "description": "d",
                         "synonyms": [], "ddcs": ["530"]}],
    "Physik": [{"gnd_id": "G2", "title": "Physik", "description": "", "synonyms": [], "ddcs": []}],
}


class TestFincSubjectHarvest(unittest.TestCase):

    def test_disabled_is_noop(self):
        reg = _FakeRegistry(_FINC_RESULTS, _GND_MAP)
        ctx = _ctx([{"title": "X", "sources": ["swb"], "source_count": 1}])
        out = finc_subject_harvest(["Quantenmechanik"], tool_registry=reg, context=ctx,
                                   config={"enabled": False})
        self.assertFalse(out["enabled"])
        self.assertEqual(reg.calls, [])                 # no finc/gnd calls
        self.assertEqual(len(ctx.gnd_entries), 1)       # pool untouched
        self.assertNotIn("finc_harvest", ctx.extra)

    def test_harvest_reconcile_and_merge(self):
        reg = _FakeRegistry(_FINC_RESULTS, _GND_MAP)
        # Pre-existing swb/lobid pool already contains "Quantenmechanik"
        ctx = _ctx([{"title": "Quantenmechanik", "gnd_ids": ["G1"], "gnd_id": "G1",
                     "sources": ["swb", "lobid"], "source_count": 2}])
        out = finc_subject_harvest(["Quantenmechanik"], tool_registry=reg, context=ctx,
                                   config={"enabled": True})
        self.assertTrue(out["enabled"])
        # Both subjects reconciled (Quantenmechanik + Physik)
        self.assertEqual(out["subjects_reconciled"], 2)
        # Physik is new -> appended; Quantenmechanik already present -> not re-added
        self.assertEqual(out["merged_added"], 1)
        titles = {e["title"] for e in ctx.gnd_entries}
        self.assertEqual(titles, {"Quantenmechanik", "Physik"})
        # Existing entry gained 'finc' as confirming source (rank boost)
        qm = next(e for e in ctx.gnd_entries if e["title"] == "Quantenmechanik")
        self.assertIn("finc", qm["sources"])
        self.assertEqual(qm["source_count"], 3)
        # New entry carries the finc subject frequency as its count
        phys = next(e for e in ctx.gnd_entries if e["title"] == "Physik")
        self.assertEqual(phys["gnd_id"], "G2")
        self.assertEqual(phys["sources"], ["finc"])
        # Harvest stored for the DK step (titles + DK facet distribution)
        self.assertIn("Quantenmechanik", ctx.extra["finc_harvest"])
        h = ctx.extra["finc_harvest"]["Quantenmechanik"]
        self.assertEqual(len(h["records"]), 2)
        self.assertEqual(h["dk_dist"][0]["value"], "dk 530.145")

    def test_unreconciled_subjects_are_dropped(self):
        # finc returns a subject with no GND match -> not added to the pool
        finc = {"Photokatalyse": {"records": [{"id": "x", "title": "T",
                "subjects": [["Nichtskonzept"]]}], "result_count": 1, "facets": {}}}
        reg = _FakeRegistry(finc, gnd_map={})  # empty GND map -> nothing reconciles
        ctx = _ctx()
        out = finc_subject_harvest(["Photokatalyse"], tool_registry=reg, context=ctx,
                                   config={"enabled": True})
        self.assertEqual(out["subjects_reconciled"], 0)
        self.assertEqual(out["merged_added"], 0)
        self.assertEqual(ctx.gnd_entries, [])

    def test_finc_error_breaks_gracefully(self):
        reg = _FakeRegistry(finc_error="FincSuggester not configured")
        ctx = _ctx()
        out = finc_subject_harvest(["Quantenmechanik", "Thermodynamik"],
                                   tool_registry=reg, context=ctx, config={"enabled": True})
        self.assertTrue(out["enabled"])
        self.assertEqual(out["subjects_reconciled"], 0)
        # stops after the first search_finc error (no second keyword call, no search_gnd)
        self.assertEqual([c[0] for c in reg.calls], ["search_finc"])


if __name__ == "__main__":
    unittest.main()
