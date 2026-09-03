"""RVK notations must come from rvk_lookup, not from the model's memory.

The classification prompt says twice to take RVK only from the tool and never
to invent one. Nothing enforced it in the agentic path: on 2026-09-03
ornith-1.5:35b made zero tool calls and returned QD 805, QD 810, T 215 and
T 216 — all four rejected by the RVK API as non-standard. The validation caught
them and the pipeline shipped them anyway.

The classic pipeline has this guard as ``_filter_final_rvk_classifications``;
``filter_unauthorized_rvk`` is its agentic twin, and it reads the tool's own
JSON rather than the model's retyping of it.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from types import SimpleNamespace

from src.core.agents.deterministic_functions import filter_unauthorized_rvk
from src.core.agents.workflow_loader import load_workflow

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ["alima_v51.yaml", "alima_v51_105.yaml"]

SHORTLIST = json.dumps({
    "rvk": [
        {"notation": "RVK ZM 3000", "count": 13},
        {"notation": "RVK ZM 3500", "count": 2},
    ],
    "count": 2,
})


def _context(classifications, tool_log=None, step_ran=True):
    step_results = {}
    if step_ran:
        step_results["classification"] = {"tool_log": tool_log or []}
    return SimpleNamespace(dk_classifications=classifications, step_results=step_results)


class TestFilterUnauthorizedRvk(unittest.TestCase):
    def test_invented_notations_are_dropped(self):
        ctx = _context(
            [
                {"code": "DK 620.22", "type": "DK", "rank": "core"},
                {"code": "RVK ZM 3000", "type": "RVK", "rank": "core"},
                {"code": "RVK QD 805", "type": "RVK", "rank": "additional"},
            ],
            [{"tool": "rvk_lookup", "result_full": SHORTLIST}],
        )
        out = filter_unauthorized_rvk(context=ctx)
        self.assertEqual(
            [c["code"] for c in out["classifications"]], ["DK 620.22", "RVK ZM 3000"]
        )
        self.assertEqual(out["dropped"], ["RVK QD 805"])

    def test_no_tool_call_means_no_authorised_rvk(self):
        """The ornith case: zero tool calls, four notations out of nowhere."""
        ctx = _context([
            {"code": "DK 620.22", "type": "DK"},
            {"code": "RVK QD 805", "type": "RVK"},
            {"code": "RVK QD 810", "type": "RVK"},
            {"code": "RVK T 215", "type": "RVK"},
            {"code": "RVK T 216", "type": "RVK"},
        ], tool_log=[])
        out = filter_unauthorized_rvk(context=ctx)
        self.assertEqual([c["code"] for c in out["classifications"]], ["DK 620.22"])
        self.assertEqual(len(out["dropped"]), 4)

    def test_dk_and_ddc_pass_through_untouched(self):
        entries = [
            {"code": "DK 620.1", "type": "DK", "rank": "core"},
            {"code": "DDC 540", "type": "DDC", "rank": "additional"},
            {"code": "DK 378.245", "type": "DK"},
        ]
        out = filter_unauthorized_rvk(context=_context(list(entries), tool_log=[]))
        self.assertEqual(out["classifications"], entries)
        self.assertEqual(out["dropped"], [])

    def test_rank_and_other_fields_survive(self):
        ctx = _context(
            [{"code": "RVK ZM 3000", "type": "RVK", "rank": "core", "note": "x"}],
            [{"tool": "rvk_lookup", "result_full": SHORTLIST}],
        )
        out = filter_unauthorized_rvk(context=ctx)
        self.assertEqual(
            out["classifications"],
            [{"code": "RVK ZM 3000", "type": "RVK", "rank": "core", "note": "x"}],
        )

    def test_notation_spacing_is_normalised(self):
        """"ZM3000" and "zm 3000" are the same notation as "ZM 3000"."""
        ctx = _context(
            [{"code": "RVK zm3000", "type": "RVK"}],
            [{"tool": "rvk_lookup", "result_full": SHORTLIST}],
        )
        out = filter_unauthorized_rvk(context=ctx)
        self.assertEqual(out["dropped"], [])

    def test_plain_string_classifications_work(self):
        ctx = _context(
            ["DK 620.1", "RVK ZM 3000", "RVK QD 805"],
            [{"tool": "rvk_lookup", "result_full": SHORTLIST}],
        )
        out = filter_unauthorized_rvk(context=ctx)
        self.assertEqual(out["classifications"], ["DK 620.1", "RVK ZM 3000"])
        self.assertEqual(out["dropped"], ["RVK QD 805"])

    def test_unrun_step_changes_nothing(self):
        """Nothing was claimed, so there is nothing to check."""
        entries = [{"code": "RVK QD 805", "type": "RVK"}]
        out = filter_unauthorized_rvk(context=_context(list(entries), step_ran=False))
        self.assertEqual(out["classifications"], entries)
        self.assertEqual(out["dropped"], [])

    def test_unparsable_tool_result_authorises_nothing(self):
        ctx = _context(
            [{"code": "RVK ZM 3000", "type": "RVK"}],
            [{"tool": "rvk_lookup", "result_full": "not json{{{"}],
        )
        out = filter_unauthorized_rvk(context=ctx)
        self.assertEqual(out["classifications"], [])
        self.assertEqual(out["dropped"], ["RVK ZM 3000"])

    def test_other_tools_in_the_log_are_ignored(self):
        ctx = _context(
            [{"code": "RVK ZM 3000", "type": "RVK"}],
            [
                {"tool": "search_finc", "result_full": json.dumps({"rvk": [{"notation": "RVK QD 805"}]})},
                {"tool": "rvk_lookup", "result_full": SHORTLIST},
            ],
        )
        out = filter_unauthorized_rvk(context=ctx)
        self.assertEqual(out["dropped"], [])

    def test_it_warns_about_what_it_dropped(self):
        import logging

        prev = logging.root.manager.disable
        logging.disable(logging.NOTSET)
        try:
            ctx = _context([{"code": "RVK QD 805", "type": "RVK"}], tool_log=[])
            with self.assertLogs(
                "src.core.agents.deterministic_functions", level="WARNING"
            ) as logs:
                filter_unauthorized_rvk(context=ctx)
        finally:
            logging.disable(prev)
        joined = "\n".join(logs.output)
        self.assertIn("QD 805", joined)
        self.assertIn("kein Tool-Aufruf", joined)

    def test_context_is_required(self):
        with self.assertRaises(RuntimeError):
            filter_unauthorized_rvk()


class TestGuardIsWiredIntoBothWorkflows(unittest.TestCase):
    def test_the_step_runs_between_classification_and_postprocess(self):
        for name in WORKFLOWS:
            wf = load_workflow(REPO_ROOT / "workflows" / name, strict=False)
            ids = [s.id for s in wf.steps]
            self.assertLess(ids.index("classification"), ids.index("rvk_guard"), name)
            self.assertLess(ids.index("rvk_guard"), ids.index("dk_postprocess"), name)

            guard = next(s for s in wf.steps if s.id == "rvk_guard")
            self.assertEqual(guard.depends_on, ["classification"], name)
            self.assertEqual(
                guard.outputs.get("dk_classifications"), "result.classifications", name
            )

            post = next(s for s in wf.steps if s.id == "dk_postprocess")
            self.assertEqual(post.depends_on, ["rvk_guard"], name)
            # dk_postprocess must read the FILTERED list, not the step output.
            self.assertEqual(
                post.inputs.get("dk_classifications"), "${dk_classifications}", name
            )


if __name__ == "__main__":
    unittest.main()
