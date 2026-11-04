"""scripts/bench_decision_batching.py: Auswertung und Ablauf gegen den Fake-Server. Claude Generated."""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import os
import tempfile
import unittest

from tests.test_decision_prefilter import _Fake, _ServerCase

_spec = importlib.util.spec_from_file_location("bench_decision_batching", "scripts/bench_decision_batching.py")
bd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bd)


class PureTest(unittest.TestCase):
    def test_compare_reports_worst_difference_and_top_overlap(self):
        ref = {"a": 0.9, "b": 0.8, "c": 0.1, "d": 0.05}
        other = {"a": 0.88, "b": 0.3, "c": 0.7, "d": 0.05}
        worst, overlap = bd.compare(ref, other, top=2)
        self.assertAlmostEqual(worst, 0.6)           # c: 0.1 gegen 0.7
        self.assertEqual(overlap, 1)                 # {a, b} gegen {a, c}

    def test_configs_run_serial_for_all_sizes_and_parallel_only_for_the_chosen(self):
        self.assertEqual(bd.configs([1, 16, 64], [2, 4], [16]), [(1, 1), (16, 1), (64, 1), (16, 2), (16, 4)])
        self.assertEqual(bd.configs([8], [1, 2], [8]), [(8, 1), (8, 2)])


class RunTest(_ServerCase):
    def setUp(self):
        super().setUp()
        self._td = tempfile.TemporaryDirectory()
        self.tmp = self._td.name
        json.dump({"scores": {f"Cadmium {i}": 0.5 for i in range(20)}}, open(f"{self.tmp}/s.json", "w"))
        open(f"{self.tmp}/t.txt", "w").write("Cadmium im Boden")

    def tearDown(self):
        self._td.cleanup()
        super().tearDown()

    def _main(self, *extra):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = bd.main(["--scores-file", f"{self.tmp}/s.json", "--text-file", f"{self.tmp}/t.txt",
                            "--models", f"{self.url}#d1", "--n", "20", "--batch-sizes", "1,10,20",
                            "--parallel", "2", "--parallel-sizes", "10", "--out", f"{self.tmp}/o.json", *extra])
        return code, buf.getvalue()

    def test_without_yes_only_the_plan_is_printed(self):
        code, out = self._main()
        self.assertEqual((code, _Fake.calls), (0, []))
        self.assertIn("mindestens 20+2+1+2 Anfragen".replace("20+2+1+2", "25"), out)

    def test_the_sweep_counts_requests_and_checks_that_scores_do_not_depend_on_block_size(self):
        code, out = self._main("--yes")
        rep = json.load(open(f"{self.tmp}/o.json"))[f"{self.url}#d1"]
        by = {(r["batch"], r["workers"]): r for r in rep}
        self.assertEqual(by[(1, 1)]["requests"], 20)
        self.assertEqual(by[(10, 1)]["requests"], 2)
        self.assertEqual(by[(20, 1)]["requests"], 1)
        self.assertEqual(by[(10, 2)]["requests"], 2)
        self.assertTrue(all(r["max_diff"] == 0 and r["top40_overlap"] == 20 for r in rep))
        self.assertIn("ms/Frage", out)

    def test_the_request_cap_protects_the_host(self):
        with self.assertRaises(SystemExit):
            self._main("--yes", "--max-requests", "10")
        self.assertEqual(_Fake.calls, [])


if __name__ == "__main__":
    unittest.main()
