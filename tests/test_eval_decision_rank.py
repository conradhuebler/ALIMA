"""scripts/eval_decision_rank.py: Auswertung und Ablauf gegen den Fake-Server. Claude Generated."""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from tests.test_decision_prefilter import _Fake, _ServerCase

_spec = importlib.util.spec_from_file_location("eval_decision_rank", "scripts/eval_decision_rank.py")
ev = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ev)

CANDS = [{"keyword": "Cadmium", "gnd_id": "4009274-4"}, {"keyword": "Reis", "gnd_id": "4177603-3"},
         {"keyword": "Boden", "gnd_id": "4007348-8"}]


class PureTest(unittest.TestCase):
    def test_gold_lines_in_all_three_forms(self):
        gold = ev.parse_gold(["# Kommentar", "", "Cadmium", "Reis (GND-ID: 4177603-3)", "4007348-8"])
        self.assertEqual(gold, [{"label": "Cadmium", "gnd_id": ""},
                                {"label": "Reis", "gnd_id": "4177603-3"},
                                {"label": "", "gnd_id": "4007348-8"}])

    def test_matching_by_id_and_by_label_and_missing(self):
        found = ev.find_gold(ev.parse_gold(["cadmium", "4177603-3", "Itai-Itai-Krankheit"]), CANDS)
        self.assertEqual(found, {"cadmium": "Cadmium", "4177603-3": "Reis", "Itai-Itai-Krankheit": None})

    def test_umlauts_match_in_decomposed_and_composed_form(self):
        decomposed = "Cytotoxizita\u0308t"                      # so liefert der Pool es
        found = ev.find_gold(ev.parse_gold(["Cytotoxizit\u00e4t"]),
                             [{"keyword": decomposed, "gnd_id": "1"}])
        self.assertEqual(found, {"Cytotoxizit\u00e4t": decomposed})

    def test_recall_and_rank_global_and_balanced(self):
        texts = [f"Cd{i}" for i in range(6)] + ["Reis1"]
        scores = {t: 0.9 - 0.1 * i for i, t in enumerate(texts)}      # Reis1 ist am schlechtesten
        groups = {**{t: ["Cadmium"] for t in texts[:6]}, "Reis1": ["Reis"]}
        found = {"Cd0": "Cd0", "Reis1": "Reis1", "fehlt": None}
        plain = ev.evaluate(scores, texts, groups, found, [3, 7], balanced=False)
        self.assertEqual((plain["recall"][3], plain["recall"][7], plain["in_pool"], plain["gold"]), (1, 2, 2, 3))
        self.assertEqual(plain["rank"], {"Cd0": 1, "Reis1": 7})
        bal = ev.evaluate(scores, texts, groups, found, [3], balanced=True)
        self.assertEqual(bal["recall"][3], 2)                          # Reis1 kommt reihum früh


class RunTest(_ServerCase):
    def _run(self, *extra):
        tmp = self.tmp
        argv = ["--pool", f"{tmp}/pool.json", "--text-file", f"{tmp}/t.txt", "--gold", f"{tmp}/g.txt",
                "--models", "srv/d1", "--variants", "geeignet", "--ks", "5,10", "--out", f"{tmp}/out", *extra]
        cm = MagicMock()
        cm.return_value.get_unified_config.return_value = self.config()
        buf = io.StringIO()
        with patch("src.utils.config_manager.ConfigManager", cm), contextlib.redirect_stdout(buf):
            code = ev.main(argv)
        return code, buf.getvalue()

    def setUp(self):
        super().setUp()
        self._td = tempfile.TemporaryDirectory()
        self.tmp = self._td.name
        pool = [{"title": f"Cadmium {i}", "gnd_ids": [str(i)], "count": 1} for i in range(70)]
        pool += [{"title": "Reis", "gnd_ids": ["9"], "count": 1}]
        json.dump({"pool": pool, "terms_map": {}}, open(f"{self.tmp}/pool.json", "w"))
        open(f"{self.tmp}/t.txt", "w").write("Cadmium im Boden")
        open(f"{self.tmp}/g.txt", "w").write("Cadmium 3\nReis\nItai-Itai-Krankheit\n")

    def tearDown(self):
        self._td.cleanup()
        super().tearDown()

    def test_without_yes_nothing_is_sent_and_the_plan_is_printed(self):
        code, out = self._run()
        self.assertEqual(code, 0)
        self.assertEqual(_Fake.calls, [])
        self.assertIn("Plan: 1 Läufe × 2 Anfragen = 2 Anfragen", out)
        self.assertIn("nicht im Pool: Itai-Itai-Krankheit", out)

    def test_yes_scores_saves_and_reports_then_reuse_needs_no_requests(self):
        code, out = self._run("--yes")
        self.assertEqual(len(_Fake.calls), 2)                          # 71 Kandidaten = 64 + 7
        self.assertIn("R@5", out)
        saved = json.load(open(f"{self.tmp}/out/srv_d1__geeignet.json"))
        self.assertEqual((saved["n"], saved["input_tokens"]), (71, 7100))
        _Fake.calls = []
        code, out = self._run("--reuse")
        self.assertEqual(_Fake.calls, [])
        self.assertIn("srv/d1 / geeignet / nein", out)


if __name__ == "__main__":
    unittest.main()
