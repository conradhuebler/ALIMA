"""Entscheidungs-Client und Vorfilter gegen einen echten lokalen HTTP-Server.

Der Server spielt ``/v1/systemone`` nach (Grenzen wie gemessen: 1 bis 64 Fragen,
sonst 400). Getestet wird der echte ``requests``-Pfad, nicht ein Mock der
Funktion. Claude Generated.
"""

from __future__ import annotations

import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer

from src.utils.config_models import UnifiedProvider, UnifiedProviderConfig
from src.utils.decision_client import (
    MAX_QUESTIONS,
    DecisionError,
    list_decision_models,
    provider_base_url,
    relevance_questions,
    score_items,
    systemone,
)
from src.utils.decision_prefilter import prefilter_by_decision


class _Fake(BaseHTTPRequestHandler):
    calls: list = []
    fail_first: int = 0       # so viele 503 vor der ersten Antwort
    status_override = None    # z. B. 400 für jede Anfrage
    max_questions = None      # mehr Fragen je Aufruf: 500 "Context size has been exceeded"
    lock = threading.Lock()

    def log_message(self, *a):  # still
        pass

    def do_GET(self):
        cls = type(self)
        with cls.lock:
            cls.calls.append({"path": self.path, "auth": self.headers.get("Authorization", "")})
        out = json.dumps({"models": [
            {"name": "qwen:chat", "capabilities": ["completion", "tools"]},
            {"name": "d3:latest", "capabilities": ["decision", "vision"]},
            {"name": "d1:latest", "capabilities": ["decision", "vision"]},
            {"name": "old:model"},
        ]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(out)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        cls = type(self)
        with cls.lock:
            cls.calls.append({"path": self.path, "n": len(body["questions"]),
                              "auth": self.headers.get("Authorization", ""),
                              "state": body["state"], "model": body["model"],
                              "instr": next(iter(body["questions"].values()))["instructions"]})
            fail = cls.fail_first > 0
            if fail:
                cls.fail_first -= 1
        n = len(body["questions"])
        if cls.max_questions is not None and n > cls.max_questions:
            self.send_response(500)
            self.end_headers()
            self.wfile.write(b'{"error":"Context size has been exceeded."}')
            return
        if cls.status_override or not 1 <= n <= 64 or fail:
            code = cls.status_override or (503 if fail else 400)
            self.send_response(code)
            self.end_headers()
            self.wfile.write(b'{"error":"x"}')
            return
        answers = {}
        for name, q in body["questions"].items():
            hot = "cadmium" in q["instructions"].lower()
            if q["type"] == "choice":
                probs = {k: (0.8 if i == 0 else 0.1) for i, k in enumerate(q["criteria"])}
                answers[name] = {"type": "choice", "choice": next(iter(probs)),
                                 "probabilities": probs}
            else:
                answers[name] = {"type": "noul", "noul": 0.95 if hot else 0.05}
        out = json.dumps({"model": body["model"], "answers": answers,
                          "usage": {"input_tokens": 100 * n, "output_tokens": 0}}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(out)


class _ServerCase(unittest.TestCase):
    def setUp(self):
        _Fake.calls = []
        _Fake.fail_first = 0
        _Fake.status_override = None
        _Fake.max_questions = None
        self.server = HTTPServer(("127.0.0.1", 0), _Fake)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.server.server_port}"
        import src.utils.decision_client as dc
        self._delays = dc._RETRY_DELAYS
        dc._RETRY_DELAYS = (0.0, 0.0, 0.0)

    def tearDown(self):
        import src.utils.decision_client as dc
        dc._RETRY_DELAYS = self._delays
        self.server.shutdown()
        self.server.server_close()

    def config(self, **kw) -> UnifiedProviderConfig:
        cfg = UnifiedProviderConfig(providers=[UnifiedProvider(
            name="srv", provider_type="ollama", enabled=True, preferred_model="chat",
            base_url=self.url, api_key="k")])
        cfg.decision_default_provider = "srv"
        cfg.decision_default_model = "d1"
        for k, v in kw.items():
            setattr(cfg, k, v)
        return cfg


class ClientTest(_ServerCase):
    def test_batches_stay_within_the_endpoint_limit(self):
        items = [f"Begriff {i}" for i in range(150)]
        scores = score_items(self.url, "d1", "state", items)
        self.assertEqual(len(scores), 150)
        self.assertEqual([c["n"] for c in _Fake.calls], [64, 64, 22])
        self.assertTrue(all(c["n"] <= MAX_QUESTIONS for c in _Fake.calls))

    def test_scores_are_assigned_to_the_right_item_with_parallel_workers(self):
        items = [("Cadmium %d" % i) if i % 3 == 0 else f"Boden {i}" for i in range(200)]
        scores = score_items(self.url, "d1", "s", items, max_parallel=3)
        for i, t in enumerate(items):
            self.assertEqual(scores[t] > 0.5, i % 3 == 0, t)

    def test_503_is_retried_then_succeeds(self):
        _Fake.fail_first = 2
        out = systemone(self.url, "d1", "s", relevance_questions(["Cadmium"]))
        self.assertIn("answers", out)
        self.assertEqual(len(_Fake.calls), 3)

    def test_a_batch_that_exceeds_the_context_is_split_until_it_fits(self):
        """Der Host zählt den State je Frage; 64 Fragen mit langem State sprengten den Kontext."""
        _Fake.max_questions = 20
        items = [f"Cadmium {i}" for i in range(64)]
        scores = score_items(self.url, "d1", "state", items)
        self.assertEqual(len(scores), 64)
        self.assertTrue(all(v > 0.5 for v in scores.values()))
        self.assertTrue(all(c["n"] <= 20 for c in _Fake.calls if c["n"] <= 20) and len(_Fake.calls) > 4)

    def test_a_single_question_that_still_exceeds_the_context_is_an_error(self):
        _Fake.max_questions = 0
        with self.assertRaises(DecisionError):
            score_items(self.url, "d1", "state", ["a"])

    def test_a_400_is_not_retried(self):
        _Fake.status_override = 400
        with self.assertRaises(DecisionError):
            systemone(self.url, "d1", "s", relevance_questions(["x"]))
        self.assertEqual(len(_Fake.calls), 1)

    def test_persistent_503_raises_after_the_retries(self):
        _Fake.fail_first = 99
        with self.assertRaises(DecisionError):
            systemone(self.url, "d1", "s", relevance_questions(["x"]))
        self.assertEqual(len(_Fake.calls), 4)

    def test_more_than_64_questions_in_one_call_is_refused_locally(self):
        with self.assertRaises(DecisionError):
            systemone(self.url, "d1", "s", relevance_questions([str(i) for i in range(65)]))
        self.assertEqual(_Fake.calls, [])

    def test_stop_aborts_instead_of_returning_partial_scores(self):
        with self.assertRaises(DecisionError):
            score_items(self.url, "d1", "s", ["a"] * 100, should_stop=lambda: True)

    def test_html_escapes_in_pool_titles_reach_the_model_unescaped(self):
        q = relevance_questions(["Gewässer &lt;Motiv&gt;"])
        self.assertIn("<Motiv>", q["q0"]["instructions"])

    def test_non_ollama_provider_is_refused(self):
        p = UnifiedProvider(name="o", provider_type="openai_compatible", base_url="http://x/v1")
        with self.assertRaises(DecisionError):
            provider_base_url(p)

    def test_model_list_offers_only_decision_models(self):
        provider = self.config().get_provider_by_name("srv")
        self.assertEqual(list_decision_models(provider), ["d1:latest", "d3:latest"])
        self.assertEqual(_Fake.calls[0]["path"], "/api/tags")
        self.assertEqual(_Fake.calls[0]["auth"], "Bearer k")

    def test_model_list_is_empty_for_a_dead_host_or_a_non_ollama_provider(self):
        dead = UnifiedProvider(name="x", provider_type="ollama", base_url="http://127.0.0.1:9")
        self.assertEqual(list_decision_models(dead, timeout=1), [])
        other = UnifiedProvider(name="o", provider_type="openai_compatible", base_url="http://x/v1")
        self.assertEqual(list_decision_models(other), [])

    def test_api_key_is_sent_as_bearer(self):
        score_items(self.url, "d1", "s", ["a"], api_key="  Bearer abc ")
        self.assertEqual(_Fake.calls[0]["auth"], "Bearer abc")


class PrefilterTest(_ServerCase):
    ITEMS = [{"title": ("Cadmium %d" % i) if i % 10 == 0 else f"Boden {i}"} for i in range(300)]

    def run_prefilter(self, cfg, **kw):
        msgs: list = []
        out = prefilter_by_decision(
            self.ITEMS, lambda d: d["title"], "Cadmium im Boden", cfg,
            notify=msgs.append, **kw)
        return out, msgs

    def test_keeps_the_best_in_original_order_and_records_scores(self):
        out, msgs = self.run_prefilter(self.config(), top_k=30)
        self.assertEqual(len(out.kept), 30)
        self.assertTrue(all("Cadmium" in d["title"] for d in out.kept))  # alle 30 heißen
        positions = [self.ITEMS.index(d) for d in out.kept]
        self.assertEqual(positions, sorted(positions))
        r = out.record
        self.assertEqual((r["provider"], r["model"], r["n_in"], r["n_out"]), ("srv", "d1", 300, 30))
        self.assertEqual(len(r["scores"]), 300)
        self.assertTrue(msgs and "300 → 30" in msgs[0])

    def test_ties_resolve_by_original_position(self):
        out, _ = self.run_prefilter(self.config(), top_k=40)  # 30 heiß + 10 vom Rest
        rest = [d for d in out.kept if "Cadmium" not in d["title"]]
        self.assertEqual([d["title"] for d in rest], [f"Boden {i}" for i in range(1, 12) if i % 10][:10])

    def test_state_is_truncated(self):
        self.run_prefilter(self.config(), top_k=30, max_state_chars=7)
        self.assertTrue(all(len(c["state"]) == 7 for c in _Fake.calls))

    def test_role_off_means_no_call_and_none(self):
        cfg = self.config(decision_default_provider="", decision_default_model="")
        out, _ = self.run_prefilter(cfg, top_k=30)
        self.assertIsNone(out)
        self.assertEqual(_Fake.calls, [])

    def test_small_pool_needs_no_call(self):
        out, _ = self.run_prefilter(self.config(), top_k=300)
        self.assertIsNone(out)
        self.assertEqual(_Fake.calls, [])

    def test_unreachable_host_falls_back_with_a_notice(self):
        cfg = self.config()
        cfg.get_provider_by_name("srv").base_url = "http://127.0.0.1:9"
        out, msgs = self.run_prefilter(cfg, top_k=30)
        self.assertIsNone(out)
        self.assertTrue(msgs and "Weiter mit der bisherigen Auswahl" in msgs[0])

    def test_server_error_falls_back_with_a_notice(self):
        _Fake.status_override = 400
        out, msgs = self.run_prefilter(self.config(), top_k=30)
        self.assertIsNone(out)
        self.assertTrue(msgs)


if __name__ == "__main__":
    unittest.main()
