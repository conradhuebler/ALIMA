"""Tests for the merged ModelLoadWorker (F-7) - Claude Generated.

Replaces the former ProviderModelSelector._ModelLoadWorker (single provider +
force) and UnifiedProviderTab.ModelFetchWorker (batch). Verifies both signal
shapes, force passthrough, single-id acceptance, defensive per-provider error
handling, and cooperative cancellation. run() is invoked directly (synchronous;
no QThread.start) so emissions are captured deterministically.
"""

import unittest

try:
    from PyQt6.QtCore import QCoreApplication
    from src.ui.workers import ModelLoadWorker, StoppableWorker
    _APP = QCoreApplication.instance() or QCoreApplication([])
    IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover
    IMPORT_ERROR = exc


class _FakeService:
    def __init__(self, mapping, raise_for=None):
        self.mapping = mapping
        self.raise_for = set(raise_for or ())
        self.calls = []

    def get_available_models(self, provider, force_check=False):
        self.calls.append((provider, force_check))
        if provider in self.raise_for:
            raise RuntimeError("boom")
        return self.mapping.get(provider, [])


@unittest.skipIf(IMPORT_ERROR is not None, f"PyQt6 unavailable: {IMPORT_ERROR}")
class ModelLoadWorkerTest(unittest.TestCase):
    def _run(self, worker):
        inc, final = [], []
        worker.fetched.connect(lambda p, m: inc.append((p, m)))
        worker.models_fetched.connect(lambda d: final.append(d))
        worker.run()
        return inc, final

    def test_is_stoppable_worker(self):
        self.assertTrue(issubclass(ModelLoadWorker, StoppableWorker))

    def test_batch_emits_incremental_then_final(self):
        svc = _FakeService({"a": ["m1"], "b": ["m2", "m3"]})
        inc, final = self._run(ModelLoadWorker(svc, ["a", "b"]))
        self.assertEqual(inc, [("a", ["m1"]), ("b", ["m2", "m3"])])
        self.assertEqual(final, [{"a": ["m1"], "b": ["m2", "m3"]}])

    def test_single_provider_string_and_force(self):
        svc = _FakeService({"a": ["m1"]})
        inc, final = self._run(ModelLoadWorker(svc, "a", force=True))
        self.assertEqual(inc, [("a", ["m1"])])
        self.assertEqual(final, [{"a": ["m1"]}])
        self.assertEqual(svc.calls, [("a", True)])  # force passed through

    def test_exception_yields_empty_list(self):
        svc = _FakeService({"a": ["m1"]}, raise_for={"b"})
        inc, final = self._run(ModelLoadWorker(svc, ["a", "b"]))
        self.assertEqual(inc, [("a", ["m1"]), ("b", [])])
        self.assertEqual(final, [{"a": ["m1"], "b": []}])

    def test_interruption_stops_before_fetch(self):
        svc = _FakeService({"a": ["m1"], "b": ["m2"]})
        worker = ModelLoadWorker(svc, ["a", "b"])
        worker.request_stop()
        inc, final = self._run(worker)
        self.assertEqual(inc, [])
        self.assertEqual(final, [{}])
        self.assertEqual(svc.calls, [])  # no provider queried


if __name__ == "__main__":
    unittest.main()
