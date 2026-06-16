"""Regression: shared TTL model cache in ProviderDetectionService.

Claude Generated (provider-system cleanup, Phase 4).

UI surfaces each construct their own ProviderDetectionService, so the model-list
cache is class-level and shared. Verifies cache hits, sharing across instances,
``force_check`` bypass, and stale-on-error fallback (models must not vanish on a
transient detection failure).
"""
from __future__ import annotations

import unittest

from src.utils.config_manager import ProviderDetectionService


class _FakeLlm:
    def __init__(self, models):
        self.models = models
        self.calls = 0

    def get_available_models(self, provider, force_check=False):
        self.calls += 1
        return list(self.models)


class _BoomLlm:
    def get_available_models(self, provider, force_check=False):
        raise RuntimeError("network down")


class TestProviderModelCache(unittest.TestCase):
    def setUp(self):
        ProviderDetectionService.clear_model_cache()

    def tearDown(self):
        ProviderDetectionService.clear_model_cache()

    def _service(self, llm):
        svc = ProviderDetectionService()
        svc._get_llm_service = lambda: llm
        return svc

    def test_second_read_is_cached(self):
        llm = _FakeLlm(["m1", "m2"])
        svc = self._service(llm)
        self.assertEqual(svc.get_available_models("p"), ["m1", "m2"])
        self.assertEqual(svc.get_available_models("p"), ["m1", "m2"])
        self.assertEqual(llm.calls, 1, "second read must hit the cache")

    def test_cache_shared_across_instances(self):
        llm = _FakeLlm(["m1"])
        self._service(llm).get_available_models("p")
        # A different instance using a different (would-fail) llm still gets cache.
        other = self._service(_BoomLlm())
        self.assertEqual(other.get_available_models("p"), ["m1"])
        self.assertEqual(llm.calls, 1)

    def test_force_check_bypasses_and_refreshes(self):
        llm = _FakeLlm(["m1"])
        svc = self._service(llm)
        svc.get_available_models("p")
        svc.get_available_models("p", force_check=True)
        self.assertEqual(llm.calls, 2)

    def test_stale_returned_on_fetch_error(self):
        llm = _FakeLlm(["m1", "m2"])
        svc = self._service(llm)
        svc.get_available_models("p")          # populate cache
        svc._get_llm_service = lambda: _BoomLlm()
        # force_check triggers a fetch that fails -> last good list is returned
        self.assertEqual(svc.get_available_models("p", force_check=True), ["m1", "m2"])

    def test_clear_specific_provider(self):
        llm = _FakeLlm(["m1"])
        svc = self._service(llm)
        svc.get_available_models("p")
        ProviderDetectionService.clear_model_cache("p")
        svc.get_available_models("p")
        self.assertEqual(llm.calls, 2, "cleared provider must re-fetch")


if __name__ == "__main__":
    unittest.main()
