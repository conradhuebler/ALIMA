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


class _ReloadableLlm:
    """LlmService stand-in whose model list only updates after reload_providers().

    Mirrors the real bug: a provider added in Settings has no client until the
    wrapped service is reloaded, so its models stay empty until then.
    """

    def __init__(self, before, after):
        self._before = before
        self._after = after
        self.reloaded = False

    def get_available_models(self, provider, force_check=False):
        return list(self._after if self.reloaded else self._before)

    def reload_providers(self):
        self.reloaded = True


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

    def test_reload_reloads_wrapped_service_and_invalidates_cache(self):
        # Regression: provider added in Settings; its models only appear after the
        # wrapped LlmService is reloaded (was: only after a process restart).
        llm = _ReloadableLlm(before=[], after=["new-model"])
        svc = ProviderDetectionService()
        svc._llm_service = llm
        svc._get_llm_service = lambda: llm

        self.assertEqual(svc.get_available_models("newprov"), [])  # caches empty
        svc.reload()
        self.assertTrue(llm.reloaded, "reload must reload the wrapped service")
        self.assertEqual(
            svc.get_available_models("newprov"), ["new-model"],
            "reload must invalidate the cache so fresh models are fetched",
        )

    def test_reload_is_safe_without_instantiated_service(self):
        # reload() must not crash when the wrapped service was never created.
        svc = ProviderDetectionService()
        self.assertIsNone(svc._llm_service)
        svc.reload()  # should be a no-op that just clears the cache


if __name__ == "__main__":
    unittest.main()
