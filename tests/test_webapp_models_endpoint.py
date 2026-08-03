"""Tests for /api/models: parallel probes, timeout, fallbacks - Claude Generated.

Operator-reported bug (Aug 4, 2026): the model dropdown stayed empty/loading
forever — the former sync probe loop ran IN the event loop and one unreachable
provider froze the whole webapp until its network timeout ("findet keine
Provider-Config"). These tests pin the fix: probes run in threads with a
per-provider budget; a hanging provider degrades to its persisted fallback
while fast providers deliver their live list.
"""

from __future__ import annotations

import time
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch


def _provider(name, preferred=None, persisted=None):
    return SimpleNamespace(
        name=name,
        preferred_model=preferred,
        available_models=list(persisted or []),
    )


class _Ctx:
    """AppContext stub wired to a fake config manager + detection service."""

    def __init__(self, providers, detection):
        cm = MagicMock()
        cm.get_provider_detection_service.return_value = detection
        cm.get_unified_config.return_value = SimpleNamespace(
            get_enabled_providers=lambda: providers
        )
        self._services = {"config_manager": cm}

    def get_services(self):
        return self._services


class TestModelsEndpoint(unittest.IsolatedAsyncioTestCase):
    async def _call(self, providers, detection, timeout=None):
        from src.webapp.routers import models as mod

        patches = [patch.object(mod, "AppContext", lambda: _Ctx(providers, detection))]
        if timeout is not None:
            patches.append(patch.object(mod, "_DETECT_TIMEOUT_S", timeout))
        with patches[0]:
            if timeout is not None:
                with patches[1]:
                    return await mod.get_available_models()
            return await mod.get_available_models()

    async def test_live_models_and_value_shape(self):
        detection = MagicMock()
        detection.get_available_models.return_value = ["m1", "m2"]
        out = await self._call([_provider("Mistral")], detection)
        self.assertEqual(
            out,
            [
                {"provider": "Mistral", "model": "m1", "value": "Mistral|m1"},
                {"provider": "Mistral", "model": "m2", "value": "Mistral|m2"},
            ],
        )

    async def test_hanging_provider_degrades_without_stalling_fast_ones(self):
        """The bug pin: one dead host must not stall the response. The hanging
        provider falls back to its preferred model within the probe budget."""

        def probe(name):
            if name == "GWDG":
                time.sleep(1.0)  # hängt weit über dem Test-Budget
                return ["never-delivered"]
            return ["fast-model"]

        detection = MagicMock()
        detection.get_available_models.side_effect = probe
        providers = [
            _provider("GWDG", preferred="gemma-fallback"),
            _provider("Mistral"),
        ]
        start = time.monotonic()
        out = await self._call(providers, detection, timeout=0.15)
        elapsed = time.monotonic() - start
        self.assertLess(elapsed, 0.9)  # parallel + Budget, nicht 1s+ sequenziell
        by_provider = {row["provider"]: row["model"] for row in out}
        self.assertEqual(by_provider["GWDG"], "gemma-fallback")
        self.assertEqual(by_provider["Mistral"], "fast-model")

    async def test_detection_failure_falls_back_to_persisted_list(self):
        detection = MagicMock()
        detection.get_available_models.side_effect = OSError("unreachable")
        out = await self._call(
            [_provider("Ollama", persisted=["p1", "p2"])], detection
        )
        self.assertEqual([r["model"] for r in out], ["p1", "p2"])

    async def test_nothing_known_yields_empty_not_error(self):
        detection = MagicMock()
        detection.get_available_models.return_value = []
        out = await self._call([_provider("Leer")], detection)
        self.assertEqual(out, [])


if __name__ == "__main__":
    unittest.main()
