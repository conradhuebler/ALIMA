"""Modellquelle für die Zeile "Entscheidungsmodell" der Auflösungstabelle.

Eigenes Modul, weil ``resolution_view`` beim Öffnen nie das Netz berühren darf
(``tests/test_resolution_view.py`` prüft den Quelltext auf Modellabfragen); die
Abfrage läuft erst im Hintergrund-Worker des Wählers. - Claude Generated
"""

from __future__ import annotations

from typing import Any


class DecisionModelDetection:
    """Modellquelle für die Zeile "Entscheidungsmodell" - Claude Generated.

    Gleiche Schnittstelle wie ``ProviderDetectionService`` (was der
    ``ProviderModelSelector`` aufruft), aber nur Ollama-Anbieter und nur Modelle
    mit der Fähigkeit ``decision``: ein Chat-Modell wäre hier eine Fehlwahl.
    """

    def __init__(self, unified_config: Any) -> None:
        self._unified = unified_config

    def get_available_providers(self) -> list:
        return [p.name for p in self._unified.get_enabled_providers()
                if getattr(p, "provider_type", "") == "ollama"]

    def get_available_models(self, provider: str, force_check: bool = False) -> list:
        from src.utils.decision_client import list_decision_models

        obj = self._unified.get_provider_by_name(provider)
        return list_decision_models(obj) if obj is not None else []
