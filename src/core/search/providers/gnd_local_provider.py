"""Local GND database search provider - Claude Generated.

Wraps ``UnifiedKnowledgeManager.search_local_gnd`` (no network). GND-keyword
candidates straight from the local ``gnd_entries`` table — not cache-wrapped
(it *is* the local store).
"""

from __future__ import annotations

from typing import Any, Callable, List, Optional

from ..provider import ProviderResult, ResultItem, SearchCapability
from ..registry import register_provider


@register_provider
class GndLocalProvider:
    """GND-keyword candidates from the local GND database."""

    id = "gnd_local"
    label = "Lokale GND-DB"
    capabilities = {SearchCapability.GND_KEYWORDS}

    def __init__(self, **config: Any):
        self._config = config or {}
        self._ukm = None

    @classmethod
    def config_fields(cls):
        """No user-facing config — uses the singleton UnifiedKnowledgeManager."""
        return []

    @classmethod
    def doc(cls):
        from src.core.plugins.schema import PluginDoc

        return PluginDoc(
            description="Lokale GND-Datenbank (kein Netzwerk): Schlagwort-Kandidaten "
            "direkt aus dem gnd_entries-Cache.",
            input="Suchbegriffe.",
            output="GND-Schlagwort-Kandidaten aus dem lokalen Cache (mit GND-IDs, DDC).",
        )

    @property
    def ukm(self):
        if self._ukm is None:
            from src.core.unified_knowledge_manager import UnifiedKnowledgeManager

            self._ukm = UnifiedKnowledgeManager()
        return self._ukm

    def is_available(self, cfg: Any = None) -> bool:
        return True

    def search(
        self,
        capability: SearchCapability,
        query: List[str],
        *,
        progress: Optional[Callable[[str], None]] = None,
        min_results: int = 3,
        **opts: Any,
    ) -> ProviderResult:
        if capability is not SearchCapability.GND_KEYWORDS:
            raise ValueError(f"Provider '{self.id}' does not support {capability}")
        per_term = {}
        for term in query:
            if progress is not None:
                try:
                    progress(term)
                except Exception:
                    pass
            entries = self.ukm.search_local_gnd(term, min_results=min_results)
            items = []
            for e in entries:
                ddcs = getattr(e, "ddcs", None)
                ddc = set(ddcs) if isinstance(ddcs, (list, set, tuple)) else set()
                items.append(
                    ResultItem(
                        label=e.title,
                        gnd_ids={e.gnd_id} if e.gnd_id else set(),
                        count=0,  # local DB has no occurrence count
                        ddc=ddc,
                    )
                )
            per_term[term] = items
        return ProviderResult(SearchCapability.GND_KEYWORDS, per_term=per_term)
