"""Shared base for providers that wrap a legacy ``BaseSuggester`` - Claude Generated.

The lobid / swb / catalog sources already implement the GND-keyword
``search(terms, ...) -> {term: {keyword: {...}}}`` contract; these providers are
thin adapters that lazily build the suggester and convert its output to a typed
:class:`ProviderResult`. Behaviour is unchanged (P1 is facade-preserving).
"""

from __future__ import annotations

from typing import Any, Callable, List, Optional

from ..provider import ProviderResult, SearchCapability


class SuggesterBackedProvider:
    """Base for GND-keyword providers wrapping a ``BaseSuggester``.

    Subclasses set ``id`` / ``label`` / ``capabilities`` and implement
    :meth:`_build_suggester`. The underlying suggester is built lazily so importing
    the provider module (for registration) does not construct network clients.
    """

    id: str = ""
    label: str = ""
    capabilities: set = set()

    def __init__(self, **config: Any):
        self._config = config or {}
        self._suggester = None

    def _build_suggester(self):  # pragma: no cover - overridden
        raise NotImplementedError

    @property
    def suggester(self):
        if self._suggester is None:
            self._suggester = self._build_suggester()
        return self._suggester

    def is_available(self, cfg: Any = None) -> bool:
        """Default: always available. Network sources with config gating override."""
        return True

    def _require(self, capability: SearchCapability) -> None:
        if capability not in self.capabilities:
            raise ValueError(
                f"Provider '{self.id}' does not support {capability}; "
                f"declares {sorted(c.value for c in self.capabilities)}"
            )

    def _wire_progress(self, progress: Optional[Callable[[str], None]]) -> None:
        """Connect the suggester's ``currentTerm`` signal to ``progress`` if given.

        Best-effort: works for both the real Qt signal and the no-Qt dummy in
        ``base_suggester``. Providers are short-lived (built per use), so repeated
        connects are not a concern.
        """
        if progress is None:
            return
        try:
            self.suggester.currentTerm.connect(progress)
        except Exception:
            pass

    def _gnd_search(
        self,
        query: List[str],
        progress: Optional[Callable[[str], None]],
        **suggester_kwargs: Any,
    ) -> ProviderResult:
        """Run the wrapped suggester and wrap its GND-keyword output."""
        self._wire_progress(progress)
        raw = self.suggester.search(list(query), **suggester_kwargs)
        errors = dict(getattr(self.suggester, "last_errors", {}) or {})
        return ProviderResult.from_gnd_keywords(raw, errors=errors)
