"""Finding a setting without knowing where it lives - Claude Generated.

Every control registers itself with the page and section it sits in. The search
field then reaches into folded sections, which is what makes folding acceptable:
a setting that is folded away is still findable.

Synonyms are not decoration. The interface is German and the documentation of
the services it talks to is English, so someone looking for "API key", "Cache",
"Timeout" or "SQLite" must land on "API-Schlüssel", "Zwischenspeicher",
"Zeitlimit" and "Datei auf diesem Rechner".
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional


def _normalise(text: str) -> str:
    """Casefold and strip accents/umlaut marks so „Schlüssel" matches "schlussel"."""
    text = unicodedata.normalize("NFKD", str(text or "")).casefold()
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return text.replace("ß", "ss")


#: Shared German↔English pairs. A term on either side finds the other.
COMMON_SYNONYMS = {
    "api-schlüssel": ["api key", "apikey", "token", "schluessel"],
    "zwischenspeicher": ["cache"],
    "zeitlimit": ["timeout"],
    "datei auf diesem rechner": ["sqlite", "lokale datei"],
    "sprachmodell": ["llm", "model", "modell"],
    "anbieter": ["provider"],
    "protokoll": ["log", "logging", "logdatei", "debug"],
    "datenquelle": ["plugin", "source", "quelle"],
    "schriftgröße": ["font size", "schrift"],
    "erschließung": ["indexing", "sacherschliessung"],
}


@dataclass
class SettingsEntry:
    """One findable control."""

    page: str
    label: str
    section: str = ""
    synonyms: List[str] = field(default_factory=list)
    widget: Any = None
    #: Called before focusing, to unfold whatever hides this entry.
    reveal: Optional[Callable[[], None]] = None

    def path(self) -> str:
        return f"{self.page} › {self.section}" if self.section else self.page

    def _haystack(self) -> List[str]:
        terms = [self.label, self.section, self.page, *self.synonyms]
        extra: List[str] = []
        for term in terms:
            for key, alts in COMMON_SYNONYMS.items():
                if _normalise(key) in _normalise(term):
                    extra.extend(alts)
                elif any(_normalise(a) in _normalise(term) for a in alts):
                    extra.append(key)
        return [_normalise(t) for t in terms + extra]

    def matches(self, needle: str) -> bool:
        n = _normalise(needle)
        if not n:
            return False
        return any(n in hay for hay in self._haystack())


class SettingsIndex:
    """Every control in the dialog, searchable across folded sections."""

    def __init__(self) -> None:
        self._entries: List[SettingsEntry] = []

    def register(self, entry: SettingsEntry) -> SettingsEntry:
        self._entries.append(entry)
        return entry

    def add(self, page: str, label: str, **kw) -> SettingsEntry:
        return self.register(SettingsEntry(page=page, label=label, **kw))

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def entries(self) -> List[SettingsEntry]:
        return list(self._entries)

    def search(self, needle: str, limit: int = 12) -> List[SettingsEntry]:
        """Hits, best first: label matches before section/page matches."""
        n = _normalise(needle)
        if not n:
            return []
        exact, partial = [], []
        for entry in self._entries:
            if not entry.matches(needle):
                continue
            (exact if n in _normalise(entry.label) else partial).append(entry)
        return (exact + partial)[:limit]

    def pages(self) -> List[str]:
        seen: List[str] = []
        for entry in self._entries:
            if entry.page not in seen:
                seen.append(entry.page)
        return seen
