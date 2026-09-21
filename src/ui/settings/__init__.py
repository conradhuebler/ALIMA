"""Building blocks for the settings dialog - Claude Generated.

The dialog is one sidebar with five pages and no nested navigation. Two things
make that possible without hiding anything:

* :class:`CollapsibleSection` — a section that folds away, named after what is
  inside it rather than "Erweitert". A section called "Modellgrenzen" can be
  found; a section called "Erweitert" cannot.
* :class:`SettingsIndex` — every control registers itself, so the search field
  reaches into folded sections. This is the condition for folding to be
  acceptable at all.
"""

from .collapsible import CollapsibleSection
from .search import SettingsIndex, SettingsEntry
from .shell import SettingsShell
from .resolution_view import ResolutionView
from .provider_list import ProviderList

__all__ = ["CollapsibleSection", "SettingsIndex", "SettingsEntry", "SettingsShell", "ResolutionView", "ProviderList"]
