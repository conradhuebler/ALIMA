"""Declarative config-field schema for plugins - Claude Generated.

A plugin (search provider, input source, …) declares the configuration it needs
as a list of :class:`ConfigField`. This *single* declaration is the source of
truth for two things at once:

* the settings-UI form (``src/ui/plugin_settings_tab.py`` renders one widget per
  field), and
* availability gating (a field with ``gates_availability=True`` that is left
  empty makes the owning instance unavailable, replacing ad-hoc checks like
  ``bool(self._config.get("base_url"))``).

The module is **Qt-free** and dependency-light (stdlib only) so ``core`` and the
MCP tool layer can import it without pulling in PyQt or the config package. It is
a leaf: nothing here imports back into the plugin framework.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# Field kinds — kept as plain strings so manifests/JSON can name them directly.
TEXT = "text"
SECRET = "secret"
INT = "int"
BOOL = "bool"
URL = "url"
CHOICE = "choice"

_KINDS = {TEXT, SECRET, INT, BOOL, URL, CHOICE}


@dataclass
class ConfigField:
    """One configurable setting a plugin instance exposes.

    ``key`` is the canonical settings key — it must match what the plugin's
    constructor reads from its ``settings`` dict (e.g. the ``catalog`` provider
    reads ``token`` / ``catalog_search_url`` today, so those are the keys here).
    """

    key: str
    label: str
    kind: str = TEXT
    required: bool = False
    default: Any = None
    secret: bool = False
    gates_availability: bool = False
    choices: Optional[List[str]] = None
    help: str = ""

    def __post_init__(self) -> None:
        if not self.key or not isinstance(self.key, str):
            raise ValueError("ConfigField.key must be a non-empty string")
        if self.kind not in _KINDS:
            raise ValueError(
                f"ConfigField '{self.key}': unknown kind '{self.kind}'. "
                f"Allowed: {sorted(_KINDS)}"
            )
        if self.kind == SECRET:
            self.secret = True
        if self.kind == CHOICE and not self.choices:
            raise ValueError(
                f"ConfigField '{self.key}': kind 'choice' requires non-empty choices"
            )
        if self.default is None:
            self.default = self._kind_default()

    def _kind_default(self) -> Any:
        if self.kind == INT:
            return 0
        if self.kind == BOOL:
            return False
        return ""  # text/secret/url/choice

    def coerce(self, value: Any) -> Any:
        """Coerce a raw (e.g. JSON- or widget-sourced) value to the field type.

        Empty / ``None`` falls back to :attr:`default`. Never raises for
        malformed scalars — it degrades to the default so a bad config value can
        not crash provider construction (surfaced instead via ``is_available``).
        """
        if value is None or (isinstance(value, str) and value.strip() == ""):
            return self.default
        try:
            if self.kind == INT:
                return int(value)
            if self.kind == BOOL:
                if isinstance(value, str):
                    return value.strip().lower() in ("1", "true", "yes", "on")
                return bool(value)
        except (TypeError, ValueError):
            return self.default
        return value  # text/secret/url/choice pass through as str-ish

    def is_satisfied(self, value: Any) -> bool:
        """Whether ``value`` counts as "set" for gating/required checks."""
        coerced = self.coerce(value)
        if self.kind == BOOL:
            return bool(coerced)
        if self.kind == INT:
            return coerced is not None
        return bool(str(coerced).strip())


@dataclass
class PluginDoc:
    """A plugin's *natural-language* self-description — Claude Generated.

    Design principle: a plugin must explain itself in prose, not leave its purpose
    implicit in code. Every plugin type declares what it does and, explicitly, what
    it consumes (``input``) and produces (``output``). This text is what the
    settings UI shows the operator and what an agent reads to decide when to use
    the plugin — it is the authoritative answer to "what does this plugin do?".
    """

    description: str = ""
    input: str = ""
    output: str = ""

    def is_complete(self) -> bool:
        return bool(self.description.strip() and self.input.strip() and self.output.strip())

    def as_text(self) -> str:
        """Render as a compact block for tool descriptions / tooltips."""
        parts = []
        if self.description.strip():
            parts.append(self.description.strip())
        if self.input.strip():
            parts.append(f"Input: {self.input.strip()}")
        if self.output.strip():
            parts.append(f"Output: {self.output.strip()}")
        return "\n".join(parts)


def defaults(fields: List[ConfigField]) -> Dict[str, Any]:
    """Return ``{key: default}`` for a field list — the seed for a new instance."""
    return {f.key: f.default for f in fields}


def coerce_settings(
    fields: List[ConfigField], settings: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    """Coerce a raw settings dict against a field list.

    Known keys are type-coerced; unknown keys are preserved untouched (forward
    compatibility with plugin-declared extras the core does not model).
    """
    settings = dict(settings or {})
    out: Dict[str, Any] = {}
    by_key = {f.key: f for f in fields}
    for key, fld in by_key.items():
        out[key] = fld.coerce(settings.get(key))
    for key, val in settings.items():
        if key not in by_key:
            out[key] = val
    return out


def availability_ok(fields: List[ConfigField], settings: Optional[Dict[str, Any]]) -> bool:
    """True when every ``gates_availability`` field has a satisfied value."""
    settings = settings or {}
    for fld in fields:
        if fld.gates_availability and not fld.is_satisfied(settings.get(fld.key)):
            return False
    return True
