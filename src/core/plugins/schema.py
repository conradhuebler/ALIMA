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

import os
import re
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
        # text/secret/url/choice are string-typed: coerce stray scalars so a
        # mis-typed config value (e.g. TOML int) can not leak a non-str into
        # provider constructors. - Claude Generated
        if not isinstance(value, str):
            value = str(value)
        if self.kind == CHOICE and self.choices and value not in self.choices:
            return self.default
        return value

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


def env_var_name(instance_id: str, key: str) -> str:
    """Env-var name overriding a secret setting: ``ALIMA_PLUGIN_<ID>_<KEY>``.

    Non-alphanumeric characters map to ``_``, uppercased — e.g. instance
    ``catalog`` field ``token`` → ``ALIMA_PLUGIN_CATALOG_TOKEN``. - Claude Generated
    """
    clean = lambda s: re.sub(r"[^A-Za-z0-9]+", "_", str(s)).strip("_").upper()  # noqa: E731
    return f"ALIMA_PLUGIN_{clean(instance_id)}_{clean(key)}"


def apply_env_overrides(
    instance_id: str,
    settings: Optional[Dict[str, Any]],
    fields: List[ConfigField],
) -> Dict[str, Any]:
    """Return ``settings`` with each *secret* field overridden by its env var.

    Runtime-only: called at provider/source construction, never at config
    load/save — an env-supplied secret can therefore never be persisted into
    ``config.json`` or the legacy mirrors. The input dict is not mutated.
    - Claude Generated
    """
    out = dict(settings or {})
    for fld in fields:
        if not fld.secret:
            continue
        env_val = os.environ.get(env_var_name(instance_id, fld.key))
        if env_val:
            out[fld.key] = env_val
    return out


# --- Standard per-plugin raw-response cache toggle ------------------------- #
CACHE_RESPONSES_KEY = "cache_responses"


def cache_field() -> "ConfigField":
    """The standard per-plugin cache toggle, injected into every plugin's settings
    form by the category adapters (search + input).

    Tri-state: ``auto`` (follow the global ``enable_response_cache`` switch), ``on``
    (always cache this plugin's raw responses), ``off`` (never). Read at execution
    via :func:`cache_pref_enabled`. - Claude Generated"""
    return ConfigField(
        key=CACHE_RESPONSES_KEY,
        label="Antworten cachen",
        kind=CHOICE,
        choices=["auto", "on", "off"],
        default="auto",
        help="Rohantworten dieses Plugins im lokalen Response-Cache speichern. "
        "'auto' folgt dem globalen Cache-Schalter; 'on'/'off' überschreibt ihn "
        "für dieses Plugin.",
    )


def cache_pref_enabled(value: Any, *, global_enabled: bool) -> bool:
    """Interpret a plugin's ``cache_responses`` setting → effective on/off.

    Accepts the tri-state string (``auto``/``on``/``off``) or a legacy bool.
    ``auto``/absent falls back to ``global_enabled``. - Claude Generated"""
    if isinstance(value, bool):
        return value
    if value is None:
        return global_enabled
    v = str(value).strip().lower()
    if v in ("on", "true", "1", "yes"):
        return True
    if v in ("off", "false", "0", "no"):
        return False
    return global_enabled
