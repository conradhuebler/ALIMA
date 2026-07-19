"""Migration from the flat legacy config to the per-instance plugin model - Claude Generated.

The authoritative store is ``AlimaConfig.plugins`` (a list of
:class:`PluginInstanceConfig`). ``synthesize_search_instances`` upgrades a config
written before the plugin model: it reads the raw ``catalog_config`` /
``search_provider_config`` JSON sections and turns them into instances, once, on
load. The ``CatalogConfig``/``SearchProviderConfig`` *mirrors* those sections used
to be parsed into are gone (WP P7) — nothing reads them back, so the keys are
simply dropped on the next save.

``SEARCH_FIELD_MAP`` is therefore a one-way legacy-JSON-key → instance-setting-key
map, not a mirror definition. A key absent from the legacy section is **omitted**
from the instance settings rather than written as ``None``, so the plugin's own
``ConfigField`` default applies — the dataclass defaults that used to fill those
gaps no longer exist.

The DOI ``SystemConfig`` fields *are* still a live mirror
(``synthesize_input_instances`` / ``derive_input_mirrors``); they are out of P7's
scope.
"""

from __future__ import annotations

from dataclasses import fields as dataclass_fields
from typing import Any, Dict, List

# provider_id -> {instance-setting key: legacy catalog_config JSON key}
SEARCH_FIELD_MAP: Dict[str, Dict[str, str]] = {
    "catalog": {
        "token": "catalog_token",
        "catalog_search_url": "catalog_search_url",
        "catalog_details": "catalog_details_url",
        "catalog_web_search_url": "catalog_web_search_url",
        "catalog_web_record_url": "catalog_web_record_url",
        "catalog_type": "catalog_type",
        "strict_gnd_validation_for_dk_search": "strict_gnd_validation_for_dk_search",
    },
    "finc": {
        "base_url": "finc_base_url",
        "web_record_url": "finc_web_record_url",
        "catalog_web_record_url": "catalog_web_record_url",
        "institution_filter": "finc_institution_filter",
        "default_limit": "finc_default_limit",
        "timeout": "finc_timeout",
        "dk_enabled": "finc_dk_enabled",
        "harvest_enabled": "finc_harvest_enabled",
    },
    "sru": {
        "preset": "sru_preset",
        "base_url": "sru_base_url",
        "database": "sru_database",
        "schema": "sru_schema",
        "max_records": "sru_max_records",
    },
    "lobid": {},
    "swb": {},
    "gnd_local": {},
}

# Stable presentation order + built-in labels (avoids importing the Qt-free
# provider classes just for a label during config parsing).
_SEARCH_ORDER = ["lobid", "swb", "catalog", "finc", "sru", "gnd_local"]
_SEARCH_LABELS = {
    "lobid": "Lobid (GND/DNB)",
    "swb": "SWB (BSZ)",
    "catalog": "Katalog (Libero)",
    "finc": "finc (VuFind)",
    "sru": "SRU / MARC-XML",
    "gnd_local": "Lokale GND-DB",
}

SEARCH_CATEGORY = "search_provider"


def instance_from_dict(data: Dict[str, Any]):
    """Build a PluginInstanceConfig from a JSON dict, ignoring unknown keys."""
    from src.utils.config_models import PluginInstanceConfig

    known = {f.name for f in dataclass_fields(PluginInstanceConfig)}
    filtered = {k: v for k, v in (data or {}).items() if k in known}
    return PluginInstanceConfig(**filtered)


def synthesize_search_instances(catalog_section, gate_section=None) -> List:
    """One primary instance per built-in search type, from the legacy JSON sections.

    ``catalog_section`` is the raw ``catalog_config`` dict of a pre-plugin config
    (``{}`` for a fresh one); ``gate_section`` the raw ``search_provider_config``
    dict, whose ``providers`` map disables a type (absent ⇒ enabled).

    Keys absent from ``catalog_section`` are **left out** of the instance settings
    instead of being written as ``None``: the plugin's ``ConfigField``/constructor
    default must apply. This used to come for free from the ``CatalogConfig``
    dataclass defaults (``getattr`` on an unset field returned e.g.
    ``catalog_type='libero_soap'``); with the dataclass gone, a ``dict.get`` would
    hand ``None`` to the provider constructor instead. - Claude Generated
    """
    from src.utils.config_models import PluginInstanceConfig

    catalog_section = catalog_section or {}
    providers = (gate_section or {}).get("providers", {}) or {}

    instances: List = []
    for pid in _SEARCH_ORDER:
        settings = {
            key: catalog_section[attr]
            for key, attr in SEARCH_FIELD_MAP.get(pid, {}).items()
            if attr in catalog_section
        }
        instances.append(
            PluginInstanceConfig(
                instance_id=pid,
                category=SEARCH_CATEGORY,
                provider_id=pid,
                label=_SEARCH_LABELS.get(pid, pid),
                enabled=bool(providers.get(pid, True)),
                is_primary=True,
                settings=settings,
            )
        )
    return instances


def ensure_search_instances(plugins: List) -> None:
    """Seed one primary instance per built-in search type when the category is
    empty - Claude Generated.

    The search twin of :func:`ensure_lookup_instances`, for call sites that build a
    config from scratch (the setup wizards). They must not leave the category empty
    *nor* add a lone hand-made instance: the load/save synthesis guard is
    per-category, so one catalog instance would strand the other five built-ins.
    Mutates ``plugins`` in place.
    """
    if any(p.category == SEARCH_CATEGORY for p in plugins):
        return
    plugins += synthesize_search_instances({})


# ---------------------------------------------------------------------------
# Input sources (DOI resolver split + url_fetch)
# ---------------------------------------------------------------------------

INPUT_CATEGORY = "input_source"


def synthesize_input_instances(system_config) -> List:
    """Create input-source instances from the legacy ``SystemConfig`` DOI flags.

    The DOI resolver is split into three separately-configurable instances;
    ``contact_email`` mirrors the shared SystemConfig field, ``doi_use_*`` map to
    per-instance enabled state. ``url_fetch`` is added as an always-on instance.
    """
    from src.utils.config_models import PluginInstanceConfig

    email = getattr(system_config, "contact_email", "") or ""
    return [
        PluginInstanceConfig(
            instance_id="doi_crossref", category=INPUT_CATEGORY, provider_id="doi_crossref",
            label="DOI: Crossref", enabled=bool(getattr(system_config, "doi_use_crossref", True)),
            is_primary=True, settings={"contact_email": email},
        ),
        PluginInstanceConfig(
            instance_id="doi_openalex", category=INPUT_CATEGORY, provider_id="doi_openalex",
            label="DOI: OpenAlex", enabled=bool(getattr(system_config, "doi_use_openalex", True)),
            is_primary=False, settings={"contact_email": email},
        ),
        PluginInstanceConfig(
            instance_id="doi_datacite", category=INPUT_CATEGORY, provider_id="doi_datacite",
            label="DOI: DataCite", enabled=bool(getattr(system_config, "doi_use_datacite", True)),
            is_primary=False, settings={},
        ),
        PluginInstanceConfig(
            instance_id="url_fetch", category=INPUT_CATEGORY, provider_id="url_fetch",
            label="URL (Web-Scrape)", enabled=True, is_primary=True, settings={},
        ),
    ]


def derive_input_mirrors(plugins: List, system_config) -> None:
    """Update ``SystemConfig`` DOI flags + contact_email in place from instances."""
    inp = [p for p in plugins if p.category == INPUT_CATEGORY]

    def _find(pid):
        pool = [p for p in inp if p.provider_id == pid]
        for p in pool:
            if p.is_primary:
                return p
        return pool[0] if pool else None

    cr, oa, dc = _find("doi_crossref"), _find("doi_openalex"), _find("doi_datacite")
    if cr is not None:
        system_config.doi_use_crossref = bool(cr.enabled)
    if oa is not None:
        system_config.doi_use_openalex = bool(oa.enabled)
    if dc is not None:
        system_config.doi_use_datacite = bool(dc.enabled)
    # Shared contact_email mirror from the primary crossref, else openalex.
    for p in (cr, oa):
        if p is not None and "contact_email" in (p.settings or {}):
            system_config.contact_email = p.settings.get("contact_email") or ""
            break


# ---------------------------------------------------------------------------
# Lookups (RVK / k10plus / DNB … — external-authority query plugins)
# ---------------------------------------------------------------------------

LOOKUP_CATEGORY = "lookup"


def synthesize_missing_lookup_instances(existing_lookup_instances: List) -> List:
    """One enabled primary instance per registered lookup type that has no instance
    yet - Claude Generated.

    Unlike search/input, lookups have **no legacy config section** to mirror, so the
    list is built directly from the live ``LOOKUP_REGISTRY`` (importing the package
    self-registers the built-ins). Backfilling only the *missing* types means it
    both seeds a fresh config (pass ``[]``) and adds a lookup plugin registered in a
    later release (e.g. ``webindex``) without disturbing operator-configured
    instances — the former all-or-nothing category gate stranded newly-registered
    lookups in the type combobox.
    """
    from src.utils.config_models import PluginInstanceConfig
    from src.utils.lookups import get_lookup, list_lookups

    present = {p.provider_id for p in existing_lookup_instances}
    out: List = []
    for lid in list_lookups():
        if lid in present:
            continue
        cls = get_lookup(lid)
        out.append(
            PluginInstanceConfig(
                instance_id=lid,
                category=LOOKUP_CATEGORY,
                provider_id=lid,
                label=getattr(cls, "label", lid),
                enabled=True,
                is_primary=True,
                settings={},
            )
        )
    return out


def synthesize_lookup_instances() -> List:
    """One primary instance per registered lookup (all of them) — the fresh-config
    seed. Thin wrapper over :func:`synthesize_missing_lookup_instances`. - Claude Generated"""
    return synthesize_missing_lookup_instances([])


def ensure_lookup_instances(plugins: List) -> None:
    """Make sure every registered lookup type has at least one instance - Claude Generated.

    Seeds a fresh config and backfills types registered in a later release, in one
    pass (missing-of-the-empty-set is all of them). Mutates ``plugins`` in place.
    """
    lookup = [p for p in plugins if p.category == LOOKUP_CATEGORY]
    plugins += synthesize_missing_lookup_instances(lookup)
