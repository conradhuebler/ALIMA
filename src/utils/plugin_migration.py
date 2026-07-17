"""Migration between the flat legacy config and the per-instance plugin model - Claude Generated.

The authoritative store is ``AlimaConfig.plugins`` (a list of
:class:`PluginInstanceConfig`). For the ~53 legacy readers, ``CatalogConfig`` (and,
later, DOI ``SystemConfig`` fields) are kept as *derived mirrors*:

* :func:`synthesize_search_instances` — build the initial instance list from a
  legacy ``CatalogConfig`` + ``SearchProviderConfig`` bool-gate (run once, when no
  ``plugins`` section exists yet).
* :func:`derive_search_mirrors` — push the primary instances' settings back into a
  ``CatalogConfig`` + ``SearchProviderConfig`` on save, so the mirror stays exact.

The field maps below are the single definition of which provider setting mirrors
which ``CatalogConfig`` attribute. Only mapped fields are mirrored — the map now
covers the web URLs, ``catalog_type`` and the strict-validation flag too, so a
load→save→load round-trip is diff-free. (An earlier version of this docstring
claimed those were *unmapped*; they are mapped, see ``SEARCH_FIELD_MAP``.)
"""

from __future__ import annotations

from dataclasses import fields as dataclass_fields
from typing import Any, Dict, List

# provider_id -> {instance-setting key: CatalogConfig attribute}
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


def synthesize_search_instances(catalog_config, search_provider_config) -> List:
    """Create one primary instance per built-in search provider type from legacy config."""
    from src.utils.config_models import PluginInstanceConfig

    instances: List = []
    for pid in _SEARCH_ORDER:
        settings = {
            key: getattr(catalog_config, attr, None)
            for key, attr in SEARCH_FIELD_MAP.get(pid, {}).items()
        }
        enabled = True
        if search_provider_config is not None:
            enabled = search_provider_config.is_enabled(pid)
        instances.append(
            PluginInstanceConfig(
                instance_id=pid,
                category=SEARCH_CATEGORY,
                provider_id=pid,
                label=_SEARCH_LABELS.get(pid, pid),
                enabled=enabled,
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
    from src.utils.config_models import CatalogConfig

    plugins += synthesize_search_instances(CatalogConfig(), None)


def derive_search_mirrors(plugins: List, catalog_config, search_provider_config) -> None:
    """Update ``catalog_config`` + ``search_provider_config`` in place from instances.

    Mirrors the *primary* instance of each mapped type into its CatalogConfig
    fields, and reflects per-type enabled state into the bool gate. Unmapped
    CatalogConfig fields are left as-is.
    """
    search = [p for p in plugins if p.category == SEARCH_CATEGORY]

    # Bool gate: a type is "enabled" if any of its instances is enabled.
    for pid in _SEARCH_ORDER:
        insts = [p for p in search if p.provider_id == pid]
        if insts:
            search_provider_config.set_enabled(pid, any(p.enabled for p in insts))

    # Endpoint mirror: from the primary instance of each type.
    for pid, mapping in SEARCH_FIELD_MAP.items():
        if not mapping:
            continue
        primary = _primary_of(search, pid)
        if primary is None:
            continue
        for key, attr in mapping.items():
            if key in primary.settings:
                setattr(catalog_config, attr, primary.settings.get(key))


def _primary_of(instances: List, provider_id: str):
    pool = [p for p in instances if p.provider_id == provider_id]
    for p in pool:
        if p.is_primary:
            return p
    return pool[0] if pool else None


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
