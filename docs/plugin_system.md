# Generic Plugin System

> **Status:** ✅ Implemented (July 1, 2026). Category-agnostic framework in
> `src/core/plugins/`, with two concrete categories: **search providers**
> (`src/core/search/`) and **input sources** (`src/utils/input_sources/`).
> Suite: `907 passed`. GUI (`PluginSettingsTab`) is operator-click-test-gated.

## Motivation

ALIMA has several extension points that each used the same registry idiom but were
configured and presented inconsistently: search providers exposed only enable/disable
checkboxes while their endpoints/tokens lived scattered in a flat `CatalogConfig`;
input acquisition (text/PDF/URL/DOI) was a non-extensible if/elif with the URL scraper
buried in `batch_processor` and the DOI resolver driven by loose `SystemConfig` flags.

The plugin system makes *a plugin* one concept across categories: it declares its own
config, is configured per-instance in one place, and can be added from a directory.

## Architecture

### Generic framework — `src/core/plugins/` (Qt-free)

| Module | Purpose |
|---|---|
| `schema.py` | `ConfigField` — a plugin's declarative config schema. One source for both the settings form **and** availability gating (`gates_availability`). |
| `category.py` | `PluginCategory` adapter + `PLUGIN_CATEGORY_REGISTRY`. Bridges the framework to each category's concrete registry. |
| `manifest.py` | `plugin.toml` parsing/validation (`PluginManifest`). Structural only — never imports plugin code. |
| `security.py` | AST risk scanner (`scan_dir`) + trust-on-first-use hashing (`hash_dir`). |
| `loader.py` | Directory scan → declarative instances + gated code plugins. |

`PluginInstanceConfig` (in `config_models.py`) is the per-instance record shared by all
categories: `instance_id, category, provider_id, label, enabled, is_primary, usage_hint,
settings`. `AlimaConfig.plugins` is the authoritative list.

### Categories

A category owns a concrete registry and a `PluginCategory` adapter that self-registers:

| Category | Registry | Adapter |
|---|---|---|
| `search_provider` | `PROVIDER_REGISTRY` (`src/core/search`) | `SearchProviderCategory` (`search/factory.py`) |
| `input_source` | `INPUT_SOURCE_REGISTRY` (`src/utils/input_sources`) | `InputSourceCategory` (`input_sources/category.py`) |

Adding a new category later = one adapter; the framework, config model, settings UI and
directory loader are already category-agnostic.

## Config model + facade

Instances are authoritative. For the ~298 legacy readers, `CatalogConfig` and the DOI
`SystemConfig` fields are kept as **derived mirrors**:

- **Load** (`config_manager._parse_config`): if a category has no instances yet, they are
  synthesised from the legacy mirrors (`plugin_migration.synthesize_*`).
- **Save** (`config_manager.save_config`): mirrors are re-derived from the primaries
  (`plugin_migration.derive_*_mirrors`) — a `load → save → load` round-trip is diff-free.
- The legacy Catalog/System settings tabs keep working via
  `plugin_migration.sync_instances_from_mirrors` (reverse capture), with the new
  Plugins tab authoritative on save.

**Settings UI consolidation.** The `PluginSettingsTab` is now the single editor for all
provider/source configuration. The old **Catalog tab** and the **DOI-resolution entries
in the System tab** were removed: every catalog field (token, SOAP/web URLs, `catalog_type`
DK-backend selector, `strict_gnd_validation`) is now a `config_field` on the `catalog`
plugin, finc/SRU fields on their plugins, and `contact_email` + the crossref/openalex/
datacite toggles are on the three DOI input-source plugins. `_get_config_from_ui` no longer
builds `CatalogConfig`/the DOI `SystemConfig` fields — they are derived from the instances
on save.

**Multiple instances / primary / usage_hint.** Several instances of one type may coexist
(e.g. two finc endpoints). Exactly one `is_primary` per type drives the classic
single-result path + the mirror; `usage_hint` is appended to the generated MCP tool
description so an agent can steer between siblings. The primary instance of a type keeps
the canonical tool name (`search_lobid`); additional instances get a unique name
(`search_finc_<id>`) and a factory-built handler.

## Directory plugins + two-tier security

Scanned at `~/.config/alima/plugins/<name>/plugin.toml` (merged into the config on load;
zero cost when the dir is absent).

- **Tier 1 — declarative (safe, default).** `kind` references a built-in type of the
  category; the manifest supplies `[settings]`. **No code is executed** → this is how all
  current strategies (extra finc/catalog/SRU/DOI instances) are added.
- **Tier 2 — code (experimental, consent-gated).** A Python class implementing the
  category contract. Gated by, in order: `SystemConfig.enable_code_plugins` (default
  `False`) → AST scan (`security.scan_dir` flags process/network/filesystem/dynamic-code
  use) → SHA-256 trust-on-first-use → explicit operator approval (`approve_cb`) → import.
  Approved hashes are stored in `AlimaConfig.approved_plugins`; a changed hash re-prompts.

> **Honest limit (conservative-assessment rule).** Tier-2 is *informed consent +
> tamper-detection, not a sandbox*. An approved plugin is imported in-process with full
> privileges; the AST scan is a deterrent, not a proof. Prefer Tier-1 for anything shared.

### Manifest example

```toml
# Tier-1 declarative: a second finc endpoint
[plugin]
id = "finc_zbw"
label = "finc ZBW"
category = "search_provider"
type = "declarative"
kind = "finc"

[settings]
base_url = "https://finc.zbw.eu/proxy"
```

```toml
# Tier-2 code: a new input source
[plugin]
id = "arxiv"
label = "arXiv resolver"
category = "input_source"
type = "code"

[entry]
module = "provider.py"
class = "ArxivInputSource"
```

## Adding a provider setting

Declare it once as a `ConfigField` on the provider/source class (`config_fields()`); it
then appears in the settings form, gates availability if `gates_availability=True`, and
round-trips through the config — no other change needed.

## Tests

`tests/test_plugins.py` (framework: schema, security, manifest, loader), 
`tests/test_search_plugins.py` (config_fields, factory, search migration),
`tests/test_input_sources.py` (registry dispatch, url_fetch, DOI split),
`tests/test_plugin_config_roundtrip.py` (end-to-end migration + idempotency).
