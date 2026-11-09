# Generic Plugin System

> **Status:** ✅ Implemented (July 1, 2026); **hardened + self-contained
> blueprint dirs** (July 6, 2026). Category-agnostic framework in
> `src/core/plugins/`, with two concrete categories: **search providers**
> (`src/core/search/`) and **input sources** (`src/utils/input_sources/`).
> Every built-in search provider is now a copyable plugin directory
> (`src/core/search/providers/<name>/` with plugin.toml + README) — authoring
> guide: [`plugin_authoring.md`](plugin_authoring.md).
> GUI (`PluginSettingsTab`) is operator-click-test-gated.

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
categories: `instance_id, category, provider_id, label, enabled, usage_hint,
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

Instances are authoritative. For the ~53 legacy readers, `CatalogConfig` and the DOI
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

**Multiple instances / usage_hint.** Several instances of one type may coexist
(e.g. two finc endpoints). Where exactly one has to represent the type — the classic
single-result path, the mirrors, the canonical tool name — the **first** instance in
config order is used (`AlimaConfig.type_instance`). It keeps the canonical tool name
(`search_lobid`); additional instances get a unique name (`search_finc_<id>`) and a
factory-built handler. `usage_hint` is appended to the generated MCP tool description so
an agent can steer between siblings.

The `is_primary` flag this replaced was removed in September 2026: every consumer already
fell back to "the first", real configs had it set on every instance, and no install ever
had two instances of one type — so it discriminated nothing while occupying a checkbox on
every plugin panel.

## Directory plugins + two-tier security

Scanned at `~/.config/alima/plugins/<name>/plugin.toml` (merged into the config on load;
zero cost when the dir is absent).

- **Tier 1 — declarative (safe, default).** `kind` references a built-in type of the
  category; the manifest supplies `[settings]`. **No code is executed** → this is how all
  current strategies (extra finc/catalog/SRU/DOI instances) are added.
- **Tier 2 — code (experimental, consent-gated).** A Python class implementing the
  category contract. Gated by, in order: `SystemConfig.enable_code_plugins` (default
  `False`) → AST scan (`security.scan_dir` flags process/network/filesystem/dynamic-code
  use, `requests` calls without timeout, and any symlink) → SHA-256 trust-on-first-use →
  explicit operator approval (`approve_cb`) → import.
  Approved hashes are stored in `AlimaConfig.approved_plugins`; a changed hash re-prompts.

### Hardening (July 2026)

- **Multi-file package loading**: the loader mounts the plugin dir as a synthetic
  package `alima_plugin_<id>` and imports only the manifest's entry module —
  single-level relative imports (`from .suggester import X`) work; the plugin's
  `__init__.py` is never executed. Failed imports clean up `sys.modules`.
- **Entry rules**: `entry.module` must be exactly one top-level `NAME.py`
  (no subpaths, not `__init__.py`, no symlink); `entry.class` must be a valid
  identifier; the loaded class's `id` attribute must equal the manifest id
  (an un-renamed blueprint copy is rejected with a rename hint).
- **Trust hash covers all regular files** (code + manifests + data), relative
  paths included; symlinks are excluded from the hash and reported as high
  findings (they can point outside the approved dir). Widening the coverage
  invalidated pre-existing approvals once (intentional re-approval).
- **Id collisions** with built-ins/registered types fail loudly (`status="error"`)
  and leave the registry untouched.
- **Secrets**: `ConfigField(kind=SECRET)` values can be overridden at runtime via
  `ALIMA_PLUGIN_<INSTANCE_ID>_<KEY>` env vars (applied only at provider/source
  construction — never persisted to config.json/mirrors); GUI masks + shows the
  active override; `list_plugins` excludes secret keys schema-based.
- **Network**: `src/utils/net_guard.py` — operator-configured base URLs get a
  scheme gate (`require_http_url`, intranet allowed) + save-time warnings;
  runtime/LLM-supplied URLs (`scrape_url`, URL input) go through `fetch_guarded`
  (public-address check per redirect hop, size cap,
  `SystemConfig.url_fetch_allowlist` for intranet exceptions). Residual risk
  (DNS rebinding) documented in the module.

> **Honest limit (conservative-assessment rule).** Tier-2 is *informed consent +
> tamper-detection, not a sandbox*. An approved plugin is imported in-process with full
> privileges; the AST scan is a deterrent, not a proof. Prefer Tier-1 for anything shared.
> Headless contexts (webapp/CLI without an approval callback) always deny code plugins.

### Built-in dirs are the blueprints

Every built-in search provider lives in a self-contained dir
`src/core/search/providers/<name>/` (`plugin.toml` + `README.md` + `provider.py`
[+ `suggester.py`]). The same dir, copied to `~/.config/alima/plugins/` and
renamed (manifest `id` **and** class `id`), loads through the Tier-2 path —
proven by `tests/test_plugin_blueprint_e2e.py`. Import rule: framework imports
absolute, intra-plugin imports single-level relative. Shared transport clients
(`src/utils/clients/`) deliberately stay shared (multi-consumer; see the
per-plugin READMEs). `SuggesterBackedProvider` is public API at
`src/core/search/provider_base.py`.

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

## Search sources: state, timeout, run-wide skip, rules in the tab (Nov 2026)

**State per instance ("Zustand": An / Nur Cache / Aus).** Stored as before:
`enabled` plus `settings["cache_only"]` (`src/core/search/cache_only.py`); no
migration, no new key. The tab maps the two values onto one combo
(`instance_state` / `apply_state` in `src/core/search/cache_only.py`, shared
with the institution presets) and does not show the `cache_only` field a second
time. "Aus" leaves a stored `cache_only` as it is;
"An" writes `cache_only = False` only where the key already exists. "Nur Cache"
is offered where the type declares the field: `catalog`, `finc`, `lobid`, `swb`
(raw response cache, `GND_KEYWORDS`), `kvk` (title records) and, since Nov 9,
`sru` (title records and the PICA DK answers; see below). A type without a
cache path can carry a `cache_only_note` and shows the entry disabled; no
built-in type does at present. `gnd_local`, input sources and lookups show only
An / Aus. `NetworkSourceMixin.cache_only` honours the key only on types that
declare it.

**Timeout per network source.** `timeouts.timeout_field` (INT, 1 to 120 s)
with each client's earlier limit as default, read by `timeouts.timeout_from`:
`timeout_s` on lobid (30), swb (15), catalog (10, SOAP search; retries and
backoff unchanged), sru (30); finc and kvk keep their existing key `timeout`
(30). Clients built from an instance get the value, the DK extractors from
`resolve_dk_extractor` included. Fixed limits remain for: Libero title details
(10 s) and both Libero web fallbacks (30 s), the DNB subject dump download of
lobid (60 s), and clients built outside an instance (`alima test-catalog`,
`input_sources/bib_lookup`). A value above 120 stored for finc/kvk `timeout`
is now cut to 120. The tab leaves `timeout_s` (like every other setting, see
"Plugin tab: the instance list") out of an instance that never had it while the
form shows the default, so saving the dialog does not change such a
config.json.

**Run-wide skip (`src/core/search/source_health.py`).** Per instance id,
consecutive connection failures (DNS, refused, connect timeout, TLS/proxy) are
counted; HTTP error statuses, empty hit lists and read timeouts are not, and
every answer from the server, an HTTP error included, resets the count. After
`FAILURE_THRESHOLD = 3` the source is down: `source_health.call` raises
`SourceDownError` without a request, providers return the per-term marker
`source_down:` in `errors` (unknown, never written as "no match" into the
mapping, raw or DK cache), and one notice per source and episode goes to the
log and the state bus. Libero counts each of its three attempts. Reset: at
`PipelineManager.start_pipeline`, at the outermost `WorkflowExecutor.run` /
`MetaAgent.run` (`run_scope`), and `DOWN_EXPIRY_S = 300` after the source went
down (or after the last failure of a partial count). Covered paths: every
request of lobid, swb, BiblioClient (SOAP search, title details, both web
fallbacks), FincClient (search, records, field discovery), MarcXmlClient search
and KvkClient, for clients built from an instance (provider, agent tools, the
classic GND and DK steps). BiblioClient's own circuit breaker (3 failures,
60 s) stays as it was beside it; it only counts exceptions that escape
`search`. An empty Libero search after a connection failure is no longer cached
as `no_results`.

**Rules in the tab.** Each type carries a read-only "Wirkung" line
(`PluginTypeMeta.effect`, filled by the category adapters from capabilities and
agent tools) and, with several instances of the type in the list,
`effect_multi`. Above the search list: "DK-Quelle(n) im Moment: …" from
`factory.describe_dk_source`, which computes the same chain as
`resolve_dk_extractor` (see "DK sources in list order" below) without building
an extractor, and the order rule ("Reihenfolge der Liste gilt: erste aktive
Instanz je Typ; DK-Schritt und search_titles fragen ihre Quellen in dieser
Reihenfolge. Zum Ändern ziehen oder Alt+↑ / Alt+↓."). The line follows unsaved
form edits. A candidate whose extractor fails to build at run time is not
foreseen by it.

Tests: `tests/test_source_states_and_health.py`.

## DK sources in list order, `search_titles`, source overview (Nov 2026)

**DK sources.** `factory.dk_candidates(config)` lists, in the order of
`config.plugins`, the active instances (state An or Nur Cache) whose type
declares CLASSIFICATION and whose DK switch is on: the setting `dk_enabled` on
finc, sru and catalog (absent = off), always on for an active non-built-in
plugin. A catalog instance that never stored `dk_enabled` counts only when no
other instance qualifies, and only the first such instance; a stored value is
taken literally. `resolve_dk_extractor` drops candidates that are not
available (finc without base URL, sru without endpoint; catalog has no gate),
then returns the one extractor unchanged, or a `FallbackDkExtractor`
(`src/core/search/dk_fallback.py`) over two or more. Without any usable
candidate: the first catalog instance without a stored switch, else the
registry catalog with its built-in defaults when no catalog instance is
active, else `NoDkSourceError` (the DK step reports "Keine DK-Quelle").

| Configuration | DK sources |
|---|---|
| catalog, finc, sru active; finc `dk_enabled` off; sru without the key; catalog without the key | Libero alone (as before) |
| a new installation (all defaults; order lobid, swb, finc, catalog, sru, gnd_local, kvk) | Libero alone |
| finc `dk_enabled` on, catalog without the key | finc alone |
| finc on and catalog `dk_enabled` stored on | both, in list order |

**Fallback per keyword.** The first source is asked for every keyword. A
keyword goes on to the next source when the source was down (`source_down:`),
the state Nur Cache had no entry, the institution was not set and nothing came
back, the extractor could not be built, a connection failure was counted by
`source_health` during the call (below the skip threshold), Libero's own
circuit breaker was open, or the extractor raised. "No hits" is an answer and
is not handed on. Each keyword is answered by one source, nothing is merged
within a keyword. The DK step writes `statistics.dk_sources`,
`statistics.dk_source_by_keyword` (keyword → instance id, answered keywords
including those without hits) and `statistics.dk_fallbacks`
(`[{from, to, keywords, reasons}]`), and one line per hand-over, e.g.
"Katalog (Libero): 14 Schlagwörter nicht beantwortet (gesperrt), weiter mit
finc (VuFind)". Keywords no source answered are reported under the reason the
last source gave. Classic pipeline, `dk_search_agentic` and the `rvk_lookup`
tool all go through `execute_notation_search`. finc and Libero share
`catalog_dk_cache`, keyed by keyword only: a cached title list does not record
which of the two wrote it.

**`search_titles`.** Agent tool, generated when at least one active instance
declares TITLE_RECORDS (`src/mcp/_title_search.py`). Parameters `terms`,
`search_type` (kw, title, author, subject, isbn), `max_results` (10, at most
50). Per term the title sources are asked in list order through
`search(TITLE_RECORDS)` of their factory-built providers; the first with a
record answers. Each type maps the axes in `title_search_axes` (catalog:
author `kb`, subject `sk`, isbn `i`; finc has no isbn; a type without the
attribute gets the axis unchanged). Per term: `records` (with `source`),
`answered_by`, `tried` (`ok`, `empty`, `down`, `cache_miss`, `unsupported`
for an unknown axis or a source that is not available, `error`), and `total`,
`truncated`, `notice` when the answering source reports them. The single title
tools are unchanged.

**Source overview.** `src/core/search/source_usage.py` counts per instance id
and run: terms asked, of them from a cache and over the network, keywords with
hits or title records, errors, terms skipped (source down), Nur-Cache misses,
and whether the source went down. Reset together with `source_health`
(`begin_run`, outermost `run_scope`). Counted at: the provider base (GND
keyword search), the mapping cache, the DK extractors (BiblioClient,
FincCatalogClient, MarcXmlClient, PicaDkExtractor), the title tool handlers,
`search_titles`, the GND tool paths that call the suggester directly, and the
lookup tool handler (as calls: whether a lookup asked a server is not known
there). Not counted: `gnd_local`, the DOI input sources, lookups called
directly by the pipeline (`lookups/cache.py`, e.g. the RVK API fallback of the
DK step), `k10plus_resolve` and `bib_lookup` outside the tool handler,
clients built without an instance id (`alima test-catalog`). The rows go into
`KeywordAnalysisState.source_usage` (left out of a saved file while empty),
the export as `results.source_usage`, the result card ("Quellen dieses
Laufs", collapsed above five rows), `alima show-protocol` (detailed), and,
after a chat turn (one `run_scope` per turn in the GUI chat worker and the
headless runner), one `state.notice` line. A search started elsewhere in the
GUI while a run is going on is counted into that run.

Tests: `tests/test_dk_source_chain.py`, `tests/test_search_titles_tool.py`,
`tests/test_source_usage.py`.

## Plugin tab: the instance list (Nov 2026)

**State checkbox.** Each row starts with a three-state checkbox: checked = An,
partially checked = Nur Cache, unchecked = Aus. A click moves on An → Nur Cache
→ Aus → An; Nur Cache is skipped where `cache_only_offer` does not offer it
(gnd_local, input sources, lookups, a type with `cache_only_note`). The state is
kept in one place, the panel's instance list (`enabled` and
`settings["cache_only"]` through `apply_state`); checkbox and the form field
"Zustand" show it and take effect for `collect()` / `apply_to()` without a row
change. After a change a line under the list names the new state (e.g. "finc:
nur Cache (keine Abfragen im Netz)"); it is emptied when another row is
selected. The checkbox has a tooltip of its own (meaning of the state, the state
a click moves to). With Qt 6.10 a click on the checkbox of another row switches
that row without selecting it. An instance whose type is not registered cannot
be switched: no `ItemIsUserCheckable`, the row is drawn with the muted text
token, the tooltip names the reason. The delegate draws the checkbox from the
tokens of `QCheckBox::indicator` (`indicator_colors`): frame `border`, fill
`background`, An filled `primary` with a mark in `on_primary`, Nur Cache an
inner `primary` square, on a selected row the frame in `on_primary`, a locked
row `text_muted` instead of `primary`.

**Order.** Drag and drop (`InternalMove`) and Alt+↑ / Alt+↓ move the selected
row; both arrive as `rowsMoved` of the list model (Alt+arrow calls `moveRow`).
The panel writes the form into its instance, reorders its instance list by the
row keys and keeps the selected instance selected; `collect()` and `apply_to()`
write the list order. Between categories `apply_to` keeps its order (the
managed categories, then all others). Each list has the order rule above it.

**Origin tag.** Derived when the tab loads, not stored:

| Tag | Rule |
|---|---|
| eingebaut | the registered type class lives in a module under `src.` |
| Verzeichnis | as above, and the instance id is the id of a declarative `plugin.toml` in the plugin directory with the same `kind` and category |
| Code-Plugin, freigegeben | the type class lives in `alima_plugin_<id>` (the loader's package for code plugins, loaded only after approval) |
| Code-Plugin, gesperrt | type not registered, a code manifest with that id exists, and code plugins are not allowed (checkbox in the tab, followed live), or the id is not approved, or the approved hash differs from `security.hash_dir` of the directory |
| Code-Plugin, nicht geladen | type not registered, code manifest present, allowed and approved with a matching hash; the error of a scan run in this dialog is quoted |
| unbekannt, nicht geladen | type not registered and no code manifest with that id |
| unbekannt | type registered from any other module |

The class comes from `PluginCategory.type_class` (read-only, implemented by
the search, input-source and lookup adapters). The plugin directory is the one
given to `PluginSettingsTab.load(config, plugins_dir=…)`, else the one of the
running ConfigManager (none is created for it); manifests are read
structurally, no plugin code runs. Not detected: a declarative instance whose
`plugin.toml` was removed (shown as "eingebaut"), and a manifest edited after
the tab was loaded.

**DK.** `dk_source_info` returns the line above the list and the position of
each instance id in the chain from one `describe_dk_source` call over the
dialog state (stored instances plus the unsaved form); rows in the chain carry
"DK 1", "DK 2", …. The field `dk_enabled` of an instance that does not store
the key shows the effective state (the instance is in that chain, e.g. Libero
by the rule for a never stored catalog switch) and follows changes of the
dialog state; unchanged it is not written, a change is written literally. A
stored value is shown as stored, also when the instance is not usable (finc
without base URL). Positions are matched by instance id.

**Unstored settings.** A setting an instance does not store stays out of
config.json as long as its field shows the value it showed when the form was
built. Until Nov 2026 this held for `timeout_s` and `dk_enabled` only; every
other field wrote its default on the first save of the selected row (row 0 of
each category on every save).

Tests: `tests/test_plugin_tab_list.py`.

## Institution: ISIL and ELN (Nov 2026)

`SystemConfig.institution_isil` (e.g. `DE-105`) and `institution_eln` (e.g.
`105`), edited under Einstellungen ▸ Datenquellen ▸ Einrichtung. The ELN is
not derived from the ISIL; empty means not set. `src/utils/institution.py`
derives from them and returns `None` for an empty field:

| Helper | Value | Used by |
|---|---|---|
| `chain_end_marker()` | `$A<ISIL>` | closing line of each chain in the WinIBW entry (`winibw_record`) |
| `finc_dk_facet()` | `udk_raw_<isil, lower case, no hyphen>` | finc DK extractor (per-title facet), finc `search_type="dk"` and the auto-facets of `dk`/`rvk`, `finc_subject_harvest`, the DK links of the Libero web fallback |
| `institution_filter()` | the ISIL | holding filter of the Libero web search (`hiddenFilters institution:"<ISIL>"`) |
| `institution_eln()` | the ELN | copy of the library in a K10plus PICA record (`209A $B`) |

The values in use are process-wide, set by `ConfigManager` on every load and
save. Without ISIL: the WinIBW entry has no closing line and carries
`notes`; the finc DK extractor sends no request for the keyword, the Libero
web fallback skips its search and reads no DK from a record page; both note it
in `last_not_configured`, which the DK step reports once as "Einrichtung nicht
gesetzt" (`statistics.institution_not_set`);
`search_type="dk"` on finc returns the marker `institution_not_set:` per term.
One-time migration on load: a `system_config` without both keys and a
catalog, finc or sru instance with a `tu-freiberg.de` URL gets `DE-105`/`105`
(log line). Both keys stay out of config.json while they are empty, the file
never had them and the migration would not fire. The finc instance setting
`institution_filter` is separate and unchanged.

## K10plus over SRU in PICA-XML: `search_k10plus` (Nov 2026)

An `sru` instance with preset `k10plus` gets the agent tool `search_k10plus`
(`SruProvider.instance_tool_spec`); another preset or a base URL gets
`search_sru` (MARC-XML through `MarcXmlClient`), an instance without endpoint no
tool. The hook `instance_tool_spec(spec, settings)` is optional on any search
type; `None` means no tool for that instance.

- Endpoint `https://sru.k10plus.de/k10plus`, SRU 1.1, always
  `recordSchema=picaxml` (`src/utils/clients/picaxml_client.py`, Qt-free,
  scheme gate of `net_guard`, requests through `source_health`).
- `search_type`: `kw` → `pica.all`, `title` → `pica.tit`, `author` →
  `pica.per` (phrase; "Vorname Nachname" without comma and with exactly two
  words is sent as "Nachname, Vorname"), `subject` → `pica.slw`, `isbn` →
  `pica.isb`, `ppn` → `pica.ppn`. Quotes and backslashes are escaped.
  `max_results` 10 by default, at most 50.
- Per record: `ppn`, `title`, `authors`, `year`, `publisher`, `isbn`, `form`
  (013D), `thesis` (037C), `ddc` (045F/G/H), `subjects` (044K chains with GND
  id and PPN, `source` = the ISIL of the closing `$A` line). With an ELN and a
  copy of that ELN: `local` = `signature`, `notations` (245Z of that copy,
  `{text, ppn}`), `subject_chains` (chains closed by `$A<ISIL>`); otherwise
  `local_hint`. Per term `meta.total` (numberOfRecords), `returned`, and
  `truncated: true` when more exist than came back.
- DK extractor: with preset `k10plus` and an ELN, `PicaDkExtractor` counts
  per keyword (`pica.slw`) the DK/RVK notations of 245Z on the library's
  copies (filter in the client on `209A $B<ELN>`); otherwise the MARC path as
  before. sru is still not a DK source unless `dk_enabled` is set.
- Raw cache: answers of factory-built instances are stored under source `sru`
  with search type, endpoint and schema in the key. "Nur Cache" answers from
  there without request (miss marker `cache_only:`); the MARC DK path has no
  cache and reports every keyword as a miss in that state.
- The field mapping rests on one real record (a printed dissertation with one
  copy); 004A, 033A `$n`, 021A `$d` and 045F/G/H are not checked against a real
  record.

## KVK: `notice` (Nov 2026)

`search_kvk` adds `notice`, one line of plain text, when a union catalog
delivered fewer records than it reported (or the KVK marks it truncated), a
catalog answered with an error other than "Keine Datensätze gefunden" (quoted
with the KVK's message), or `max_results` cut the merged list. It names
`search_k10plus` only when a K10plus catalog was cut and that tool exists in
the running registry. Hook: an optional `tool_notice(result, tools=…)` on a
title provider, read by the title-record handlers.

Tests: `tests/test_institution_setting.py`, `tests/test_k10plus_sru_tool.py`,
`tests/test_kvk_notice.py`.

## Institution presets (Nov 2026)

An institution package (folder with `alima_presets.json` and its files) sets
instance states, order and settings, ISIL/ELN, the standard workflow, and
installs shipped files (the database of `geo_sachdb`, the vocabulary of
`local_scheme`) with checksum. Settings are checked against the type's
`ConfigField` schema; SECRET fields are never set. `alima preset
{apply,revert,export,show}`, GUI group "Einrichtung (Preset)" under the
category lists. The form of a `geo_sachdb` instance shows the state of an
available database (`meta`: source, date, checksum) in the row "Datenbank"
(`register_type_info`). Spec: [`institutional_presets.md`](institutional_presets.md).

## Adding a provider setting

Declare it once as a `ConfigField` on the provider/source class (`config_fields()`); it
then appears in the settings form, gates availability if `gates_availability=True`, and
round-trips through the config — no other change needed.

## Tests

`tests/test_plugins.py` (framework: schema, security, manifest, loader), 
`tests/test_search_plugins.py` (config_fields, factory, search migration),
`tests/test_input_sources.py` (registry dispatch, url_fetch, DOI split),
`tests/test_plugin_config_roundtrip.py` (end-to-end migration + idempotency).
