# Plugin: lobid — GND-Schlagwortsuche (Lobid-API der DNB)

Dieses Verzeichnis ist ein **self-contained Suchprovider-Plugin** und zugleich die
**Blaupause** für eigene GND-Keyword-Plugins (JSON-API-Variante). Es funktioniert
in zwei Modi identisch:

* **Built-in**: statisch importiert über `providers/__init__.py` (`__init__.py`
  hier registriert die Klasse per `@register_provider`).
* **Externes Code-Plugin**: das komplette Verzeichnis nach
  `~/.config/alima/plugins/<dein-name>/` kopieren; der Loader lädt es über
  `plugin.toml` (Tier 2, consent-gated).

## Was es tut

| | |
|---|---|
| Capability | `gnd_keywords` — Suchbegriff → GND-Schlagwort-Kandidaten |
| Eingabe | Ein oder mehrere Suchbegriffe (`search_type`: `kw`/`title`/`freetext`) |
| Ausgabe | `{Begriff: {Schlagwort: {count, gnd_ids, classifications}}}` als `ProviderResult` (Vertrag v2) |
| MCP-Tool | `search_lobid` (aus `mcp_tool_specs()` generiert) |
| Konfiguration | keine (öffentliche API, kein Token) |

## Anatomie

```
lobid/
├── plugin.toml   # Manifest: id, category, [entry] module/class, [doc]
├── __init__.py   # NUR Built-in-Glue (Re-Export); extern nie ausgeführt
├── provider.py   # LobidProvider: Contract (id/label/capabilities/search/doc)
└── suggester.py  # HTTP + Transform: fetch() (I/O) getrennt von transform() (pur)
```

**Import-Regel** (macht das Verzeichnis kopierbar):
- Framework-Imports **absolut**: `from src.core.search.provider import …`,
  `from src.core.search.provider_base import SuggesterBackedProvider`,
  `from src.core.search.registry import register_provider`.
- Innerhalb des Plugins **relativ einstufig**: `from .suggester import LobidSuggester`.

## Eigenes Plugin aus dieser Blaupause

1. Verzeichnis kopieren: `cp -r lobid ~/.config/alima/plugins/mein_katalog`
2. **Umbenennen (beide Stellen, müssen übereinstimmen):**
   - `plugin.toml` → `id = "mein_katalog"`
   - `provider.py` → `id = "mein_katalog"` (Klassenattribut)
3. `suggester.py` anpassen: `fetch()` (HTTP zur eigenen Quelle) und
   `transform(raw)` (Verbatim-Antwort → `{schlagwort: {count, gnd_ids,
   classifications}}`; `classifications` ist `{System: Codes}` mit den
   kanonischen System-Keys `DK`/`DDC`/`RVK`).
4. In ALIMA: Einstellungen → **Plugins-Tab** → „Code-Plugins (Tier 2) erlauben"
   anhaken, dann „📂 Plugin-Verzeichnis scannen…" und das Plugin im
   Approval-Dialog freigeben (AST-Scan-Findings + Hash werden angezeigt; jede
   Dateiänderung erfordert erneute Freigabe).

## Konventionen (Pflicht für Plugins)

- **HTTP**: jeder Request mit explizitem `timeout` (der Security-Scanner flaggt
  `requests.*` ohne Timeout); URLs zur Laufzeit nur http(s) —
  Helfer: `src/utils/net_guard.py`.
- **Secrets**: als `ConfigField(kind=SECRET)` deklarieren — sie werden in der GUI
  maskiert, im `list_plugins`-Tool ausgeblendet und können zur Laufzeit per
  `ALIMA_PLUGIN_<ID>_<FELD>`-Umgebungsvariable überschrieben werden. Nie in
  Log-Ausgaben oder Cache-Parameter schreiben.
- **Raw-Cache (WP2)**: für Response-Caching stellt der Suggester `last_raw`
  (`{term: raw_json_str}`), `last_http_status` und ein *pures* `transform(raw)`
  bereit; `SuggesterBackedProvider._gnd_search` übernimmt den Dual-Write.
- **Sicherheitsmodell**: Code-Plugins laufen nach Freigabe **in-process mit
  vollen Rechten** — Consent + Manipulationserkennung, KEIN Sandbox. Details:
  `docs/plugin_system.md`, Anleitung: `docs/plugin_authoring.md`.
