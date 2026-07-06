# Plugin Authoring Guide

> Zielgruppe: Dritte, die eine eigene Schnittstelle (Katalog, API, Datenquelle)
> an ALIMA anbinden wollen. Grundlagenarchitektur: [`plugin_system.md`](plugin_system.md).

## 1. Zwei Wege zum Plugin

| | Tier 1 — deklarativ | Tier 2 — Code |
|---|---|---|
| Was | Neue *Instanz* eines eingebauten Typs (z. B. zweiter finc-Endpoint) | Neuer *Typ* (eigenes Protokoll, eigene Quelle) |
| Es läuft | kein fremder Code | Python-Code in-process |
| Braucht | nur `plugin.toml` mit `kind` + `[settings]` | Verzeichnis mit `provider.py` (+ Hilfsmodule) |
| Freigabe | keine | `enable_code_plugins` + AST-Scan + Hash + Operator-Approval |

**Erst prüfen, ob Tier 1 reicht** — ein weiterer VuFind/SRU/Libero-Endpoint ist
eine deklarative Instanz, kein Code-Plugin.

## 2. Quickstart: Blaupause kopieren

Jedes Built-in-Verzeichnis unter `src/core/search/providers/<name>/` ist ein
vollständiges, kopierbares Code-Plugin. Wähle die passende Blaupause:

| Blaupause | Muster |
|---|---|
| `lobid/` | GND-Keywords aus einer JSON-API (`SuggesterBackedProvider`) |
| `swb/` | GND-Keywords per HTML-Scraping (Paginierung, Fehler-Handling) |
| `catalog/` | Token-Auth (SECRET), mehrere Capabilities, SOAP + Web-Fallback |
| `finc/` | Standalone-Provider: Titel-Records + Facetten, per-Term-Metadaten |
| `sru/` | Endpoint-Presets + freie URL, Standardprotokoll (SRU/MARC-XML) |
| `gnd_local/` | Minimalfall: kein Netzwerk, kein Suggester, kein MCP-Tool |

```bash
cp -r src/core/search/providers/lobid ~/.config/alima/plugins/mein_katalog
```

Dann (Reihenfolge egal, alles Pflicht):
1. `plugin.toml`: `id = "mein_katalog"` (nur Kleinbuchstaben/Ziffern/`_`/`-`).
2. `provider.py`: Klassenattribut `id = "mein_katalog"` — **muss dem Manifest
   entsprechen**, sonst verweigert der Loader den Import.
3. Logik anpassen (`suggester.py` bzw. `provider.py`).
4. In ALIMA: Einstellungen → **Plugins-Tab** → „Code-Plugins (Tier 2) erlauben"
   anhaken → „📂 Plugin-Verzeichnis scannen…" → Plugin im Approval-Dialog
   freigeben. (Headless/CLI: `system_config.enable_code_plugins` in
   `~/.config/alima/config.json`; ohne Approval-Callback bleiben Code-Plugins
   dort trotzdem gesperrt — die Freigabe selbst ist GUI-gebunden.)

## 3. Verzeichnis-Anatomie & Regeln

```
mein_katalog/
├── plugin.toml   # Pflicht: [plugin] id/label/category/type="code", [entry], [doc]
├── provider.py   # entry.module: GENAU eine Top-Level-Datei (kein Unterordner,
│                 # kein __init__.py); entry.class: gültiger Python-Identifier
├── suggester.py  # optional: beliebige weitere Module im selben Verzeichnis
└── README.md     # empfohlen: was/Eingabe/Ausgabe/Konfiguration
```

- Der Loader lädt das Verzeichnis als synthetisches Package
  `alima_plugin_<id>` und importiert **nur** das entry-Modul.
  Ein `__init__.py` wird extern **nie** ausgeführt (bei Built-ins ist es
  Registrierungs-Glue).
- **Symlinks sind verboten** (High-Finding, vom Trust-Hash ausgeschlossen;
  ein Symlink als entry-Modul bricht den Load).
- Der Trust-Hash (SHA-256) deckt **alle** Dateien ab — auch Datendateien.
  Jede Änderung erfordert erneute Operator-Freigabe.

**Import-Regel** (macht das Verzeichnis in beiden Modi lauffähig):
- Framework absolut: `from src.core.search.provider import …`,
  `from src.core.search.provider_base import SuggesterBackedProvider`,
  `from src.core.search.registry import register_provider`,
  `from src.core.plugins.schema import ConfigField, PluginDoc`.
- Intra-Plugin relativ, einstufig: `from .suggester import MeinSuggester`.

## 4. Der Provider-Contract (`search_provider`)

```python
@register_provider
class MeinProvider:                      # oder SuggesterBackedProvider-Subklasse
    id = "mein_katalog"                  # == Manifest-id
    label = "Mein Katalog"
    capabilities = {SearchCapability.GND_KEYWORDS}

    @classmethod
    def config_fields(cls) -> list[ConfigField]: ...   # Settings-Form + Gating
    @classmethod
    def doc(cls) -> PluginDoc: ...                     # Selbstbeschreibung (Pflicht)
    @classmethod
    def mcp_tool_specs(cls) -> list[ProviderToolSpec]: ...  # optional: Agent-Tools

    def is_available(self, cfg=None) -> bool: ...
    def search(self, capability, query, *, progress=None, **opts) -> ProviderResult: ...
```

- `SearchCapability`: `GND_KEYWORDS` | `TITLE_RECORDS` | `SUBJECT_FACETS` | `CLASSIFICATION`.
- `ProviderResult.errors[term]` für Quellenfehler (nicht „kein Treffer").
- Kein `mcp_tool_specs()` → Plugin bleibt agent-unsichtbar (nur Pipeline).
- **Nach der Freigabe** erscheint das Plugin automatisch als Instanz im
  Plugins-Tab (Einstellungen dort pflegen; `[settings]` im Manifest liefert
  Startwerte). Seine MCP-Tools werden generiert; behält die Kopie den Toolnamen
  der Blaupause (z. B. `search_catalog`), wird er automatisch suffigiert
  (`search_catalog_<id>`) statt das Built-in zu überschatten — für saubere
  Namen die `name`-Felder in `mcp_tool_specs()` anpassen.
- Für GND-Keyword-Quellen: `SuggesterBackedProvider` erben,
  `_build_suggester()` überschreiben, in `search()` `self._gnd_search(...)`
  aufrufen — Konvertierung, Progress und Raw-Cache-Dual-Write sind geerbt.

## 5. Konfiguration & Secrets

- Jede Einstellung als `ConfigField` deklarieren; `gates_availability=True`
  deaktiviert die Instanz, solange das Feld leer ist.
- `kind=SECRET` ⇒ GUI maskiert, `list_plugins` blendet aus, und zur Laufzeit
  gewinnt die Umgebungsvariable **`ALIMA_PLUGIN_<INSTANCE_ID>_<FELD>`**
  (z. B. `ALIMA_PLUGIN_CATALOG_TOKEN`). Env-Werte werden nie in `config.json`
  persistiert.
- Secrets nie loggen und nie in Cache-Parameter (`raw_cache_params_for`) oder
  Tool-Beschreibungen aufnehmen.

## 6. HTTP-Konventionen

- **Jeder Request mit explizitem `timeout`** — der Approval-Scanner flaggt
  `requests.*()`-Aufrufe ohne Timeout (medium).
- Konfigurierte Basis-URLs: `net_guard.require_http_url` als Schema-Gate
  (killt `file://` & Co.; Intranet bleibt erlaubt).
- Zur Laufzeit gelieferte URLs (User/LLM): `net_guard.fetch_guarded`
  (SSRF-Guard: nur public Hosts, Redirect-Prüfung pro Hop, Größen-Cap;
  Ausnahmen über `SystemConfig.url_fetch_allowlist`).

## 7. Raw-Response-Cache (WP2) — optional, empfohlen

Wenn die Quelle gecacht werden soll, stellt der Suggester bereit:
- `last_raw: {term: raw_json_str}` — Verbatim-Antwort pro Begriff,
- `last_http_status: {term: int}`, `last_errors: {term: str}`,
- ein **pures** `transform(raw) -> {schlagwort: {count, gndid, ddc, dk}}`
  (kein I/O — es läuft auch beim Cache-Read).

`SuggesterBackedProvider._gnd_search` übernimmt den Dual-Write; Standalone-
Provider rufen `ukm.store_raw_response(...)` selbst (Muster: `finc/provider.py`).

## 8. Sicherheitsmodell — ehrlich

Ein freigegebenes Code-Plugin läuft **in-process mit vollen Rechten**. AST-Scan
(Findings nach Severity), Hash-Pinning (Re-Approval bei jeder Änderung) und der
Approval-Dialog sind *informed consent + Manipulationserkennung* — **kein
Sandbox**. Headless-Betrieb (Webapp/CLI ohne Approval-Callback) verweigert
Code-Plugins grundsätzlich. Teile nur Tier-1-Manifeste, wenn du dem Code nicht
traust.

## 9. Plugin testen

Das End-to-End-Muster steht in `tests/test_plugin_blueprint_e2e.py`:
Verzeichnis nach `tmp/plugins/` kopieren, id umbenennen,
`loader.discover(root, approve_cb=lambda *a: True, enable_code_plugins=True)`,
`status == "loaded"` prüfen, Provider aus `get_provider(<id>)` instanziieren
und `search()` gegen einen Fake-Client asserten. Framework-Grenzfälle
(Symlinks, id-Kollision, Multi-File-Import) deckt `tests/test_plugins.py` ab.
