# Plugin: catalog — Lokaler Bibliothekskatalog (Libero/SOAP)

Self-contained Suchprovider-Plugin; Blaupause für **Token-authentifizierte,
Multi-Capability-Plugins** (SOAP/HTTP-Katalog mit drei Gesichtern). Struktur,
Modi und Konventionen wie in [`../lobid/README.md`](../lobid/README.md).

## Was es tut

| | |
|---|---|
| Capabilities | `gnd_keywords`, `title_records`, `classification` (DK) |
| Eingabe | Suchbegriffe / Titelanfragen (je nach Capability) |
| Ausgabe | GND-Schlagworte, Titel-Records (rsn, web_url, …) oder DK-Notationen |
| MCP-Tools | `search_catalog`, `search_catalog_titles` |
| Konfiguration | `token` (SECRET, gated), SOAP-/OPAC-URLs, DK-Backend-Wahl |

## Spezifisch für diese Blaupause

- **Mehrere Capabilities**: `search()` verzweigt auf die angefragte Capability
  (`_require` + pro-Capability-Zweig in `provider.py`) — Muster für Plugins,
  die mehr als GND-Keywords liefern.
- **Secret-Konfig**: `token` ist `ConfigField(kind=SECRET, gates_availability=True)`
  → ohne Token ist die Instanz inaktiv; GUI maskiert; Laufzeit-Override via
  `ALIMA_PLUGIN_CATALOG_TOKEN`.
- **Shared Transport-Client**: `suggester.py` (ehem. BiblioSuggester) nutzt den
  geteilten `src/utils/clients/biblio_client.py` (SOAP + Web-Fallback), der auch
  von der DK-Suche der Pipeline direkt konsumiert wird. Beim Kopieren als
  eigenes Plugin: entweder ALIMAs Client absolut importieren oder einen eigenen
  Client als weitere Datei ins Plugin-Verzeichnis legen.
- Beim Kopieren umbenennen (müssen übereinstimmen): `id` in `plugin.toml`
  **und** das `id`-Klassenattribut in `provider.py`.
