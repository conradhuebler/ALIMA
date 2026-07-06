# Plugin: gnd_local — Lokale GND-Datenbank

Self-contained Suchprovider-Plugin; **Minimal-Blaupause** für Provider ohne
Netzwerk und ohne Suggester-Zwischenschicht (implementiert den
`SearchProvider`-Contract direkt, ohne `SuggesterBackedProvider`). Struktur,
Modi und Konventionen wie in [`../lobid/README.md`](../lobid/README.md).

## Was es tut

| | |
|---|---|
| Capability | `gnd_keywords` |
| Eingabe | Ein oder mehrere Suchbegriffe |
| Ausgabe | GND-Kandidaten aus der lokalen `gnd_entries`-Tabelle |
| MCP-Tool | keines (`mcp_tool_specs()` fehlt bewusst — rein pipeline-intern) |
| Konfiguration | keine |

## Spezifisch für diese Blaupause

- Zeigt das **kleinste vollständige Plugin**: `id`/`label`/`capabilities` +
  `is_available()` + `search()` in einer Datei; kein Caching-Wrapper nötig
  (es *ist* der lokale Bestand).
- Kein `mcp_tool_specs()` → das Plugin ist für Agenten unsichtbar, wird aber
  von der Pipeline über die Registry gefunden — Muster für interne Quellen.
- Beim Kopieren umbenennen (müssen übereinstimmen): `id` in `plugin.toml`
  **und** das `id`-Klassenattribut in `provider.py`.
