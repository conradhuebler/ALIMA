# Plugin: sru — SRU/MARC-XML-Endpunkte

Self-contained Suchprovider-Plugin; Blaupause für **standardprotokoll-basierte
Plugins mit Endpoint-Presets** (SRU 1.1, MARC-XML). Struktur, Modi und
Konventionen wie in [`../lobid/README.md`](../lobid/README.md).

## Was es tut

| | |
|---|---|
| Capabilities | `title_records`, `classification` |
| Eingabe | Suchbegriff + Suchtyp (`keyword`/`title`/`author`/`subject`/`isbn`) |
| Ausgabe | Geparste MARC-Records (Titel, Autoren, Klassifikationen …) |
| MCP-Tool | keines (pipeline-intern; `mcp_tool_specs()` bewusst nicht deklariert) |
| Konfiguration | `preset` (CHOICE: dnb/k10plus/loc/…) oder freie `base_url` (URL), Schema, Limits |

## Spezifisch für diese Blaupause

- **Preset + freie URL**: zeigt das Muster „bekannte Endpunkte als CHOICE,
  Custom-Endpoint als URL-Feld" — die URL wird zur Laufzeit per
  `net_guard.require_http_url` (Schema-Gate) geprüft.
- **Shared Transport-Client**: nutzt `src/utils/clients/marcxml_client.py`
  (CQL-Query-Bau, SRU-Paging, MARC-Parsing).
- Beim Kopieren umbenennen (müssen übereinstimmen): `id` in `plugin.toml`
  **und** das `id`-Klassenattribut in `provider.py`.
