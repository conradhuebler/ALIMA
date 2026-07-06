# Plugin: finc — VuFind-/JSON-API-Katalog

Self-contained Suchprovider-Plugin; Blaupause für **Record-/Facetten-Plugins**
(liefert bibliografische Datensätze und Klassifikations-Verteilungen statt
GND-Keywords) und für **Standalone-Provider ohne `SuggesterBackedProvider`**.
Struktur, Modi und Konventionen wie in [`../lobid/README.md`](../lobid/README.md).

## Was es tut

| | |
|---|---|
| Capabilities | `title_records`, `subject_facets` |
| Eingabe | Suchanfrage (Subject/Titel/Autor/Freitext), optional `facets` (z. B. `udk_raw_de105`, `rvk_facet`) |
| Ausgabe | Titel-Records (id, Titel, Autoren, Subjects, web_url …) bzw. Facetten-Verteilungen |
| MCP-Tool | `search_finc` |
| Konfiguration | `base_url` (URL, gated — ohne URL inaktiv), `web_record_url`, Limits/Timeout |

## Spezifisch für diese Blaupause

- **Standalone-Contract**: `provider.py` implementiert `SearchProvider` direkt
  (eigene Config-Verwaltung, `availability_ok` gegen `config_fields`,
  Raw-Dual-Write über `_store_finc_raw`) — Muster für Quellen, die nicht auf
  dem GND-Keyword-Suggester-Schema aufsetzen.
- **Per-Term-Metadaten**: `ProviderResult.per_term_meta` transportiert
  `result_count`/`facets` zusätzlich zu den Records.
- **Shared Transport-Client**: `suggester.py` nutzt den geteilten
  `src/utils/clients/finc_client.py` (Rate-Limit, Timeout, Fehler-Dicts), der
  auch von der DK-Suche der Pipeline direkt konsumiert wird. Eigenes Plugin:
  Client absolut importieren oder eigenen Client ins Verzeichnis legen.
- Beim Kopieren umbenennen (müssen übereinstimmen): `id` in `plugin.toml`
  **und** das `id`-Klassenattribut in `provider.py`.
