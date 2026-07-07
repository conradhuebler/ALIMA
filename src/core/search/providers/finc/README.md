# Plugin: finc — VuFind-/JSON-API-Katalog

Self-contained Suchprovider-Plugin; Blaupause für **Record-/Facetten-Plugins**
(liefert bibliografische Datensätze und Klassifikations-Verteilungen statt
GND-Keywords) und für **Standalone-Provider ohne `SuggesterBackedProvider`**.
Struktur, Modi und Konventionen wie in [`../lobid/README.md`](../lobid/README.md).

## Was es tut

| | |
|---|---|
| Capabilities | `title_records`, `subject_facets`, `classification` (DK/RVK pro Keyword) |
| Eingabe | Suchanfrage (Subject/Titel/Autor/Freitext), optional `facets` (z. B. `udk_raw_de105`, `rvk_facet`) |
| Ausgabe | Titel-Records (id, Titel, Autoren, Verlag, Auflage, Jahr, ISBN, Subjects, web_url, resource_url …) bzw. Facetten-Verteilungen |
| MCP-Tool | `search_finc` |
| Konfiguration | `base_url` (URL, gated — ohne URL inaktiv), `web_record_url`, Limits/Timeout |

## Spezifisch für diese Blaupause

- **Standalone-Contract**: `provider.py` implementiert `SearchProvider` direkt
  (eigene Config-Verwaltung, `availability_ok` gegen `config_fields`,
  Raw-Dual-Write über `_store_finc_raw`) — Muster für Quellen, die nicht auf
  dem GND-Keyword-Suggester-Schema aufsetzen.
- **Per-Term-Metadaten**: `ProviderResult.per_term_meta` transportiert
  `result_count`/`facets` zusätzlich zu den Records.
- **`classification`-Capability (DK/RVK)**: `provider.dk_extractor()` liefert den
  `FincCatalogClient` (im Plugin-Dir: `finc_catalog_client.py`), der die geteilte
  `extract_dk_classifications_for_keywords`-Schnittstelle des klassischen
  DK-Schritts implementiert (Titelliste per finc-Subject-Suche, dann `udk_raw`/
  `rvk_facet` pro Titel). `execute_dk_search` wählt die DK-Quelle über die
  `CLASSIFICATION`-Capability aus den aktiven Plugins — ein anderes Katalog-Plugin
  mit dieser Capability + `dk_extractor()` wird ohne Core-Änderung zur DK-Quelle.
- **Vendored Transport-Client**: `finc_client.py` liegt **im Plugin-Verzeichnis**
  (Rate-Limit, Timeout, Fehler-Dicts) und wird von `suggester.py` relativ
  importiert (`from .finc_client import FincClient`) — dadurch ist das
  Verzeichnis kopierbar/self-contained. `src/utils/clients/finc_client.py` ist
  nur noch ein Re-Export-Shim auf diese Datei (single source of truth), damit die
  DK-Suche der Pipeline den Client weiter über den alten Pfad konsumieren kann.
- Beim Kopieren umbenennen (müssen übereinstimmen): `id` in `plugin.toml`
  **und** das `id`-Klassenattribut in `provider.py`.
- **`FincClient.get_records(ids)`**: exakter Lookup über `{base_url}/api/v1/record`
  (ein oder mehrere finc-IDs, z.B. nach Abgleich einer Libero-RSN gegen die
  "0-"-prefixte finc-ID), gleiche normalisierte Record-Form wie `search()`.
  Bisher nur Client-Ebene — kein eigenes MCP-Tool/`ProviderToolSpec` dafür
  (würde `search_finc` als einzigen Tool-Namen der Blaupause ergänzen).
- **Feldliste ist instanzspezifisch verifiziert, nicht garantiert portabel**:
  `FincClient.DEFAULT_FIELDS` (Jahr/Verlag/Auflage/ISBN/DOI zusätzlich zum
  proxy-seitigen 8-Feld-Default) ist gegen die TU-Freiberg-Instanz per
  Swagger-Spec verifiziert (`{base_url}/api?swagger`). Andere finc/VuFind-
  Instanzen können ein anderes Solr-Schema haben. Vor dem Anpassen dieser
  Blaupause auf eine andere Instanz: `FincClient(base_url=...).discover_fields()`
  aufrufen (liest die Swagger-`Record`-Schema-Properties live aus) und mit
  `DEFAULT_FIELDS` abgleichen, bevor man sich auf die Feldliste verlässt.
