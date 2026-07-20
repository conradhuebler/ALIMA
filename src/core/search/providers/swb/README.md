# Plugin: swb — GND-Schlagwortsuche (SWB-Verbundkatalog, BSZ)

Self-contained Suchprovider-Plugin; Blaupause für die **HTML-Scraping-Variante**
eines GND-Keyword-Plugins (Quelle liefert kein JSON, sondern OPAC-Seiten, die
mit BeautifulSoup geparst werden). Struktur, Modi (built-in / externes
Code-Plugin) und Konventionen sind identisch zur JSON-API-Blaupause —
siehe [`../lobid/README.md`](../lobid/README.md).

## Was es tut

| | |
|---|---|
| Capability | `gnd_keywords` |
| Eingabe | Suchbegriffe; `search_type` `kw`/`title`/`freetext`, `max_pages` (Paginierung) |
| Ausgabe | `{Begriff: {Schlagwort: {count, gnd_ids, classifications}}}` als `ProviderResult` (Vertrag v2) |
| MCP-Tool | `search_swb` |
| Konfiguration | keine (öffentlicher Katalog) |

## Spezifisch für diese Blaupause

- `suggester.py` zeigt: Session-loses `requests.get` mit Modul-Konstante
  `REQUEST_TIMEOUT_S`, HTML-Entschärfung (`html.unescape`), Paginierung über
  Folgeseiten-Links (`max_pages`-Deckel), Fehler-Seite ≠ leeres Ergebnis
  (`had_error` verhindert Cache-Vergiftung).
- Beim Kopieren umbenennen (müssen übereinstimmen): `id` in `plugin.toml`
  **und** das `id`-Klassenattribut in `provider.py`.
