# Plugin: kvk — Karlsruher Virtueller Katalog

Self-contained Suchprovider-Plugin; Blaupause für **NDJSON-Meta-Suchen über
mehrere Kataloge**. Struktur, Modi und Konventionen wie in
[`../lobid/README.md`](../lobid/README.md).

## Was es tut

| | |
|---|---|
| Capabilities | `title_records` |
| Eingabe | Suchbegriffe + Suchachse (`kw`/`title`/`author`/`subject`/`isbn`) |
| Ausgabe | Titel-Datensätze je Begriff (Titel, Autor, Jahr, Katalog, Link, `ppn`/`idn`/`bvnumber`) |
| MCP-Tool | `search_kvk` |
| Konfiguration | `catalogs` (Katalog-Ids), `base_url`, `timeout`, `max_results` |

## Grenzen — vor der Nutzung lesen

Die KVK-JSON-Antwort enthält je Treffer nur `title`, `author`, `year`, `text`
(Impressum-Zeile), den Link und ein `digital`-Flag. **Keine Schlagworte, keine
DK/RVK/DDC-Notationen.** Damit kann diese Quelle den GND-Pool und den
Klassifikationsschritt nicht bedienen — dafür bleiben `lobid`, `catalog`,
`finc`, `sru`. Was der KVK kann, ist Breite: eine Anfrage erreicht acht
Verbundkataloge.

Weitere Eigenheiten des Formats:

- **Die Feldbelegung wechselt je Katalog.** Die DNB füllt `author`/`year`,
  K10plus lässt beide leer und schreibt alles in `text`
  (`"Quintes, Florian. - Freiburg im Breisgau, 06.07.2026"`). `parse_item`
  liest deshalb notfalls aus `text`.
- **Je Katalog nur die erste Seite.** `catalog_stats[].results` nennt die
  Gesamttrefferzahl, `truncated` sagt, dass mehr existiert.
- **K10plus-Links tragen eine Session-ID** und sind nicht dauerhaft zitierfähig.
- **Kein Raw-Cache.** Das Ergebnis hängt von der Katalogauswahl ab, die der
  Cache-Key (`raw_cache_params_for`: `search_type`/`max_pages`/`facets`) nicht
  ausdrücken kann. Ein stiller Falschtreffer wäre schlimmer als kein Cache.

## Die Brücke zur Anreicherung

Die Identifier stehen im Record-**Link**, nicht in einem JSON-Feld
(`client.extract_identifiers`):

| Katalog | Linkform | Ergebnis |
|---|---|---|
| K10plus | `…?bibtip_docid=1981371435` | `ppn` |
| StaBi Berlin | `stabikat.de/Record/366303287` | `ppn` |
| KOBV | `portal.kobv.de/KobvIndexRecord/gbv_537048642` | `ppn` |
| KOBV (Alma-Quelle) | `…/KobvIndexRecord/almahu_9949983928502882` | — |
| hbz/NRW | `nrw.digibib.net/search/hbzvk/record/9937…` | — |
| DNB | `portal.dnb.de/…?bibtip_docid=1415663890` | `idn` |
| BVB | `gateway-bayern.de/BV044038433` | `bvnumber` |

Eine so gewonnene PPN ist ein **Kandidat** für `k10plus_resolve`, keine Zusage:
sie ist im ausgebenden Katalog gültig, aber nicht jede ist über den
K10plus-SRU-Endpunkt abrufbar, den ALIMA anfragt.

Kataloge, deren Link eine für ALIMA nicht auflösbare Id trägt, liefern bewusst
**keinen** Identifier. KOBV ist der Fall, an dem das hängt: dort stehen
`gbv_<ppn>` und `almahu_<mms-id>` nebeneinander. Eine Alma-Id als „ppn"
weiterzureichen sähe aus wie ein Lookup-Fehlschlag statt wie die falsche Id,
die sie ist.

## Katalog-Ids

`catalogs` erwartet die Ids aus dem eigenen KVK-Suchlink (`kataloge=…`),
Default: `K10PLUS, BVB, NRW, HEBIS, HEBIS_RETRO, KOBV_SOLR, DDB, STABI_BERLIN`.
Die KVK-Startseite liegt hinter einer JS-Bot-Challenge, die vollständige Liste
ist also nicht maschinell abrufbar — Ids aus der Browser-URL übernehmen.

## Spezifisch für diese Blaupause

- **NDJSON statt einem JSON-Dokument**: ein Objekt je Zeile, je Katalog eines,
  plus ein abschließender `{"type":"error"}`-Block. `parse_response` überspringt
  eine kaputte Zeile, statt die ganze Antwort zu verlieren.
- **Trennung Fehler ↔ kein Treffer**: `ProviderResult.errors` bekommt nur den
  Transportfehler (die Quelle hat versagt). Kataloge, die nichts fanden, stehen
  in `per_term_meta[term]["catalog_errors"]` — sonst läse sich „nichts gefunden"
  wie ein Ausfall.
- **Reines Parsing getrennt vom HTTP** (`client.parse_response`/`parse_item`),
  daher ohne Netz testbar.
- Beim Kopieren umbenennen (müssen übereinstimmen): `id` in `plugin.toml`
  **und** das `id`-Klassenattribut in `provider.py`.
