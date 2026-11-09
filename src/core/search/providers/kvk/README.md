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
| Konfiguration | `catalogs` (Katalog-Ids), `base_url`, `timeout` (1 bis 120 s), `max_results`, `cache_only` (Zustand „Nur Cache“: Titel aus dem Rohantwort-Cache, Schlüssel Suchachse; die Katalogauswahl gehört nicht dazu) |

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
  Gesamttrefferzahl, `truncated` sagt, dass mehr existiert. Das Werkzeug fasst
  das in `notice` zusammen (eine Zeile Klartext), wenn ein Katalog weniger
  lieferte, als er meldete, ein Katalog mit einer anderen Meldung als „Keine
  Datensätze gefunden“ antwortete (zitiert) oder `max_results` die
  zusammengeführte Liste kürzte; `search_k10plus` wird nur genannt, wenn ein
  K10plus-Katalog gekürzt war und das Werkzeug in der laufenden Konfiguration
  existiert (`KvkProvider.tool_notice`).
- **K10plus-Links tragen eine Session-ID** und sind nicht dauerhaft zitierfähig.
- **Raw-Cache nur nach Suchachse.** Fabrikgebaute Instanzen legen jede
  beantwortete Suche ab (Schlüssel: `search_type`), der Zustand „Nur Cache“
  liest dort. Die Katalogauswahl gehört nicht zum Schlüssel: nach einer
  Änderung der Kataloge liefert „Nur Cache“ die ältere Auswahl.

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
