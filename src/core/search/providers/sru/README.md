# Plugin: sru — SRU/MARC-XML-Endpunkte

Self-contained Suchprovider-Plugin; Blaupause für **standardprotokoll-basierte
Plugins mit Endpoint-Presets** (SRU 1.1, MARC-XML). Struktur, Modi und
Konventionen wie in [`../lobid/README.md`](../lobid/README.md).

## Was es tut

| | |
|---|---|
| Capabilities | `title_records`, `classification` |
| Eingabe | Suchbegriff + Suchtyp (`kw`/`title`/`author`/`subject`/`isbn`, bei k10plus auch `ppn`) |
| Ausgabe | Preset k10plus: PICA-Sätze mit lokalen Daten der Bibliothek; sonst geparste MARC-Records |
| MCP-Tool | Preset k10plus: `search_k10plus`; anderes Preset oder `base_url`: `search_sru`; ohne Endpunkt keines (`instance_tool_spec`) |
| Konfiguration | `preset` (CHOICE: dnb/k10plus/loc/…) oder freie `base_url` (URL), Schema, Limits, `timeout_s`, `dk_enabled`, `cache_only` (Zustand „Nur Cache“) |

## Preset k10plus: PICA-XML

- Endpunkt `https://sru.k10plus.de/k10plus` (Datenbank mit lokalen Daten),
  SRU 1.1, immer `recordSchema=picaxml`; Client und Parser:
  `src/utils/clients/picaxml_client.py`.
- Indizes: `kw` → `pica.all`, `title` → `pica.tit`, `author` → `pica.per`
  (Phrase; „Vorname Nachname“ ohne Komma und mit genau zwei Wörtern wird zu
  „Nachname, Vorname“), `subject` → `pica.slw`, `isbn` → `pica.isb`, `ppn` →
  `pica.ppn`. Anführungszeichen und Backslash werden maskiert. `max_results`
  10, höchstens 50.
- Je Satz `ppn`, `title`, `authors`, `year`, `publisher`, `isbn`, `form`,
  `thesis`, `ddc`, `subjects` (044K-Ketten mit GND-ID und PPN). Mit ELN aus der
  Einrichtung (Einstellungen ▸ Datenquellen) und einem Exemplar `209A $B<ELN>`
  zusätzlich `local` (`signature`, `notations` aus 245Z dieses Exemplars,
  `subject_chains` mit `$A<ISIL>`), sonst `local_hint`. Je Begriff `meta.total`
  und `truncated`.
- DK-Schritt (nur mit `dk_enabled`): mit ELN zählt `PicaDkExtractor` je
  Schlagwort die lokalen Notationen der Exemplare der Bibliothek (Filter im
  Client, nicht im Suchindex; nur die ersten `max_results` Treffer). Ohne ELN
  der MARC-Weg über `MarcXmlClient.KNOWN_ENDPOINTS["k10plus"]` (`opac-de-627`).
- Feldzuordnung aus einem echten Satz abgeleitet; 004A, 033A `$n`, 021A `$d`
  und 045F/G/H sind nicht an einem echten Satz geprüft.

## Nur Cache

Antworten fabrikgebauter Instanzen (Titelsuche, PICA-DK-Suche) stehen im
Rohantwort-Cache unter `sru` mit Suchtyp, Endpunkt und Schema im Schlüssel. Im
Zustand „Nur Cache“ kommt alles von dort, ohne Anfrage; ein fehlender Begriff
trägt den Marker `cache_only:`. Der MARC-DK-Weg hat keinen Cache und meldet in
diesem Zustand jeden Begriff als nicht im Cache.

## Spezifisch für diese Blaupause

- **Preset + freie URL**: zeigt das Muster „bekannte Endpunkte als CHOICE,
  Custom-Endpoint als URL-Feld" — die URL wird zur Laufzeit per
  `net_guard.require_http_url` (Schema-Gate) geprüft.
- **Shared Transport-Clients**: `src/utils/clients/marcxml_client.py`
  (CQL-Query-Bau, MARC-Parsing) und `src/utils/clients/picaxml_client.py`
  (K10plus in PICA-XML).
- Beim Kopieren umbenennen (müssen übereinstimmen): `id` in `plugin.toml`
  **und** das `id`-Klassenattribut in `provider.py`.
