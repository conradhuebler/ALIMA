# WP: Sacherschließungsdatenbank Geowissenschaften als ALIMA-Plugin (`geo_sachdb`)

> **Stand (4.11.2026, Branch `feature/geo-sachdb-plugin`):** Vorbereitung.
> Vorhanden: Umwandler `scripts/access_to_sql.py`, Lookup-Plugin-Gerüst
> `src/utils/lookups/geo_sachdb/` (nicht als Built-in registriert), kleines
> Verfügbarkeits-Gate im Werkzeuggenerator, deterministischer Schritt
> `geo_candidates` in `alima_v51_105`. Nicht vorhanden: Registrierung als
> Built-in, GUI-Anbindung, Übernahme ins Ergebnis (`KeywordAnalysisState`),
> LLM-Auswahlschritt. Die Rollen-Abbildung (welche Spalte ist Schlagwort, welche
> Notation) ist ein **Vorschlag** und vom Entwickler zu bestätigen (Abschnitt 8).
>
> Kennzeichnung: **[gemessen]** = mit Werkzeug an den Kopien in `.scratch/`
> ermittelt, **[gelesen]** = aus Code/Schema, **[vermutet]** = Deutung ohne Beleg.

## 1. Ziel und Umfang

Die Access-Datenbank `SACH-GEO1.mdb` (lokale Sacherschließung Geowissenschaften)
soll ALIMA als Nachschlagequelle dienen: lokale Schlagwortsätze zu einem Begriff,
lokale DK-Notationen zu einem Schlagwort, Abgleich vergebener Notationen mit dem
lokalen Bestand. Lokale Daten schlagen vor, entscheiden nie (Grundsatz wie bei
`local_scheme`/`shelf_*`).

## 2. Ist-Analyse des Plugin-Systems [gelesen]

| Baustein | Beleg | Bedeutung für dieses WP |
|---|---|---|
| Drei Kategorien `search_provider`, `input_source`, `lookup` | `src/utils/lookups/__init__.py:17-21`, `docs/wp_search_tool_plugin_potential.md` B4 | `lookup` ist die Kategorie für Autoritäts-/Nachschlagefragen |
| Lookup-Vertrag: `id`, `label`, `config_fields`, `doc`, `mcp_tool_specs`, `__init__(**config)` | `src/utils/lookups/registry.py:48-73` | Ergebnis ist ein freies Dict; Dispatch über `LookupToolSpec.method` (`registry.py:27-45`) |
| Konstruktion mit Instanz-Settings + Env-Overrides | `src/utils/lookups/category.py:44-54` | Rollen-Abbildung kommt als Settings in den Konstruktor |
| Ein Konstruktionspunkt für Nicht-Agent-Aufrufer | `src/utils/lookups/resolve.py:67-87` (`build_lookup`, `None` bei abgeschalteter oder fehlender Instanz) | Workflow-Funktion nutzt ihn wie `shelf_candidates` |
| Werkzeugerzeugung je aktivierter Instanz | `src/mcp/_tool_generation.py:156-219` | vorher ohne Verfügbarkeitsprüfung für Lookups |
| Instanz-Seeding: jede registrierte Lookup-Art bekommt beim Laden eine **aktivierte** Instanz | `src/utils/plugin_migration.py:283-330` | Grund, das Plugin **nicht** als Built-in zu registrieren: es stünde sonst sofort in jeder Konfiguration |
| `gates_availability` wird nur für Suchquellen ausgewertet | `src/core/search/factory.py:131-143`, `src/core/search/provider_base.py:82` | für Lookups wirkungslos, daher eigenes `is_available()` |
| Suchquellen prüfen `is_available()` erst beim Aufruf | `src/mcp/_tool_generation.py:317`, `:382` | Werkzeug bleibt sichtbar und meldet Fehler |
| Vorbild lokale Datei: `local_scheme` | `src/utils/lookups/local_scheme.py:66-70` (Suchpfade), `:80` (`digits_of`), `:276` (`cacheable=False`) | Notationsnähe per Ziffernpräfix wird wiederverwendet |
| Vorbild eigene SQLite: `webindex` | `src/utils/lookups/webindex/store.py` (eigene DB über `DatabaseManager`) | hier statt QtSql das Standardmodul `sqlite3` mit `mode=ro`, weil nur gelesen wird |
| Tier-2-Laden registriert die Klasse selbst | `src/core/plugins/loader.py:281` (`register_code_type`) | Plugin braucht keinen Import-Seiteneffekt |
| Lokale Daten im Workflow: `shelf_candidates` → `shelf_select` → `shelf_guard` | `src/core/agents/deterministic_functions.py:1843-1909`, `:1912`; `workflows/alima_v51_105.yaml:897-` | `build_lookup` → `None` ergibt leeres Ergebnis (`deterministic_functions.py:1890-1892`) |
| Aufstellung gelangt ins Ergebnis über `extra.shelf_final` | `src/core/agents/shared_context.py:511-516` | für `geo_sachdb` gibt es (noch) keine solche Übernahme |

### Entscheidung: Kategorie `lookup`

- **Passt:** Die Daten beantworten Nachschlagefragen (Begriff → lokaler Satz,
  Notation → lokale Einträge). Lookups sind genau dafür gebaut, haben Werkzeuge,
  Instanz-Settings und mit `build_lookup` einen Pfad für Pipeline/Workflow.
- **Suchquelle (`search_provider`) passt jetzt nicht:** `GND_KEYWORDS` erwartet
  GND-IDs je Schlagwort (`docs/plugin_authoring.md` §7). Die Datei enthält
  `!…!`-Kennungen, deren Zuordnung zu GND-IDs nicht geklärt ist (Frage F3).
  `CLASSIFICATION` (Schlagwort → DK) wäre fachlich denkbar, setzt aber geklärte
  Semantik voraus und speist dann die klassische Pipeline. Phase 4.
- **Neue Kategorie `local_database`: nicht nötig.** Sie würde Konfiguration,
  Werkzeugerzeugung und Seeding der Lookup-Kategorie verdoppeln. Gefehlt hat nur
  ein Verfügbarkeits-Gate.

### Erweiterung des Rahmens (umgesetzt, klein)

`ToolGenerationMixin._lookup_instance_available` (`src/mcp/_tool_generation.py:221-242`,
aufgerufen in `:199`): Deklariert eine Lookup-Klasse `is_available()`, bekommt
eine Instanz nur Werkzeuge, solange sie True liefert. Klassen ohne die Methode
werden dort nicht gebaut und behalten ihre Werkzeuge (alle fünf vorhandenen).
Dokumentiert am Protokoll (`src/utils/lookups/registry.py:56-58`). Test:
`tests/test_geo_sachdb_lookup.py::ToolGenerationGate`.

## 3. Quelldaten: Befunde

### 3.1 Die beiden Dateien [gemessen]

| | `/home/conrad/src/ALIMA/SACH-GEO1.mdb` | `…/Dokumente/Bibliothek/datenbank/SACH-GEO1.mdb` |
|---|---|---|
| Größe / Stand | 39 796 736 B, 4.11.2026 | 19 501 056 B, 4.9.2025 |
| SHA-256 (Anfang) | `8b5dccfe04172591` | `a559c08f1f609c05` |
| Format (`mdb-ver`) | JET4 | JET4 |
| Seiten à 4096 B | 9716 | 4761 |
| Benutzertabellen (`mdb-tables`) | 11, dieselben Namen | 11 |
| Schema (`mdb-schema … access`) | identisch (`diff` leer) | |

Nur `SWB` und `DK` unterscheiden sich im Inhalt; die übrigen 9 Tabellen sind als
Zeilen-Multimenge identisch.

| Tabelle | Zeilen neu | Zeilen alt | Unterschied |
|---|---|---|---|
| `SWB` | 16 664 | 14 509 | nach `LAUF_NR`: 2319 neu, 164 entfallen, 5094 geändert (davon `SWB_AEQUIV` 4693, `OBER_BEGR` 641, `DK_ID…DK_ID3` 771/777/372/152, `NOTES` 1) |
| `DK` | 7 365 (`mdb-count`: 7 355) | 6 653 (`mdb-count`: 6 653) | nach `DK_ID`: 723 neu, 7 entfallen, 1664 geändert (`DK` 1509, `ERL_UTER` 258) |

**Die neue Datei (4.11.2026) ist die aktuellere und vollständigere:** gleiche
Struktur, mehr Sätze, keine Tabelle fehlt.

Die Verdopplung der Dateigröße passt zu gelöschten Datensatz-Slots, die Access
erst beim Komprimieren freigibt: gelöschte Slots `SWB` 1931 → 5450, `DK`
1071 → 2496, `MSysAccessStorage` 321 → 2002; Datenseiten `SWB` 2644 → 5967,
`DK` 1306 → 2708 [gemessen mit `access_parser`; Deutung „nicht komprimiert"
vermutet].

**Abweichung `DK` neu:** `mdb-export` und `access_parser` lesen übereinstimmend
7365 Datensätze, `mdb-count` (Zeilenzähler im Tabellenkopf) meldet 7355. Die
Ursache ist nicht geklärt [gemessen; Erklärung offen, Frage F7].

### 3.2 Gegenprobe der Leser [gemessen]

- Neue Datei mit mdbtools und mit `access_parser` umgewandelt: alle 11 Tabellen
  als Zeilen-Multimenge identisch (9 auch in gleicher Reihenfolge, `SWB`/`DK`
  in anderer Reihenfolge).
- Die vorhandene `SACH-GEO1.db` (Ergebnis von `convert2.py`: mdb-export → CSV →
  pandas `to_sql`) hat für alle 11 Tabellen dieselben Zeilenzahlen wie die alte
  .mdb. Sie hat keine Herkunftsangaben, Typen von pandas geraten (Ganzzahlen mit
  NULL als REAL), Spaltennamen wie in Access (`ERL_UTER`, `ERLÄUTER`).
  `convert.py` (pyodbc mit Windows-Access-Treiber) und `convert2.py` schreiben
  beide nach `SACH-GEO1.sqlite`, nicht `.db` [gelesen].

### 3.3 Schema und Zweck (neue Datei)

Spalten und Typen [gelesen aus `mdb-schema`], Zweck [vermutet, soweit nicht anders vermerkt]:

| Tabelle | Zeilen | Spalten (Access-Typ) | Zweck |
|---|---|---|---|
| `SWB` | 16 664 | `LAUF_NR` Long (Autowert, **ohne** Index), `OBER_BEGR` Text 60, `SWB_AEQUIV` Text 254, `SWB_NOTES` Memo, `NOTES` Text 100, `DK_ID`, `DK_ID1`, `DK_ID2`, `DK_ID3` Double, `VERWANDTE` Text 100 | lokale Schlagwortsätze mit bis zu vier DK-Verweisen |
| `DK` | 7 365 | `DK_ID` Double, `ERL_UTER` Text 212, `DK` Text 80 | DK-Notationen mit Erläuterung, Ziel der `SWB.DK_ID*`-Verweise |
| `Geographica` | 2 526 | `SWB_AEQUIV`, `OBER_BEGR`, `SWB_NOTES` Memo, `NOTES`, `DK_ID` Double, `DK_DK`/`DK_ERL_UTER` und `DK_1_…` bis `DK_3_…` Text | geografische Schlagwörter, ältere Schreibweise; die `DK_n_*`-Spalten sehen aus wie das Ergebnis einer Tabellenerstellungsabfrage mit vier DK-Aliasen |
| `DK-Geographica` | 550 | `DK` Text 33, `ERL_UTER` Text 212, `DK_ID` Double | DK-Notationen der Geografie (DK 91…), eigener ID-Raum |
| `DK_alt` | 2 803 | `LAUF_NR` Long (eindeutiger Index), `ERLÄUTER` Text 200, `DK_ID` Integer, `DK` Text 255 | ältere DK-Liste, Umlaute ausgeschrieben als Buchstaben |
| `DK_BERG` | 517 | `DK_BERGBAU` Text 254, `ERLAEUTERU` Text 254 | DK Bergbau (622…) |
| `DK_BIO` | 1 775 | `DK_MITTLER` Text 22, `ERLAEUTERU` Text 85 | DK-Mittelausgabe Biologie (57…), ohne Präfix `DK` |
| `Geogr_Anhaengezahlen` | 1 210 | `ID` Long (PK), `Nummer` Text (Index), `Inhalt` Text, `Feld3` Text (immer NULL) | Anhängezahlen des Ortes |
| `EAS` | 79 | 37 Spalten: `AUTOR`, `TITEL`, `VERLAG`, `ISBN`, `PREIS`, `LIEFER`, `BESTDAT`, `RGNR` … | Erwerbungs-/Bestellliste, keine Sacherschließung |
| `SS_SG` | 98 | wie `EAS` | wie `EAS` |
| `Einfügefehler` | 1 | `Feld0` Memo | von Access beim Einfügen angelegte Fehlertabelle |

Dazu kommen Systemtabellen (`MSys*`) und eine Ressourcentabelle
`f_…_Data`, die `access_parser` auflistet und `mdb-tables` nicht.
**Schlüssel und Beziehungen [gemessen]:** eindeutiger Index nur auf
`DK_alt.LAUF_NR`, Primärschlüssel nur `Geogr_Anhaengezahlen.ID`, keine
Fremdschlüssel. Die Feldnamen mit 10 Zeichen (`ERLAEUTERU`, `DK_BERGBAU`) und
`ERL_UTER` passen zu einem früheren dBase-Import [vermutet].

### 3.4 Verknüpfungen [gemessen]

- `SWB.DK_ID*` → `DK.DK_ID`: belegt 16 619 / 5 787 / 2 290 / 845, davon ohne
  `DK`-Zeile 11 / 26 / 12 / 3. Alle IDs ganzzahlig, obwohl Typ Double.
- `DK.DK_ID` ist **nicht eindeutig**: 8 IDs kommen doppelt mit verschiedenem
  Inhalt vor, 1 Zeile ist ganz leer; 29 Notationen stehen mehrfach. Die alte
  Datei hatte 12 doppelte IDs.
- `Geographica.DK_ID` zeigt auf `DK-Geographica.DK_ID` (2091 von 2526
  auflösbar; `DK_DK` stimmt dort in 2088 Fällen), nicht auf `DK.DK_ID`
  (0 Übereinstimmungen der Notation).
- 2072 von 2526 `Geographica`-Ansetzungen kommen normalisiert (Typ-Präfix,
  `!…!`, Zahl entfernt) auch in `SWB` vor.

### 3.5 Inhalt der Schlagwortfelder

Gepacktes Format in `SWB.SWB_AEQUIV` und `Geographica.SWB_AEQUIV`; drei gekürzte Beispiele:

```
|s|Korund !105743402! !a209572930! 74899 ‡
DK 624.137.2 !1293463876! !a223463876!
g.Chile <Nord> 44247 ‡ g.Nordchile
```

- `|s|`, `|g|`, `|p|`, `|k|`, `|f|`, `|c|`, `|t|`, `|z|` am Anfang;
  Verteilung in `SWB` (Plugin-Zerlegung): s 10 296, g 5 867, k 131, p 108,
  f 84, c 68, t 29, z 16, ohne Kennzeichen 62 [gemessen]. Deutung als
  RSWK-Schlagworttypen [vermutet].
- `‡` (U+2021) trennt die Ansetzung von weiteren `|x|`-Formen: in 7 344 Werten
  [gemessen]. Deutung als Verweisungsformen [vermutet]; das Plugin behandelt sie so.
- `!…!`-Kennungen in 16 364 `SWB`-Sätzen und 7 206 `DK`-Notationen, ein Teil mit
  `a`-Präfix [gemessen]. Deutung als PPN von Normdatensätzen [vermutet; `!a…!`
  unbekannt, Frage F3].
- Eine Zahl hinter der letzten Kennung bzw. vor `‡` in 4 488 `SWB`- und 2 277
  `Geographica`-Sätzen [gemessen, nach der Regel des Plugins]. Bedeutung unbekannt.
- `Geographica` nutzt die ältere Schreibweise `g.` statt `|g|` und hat keine Kennungen.
- `OBER_BEGR` enthält in 5 837 `SWB`-Sätzen den Wert `kein`; das Plugin liest
  ihn im Format `swb_packed` als „kein Oberbegriff" [vermutet].
- `SWB_NOTES` beginnt in 939 Sätzen mit einem Ländercode der Form `XA-…`.

### 3.6 Zeichensatz [gemessen]

- mdbtools liefert UTF-8; der Kopf von `mdb-export` und die Tabellennamen
  (`Einfügefehler`) kommen korrekt an.
- Umlaute sind meist ausgeschrieben: 79 `SWB_AEQUIV`-Werte enthalten ä/ö/ü/ß,
  4 225 enthalten keine, aber `ae`/`oe`/`ue` (grobe Zählung, `ae` kommt auch in
  normalen Wörtern vor). Das Plugin faltet beide Seiten auf `ae/oe/ue/ss`.
- `DK-Geographica.ERL_UTER`: U+2580 (4×) und U+2500 (3×); das passt zu
  Latin-1-`ß`/`Ä`, die einmal als CP437 gelesen wurden [vermutet]. Nicht korrigiert.
- Steuerzeichen U+0014 in 2 `SWB_NOTES`-Werten; Zeilenumbrüche in 381
  `SWB.SWB_NOTES`- und 73 `Geographica.SWB_NOTES`-Werten (Memo, im CSV gequotet).
- Führende/folgende Leerzeichen u. a. in 937 `Geogr_Anhaengezahlen.Nummer`-Werten.

## 4. Umwandlung: `scripts/access_to_sql.py`

- `inspect`: Tabellen, Spalten (Access-Typ, Länge, Ziel-Name, SQLite-Typ),
  Indizes, Zeilenzahl aus dem csv-Modul und `mdb-count`, optional bis zu 5
  Stichproben (je Wert 60 Zeichen). Schreibt nichts außer `--json`.
- `convert`: je Benutzertabelle eine SQLite-Tabelle mit bereinigten Namen
  (`DK-Geographica` → `dk_geographica`, `ERLÄUTER` → `erlaeuter`); Typen
  Long/Integer/Byte/Boolean → INTEGER, Single/Double → REAL, Text/Memo/DateTime
  → TEXT, Currency/Numeric → TEXT (exakt), OLE/Binary → BLOB. NULL bleibt NULL
  (Sentinel `-0` in `mdb-export`), `""` bleibt `""`. Nicht umwandelbare Werte
  bleiben als Text erhalten und werden gezählt. Indizes aus Access werden
  nachgebaut; ist Eindeutigkeit verletzt, als einfacher Index mit Vermerk.
- Herkunft: `alima_import_run` (Pfad, SHA-256, Größe, mtime, Leser, Version),
  `alima_import_table` (Zeilen Quelle/geschrieben/`mdb-count`, Prüfsummen,
  Warnungen, Indizes), `alima_import_column` (Abbildung je Spalte).
- Prüfung: jede Tabelle wird zurückgelesen; Zeilenzahl und SHA-256 über die
  umgewandelten Werte müssen gleich sein, `mdb-count` ebenso (abschaltbar mit
  `--allow-count-mismatch`, dann protokolliert). Erst dann wird die Zieldatei
  atomar ersetzt; bei Abweichung bleibt sie unverändert.

Messung an der neuen Datei [gemessen]: ohne Schalter bricht `convert` an `DK`
ab (7365 gegen `mdb-count` 7355); mit `--allow-count-mismatch` 11 Tabellen,
33 588 Zeilen, alle Prüfsummen gleich; zweiter Lauf: Prüfsummen aller
11 Tabellen identisch. Alte Datei: 30 721 Zeilen ohne Abweichung.

## 5. Zielschema

**Umgesetzt (Staging, 1:1):** eine Tabelle je Access-Tabelle plus die drei
`alima_import_*`-Tabellen. Keine Normalisierung, damit jede Zeile auf die Quelle
zurückführbar bleibt und ein erneuter Import ohne Abbildungslogik auskommt.

**Vorschlag (Phase 3, nach Klärung von F1-F4), normalisierte Schicht in derselben Datei:**

| Tabelle | Inhalt |
|---|---|
| `notation(key PK, scheme, source_table, source_id, code, code_raw, caption, identifiers, digits)` | alle DK-Tabellen zusammen, `source_id` nicht eindeutig |
| `heading(key PK, source_table, source_id, type, text, text_raw, identifiers, local_number, broader, notes, search_norm)` | `SWB` und `Geographica` |
| `heading_variant(heading_key, type, text, search_norm)` | Verweisungsformen |
| `heading_notation(heading_key, position, source_ref, notation_key, status)` | `status` ∈ aufgelöst / mehrdeutig / offen |
| `heading_fts` (FTS5 über `search_norm`) | FTS5 ist in der Python-sqlite3 des Projekt-venv vorhanden (SQLite 3.53.4) [gemessen] |

Ohne diese Schicht lädt das Plugin die Sätze einer Instanz einmal in den
Speicher: `SWB` 16 662 Sätze + 7 364 Notationen in 0,84 s, Suche über 8
Begriffe Median 57 ms, Maximum 130 ms [gemessen, ein Lauf auf diesem Rechner].
FTS5 lohnt erst bei deutlich größeren Daten oder unscharfer Suche.

## 6. Plugin-Entwurf (umgesetzt, soweit nicht als Plan markiert)

- Verzeichnis `src/utils/lookups/geo_sachdb/`: `plugin.toml`, `README.md`,
  `provider.py` (`GeoSachdbLookup`), `store.py` (Lesen, Zerlegung, Faltung).
- **Nicht registriert** beim Import; Tier-2-Laden ist getestet
  (`tests/test_geo_sachdb_lookup.py::Packaging`).
- Konfiguration = Rollen-Abbildung (Felder siehe Plugin-README). Vorschlag für
  zwei Instanzen, **unbestätigt**:
  - `geo_sachdb`: `term_table=swb`, `term_id_column=lauf_nr`,
    `term_column=swb_aequiv`, `term_format=swb_packed`,
    `broader_column=ober_begr`, `related_column=verwandte`,
    `note_columns=swb_notes,notes`, `term_notation_columns=dk_id,dk_id1,dk_id2,dk_id3`,
    `notation_table=dk`, `notation_id_column=dk_id`, `notation_code_column=dk`,
    `notation_caption_column=erl_uter`.
    Gemessen damit: 25 539 Notationsverweise, 25 487 aufgelöst, 52 offen, 16 mehrdeutig.
  - `geo_sachdb_geogr`: `term_table=geographica`, `term_column=swb_aequiv`,
    `term_notation_columns=dk_id`, `notation_table=dk_geographica` (Rest analog).
    Gemessen: 2 526 Verweise, 2 091 aufgelöst, 435 offen.
- Werkzeuge: `geo_sachdb_search`, `geo_sachdb_record`, `geo_sachdb_notation`
  (`cacheable=False`, lokale Datei).
- Ohne Datei/Abbildung: `is_available()` False → keine Werkzeuge; Methoden
  antworten mit `error`.
- **Plan:** Aufnahme in ein Werkzeug-Preset (`src/mcp/default_presets.yaml`,
  z. B. `lookup`) und Registrierung als Built-in, sobald F1-F5 geklärt sind.

## 7. Anbindung an alima_v51_105

### 7.1 Umgesetzt: Schritt 7b `geo_candidates` [gelesen aus dem Branch]

| | |
|---|---|
| Ort | nach `dk_postprocess`, vor `shelf_candidates` (`workflows/alima_v51_105.yaml:857`); `depends_on: [dk_postprocess]` |
| Funktion | `geo_sachdb_candidates` (`src/core/agents/deterministic_functions.py:2031`) |
| Eingaben | `keywords: ${extra.final_keywords}`, `notations: ${dk_classifications}`; fehlen sie, liest die Funktion dieselben Werte aus dem Kontext |
| Ausgabe | `extra.geo_sachdb` mit `status`, `keyword_matches` (lokale Sätze je Schlagwort), `notation_matches` (lokale Einträge je vergebener DK-Notation; RVK/DDC/BK bleiben draußen), `candidates` (Notationen aus den getroffenen lokalen Sätzen, `assigned` = schon vergeben), `prompt_text`, `source` |
| Schalter | die Plugin-Instanz selbst: keine Instanz/abgeschaltet → `status: disabled`; ohne Datei oder Abbildung → `status: unavailable` mit Grund; beides leeres Ergebnis, der Lauf geht weiter |
| Schutz gegen Erfindungen | kein LLM beteiligt; nur identische Ansetzungen/Verweisungsformen (`min_score: 90`); `dk_classifications` wird nicht verändert (Test) |
| Budget | `meta_agent.max_cycles` 20 → 21 (13 aktivierte Schritte); Planner- und Reflexionsregeln nennen den Schritt als Abschluss, kein Prüftor |

Was der Schritt **nicht** tut: Er geht nicht ins gespeicherte Ergebnis
(`KeywordAnalysisState`), nicht in die GUI-Anzeige und nicht in einen Prompt.
`extra.geo_sachdb` lebt im Workflow-Kontext des Laufs.

### 7.2 Vorschlag: Vorwissen vor `classification` (Plan, Phase 3)

Ein zweiter deterministischer Schritt `geo_prior` nach `dk_collect` mit
denselben Eingaben (`${extra.final_keywords}`) liefert `prompt_text` an
`classification` (neuer Input `geo_context`, Platzhalter im User-Prompt). Das
Modell sähe dann je Schlagwort die lokalen DK-Notationen mit Erläuterung als
Hinweis neben den Katalogdaten. Voraussetzung: F1/F2 geklärt, und ein
Vergleichslauf zeigt, dass die lokalen DK die Zuweisung verbessern. Dieser
Schritt ändert den Klassifikations-Prompt und gehört deshalb nicht in die
Vorbereitung.

### 7.3 Vorschlag: Auswahl + Guard (Plan, Phase 3)

`geo_select` (LLM, wählt aus `extra.geo_sachdb.candidates` die zu ergänzenden
Notationen, Gate `when: "${extra.geo_sachdb.prompt_text} != ''"`) und
`geo_guard` mit der vorhandenen Funktion `filter_unauthorized_shelf`
(`config: candidates_key: geo_sachdb, selection_key: geo_selection`): die
Kandidatenliste trägt dafür schon das Feld `notation`. Für mehrere Notationen
bräuchte der Guard eine Listenvariante.

### 7.4 Synchronhaltung mit alima_v51

`alima_v51.yaml` bleibt unverändert, `geo_candidates` ist wie `shelf_*`
institutionsspezifisch. Soll der Schritt auch dort laufen, wären nötig: der
Schritt selbst, `max_cycles` +1, die Phasenkette und die Abschluss-Regel in
`meta_agent.planning.rules` und `meta_agent.reflection.rules`, und
`workflows/CLAUDE.md` (Sync-Regel). Ohne Plugin-Instanz liefe er dort leer.

## 8. Offene Fragen an den Entwickler

- **F1 Datenmodell:** Ist `SWB` die maßgebliche Schlagworttabelle, `SWB_AEQUIV`
  die Ansetzung und sind die Formen hinter `‡` Verweisungen? Was ist
  `Geographica` gegenüber den `|g|`-Sätzen in `SWB` (2072 Überschneidungen)?
- **F2 Notationen:** Sind `DK_ID…DK_ID3` gleichrangig oder ist `DK_ID` die
  Hauptnotation? Welche der DK-Tabellen (`DK`, `DK_alt`, `DK_BERG`, `DK_BIO`,
  `DK-Geographica`) sind aktuell? Sollen die 8 doppelten `DK_ID` bereinigt werden?
- **F3 Kennungen:** Sind die `!…!`-Werte PPN der Normdaten (SWB/K10plus)? Was
  bedeutet `!a…!`, was die Zahl hinter der Ansetzung? Mit einer Zuordnung
  PPN → GND-ID könnte das Plugin GND-IDs liefern.
- **F4 Bedeutung von `EAS`, `SS_SG`, `OBER_BEGR = 'kein'`, `NOTES`/`SWB_NOTES`.**
  Sollen `EAS`/`SS_SG` (Erwerbungsdaten mit Preisen) überhaupt umgewandelt werden?
  `convert --exclude EAS SS_SG` lässt sie aus.
- **F5 Rechte:** Darf die Datei in eine ALIMA-Installation, die andere nutzen
  (Webapp), oder nur lokal? Darf ein Agent sie Cloud-Modellen zeigen?
- **F6 Aktualisierung:** Wird weiter in Access gepflegt? Dann: wer wandelt um,
  wie oft, wohin (Pfad für `db_path`)?
- **F7 Zeilenzahl `DK`:** `mdb-count` 7355, gelesen 7365. Bitte eine Kopie in
  Access „Komprimieren und Reparieren" und erneut `inspect` laufen lassen.
- **F8 Zeichensatz:** Sollen die ausgeschriebenen Umlaute und die vermuteten
  CP437-Fehler (`DK-Geographica`) in der Quelle korrigiert werden?
- **F9 Umfang:** Nur Agent-Werkzeuge, oder auch die klassische Pipeline
  (DK-Quelle, Abschnitt 2 „Suchquelle")?

## 9. Phasen und Aufwand (Schätzung)

| Phase | Inhalt | Aufwand |
|---|---|---|
| 1 ✅ | Analyse, Umwandler, Plugin-Gerüst, Gate, `geo_candidates` (dieser Branch) | erledigt |
| 2 | F1-F7 klären; Abbildung bestätigen; `convert` der aktuellen Datei; Plugin registrieren (Built-in oder Tier-2), Preset `lookup` | 0,5-1 Tag |
| 3 | Ergebnisübernahme (`KeywordAnalysisState`-Feld + Formatter wie `shelf`), optional `geo_prior`/`geo_select`/`geo_guard`, Vergleichslauf mit/ohne an ≥ 20 Titeln | 2-3 Tage |
| 4 | optional normalisierte Schicht + FTS5, `CLASSIFICATION`-Suchquelle für die klassische Pipeline, PPN → GND-ID | 2-4 Tage |

## 10. Reproduktion

Alle Messungen an Kopien in `.scratch/` (nicht im Repo):

```bash
python scripts/access_to_sql.py inspect .scratch/src_new/SACH-GEO1.mdb --json .scratch/inspect_new.json
python scripts/access_to_sql.py convert .scratch/src_new/SACH-GEO1.mdb .scratch/geo_new.db --allow-count-mismatch
python scripts/access_to_sql.py --reader access-parser convert .scratch/src_new/SACH-GEO1.mdb .scratch/geo_new_ap.db
```

Die Tabellen- und Spaltenvergleiche alt/neu, die Slot-Zählung und die
Plugin-Messung liefen als Wegwerf-Skripte; ihre Ergebnisse stehen in
`STATUS_geo_sachdb.md`.
