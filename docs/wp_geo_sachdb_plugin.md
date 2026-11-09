# WP: Sacherschließungsdatenbank Geowissenschaften als ALIMA-Plugin (`geo_sachdb`)

> **Stand (4.11.2026, Branch `feature/geo-sachdb-plugin`):** Vorbereitung.
> Vorhanden: Umwandler und Importer `scripts/access_to_sql.py` (`convert`,
> `import-geo`), Lookup-Plugin `src/utils/lookups/geo_sachdb/` mit eigenem
> Schema v1 (als Built-in registriert), kleines Verfügbarkeits-Gate im
> Werkzeuggenerator, deterministischer Schritt `geo_candidates` in
> `alima_v51_105`, Übernahme ins Ergebnis (`KeywordAnalysisState.geo_sachdb`,
> §7.1), lokale PPN im gemeinsamen WinIBW-Eintrag (`src/utils/winibw_record.py`,
> §7.1, Stand 6.11.2026), Schritt `geo_prior` mit den lokalen Zuordnungen als
> Eingabe der Klassifikation (§7.2, Stand 9.11.2026). Nicht vorhanden:
> GUI-Anbindung außer dem Plugin-Tab und der Abschlusskarte, eigener
> LLM-Auswahlschritt für lokale Notationen, Schutz gegen erfundene DK. Die
> Antworten des Entwicklers stehen in Abschnitt 8; Schema und Importer setzen sie um.
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
| Konstruktion mit Instanz-Settings + Env-Overrides | `src/utils/lookups/category.py:44-54` | `db_path` kommt als Setting in den Konstruktor |
| Ein Konstruktionspunkt für Nicht-Agent-Aufrufer | `src/utils/lookups/resolve.py:67-87` (`build_lookup`, `None` bei abgeschalteter oder fehlender Instanz) | Workflow-Funktion nutzt ihn wie `shelf_candidates` |
| Werkzeugerzeugung je aktivierter Instanz | `src/mcp/_tool_generation.py:156-219` | vorher ohne Verfügbarkeitsprüfung für Lookups |
| Instanz-Seeding: jede registrierte Lookup-Art bekommt beim Laden eine **aktivierte** Instanz | `src/utils/plugin_migration.py:283-330` | Grund, das Plugin **nicht** als Built-in zu registrieren: es stünde sonst sofort in jeder Konfiguration |
| `gates_availability` wird nur für Suchquellen ausgewertet | `src/core/search/factory.py:131-143`, `src/core/search/provider_base.py:82` | für Lookups wirkungslos, daher eigenes `is_available()` |
| Suchquellen prüfen `is_available()` erst beim Aufruf | `src/mcp/_tool_generation.py:317`, `:382` | Werkzeug bleibt sichtbar und meldet Fehler |
| Vorbild lokale Datei: `local_scheme` | `src/utils/lookups/local_scheme.py:66-70` (Suchpfade), `:80` (`digits_of`), `:276` (`cacheable=False`) | Notationsnähe per Ziffernpräfix wird wiederverwendet |
| Vorbild eigene SQLite: `webindex` | `src/utils/lookups/webindex/store.py` (eigene DB über `DatabaseManager`) | hier statt QtSql das Standardmodul `sqlite3` mit `mode=ro`, weil nur gelesen wird |
| Tier-2-Laden registriert die Klasse selbst | `src/core/plugins/loader.py:281` (`register_code_type`) | Plugin braucht keinen Import-Seiteneffekt |
| Lokale Daten im Workflow: `shelf_candidates` → `shelf_select` → `shelf_guard` | `src/core/agents/deterministic_functions.py:1843-1909`, `:1912`; `workflows/alima_v51_105.yaml:897-` | `build_lookup` → `None` ergibt leeres Ergebnis (`deterministic_functions.py:1890-1892`) |
| Aufstellung gelangt ins Ergebnis über `extra.shelf_final` | `src/core/agents/shared_context.py:637-649` | `geo_sachdb` auf demselben Weg über `extra.geo_sachdb` (`:651`, §7.1) |

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
  Verteilung in `SWB` (einfaches Kennzeichen `|x|` am Anfang): s 10 297,
  g 5 867, k 131, p 108, f 84, c 68, t 29, z 16, x 1, ohne einfaches Kennzeichen
  63 (darunter `|f/s|`) [gemessen]. Deutung als RSWK-Schlagworttypen [vermutet].
- `‡` (U+2021) trennt die Ansetzung von weiteren `|x|`-Formen: in 7 344 Werten
  [gemessen]. Deutung als Verweisungsformen [vermutet]; der Importer legt sie als
  `variants` ab.
- Weitere `|x|`-Formen stehen auch **vor** `‡`: 3 484 Werte haben vor `‡` mehr
  als ein Kennzeichen [gemessen]. Meist folgen sie der ersten Kennung als
  Verweisungsformen; in 31 Werten steht die erste PPN erst hinter einem späteren
  Kennzeichen (`|g|Ort / |s|Thema !…!`), das sind Ketten [gemessen; Deutung
  vermutet].
- `!…!`-Kennungen in 16 364 `SWB`-Sätzen und 7 206 `DK`-Notationen, ein Teil mit
  `a`-Präfix [gemessen]. Deutung als PPN von Normdatensätzen [vermutet; `!a…!`
  unbekannt, Frage F3].
- Eine Zahl hinter der letzten Kennung bzw. vor `‡` in 4 488 `SWB`- und 2 277
  `Geographica`-Sätzen [gemessen mit der Regel, die der Importer anwendet, vor
  der Erweiterung um Ketten]. Bedeutung unbekannt.
- `Geographica` nutzt die ältere Schreibweise `g.` statt `|g|` und hat keine Kennungen.
- `OBER_BEGR` enthält in 5 837 `SWB`-Sätzen den Wert `kein` [gemessen]; die
  Spalte wird nicht übernommen (F4).
- Die Tabelle `DK` enthält neben DK auch RVK-Notationen (`QP 925`, `QC 347` …):
  340 Werte beginnen nicht mit `DK`, 318 davon haben die RVK-Form [gemessen].
  Der Platzhalter `DK_ID` 9999 heißt `leer` („keine DK/Q... eingetragen“).
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
- `import-geo`: liest die Ausgabe von `convert` und schreibt das Schema des
  Plugins (Abschnitt 5). Öffnet keinen Access-Leser.

Messung an der neuen Datei [gemessen]: ohne Schalter bricht `convert` an `DK`
ab (7365 gegen `mdb-count` 7355); mit `--allow-count-mismatch` 11 Tabellen,
33 588 Zeilen, alle Prüfsummen gleich; zweiter Lauf: Prüfsummen aller
11 Tabellen identisch. Alte Datei: 30 721 Zeilen ohne Abweichung.

## 5. Zielschema

Zwei Stufen, beide als SQLite-Datei:

1. **Zwischenstand (`convert`, 1:1):** eine Tabelle je Access-Tabelle plus
   `alima_import_run`, `alima_import_table`, `alima_import_column`. Keine
   Abbildungslogik, damit jede Zeile auf die Quelle zurückführbar bleibt.
2. **Schema des Plugins, Version 1 (`import-geo`):** quellenunabhängig, definiert
   in `src/utils/lookups/geo_sachdb/schema.py` (DDL, `create_schema`,
   `check_schema`). Andere Häuser schreiben dafür eigene Importer; das
   Eingabeformat beschreibt das Plugin-README.

| Tabelle | Spalten | Inhalt |
|---|---|---|
| `meta` | `key`, `value` | `schema_version` = `1` (Pflicht), `source`, `source_date`, `source_sha256`, `created_at`, `importer` |
| `heading` | `id`, `label`, `kind`, `ppn`, `chain`, `variants` | `kind` ∈ Sachschlagwort, Geographikum, Person, Körperschaft, sonstiges; `ppn` = erste K10plus-PPN; `chain` = fertige lokale Zeichenkette; `variants` = Verweisungsformen als Rohtext |
| `classification` | `id`, `system`, `notation`, `label`, `ppn` | `system` ∈ DK, RVK; Notation ohne `DK`-Präfix bzw. als `QP 925`; nicht eindeutig |
| `heading_classification` | `heading_id`, `classification_id`, `position`, `ambiguous` | Notationen eines Schlagworts; `ambiguous = 1` bei mehrfacher `DK_ID` |

Die Notationen eines Schlagworts stehen in einer eigenen Verknüpfungstabelle und
nicht als Spalte von `heading`, weil ein Satz bis zu vier Notationen hat und eine
Verknüpfung als mehrdeutig markiert werden muss.

Der Importer schreibt zusätzlich `alima_import_run` (aus dem Zwischenstand
kopiert) und `alima_import_rule` (Zählung je Regel). Das Plugin liest beide nicht.

**Regeln von `import-geo`** (Entwicklerentscheidungen F2-F4, F8): `DK_ID` wird
intern aufgelöst und nicht ausgegeben; der Platzhalter 9999 und Verweise darauf
entfallen; `dk_alt` wird ignoriert; `EAS`, `SS_SG`, `NOTES`, `SWB_NOTES`,
`OBER_BEGR` werden nicht übernommen; nur die erste `!…!`-Nummer in PPN-Form ist
`ppn`; doppelte `DK_ID` werden markiert; Zeilenumbrüche in Texten werden
Leerzeichen; `¬…¬` bleibt in `chain` und fällt im `label` weg; Platzhalter wie
`{Balken ueber}` bleiben unverändert. Ohne Entscheidung und deshalb ebenfalls
nicht übernommen: `Geographica` (F1 ungeklärt) und `VERWANDTE`. Übernommen
werden `swb` sowie `dk`, `dk_bio`, `dk_berg`, `dk_geographica` (F2: fachliche
Zuordnungen; nur `dk` trägt PPN und ist Ziel der `SWB.DK_ID*`-Verweise).

**Lauf an der neuen Datei** (`SACH-GEO1-neu.sqlite`, Ausgabe von `convert`)
[gemessen, ein Lauf]:

| | |
|---|---|
| `heading` | 16 663 von 16 664 `SWB`-Sätzen (1 ohne Ansetzung); mit PPN 16 367 (98,2 %) |
| `kind` | Sachschlagwort 10 298, Geographikum 5 867, Körperschaft 131, Person 108, sonstiges 259 |
| `classification` | 10 203 (DK 9 885, davon 6 867 mit PPN; RVK 318, davon 316 mit PPN) |
| `heading_classification` | 23 796 (32 mit `ambiguous = 1`) |
| Platzhalter 9999 | 1 Eintrag in `dk`, 1 687 Verweise ausgeblendet |
| weitere Zählungen | 8 doppelte `DK_ID`, 54 Verweise ohne Eintrag, 19 doppelte Verweise eines Satzes, 3 Notationszeilen ohne Notation, 54 Sätze ohne Typkennzeichen, 44 Sätze mit weiteren PPN-Nummern, 82 Textfelder mit Zeilenumbruch |

Die Verweise gehen auf: 25 541 belegte `DK_ID*`-Felder minus 1 687 (9999), 54
(ohne Eintrag), 19 (doppelt) und 1 (Satz ohne Ansetzung) ergibt 23 780; die 16
Verweise auf doppelte `DK_ID` ergeben je zwei Verknüpfungen, zusammen 23 796.

Das Plugin lädt die Zeilen einer Instanz einmal in den Speicher: 0,54 s; Suche
über 8 Begriffe Median 45 ms, Maximum 48 ms [gemessen, ein Lauf auf diesem
Rechner; Begriffe Granit, Lagerstätte, Sachsen, Erzgebirge, Bergbau, Mineral,
Geologie, Freiberg]. Ein Volltextindex (FTS5, in der Python-sqlite3 des
Projekt-venv vorhanden) lohnt erst bei deutlich größeren Daten oder unscharfer
Suche.

## 6. Plugin (umgesetzt, soweit nicht als Plan markiert)

- Verzeichnis `src/utils/lookups/geo_sachdb/`: `plugin.toml`, `README.md`,
  `schema.py` (Schema v1, nur Standardbibliothek; der Importer lädt die Datei per
  Pfad), `store.py` (Lesen, Faltung, Zeilen), `provider.py` (`GeoSachdbLookup`).
- **Built-in** (`src/utils/lookups/__init__.py` importiert das Paket). Jede
  Installation erhält deshalb die Instanz `geo_sachdb` in der Lookup-Liste
  (`plugin_migration.ensure_lookup_instances`); ohne `db_path` ist sie nicht
  verfügbar und erzeugt keine Werkzeuge.
- Konfiguration: `db_path`, `max_results`. Andere Einstellungen einer Instanz
  (etwa `term_table` aus einer älteren Konfiguration) werden ignoriert.
- Werkzeuge (`cacheable=False`, lokale Datei):
  - `geo_sachdb_search(query, kinds?)`: gerankte Schlagwörter aller Arten.
  - `geo_sachdb_lines(terms, kinds?)`: nur gleiche Ansetzung oder
    Verweisungsform; `found` und `not_found` getrennt, `lines` als fertige Zeilen.
  - `geo_sachdb_notation(notation)`: Einträge zu einer DK- oder RVK-Notation, die
    Schlagwörter dazu, übergeordnete DK-Notationen.
- Zeilen (F3): Schlagwort `5550 !PPN!`, ohne PPN `5550 <chain>`, ohne `chain`
  `5550 <label>`; Notation `6700 !PPN!`, ohne PPN bei DK `6700 DK <Notation>`,
  bei RVK ohne PPN keine Zeile. Ob WinIBW den Rückfall `5550 <chain>` und
  Platzhalter wie `{Balken ueber}` annimmt, ist **ungeprüft**.
- Ohne Datei, mit anderem Schema (etwa dem Zwischenstand) oder anderer
  Schemaversion: `is_available()` False → keine Werkzeuge; Methoden antworten mit
  `error`.
- Die Kandidaten dürfen an Cloud-Modelle gehen (F5); das Plugin schränkt das
  nicht ein.
- **Plan:** Aufnahme in ein Werkzeug-Preset (`src/mcp/default_presets.yaml`,
  z. B. `lookup`).

## 7. Anbindung an alima_v51_105

Ablauf: Die Datenbank wird zweimal gefragt. Vor der Klassifikation liefert
`geo_prior` (§7.2) die lokalen Notationen der finalen Schlagwörter als
Eingabe `local_notations`; die Klassifikation wägt sie mit den Katalogdaten
ab und läuft auch, wenn nur sie vorliegen. Nach `dk_postprocess` schlägt
`geo_candidates` (§7.1) mit den vergebenen Notationen die WinIBW-Zeilen und
weitere Notationen vor. Phasenkette (Planer- und Reflexionsregeln):

```
extraction → search → selection_chunks → selection → (verify_keywords) →
dk_collect → geo_prior → classification → rvk_guard → dk_postprocess →
publication_form → geo_candidates → shelf_candidates → shelf_select → shelf_guard
```

`publication_form` (§7.5) liest vor `geo_candidates` die Publikationsform aus
dem Eingabetext; eine dabei gesetzte Formnotation prüft `geo_candidates` wie
jede vergebene Notation.

### 7.1 Umgesetzt: Schritt 7b `geo_candidates` [gelesen aus dem Branch]

| | |
|---|---|
| Ort | nach `publication_form`, vor `shelf_candidates` (`workflows/alima_v51_105.yaml:986`); `depends_on: [dk_postprocess, publication_form]` |
| Funktion | `geo_sachdb_candidates` (`src/core/agents/deterministic_functions.py:2076`) |
| Eingaben | `keywords: ${extra.final_keywords}`, `notations: ${dk_classifications}`; fehlen sie, liest die Funktion dieselben Werte aus dem Kontext |
| Ausgabe | `extra.geo_sachdb` mit `status`, `keyword_matches` (lokale Sätze je Schlagwort mit Art, PPN und Zeile), `not_found` (Schlagworte ohne lokalen Satz), `notation_matches` (lokale Einträge je vergebener Notation des Systems `scheme`, Standard DK, mit `system`; andere Systeme bleiben draußen), `candidates` (Notationen aus den getroffenen lokalen Sätzen, `assigned` = schon vergeben, mit `line`), `lines` (5550 je getroffenem Satz, 6700 je Kandidat), `prompt_text`, `source` |
| Schalter | die Plugin-Instanz selbst: keine Instanz/abgeschaltet → `status: disabled`; ohne Datei oder mit anderem Schema → `status: unavailable` mit Grund; beides leeres Ergebnis, der Lauf geht weiter. `config.kinds` begrenzt die Arten (leer = alle), `config.scheme` das System |
| Schutz gegen Erfindungen | kein LLM beteiligt; nur identische Ansetzungen/Verweisungsformen (`min_score: 90`); `dk_classifications` wird nicht verändert (Test) |
| Budget | `meta_agent.max_cycles` 20 → 21 (13 aktivierte Schritte), mit `geo_prior` 22 (14), mit `publication_form` 23 (15); Planner- und Reflexionsregeln nennen den Schritt als Abschluss, kein Prüftor |

Was der Schritt **nicht** tut: Er geht in keinen Prompt und ändert keine
Klassifikation. Seit `geo_prior` (§7.2) sind die vergebenen Notationen oft
lokale Notationen; `assigned` markiert sie, sie stehen dann nicht mehr unter
„Weitere lokale DK-Vorschläge“, und eine Notation mit genau einer lokalen PPN
erscheint im WinIBW-Eintrag als `6700 !PPN!` (unverändertes Verhalten, für
diesen Fall festgehalten in `tests/test_geo_sachdb_prior.py::EndToEnd`).

**Ergebnisübernahme (umgesetzt, Muster `shelf`):**
`SharedContext.to_keyword_analysis_state` legt in
`KeywordAnalysisState.geo_sachdb` ab (`_geo_sachdb_for_state`,
`src/core/agents/shared_context.py:122`):

- `terms`: je nachgeschlagenem Schlagwort die lokalen Sätze (`records`:
  Bezeichnung, Art, PPN), der Satz, für den es steht (`record`), `status` und
  die verwendbare `ppn`. Eindeutig (`unique`) ist ein Treffer mit genau einem
  lokalen Satz oder mit genau einem Satz der Art Sachschlagwort unter mehreren
  (`_geo_pick`, `:41`); hat dieser Satz keine PPN, heißt der Status `no_ppn`,
  sonst `ambiguous`.
- `notations`: je vergebener Notation des Systems `scheme` die lokalen
  Einträge, `status` (`unique`: genau eine PPN unter den Einträgen;
  `ambiguous`: mehrere verschiedene; `no_ppn`; `not_local`) und die verwendbare
  `ppn`. Dafür trägt jeder Eintrag von `notation_matches` im Schritt jetzt
  `system`.
- `not_found`, `candidates` und von `source` nur `source`, `source_date`,
  `source_sha256`, `schema_version`.

Die Funktion schreibt keine Zeilen. Die frühere eigene Nummerierung der lokalen
Schlagwörter (`lines`, `entries`, `exhausted`, `numbering`; 5550 + Zahl der
Ketten bis 5599) ist entfallen: lokale Treffer gehen jetzt in den
WinIBW-Eintrag ein. Ohne Schritt, mit einem Schrittergebnis ohne Plugin
(`status` ungleich `ok`) oder ohne jeden Treffer, Vorschlag und Fehlbegriff
bleibt das Feld `{}`.

**WinIBW-Eintrag (`src/utils/winibw_record.py`, `build_winibw_record`):** ein
deterministischer Aufbau aus Ketten, finalen Schlagwörtern, Formschlagwörtern
und Notationen, für einen Zustand ebenso wie für ein Export-Payload. Form:

```
5550 <Glied 1 der Kette 1>
5550 <Glied 2 der Kette 1>
5550 $ADE-105

5551 <Glied 1 der Kette 2>
5551 $ADE-105

5552 <Einzelschlagwort>
5553 <Einzelschlagwort>

1131 <Formschlagwort>

6700 DK <Notation>
6700 <RVK-Notation>
```

- Ketten ab 5550, eine Feldnummer je Kette für alle Glieder, Abschluss
  `<nr> $ADE-105`, eine Leerzeile zwischen den Ketten und zwischen den
  Abschnitten. Kettenquelle und GND-ID-Bereinigung wie im früheren K10+-Export.
- Einzelschlagwörter: finale Schlagwörter, die in keiner Kette stehen und keine
  Formschlagwörter sind, jedes einmal (Vergleich ohne Groß-/Kleinschreibung, NFC);
  je eine eigene Feldnummer, ohne `$ADE-105`.
- Formschlagwörter (`form_keywords`) als `1131 <Begriff>`, immer als Text; ein
  Formschlagwort in einer Kette wird aus der Kette genommen (Vermerk an der
  1131-Zeile).
- DK als `6700 DK <Notation>`, RVK als `6700 <Notation>` ohne Präfix. Andere
  Systeme (DDC, BK) bekommen keine Zeile und stehen in `skipped`.
- Mit `terms`/`notations`: ein Begriff mit Status `unique` erscheint als
  `<nr> !PPN!` (in Ketten wie als Einzelschlagwort), eine vergebene Notation
  mit Status `unique` als `6700 !PPN!`. Mehrdeutige Treffer und Treffer ohne PPN
  bleiben Text und tragen einen Vermerk. Zwei Begriffe mit derselben lokalen PPN
  ergeben eine Zeile. Begriff und Treffer werden über die gleiche Schreibung des
  nachgeschlagenen Schlagworts zugeordnet.
- Obergrenze der Schlagwortfelder: die Konstante `WINIBW_SUBJECT_LAST_TAG`
  (5559, wie der frühere K10+-Export). Eine Kette oder ein Einzelschlagwort
  darüber bekommt keine Zeile und steht in `overflow`; nichts wird auf die
  letzte Nummer gelegt.
- Rückgabe: `lines` (mit Leerzeilen), `text`, `entries` (je Zeile Herkunft,
  Begriff, verwendeter lokaler Satz, Vermerk), `overflow`, `skipped`,
  `last_tag`.

**Ungeprüfte Annahmen** (nicht an WinIBW geprüft): Feld 1131 für
Formschlagwörter; die Obergrenze 5559 (5599 ist ebenfalls im Gespräch, die
Entscheidung steht aus); eine Kette belegt genau eine Feldnummer; WinIBW nimmt
einen Begriff als Text statt PPN an und löst ihn selbst auf; Leerzeilen zwischen
den Feldern stören beim Einfügen nicht; Feld 6700 wird je Notation ohne Zählung
wiederholt (die Klassifikations-Prompts erwähnen `6700, 6701, …`).

Weitere Grenzen: Die Eindeutigkeit beurteilt nur die Sätze, die der Schritt
liefert (höchstens `max_per_keyword`, Standard 3, je Schlagwort). Die
GND-ID-Bereinigung schneidet ab der ersten Klammer (`strip_gnd_id`, wie bisher);
eine Ansetzung mit eigener Klammer verliert diesen Teil.

**Ausgaben mit `build_winibw_record`:** Abschlusskarte
(`format_final_result_card_html`, GUI-Log und Webapp-Log am Ende eines Laufs),
CLI `show-protocol` (`detailed`, Schritt `dk_classification`, auch ohne DK) und
`show-protocol --format k10plus`, Review-Reiter
(`AnalysisReviewTab._generate_k10plus_format`), `exporters.generate_k10plus_lines`
und damit der Export `marc` (auch über das MCP-Werkzeug `export`), `results.winibw`
im JSON-Export (`extract_results_from_analysis_state`; beim Laden verworfen und
aus den übrigen Feldern neu gebaut). **Nicht:** `rule_output` (von einem
Sprachmodell nach persönlichen Regeln geschrieben, unverändert und unabhängig),
`show-protocol --format compact`, die Ergebnisübersicht der Webapp (`app.js`),
die Konsolenausgabe nach einem Lauf.

Anzeige:

- Abschlusskarte: Block „WinIBW-Eintrag“ mit dem Eintrag als reinem Kopierblock
  (`<pre class="alima-result__winibw">`, eigener Kopierknopf; nicht
  `alima-result__pre`, weil der Knopf der Zusatzregel das erste
  `.alima-result__pre` der Karte wählt), darunter die Liste, wofür jede Zeile
  steht, und gegebenenfalls „Ohne Feldnummer (5559 erreicht)“ und „Nicht
  übernommen“. Danach der Block „Lokale Sacherschließung (geo_sachdb)“: lokale
  Treffer je Begriff und je vergebener Notation (Begriff → lokaler Satz, Art,
  PPN; mehrdeutige Fälle mit allen Sätzen), „Weitere lokale DK-Vorschläge“
  (Notationen getroffener lokaler Sätze, im Lauf nicht vergeben, mit fertiger
  Zeile nach derselben Regel in einem eigenen Kopierblock
  `alima-result__geo-proposals`, nicht im WinIBW-Eintrag) und „Ohne lokales
  Schlagwort“.
- CLI `show-protocol`: dieselben Blöcke; Eintrag und Vorschlagszeilen ohne
  Einrückung, Erklärungen eingerückt. Karte und CLI beschreiben Zeilen über
  dieselben Funktionen (`winibw_entry_parts`, `geo_sachdb_hits`,
  `geo_sachdb_proposals`).

Rückwärtskompatibilität: Ergebnisdateien ohne `geo_sachdb` oder aus dem Stand
vor `terms`/`notations` (mit `lines`/`entries`) laden. Der Eintrag wird dann
ohne lokale PPN gebaut, die gespeicherten alten Zeilen werden nicht mehr
angezeigt; Vorschläge (`candidates`) und `not_found` erscheinen weiter.
Gespeichert wird das Feld in beiden JSON-Formaten (`save_analysis_state`,
`results.geo_sachdb`).

Tests: `tests/test_winibw_record.py` (Aufbau, Obergrenze, lokale PPN, Karte,
gleicher Text über alle Ausgabewege), `tests/test_geo_sachdb_result.py`
(Zustand gegen eine synthetische Datenbank mit mehrdeutigen Fällen, Geo-Block,
Speichern/Laden, alte Dateien).

Grenzen: Die Abschlusskarte entsteht nur am Ende eines Laufs (GUI
`on_pipeline_completed`, Webapp-Lauf); geladene Dateien zeigen den Eintrag im
Review-Reiter und in der CLI. `from_keyword_analysis_state` bringt das Feld
nicht nach `extra` zurück.

### 7.2 Umgesetzt: Schritt 5b `geo_prior`, lokale Zuordnungen vor `classification`

Anlass: Liefert der Katalog keine DK-Daten (Libero/finc nicht erreichbar), war
`extra.dk_prompt_text` leer und `classification` lief nicht; die lokalen
Notationen erschienen nur als Vorschläge. Entscheidung des Operators: nur
`alima_v51_105`, die lokale Zuordnung ist immer zusätzliche Evidenz (nicht nur
Ersatz), alle Kandidaten gehen in den Prompt.

| | |
|---|---|
| Ort | nach `dk_collect`, vor `classification` (`workflows/alima_v51_105.yaml:627`); `depends_on: [dk_collect]`, `classification` hat `depends_on: [dk_collect, geo_prior]` |
| Funktion | `geo_sachdb_prior` (`src/core/agents/deterministic_functions.py:2264`): ruft `geo_sachdb_candidates` mit den Schlagwörtern allein (`context=None`, also auch bei einem späteren Lauf keine vergebenen Notationen), gleiche Konfiguration wie `geo_candidates` (`scheme: DK`, `min_score: 90`, `max_per_keyword: 3`, `max_keywords: 30`, `kinds: []`) plus `title` |
| Eingaben | `keywords: ${extra.final_keywords}`, `core_keywords: ${extra.core_keywords}`; fehlen sie, aus dem Kontext |
| Ausgabe | `extra.geo_sachdb_prior` (Nutzlast von `geo_sachdb_candidates`, je Kandidat zusätzlich `via_core_keywords`, dazu `core_keywords`) und `extra.geo_prompt_text` (der Block) |
| Block | Kopfzeile `<title>:`, eine Hinweiszeile, je Notation `- DK <Notation> \| <Bezeichnung> \| über <Schlagwörter> \| Kernschlagwörter <k> von <n>`; alle Kandidaten, geordnet nach Kernschlagwörtern, dann Schlagwörtern (beide absteigend), dann Notation; ohne PPN und WinIBW-Zeilen. Ohne Kernschlagwörter: Zeile „Kein Kern bestimmt“, Zählung `Schlagwörter <n>`. Kein Plugin, keine Eingabe, kein Kandidat → `""` |
| Klassifikation | Eingabe `local_notations: ${extra.geo_prompt_text}`; `when: "${extra.dk_prompt_text} != '' or bool(${extra.geo_prompt_text})"` (`or` kann der Auswerter schon; `bool(...)`, weil ein fehlender Wert mit `!= ''` als wahr gälte). Im System-Prompt ein Absatz „Lokale Zuordnungen aus der Freiberger Sachdatenbank“ (Evidenz neben dem Katalog, Vorschlagsliste, Kernschlagwörter höher gewichten, auch ohne Katalogdaten wählbar), im User-Prompt der Block nach den GND-Schlagworten; beide in bedingten Abschnitten `{?local_notations}…{/local_notations}` (`prompt_resolver._render`, siehe `docs/workflow_yaml_spec.md`) |
| Ohne lokale Daten | System- und User-Prompt byte-gleich zum Stand davor (einmal gegen die Datei vor der Änderung geprüft; der Test hält die Nahtstellen fest) |
| Reflexion, Planer | Zustandszeile „- Lokale Zuordnungen: n Notationen“ nur mit mindestens einem Kandidaten (`reflection_step.local_notations_line`, im Planerzustand von `MetaAgent._plan_with_llm` ebenso). Regeln: eine schwache oder leere Katalogdatenlage ist bei vorhandenen lokalen Zuordnungen kein Grund für `rerun_dk_collect` bzw. für ein erneutes `dk_collect`; `max_cycles` 21 → 22 |
| Ergebnis | `KeywordAnalysisState.geo_prior = {candidates}` (`_geo_prior_for_state`, `shared_context.py:174`): `notation`, `system`, `code`, `label`, `via_keywords`, `via_core_keywords`, `in_catalog` (gegen `extra.dk_entries` des DK-Schritts, nur gleiches System). Beide JSON-Formate (`asdict`, `results.geo_prior`), alte Dateien laden mit `{}` |
| Anzeige | DK-Karte (`format_dk_classifications_card_html`): neben der Treffer-Anzeige ein Kennzeichen „lokale Datenbank“ bzw. „Katalog und lokale Datenbank“, darunter „über <Schlagwörter>“, im Klartext in der Klammer hinter der Notation; CLI `show-protocol` (`format_step_dk_classification`): `* DK 999.5 · lokale Datenbank · über …`. Ohne Treffer in `geo_prior` unverändert |
| Quellenübersicht | je gefragtem Schlagwort ein Aufruf unter der Lookup-Kennung `geo_sachdb`, gefundene Schlagwörter als Treffer, ein Fehlschlag als Fehler je Schlagwort |
| Tests | `tests/test_geo_sachdb_prior.py` (39, synthetische Datenbank, Stub-Modell, ein Lauf über den MetaAgent von `geo_prior` bis `shelf_guard` mit leerem Katalog) |

Grenzen:

- Nicht gegen die echte Datenbank und nicht mit einem echten Modell geprüft.
  Ob die lokalen Zuordnungen die Zuweisung verbessern, ist nicht gemessen
  (Vergleichslauf mit/ohne, §9).
- Der Block hat keine Obergrenze (Entscheidung); bei 30 Schlagwörtern mit je
  bis zu 3 lokalen Sätzen kann er lang werden, die Tokenzahl ist nicht gemessen.
- Kein Schutz gegen erfundene Notationen: im agentischen Pfad prüft nichts, dass
  eine gewählte DK aus dem Katalog oder den lokalen Zuordnungen stammt
  (`rvk_guard` prüft nur RVK, `build_dk_search_results` übernimmt jede
  Notation). Die Regel steht nur im Prompt.
- Kernschlagwörter werden über die gefaltete Schreibweise (`store.fold`) den
  Schlagwörtern zugeordnet, über die eine Notation gefunden wurde.
- `rerun_dk_collect` ist eine Empfehlung der Reflexion an den Planer, keine
  Codeverzweigung; die neue Regel wirkt nur, soweit das Modell ihr folgt.
- Bei leerem Katalog steht die Überschrift „Ausschnitt aus dem
  Bibliotheksbestand (DK)“ weiter leer im User-Prompt; der Absatz sagt, dass
  lokale Notationen auch dann wählbar sind.
- Gezählt wird nur `geo_prior`, nicht `geo_candidates`; die Kennung ist der Typ
  `geo_sachdb`, nicht die Kennung einer umbenannten Instanz.
- `from_keyword_analysis_state` bringt das Feld nicht nach `extra` zurück.

### 7.3 Vorschlag: Auswahl + Guard (Plan, Phase 3)

`geo_select` (LLM, wählt aus `extra.geo_sachdb.candidates` die zu ergänzenden
Notationen, Gate `when: "${extra.geo_sachdb.prompt_text} != ''"`) und
`geo_guard` mit der vorhandenen Funktion `filter_unauthorized_shelf`
(`config: candidates_key: geo_sachdb, selection_key: geo_selection`): die
Kandidatenliste trägt dafür schon das Feld `notation`. Für mehrere Notationen
bräuchte der Guard eine Listenvariante.

### 7.4 Synchronhaltung mit alima_v51

`alima_v51.yaml` bleibt unverändert, `geo_prior` und `geo_candidates` sind wie
`shelf_*` institutionsspezifisch. Soll der Schritt auch dort laufen, wären nötig: der
Schritt selbst, `max_cycles` +1, die Phasenkette und die Abschluss-Regel in
`meta_agent.planning.rules` und `meta_agent.reflection.rules`, und
`workflows/CLAUDE.md` (Sync-Regel). Ohne Plugin-Instanz liefe er dort leer.

### 7.5 Umgesetzt: Schritt 7a `publication_form`, Form aus dem Eingabetext

Anlass (UB Freiberg, Dissertation): Eine Dissertation soll `6700 DK 378.245`
(mit lokaler PPN, wenn bekannt) und `1131 Hochschulschrift` bekommen. Vorher
verbot nur eine Prompt-Regel der Klassifikation die Formnotation bei einem
Werk ohne diese Form; vergeben wurde sie nur, wenn das Modell es tat. Die
Datenbank führt 378.245 an Formschlagwörtern, `geo_prior` fragt aber nur mit
Sachschlagwörtern. Entscheidung des Operators: die Form ergibt sich aus dem
Eingabetext, nur `alima_v51_105`.

| | |
|---|---|
| Ort | nach `dk_postprocess`, vor `geo_candidates` (`workflows/alima_v51_105.yaml:954`); `depends_on: [dk_postprocess]`, `geo_candidates` hat `depends_on: [dk_postprocess, publication_form]` |
| Erkennung | `src/utils/publication_form.py`, `detect_publication_form(text)`: Regeltabelle `RULES` (Name, Form, Reichweite `start`/`head`/`anywhere`, Muster); die erste passende Regel der Tabelle gewinnt, höchstens ein Ergebnis. Formen nur mit einer Notation aus `FORM_NOTATIONS`: `dissertation` (DK 378.245), `diplom_master` (DK 378.244), Formschlagwort jeweils „Hochschulschrift“ |
| Regeln | Titelei („genehmigte Dissertation zur Erlangung des …“, „zur Erlangung des … Grades Doktor … vorgelegt von“, „thesis submitted … for the degree of Doctor/Master“, „zur Erlangung des … Grades Diplom/Master … eingereicht von“), Selbstbezeichnung am Textanfang („Zusammenfassung der Ergebnisse der Dissertation“, „Kurzfassung der Diplomarbeit“), zitierter Titel mit Kandidat („Dissertation „…“ von Herrn M. Sc.“), Selbstbezug („die vorliegende Dissertation“, „this dissertation“, „this master's thesis“). `head` = Beginn in den ersten 600 Zeichen; Regeln mit fremdem Bezug fallen weg, wenn davor „basiert auf“, „im Rahmen“, „based on“, „part of“ o. ä. steht |
| Bewusst nicht | das bloße Wort „Dissertation“, Plural und Genitiv als Gegenstand, „Doktorarbeit von …“ ohne zitierten Titel, „Inaugural-Dissertation“ und „Dissertationsschrift“ ohne die Wendungen oben, Bachelorarbeit und Habilitation (keine Formnotation in `FORM_NOTATIONS`), Magisterarbeit |
| Wirkung | Formschlagwort in `extra.form_keywords` (einmal, Vergleich ohne Groß-/Kleinschreibung) → `1131` über `winibw_record`; Formnotation an die erste Stelle von `dk_classifications`, neu als `{code, type, rank: additional, origin: publication_form, reason: "Form (aus dem Eingabetext): „…“"}`, schon vergeben nur nach vorn gezogen; Erkennung in `extra.publication_form` |
| Ohne Treffer | Formschlagwörter, Notationen und `extra` unverändert, kein `extra.publication_form` (Test) |
| PPN | `geo_candidates` prüft die Formnotation wie jede vergebene Notation: ein lokaler Eintrag mit genau einer PPN ergibt `6700 !PPN!`, sonst `6700 DK 378.245` |
| Aufstellung | `shelf_candidates` lässt alle Notationen aus `FORM_NOTATIONS` aus und gibt die übrige Liste als `classifications` weiter; `shelf_select` liest `dk_codes` daraus statt aus `${dk_classifications}`. Kandidaten und Prompt sind mit und ohne Treffer gleich (Test). Das gilt auch für eine Formnotation, die die Klassifikation selbst vergeben hat |
| Planer, Reflexion | Phasenkette mit `publication_form`, Abschluss-Schritt, kein Prüftor; eine von ihm gesetzte Formnotation ist kein Grund für eine Wiederholung. `max_cycles` 22 → 23 |
| Ergebnis | `KeywordAnalysisState.publication_form = {form, label, system, code, form_keyword, evidence, rule}` (`_publication_form_for_state`, `shared_context.py:209`); beide JSON-Formate (`asdict`, `results.publication_form`), alte Dateien laden mit `{}` |
| Anzeige | DK-Karte: Kennzeichen „Form: Dissertation (aus dem Eingabetext)“ an der Notation, die Textstelle als Tooltip, im Klartext in der Klammer; CLI `show-protocol`: `* DK 378.245 · Form: Dissertation (aus dem Eingabetext), Textstelle „…“`. Ohne Treffer unverändert |
| Tests | `tests/test_publication_form.py` (35, davon 11 Positiv- und 15 Negativfälle als Untertests; Lauf über den MetaAgent bis zum WinIBW-Eintrag mit synthetischer Datenbank und Systematik) |

Grenzen:

- Erkannt wird nur, was die Regeltabelle kennt; es gibt keine Erkennung durch
  ein Sprachmodell. Fehlerrichtungen: nicht erkannt heißt Verhalten wie
  vorher (keine Formnotation, kein Formschlagwort); ein Fehltreffer ist im
  Ergebnis sichtbar, mit der Textstelle an der Notation.
- Geprüft nur an den erfundenen Testtexten, nicht an echten Eingaben.
- Ein Eingabesatz mit Hochschulschriftenvermerk (DOI-/PPN-Eingabe) wird nicht
  ausgewertet: `BibRecord` hat kein solches Feld, und der agentische Pfad
  bekommt nur den Text.
- Eine Rezension, die von „der vorliegenden Dissertation“ spricht, und eine
  Zusammenfassung einer fremden Dissertation, die mit „Zusammenfassung der
  Dissertation“ beginnt, ergeben einen Treffer.
- Die Formnotation steht vorn, ist aber `additional`; ohne Kernnotation
  zeigt die Zeile „Klassifikation“ der Abschlusskarte sie als erste.
- Wiederholt der Planer `classification` nach `publication_form`, überschreibt
  das `dk_classifications` samt Formnotation; `publication_form` läuft danach
  nicht von selbst erneut (Pflichtschritte gelten als erledigt, sobald sie
  einmal gelaufen sind).

## 8. Fragen an den Entwickler und Antworten

Beantwortet im Gespräch (Nov 2026). Der Entwickler arbeitet selbst nicht mit der
Datenbank; wo er es nicht wusste, gilt der Standard und der Punkt ist als
**ungeprüft** markiert.

- **F1 Datenmodell: teilweise.** `SWB_AEQUIV` ist die einfügefertige Zeichenkette
  (`|s|Korund !105743402! !a209572930! 74899 ‡ …`). Das Verhältnis von
  `Geographica` zu den `|g|`-Sätzen in `SWB` ist **ungeklärt**.
- **F2 Notationen:** `DK_ID` ist eine interne Nummer und wird nicht ausgegeben,
  das Plugin löst sie zur Notation auf (Beispiel: `DK_ID` 16 = Dissertation =
  DK 378.245). Die Tabellen `dk_bio`, `dk_berg`, `dk_geographica` sind
  fachliche Zuordnungen. `dk_alt` wird ignoriert. Der Platzhalter `DK_ID` 9999
  wird nicht angezeigt. Doppelte `DK_ID` bleiben markiert.
- **F3 Kennungen:** `!…!` ist die K10plus-Normdaten-PPN; WinIBW wandelt Eintrag
  zu PPN um. Nur die erste `!…!`-Nummer zählt. Ausgabe als fertige Zeilen
  `5550 !PPN!`; für DK `6700 !PPN!`, wenn die Notation in der Tabelle `dk` steht,
  sonst `6700 DK <Notation>` (WinIBW wandelt um). Die Datenbank deckt nur Begriffe
  ab, die das Fach lokal braucht; fehlende Begriffe sind normal.
- **F4 Tabellen und Notizen:** `EAS` und `SS_SG` (Erwerbungslisten mit Preisen)
  werden nicht umgewandelt. `NOTES`, `SWB_NOTES` und `OBER_BEGR` werden nicht
  ausgegeben.
- **F5 Rechte:** Die Daten bleiben lokal und werden als Plugin für die eigene UB
  eingebunden. Verarbeitung durch Cloud-Modelle ist erlaubt. **Entscheidung:**
  ein eigenes, schmales Zielschema (Abschnitt 5 wird danach angepasst), in das
  der Geo-Konverter als *ein* Importer umwandelt; andere Häuser füllen es mit
  eigenen Importern. Das Plugin liefert alle Arten (`kind`), Auswahl und Filter
  liegen beim Workflow.
- **F6 Aktualisierung: ungeklärt.** Wie die Pflege weiterläuft, weiß der
  Entwickler nicht. Der Importer läuft deshalb auf Abruf; das Plugin liest nur.
- **F7 Zeilenzahl: ungeprüft.** Ob die Datei komprimiert wurde und ob gelöschte
  Zeilen im Export stehen, ist unbekannt. Einschränkung bleibt dokumentiert.
- **F8 Zeichensatz:** Textfelder der Tabelle `swb` sind sauber (kein
  Mojibake). `¬…¬` sind Nichtsortierzeichen, sie bleiben in der fertigen Zeile
  und fallen im Label für das Modell weg. Platzhalter wie `{Balken ueber}` bleiben
  in der Zeile (WinIBW-Erwartung **ungeprüft**); 81 Felder mit Zeilenumbruch
  werden im Importer zu Leerzeichen.
- **F9 Umfang:** Keine Beschränkung; das Plugin liefert alle Arten, der
  Workflow wählt aus. Die Frage Agent-Werkzeuge gegenüber klassischer Pipeline
  bleibt für Phase 4 offen.

## 9. Phasen und Aufwand (Schätzung)

| Phase | Inhalt | Aufwand |
|---|---|---|
| 1 ✅ | Analyse, Umwandler, Plugin-Gerüst, Gate, `geo_candidates` (dieser Branch) | erledigt |
| 2 | F1-F7 klären; Abbildung bestätigen; `convert` der aktuellen Datei; Plugin registrieren (Built-in oder Tier-2), Preset `lookup` | 0,5-1 Tag |
| 3 | Ergebnisübernahme ✅ (`KeywordAnalysisState.geo_sachdb`, lokale PPN im WinIBW-Eintrag, Abschlusskarte, CLI `show-protocol`, Export; Grenzen §7.1); `geo_prior` ✅ (§7.2); offen: optional `geo_select`/`geo_guard`, Vergleichslauf mit/ohne an ≥ 20 Titeln | 2-3 Tage |
| 4 | optional normalisierte Schicht + FTS5, `CLASSIFICATION`-Suchquelle für die klassische Pipeline, PPN → GND-ID | 2-4 Tage |

## 10. Reproduktion

Alle Messungen an Kopien in `.scratch/` (nicht im Repo):

```bash
python scripts/access_to_sql.py inspect .scratch/src_new/SACH-GEO1.mdb --json .scratch/inspect_new.json
python scripts/access_to_sql.py convert .scratch/src_new/SACH-GEO1.mdb .scratch/geo_new.db --allow-count-mismatch
python scripts/access_to_sql.py --reader access-parser convert .scratch/src_new/SACH-GEO1.mdb .scratch/geo_new_ap.db
python scripts/access_to_sql.py import-geo .scratch/geo_new.db .scratch/geo_sachdb_v1.db
```

`import-geo` gibt die Zeilenzahlen und die Zählung jeder Regel aus (auch in
`alima_import_rule` der Zieldatei).

Die Tabellen- und Spaltenvergleiche alt/neu, die Slot-Zählung und die
Plugin-Messung liefen als Wegwerf-Skripte; ihre Ergebnisse stehen in
`STATUS_geo_sachdb.md`.
