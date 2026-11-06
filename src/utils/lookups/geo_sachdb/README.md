# Plugin: geo_sachdb (lokale Sacherschließungsdatenbank)

Lookup-Plugin über eine SQLite-Datei im eigenen Schema des Plugins (Version 1,
definiert in `schema.py`). Das Plugin liest nur (`mode=ro`) und kennt keine
Quelldatenbank: jedes Haus füllt das Schema mit einem eigenen Importer. Für die
Freiberger Access-Datenbank ist das `scripts/access_to_sql.py import-geo`.

| | |
|---|---|
| Kategorie | `lookup` (Agent-Werkzeuge, freies Dict-Ergebnis) |
| Registrierung | Built-in (`src/utils/lookups/__init__.py`). Jede Installation führt die Instanz `geo_sachdb` in der Lookup-Liste; ohne `db_path` ist sie nicht verfügbar und erzeugt keine Werkzeuge. Keine Freigabe von Code-Plugins nötig |
| Konfiguration | `db_path` (Datei im Schema v1), `max_results` (Standard 10) |
| Werkzeuge | `geo_sachdb_search`, `geo_sachdb_lines`, `geo_sachdb_notation` |
| Workflow | deterministische Funktion `geo_sachdb_candidates` (Schritt `geo_candidates` in `alima_v51_105`) |

## Eingabeformat: Schema v1

Die Tabellen legt `schema.create_schema(conn)` an; `schema.check_schema(conn)`
ist die Prüfung, die das Plugin beim Öffnen macht. Weitere Tabellen (etwa die
Herkunftstabellen eines Importers) und weitere Spalten sind erlaubt.

**`meta(key, value)`**: `schema_version` muss `1` sein. Optional und als
Herkunft angezeigt: `source` (Name der Quelle), `source_date`, `source_sha256`,
`created_at`, `importer`.

**`heading`**: ein lokales Schlagwort je Zeile.

| Spalte | Pflicht | Inhalt |
|---|---|---|
| `id` | ja | ganze Zahl, eindeutig |
| `label` | ja | Ansetzung als reiner Text; wird durchsucht und dem Modell gezeigt. Das Plugin entfernt beim Lesen noch `\|x\|`-Kennzeichen, `!…!`-Kennungen und `¬` |
| `kind` | ja | `Sachschlagwort`, `Geographikum`, `Person`, `Körperschaft` oder `sonstiges` |
| `ppn` | nein | K10plus-Normdaten-PPN (Ziffern, letzte Stelle Ziffer oder `X`), ohne Ausrufezeichen |
| `chain` | nein | fertige lokale Zeichenkette, so wie das Haus sie einträgt |
| `variants` | nein | Verweisungsformen als Text, getrennt durch `;`, Zeilenumbruch, `‡` oder ein vorangestelltes `\|x\|`-Kennzeichen |

**`classification`**: ein lokaler Notationseintrag je Zeile.

| Spalte | Pflicht | Inhalt |
|---|---|---|
| `id` | ja | ganze Zahl, eindeutig |
| `system` | ja | `DK` oder `RVK` |
| `notation` | ja | DK ohne Präfix (`553.62`), RVK als `QP 925`; `schema.normalize_notation` erzeugt diese Form. Dieselbe Notation darf mehrfach vorkommen |
| `label` | nein | Erläuterung |
| `ppn` | nein | K10plus-PPN des Notationsdatensatzes |

**`heading_classification(heading_id, classification_id, position, ambiguous)`**:
die Notationen eines Schlagworts in Reihenfolge (`position` ab 0). `ambiguous = 1`
markiert eine Verknüpfung, die die Quelle nicht eindeutig auflösen ließ; der
Importer verknüpft dann alle in Frage kommenden Einträge.

## Was die Werkzeuge liefern

Alle Ergebnisse tragen `source` (Angaben aus `meta`, Zeilenzahlen).

- `geo_sachdb_search(query, kinds?)`: gerankte lokale Schlagwörter. Gleiche
  Ansetzung 100, gleiche Verweisungsform 90, alle Wörter in der Ansetzung 60
  (Verweisung 50), Teilzeichenkette 30 (25). Vergleich nach Faltung:
  Kleinschreibung, `ä` und `ae` gleich, Diakritika entfernt. `kinds` begrenzt auf
  Arten, ohne Angabe alle.
- `geo_sachdb_lines(terms, kinds?)`: fertige WinIBW-Zeilen für eine Liste von
  Begriffen. Es zählen nur gleiche Ansetzungen und gleiche Verweisungsformen.
  `found` enthält je Begriff die Schlagwörter mit ihren Notationen, `not_found`
  die Begriffe ohne lokales Schlagwort, `lines` jede Zeile einmal.
- `geo_sachdb_notation(notation)`: lokale Einträge mit derselben Notation (DK
  oder RVK) mit Zeile, die Schlagwörter, die sie verwenden, und ohne gleichen
  Eintrag die nächsten übergeordneten DK-Notationen (Ziffernpräfix).
  `ambiguous` ist wahr, wenn die Einträge verschiedene PPN tragen.

### Zeilen

| Fall | Zeile |
|---|---|
| Schlagwort mit `ppn` | `5550 !PPN!` |
| Schlagwort ohne `ppn`, mit `chain` | `5550 <chain>` (`line_form: chain`) |
| Schlagwort ohne `ppn` und `chain` | `5550 <label>` (`line_form: label`) |
| Notation mit `ppn` | `6700 !PPN!` |
| DK-Notation ohne `ppn` | `6700 DK <notation>` |
| andere Notation ohne `ppn` | keine Zeile (`line: null`) |

Die Zeilen folgen der Praxis der UB Freiberg: WinIBW löst einen Eintrag ohne PPN
selbst zur PPN auf. Die Feldnummern 5550 und 6700 sind fest.

## Datenbank für Freiberg erzeugen

```bash
python scripts/access_to_sql.py convert SACH-GEO1.mdb zwischenstand.sqlite --allow-count-mismatch
python scripts/access_to_sql.py import-geo zwischenstand.sqlite geo_sachdb.db
```

`convert` bildet die Access-Tabellen 1:1 ab (Zwischenstand). `import-geo` liest
daraus `swb` und `dk`, `dk_bio`, `dk_berg`, `dk_geographica` und wendet an:

- `SWB_AEQUIV`: Typkennzeichen `|s|` `|g|` `|p|` `|k|`/`|b|` ergeben die Art,
  alles andere `sonstiges`. Die bevorzugte Form reicht bis zum nächsten
  Kennzeichen hinter der ersten PPN (ohne PPN bis zum zweiten Kennzeichen),
  höchstens bis `‡`; sie wird `chain` (mit `¬…¬`, Platzhaltern wie
  `{Balken ueber}` und der Zahl am Ende). Der Rest ist `variants`.
- `ppn` ist die erste `!…!`-Nummer; `!a…!`-Kennungen sind keine PPN, weitere
  Nummern werden ignoriert. Für Notationen gilt dasselbe in der Spalte `dk`.
- `label` ist `chain` ohne Kennzeichen, Kennungen, `¬` und Zahl am Ende.
- `SWB.DK_ID*` wird intern über `dk.DK_ID` aufgelöst und nicht ausgegeben. Der
  Platzhalter 9999 und Verweise darauf entfallen. Kommt eine `DK_ID` mehrfach
  vor, werden alle Einträge verknüpft und als `ambiguous` markiert.
- Notationen der Form `QP 925` (die Tabelle `dk` enthält auch RVK) werden
  `RVK`, alle übrigen `DK`.
- Zeilenumbrüche in Texten werden zu Leerzeichen.
- Nicht übernommen: `dk_alt`, `Geographica`, `EAS`, `SS_SG`, die Spalten
  `NOTES`, `SWB_NOTES`, `OBER_BEGR`, `VERWANDTE`.

Jede Regel wird gezählt (`alima_import_rule`); `alima_import_run` aus dem
Zwischenstand wird mitkopiert. Die Zieldatei wird erst ersetzt, wenn sie die
Schemaprüfung besteht und die Zeilenzahlen stimmen.

## Verfügbarkeit

`is_available()` ist False ohne `db_path`, ohne Datei, bei einem anderen Schema
(etwa dem Zwischenstand aus `convert`), bei einer anderen Schemaversion oder wenn
`heading` und `classification` beide leer sind. Dann erzeugt der
Werkzeuggenerator keine Werkzeuge für die Instanz, und jede Methode antwortet mit
`{"error": …, "available": false}`.

## Grenzen

- Ob WinIBW `5550 <chain>` und Platzhalter wie `{Balken ueber}` so übernimmt, ist
  nicht geprüft.
- Für RVK-Notationen ohne PPN gibt es keine Zeilenregel.
- Was die Zahl am Ende von `SWB_AEQUIV` bedeutet, ist nicht geklärt; sie bleibt in
  `chain` und damit in einer Zeile ohne PPN.
- Der Importer prüft die Form einer DK-Notation nicht; Einträge ohne Ziffern
  landen mit ihrem Text als DK-Notation in `classification`.
- Die Suche lädt alle Zeilen einer Instanz einmal in den Speicher und vergleicht
  in Python; es gibt keinen Volltextindex.
- Der AST-Sicherheitsscan (`security.scan_dir`) meldet für dieses Verzeichnis
  keine Befunde; das ist kein Sicherheitsnachweis (siehe `docs/plugin_system.md`).

Arbeitspaket und Befunde: [`docs/wp_geo_sachdb_plugin.md`](../../../../docs/wp_geo_sachdb_plugin.md).
