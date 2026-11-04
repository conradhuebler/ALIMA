# Plugin: geo_sachdb (lokale Sacherschließungsdatenbank)

Lookup-Plugin über eine SQLite-Datei, die `scripts/access_to_sql.py` aus einer
Access-Datenbank erzeugt. Liest nur (`mode=ro`), schreibt nie.

| | |
|---|---|
| Kategorie | `lookup` (Agent-Werkzeuge, freies Dict-Ergebnis) |
| Registrierung | **nicht** als Built-in. Laden als Tier-2-Code-Plugin (Verzeichnis nach `~/.config/alima/plugins/geo_sachdb/` kopieren, Code-Plugins erlauben, freigeben) oder per `register_lookup(GeoSachdbLookup)` |
| Werkzeuge | `geo_sachdb_search`, `geo_sachdb_record`, `geo_sachdb_notation` |
| Workflow | deterministische Funktion `geo_sachdb_candidates` (Schritt `geo_candidates` in `alima_v51_105`) |

## Datenbank erzeugen

```bash
python scripts/access_to_sql.py inspect QUELLE.mdb            # Schema + Zeilenzahlen, schreibt nichts
python scripts/access_to_sql.py convert QUELLE.mdb geo.db      # 1:1 nach SQLite, mit Prüfsummen
```

Leser ist mdbtools (`mdb-tables`, `mdb-schema`, `mdb-export`, `mdb-count`);
`--reader access-parser` nutzt stattdessen das Python-Paket `access_parser`.
Weicht `mdb-count` von den gelesenen Datensätzen ab, bricht `convert` ab;
`--allow-count-mismatch` nimmt die Abweichung hin und protokolliert sie in
`alima_import_table.rows_counted`. Prüfsumme und zurückgelesene Zeilen müssen
immer übereinstimmen.

## Rollen-Abbildung (Plugin-Konfiguration)

Das Plugin legt nicht fest, was eine Tabelle bedeutet. Die Instanz nennt die
Tabellen und Spalten:

| Feld | Rolle |
|---|---|
| `db_path` | SQLite-Datei. Leer oder nicht vorhanden: nicht verfügbar |
| `term_table`, `term_id_column`, `term_column` | Schlagwortsätze, Schlüssel (leer: rowid), Ansetzung |
| `term_format` | `plain` (nur Text) oder `swb_packed` (`\|x\|Ansetzung !ID! … ‡ \|x\|Verweisung …`) |
| `variant_column`, `broader_column`, `related_column`, `note_columns` | optionale Spalten |
| `term_notation_columns` | Spalten mit Verweisen auf die Schlüsselspalte der Notationstabelle (kommagetrennt) |
| `notation_table`, `notation_id_column`, `notation_code_column`, `notation_caption_column` | Notationssätze |
| `notation_scheme` | Präfix, das beim Vergleich abgestreift wird (Standard `DK`) |
| `max_results` | Treffer je Abfrage (Standard 10) |

Mehrere Instanzen desselben Typs sind möglich, etwa eine für Schlagwortsätze
mit ihrer Notationstabelle und eine zweite für eine andere Tabellengruppe.

## Was die Werkzeuge liefern

- `geo_sachdb_search(query)`: gerankte lokale Sätze. Gleiche Ansetzung 100,
  gleiche Verweisungsform 90, alle Wörter in der Ansetzung 60 (Verweisung 50),
  Teilzeichenkette 30 (25). Vergleich nach Faltung: Kleinschreibung, `ä`→`ae`
  usw., Diakritika entfernt. Jeder Treffer trägt Typ, Kennungen (`!…!`-Werte
  unverändert), Verweisungen, Oberbegriff und die verknüpften Notationen mit
  Erläuterung; ein nicht auflösbarer Verweis steht mit `resolved: false` da,
  ein Schlüssel mit mehreren Notationssätzen mit `ambiguous: true`.
- `geo_sachdb_record(record_id)`: alle gespeicherten Spalten eines Satzes plus
  die zerlegte Sicht.
- `geo_sachdb_notation(notation)`: lokale Einträge mit gleicher Notation, die
  damit verknüpften Sätze und, wenn es keinen gleichen Eintrag gibt, die nächsten
  übergeordneten Notationen (Ziffernpräfix).

Jedes Ergebnis enthält `source` (Dateiname, Prüfsumme und Stand der Quelle,
Importzeitpunkt, Zeilenzahlen), soweit die Herkunftstabellen vorhanden sind.

## Verfügbarkeit

`is_available()` ist False ohne `db_path`, ohne Datei, ohne Rollen-Abbildung,
bei fehlender Tabelle oder Spalte oder leerer Tabelle. Dann erzeugt der
Werkzeuggenerator keine Werkzeuge für die Instanz, und jede Methode antwortet
mit `{"error": …, "available": false}`.

## Grenzen

- Die Bedeutung von `!a…!`-Kennungen und der Zahl hinter der Ansetzung im
  Format `swb_packed` ist nicht geklärt; beide werden nur durchgereicht.
- Die Suche lädt alle Sätze der Instanz einmal in den Speicher und vergleicht in
  Python; es gibt keinen Volltextindex.
- Der AST-Sicherheitsscan (`security.scan_dir`) meldet für dieses Verzeichnis
  keine Befunde; das ist kein Sicherheitsnachweis (siehe `docs/plugin_system.md`).

Arbeitspaket und Befunde: [`docs/wp_geo_sachdb_plugin.md`](../../../../docs/wp_geo_sachdb_plugin.md).
