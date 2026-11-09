# Einrichtungs-Presets: ALIMA für eine Einrichtung ausrollen

Ein **Preset** (`alima_presets.json`) beschreibt die Einrichtung einer
Bibliothek: Zustand, Reihenfolge und Einstellungen der Plugin-Instanzen, ISIL
und ELN, den Standard-Workflow und Dateien, die mitgeliefert werden (etwa die
lokale Sacherschließungsdatenbank für `geo_sachdb` und die
Aufstellungssystematik für `local_scheme`). Dieselbe Datei belegt den
Einrichtungsassistenten vor (LLM-Adresse, Modelle, Datenbank, Libero-Adressen).

Module: [`src/utils/preset_apply.py`](../src/utils/preset_apply.py) (finden,
lesen, Vorschau, anwenden, zurücknehmen),
[`src/utils/preset_export.py`](../src/utils/preset_export.py) (exportieren),
[`src/utils/preset_loader.py`](../src/utils/preset_loader.py) (Vorbelegung des
Assistenten). CLI: `alima preset`. GUI: Einstellungen ▸ Datenquellen, Gruppe
„Einrichtung (Preset)“ unter den Plugin-Listen. Beispiel:
[`examples/presets/demo_institution/`](../examples/presets/demo_institution/).

## Paket

Ein Paket ist ein Ordner:

```
ub-musterstadt/
  alima_presets.json
  files/
    geo_sachdb/geo_sachdb_v1.db
    local_scheme/freihand.yaml
```

Pfade in `files[].source` gelten relativ zum Ordner der Preset-Datei. Ziel der
Dateien ist `<Konfigurationsverzeichnis>/data/` (unter Linux
`~/.config/alima/data/`), `files[].target` ist relativ dazu.

## Wo ALIMA das Preset findet

In dieser Reihenfolge, das erste gefundene gilt; das Log nennt Datei und Quelle:

1. Umgebungsvariable `ALIMA_PRESET`: eine Preset-Datei oder ein Ordner mit
   `alima_presets.json`. Zeigt sie ins Leere, steht eine Warnung im Log und die
   Suche geht weiter.
2. `<Konfigurationsverzeichnis>/alima_presets.json` (das Verzeichnis der
   `config.json`).
3. `alima_presets.json` im Projektverzeichnis.

Gefunden wird das Preset vom Einrichtungsassistenten (GUI und CLI), von der
Gruppe im Plugin-Reiter und von `alima preset show` und `alima preset apply`
ohne Pfad.

## Schlüssel

Alle Schlüssel sind optional. Schlüssel, die mit `_` beginnen, sind Kommentare.

| Schlüssel | Wirkung |
|---|---|
| `format` | `2`. Ein höherer Wert ergibt einen Hinweis in der Vorschau. |
| `preset` | `{"id", "version", "label"}`: erscheinen in der Vorschau und im Verlauf. Ohne `id` gilt `institution_name` (als Kennung geschrieben) oder der Ordnername. |
| `institution_name` | Name im Begrüßungsbanner des Assistenten (sonst `preset.label`). |
| `llm` | Vorbelegung des Assistenten: `provider_type`, `provider_name`, `base_url`, `default_model`, `task_models` (`pipeline`, `vision`). |
| `database` | Vorbelegung des Assistenten: `type` (`sqlite`, `mariadb`), `sqlite_path` oder `host`, `port`, `database`, `username`, `password`, `charset`, `gnd_preloaded`. |
| `catalog` | Vorbelegung der Katalogseite: `soap_search_url`, `soap_details_url`, `web_search_url`, `web_record_url`. Ohne diesen Block nimmt der Assistent die Adressen aus dem Eintrag `catalog` unter `plugins`. |
| `institution` | `{"isil": "...", "eln": "..."}` setzt `SystemConfig.institution_isil`/`institution_eln` (siehe [plugin_system.md](plugin_system.md), „Institution“). |
| `workflow` | setzt `SystemConfig.default_workflow`, wenn es den Workflow gibt (`__classic__` für die klassische Pipeline); sonst ein Hinweis, der Wert bleibt. |
| `plugins` | Liste der Instanzen, siehe unten. |
| `files` | Liste der mitgelieferten Dateien, siehe unten. |
| `secrets` | `[{"plugin", "key", "hint"}]`: nur Deklaration. Die Vorschau listet sie als „noch einzutragen“ mit der Umgebungsvariable `ALIMA_PLUGIN_<INSTANZ>_<SCHLÜSSEL>`; geschrieben wird nichts. |

Felder mit dem Namen `api_key` oder `token` (auch `*_api_key`, `*_token`)
werden beim Lesen verworfen, überall in der Datei. Hatte eines einen Wert,
steht einmal die Warnung „Im Preset gefundene Schlüssel werden nicht
verwendet“ im Log und als Hinweis in der Vorschau.

### `plugins`

```json
{"instance": "geo_sachdb", "state": "on", "settings": {"db_path": "@file:geo_db"}}
{"instance": "k10plus_sru", "type": "sru", "category": "search_provider",
 "label": "K10plus (SRU)", "state": "on", "settings": {"preset": "k10plus"}}
```

- `instance`: die Instanz-ID. Gibt es die Instanz, wird sie eingestellt (es
  entsteht keine zweite daneben). Gibt es sie nicht, wird sie angelegt, wenn
  `type` (ohne `type`: die Instanz-ID) ein registrierter Plugin-Typ ist; sonst
  ein Hinweis, der Eintrag entfällt. `category` ist nötig, wenn ID oder Typ in
  mehreren Kategorien vorkommen. Passt `type` nicht zum Typ einer vorhandenen
  Instanz: Hinweis, Eintrag entfällt.
- `state`: `on`, `cache_only` oder `off`, gespeichert wie im Plugin-Reiter als
  `enabled` und `settings.cache_only` (dieselben Helfer,
  `src/core/search/cache_only.py`). `cache_only` für einen Typ ohne Nur-Cache:
  Hinweis, der Zustand bleibt.
- `label`: Name der Instanz.
- `settings`: nur Felder des `ConfigField`-Schemas des Typs; Werte werden mit
  `ConfigField.coerce` umgewandelt. Unbekannte Felder und nicht erlaubte Auswahlwerte:
  Hinweis, nicht geschrieben. Geheime Felder (SECRET) werden nie gesetzt.
  `"@file:<id>"` wird zum Zielpfad der Datei mit dieser `id`; verweist es auf
  eine `id`, die unter `files` fehlt, ist das ein Fehler.
- Reihenfolge: Die Reihenfolge der Liste ist die Reihenfolge der genannten
  Instanzen innerhalb ihrer Kategorie; nicht genannte Instanzen folgen in ihrer
  bisherigen Reihenfolge. Die Reihenfolge wirkt dort, wo sie im Programm wirkt
  (erste aktive Instanz je Typ, DK-Kette, `search_titles`).

### `files`

```json
{"id": "geo_db", "kind": "geo_sachdb_db", "source": "files/geo_sachdb_v1.db",
 "target": "geo_sachdb/geo_sachdb_v1.db", "sha256": "…64 Hexziffern…"}
```

- `sha256` ist Pflicht. Stimmt sie nicht, fehlt sie oder fehlt die Quelldatei,
  wird die Datei nicht installiert und die Vorschau meldet einen Fehler.
- `kind`: `geo_sachdb_db` (die Datei wird nur lesend geöffnet und mit
  `geo_sachdb.schema.check_schema` geprüft), `vocabulary` (YAML mit einer Liste
  `entries`, die Form von `local_scheme`) oder `file` (nur die Prüfsumme).
- `target` muss relativ sein und darf den Datenordner nicht verlassen; ohne
  `target` gilt der Dateiname der Quelle.
- Installation: Kopie in eine temporäre Datei im Zielordner, Prüfsumme der
  Kopie, dann Ersetzen. Eine vorhandene Datei bleibt als `<name>.bak` (eine
  Generation; eine ältere `.bak` wird überschrieben). Eine gleiche Datei
  (gleiche Prüfsumme) bleibt unberührt.

**Vokabulare.** `local_scheme` sucht eine Vokabular-ID ohne Pfad in
`vocabularies/` (relativ zum Arbeitsverzeichnis), dann in
`~/.config/alima/vocabularies/`, dann im Ordner `vocabularies/` des Programms.
Das Preset kopiert nicht dorthin (der erste und der letzte Ort liegen im
Programmverzeichnis), sondern in den Datenordner, und die Instanz bekommt den Pfad über
`"vocabulary": "@file:<id>"`. Ein mitgeliefertes Vokabular ohne diese
Einstellung wird installiert, aber nicht gelesen.

## Vorschau, Anwenden, Rückgängig

Die Vorschau (`build_plan`) schreibt nichts. Sie listet je Änderung den Wert
vorher und nachher, neue, unveränderte und ersetzte Dateien, noch einzutragende
Secrets, Hinweise und Fehler. Ein Plan mit Fehlern wird nicht angewendet, auch
nicht teilweise; Hinweise halten das Anwenden nicht auf.

Anwenden (`apply_plan`):

- setzt nur, was das Preset nennt; alle anderen Werte bleiben, Secrets nie;
- prüft vorher, dass die Konfiguration noch die Werte hat, auf denen die
  Vorschau beruht (sonst: Vorschau neu erstellen);
- installiert zuerst die Dateien, speichert dann über
  `ConfigManager.save_config`; scheitert eines von beiden, werden die
  installierten Dateien zurückgenommen;
- trägt in `AlimaConfig.applied_presets` ein: ID, Version, Name, Quelle,
  Zeitpunkt, je Änderung der vorherige und der neue Wert, installierte Dateien
  (mit `.bak`). Ein leerer Verlauf steht nicht in der `config.json`.

Ein zweites Anwenden desselben Presets ist ein leerer Plan und speichert nichts.

Rückgängig (`revert_last`) nimmt den letzten Eintrag des Verlaufs zurück: jeder
Wert, den das Preset gesetzt hat, bekommt seinen vorherigen Wert, solange er
noch den Wert des Presets hat. Seitdem geänderte Werte bleiben und werden
genannt. Eine vom Preset angelegte Instanz wird entfernt, wenn sie seitdem
nicht geändert wurde. Dateien: neue werden gelöscht, ersetzte kommen aus der
`.bak` zurück, seitdem geänderte bleiben.

```bash
alima preset show                                   # gefundenes und zuletzt angewendetes Preset
alima preset apply ub-musterstadt/ --dry-run        # nur Vorschau
alima preset apply ub-musterstadt/                  # Vorschau, Rückfrage, anwenden
alima preset apply --yes                            # gefundenes Preset ohne Rückfrage
alima preset revert                                 # letztes Preset zurücknehmen
```

`alima preset` läuft ohne vorherige Einrichtung (wie `setup`). Ohne `--yes`
fragen `apply` und `revert` nach. Mit Fehlern endet `apply` mit Exit-Code 1.

**GUI.** Die Gruppe „Einrichtung (Preset)“ zeigt das gefundene Preset (Name,
Version, Quelle, Pfad) und das zuletzt angewendete. „Preset anwenden…“ bietet
das gefundene Preset, eine Datei oder einen Ordner; die Vorschau zeigt den Plan,
„Anwenden“ gibt es nur ohne Fehler. Vorschau, Kopieren, Rückgängig und Export
laufen im Hintergrund, Fehler erscheinen als Meldung und in der Statuszeile der
Gruppe. Angewendet wird sofort, unabhängig von OK oder Abbrechen des
Einstellungsdialogs; der Dialog übernimmt die gesetzten Werte (Plugin-Listen,
ISIL, ELN, Standard-Workflow) in seine Felder, nicht gespeicherte Eingaben in
den Plugin-Listen bleiben erhalten.

**Erster Start.** Der Einrichtungsassistent (GUI und `alima setup`) belegt seine
Felder aus dem gefundenen Preset vor, ohne Schlüssel und Token. Nach
„Fertig stellen“ wendet er `institution`, `workflow`, `plugins` und `files` an
(derselbe Plan, ohne weiteren Dialog). Die Zusammenfassungsseite listet vorher,
was das Preset setzt; das Ergebnis steht im Log, in der GUI-Statuszeile nach
einem Lauf aus dem Menü, im Terminal bei `alima setup`. Ein fehlerhaftes Preset
bricht den Assistenten nicht ab; es wird dann nicht angewendet. Was das Preset
unter `plugins` nennt, gilt nach dem Assistenten auch dann, wenn auf der
Katalogseite etwas anderes eingetragen wurde.

## Export

`alima preset export <ordner> [--plugin <instanz-id> …] [--id …] [--version …] [--label …]`
oder „Preset exportieren…“ (Gruppe im Plugin-Reiter und „Als Preset
exportieren“ im Einstellungsdialog; eine Checkliste wählt die Instanzen, das
Paket entsteht als Unterordner `<Preset-ID>` im gewählten Ordner). Der Ordner
muss neu oder leer sein. Exportiert wird der gespeicherte Stand.

Das Paket enthält:

- die gewählten Instanzen (ohne Auswahl alle) in Listenreihenfolge mit
  Kategorie, Typ, Name, Zustand und den Einstellungen, die sie speichern;
- die Datenbank von `geo_sachdb` und das Vokabular von `local_scheme` (auch
  das Standardvokabular, wenn keines eingestellt ist), kopiert nach
  `files/<instanz>/`, mit Prüfsumme, die Einstellung als `@file:<id>`;
- ISIL und ELN, den Standard-Workflow;
- für den Assistenten Typ, Name, Adresse und Modelle des allgemeinen
  Standard-Anbieters und eine Server-Datenbank (MariaDB/MySQL) ohne Passwort;
- `secrets` für jedes verwendete geheime Feld.

Nicht im Paket, jeweils mit Hinweis: geheime Felder und Felder namens
`api_key`/`token`, das Datenbankpasswort, der Pfad einer lokalen
SQLite-Datenbank, jede andere Einstellung mit einem lokalen Pfad (etwa
`k10plus.cache_dir`), Einstellungen, die der Typ nicht deklariert, die
Einstellungen einer Instanz, deren Typ nicht geladen ist (nur ihr Zustand).
Ein exportiertes Paket, auf einer anderen Installation angewendet, ergibt
dieselben Zustände, Einstellungen und Reihenfolgen und Dateien mit derselben
Prüfsumme (Test `tests/test_institution_presets.py::TestExport`).

## Grenzen

- **Beratend.** Nutzer können nach dem Anwenden alles ändern; ein Preset sperrt
  nichts.
- **Keine Secrets.** Ein Preset trägt keine Schlüssel und Tokens; jede Person
  trägt sie im Plugin-Reiter oder als Umgebungsvariable ein. Das
  Datenbankpasswort im Block `database` liest der Assistent weiterhin als
  Vorbelegung; der Export schreibt es nicht.
- **Dateien nur mit Prüfsumme.** Die Prüfung sagt, dass die Datei die gemeinte
  ist, nicht, dass ihr Inhalt stimmt. Bei der Geo-Datenbank wird nur das Schema
  geprüft.
- **Daten und Rechte.** Die Datenbank einer Einrichtung gehört nicht in das
  öffentliche Repository, nur in ihr eigenes Paket. Lizenz und Rechte an den
  mitgelieferten Daten liegen bei der Einrichtung.
- **Keine Plugin-Verzeichnisse.** Ein Preset stellt Instanzen registrierter
  Typen ein und legt welche an; es liefert keinen Plugin-Code und keine
  `plugin.toml` mit. Ein Code-Plugin muss auf dem Arbeitsplatz installiert und
  freigegeben sein, sonst entfällt sein Eintrag mit Hinweis.
- **Eine Generation `.bak`.** Wird dieselbe Datei zweimal ersetzt, ist die
  ursprüngliche Fassung nach dem zweiten Mal nicht mehr da; Rückgängig meldet
  dann, dass keine Sicherung vorhanden ist.
- **Nur lokale Pfade.** Presets werden nicht aus dem Netz geladen, nicht
  signiert und nicht verschlüsselt.
- Ob unter Windows eine Datei ersetzt werden kann, die ein laufendes ALIMA
  geöffnet hält, ist nicht geprüft.

## Beispielablauf (UB)

1. Auf einem eingerichteten Arbeitsplatz (Plugins, ISIL/ELN, Standard-Workflow,
   Geo-Datenbank, Aufstellungssystematik) speichern, dann
   `alima preset export /srv/alima/ub-paket --id ub-musterstadt --version 2026.1 --label "UB Musterstadt"`.
2. Den Ordner prüfen (Endpunkte, Dateien; keine Schlüssel enthalten) und intern
   verteilen, etwa auf ein Netzlaufwerk. Nicht in ein öffentliches Repository.
3. Auf jedem Arbeitsplatz entweder
   `alima preset apply /srv/alima/ub-paket` (mit Vorschau und Rückfrage) oder
   `ALIMA_PRESET=/srv/alima/ub-paket` setzen: dann findet der
   Einrichtungsassistent das Paket beim ersten Start und wendet es an, und die
   GUI-Gruppe bietet es unter „Gefundenes Preset“ an.
4. Den Libero-Token (und andere Secrets) trägt jede Person selbst ein.
5. Neue Version des Pakets: erneut anwenden; nur Abweichungen werden geändert.

Tests: [`tests/test_institution_presets.py`](../tests/test_institution_presets.py).
