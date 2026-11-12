# Einrichtungs-Presets: ALIMA für eine Einrichtung ausrollen

Ein **Preset** (`alima_presets.json`) beschreibt die Einrichtung einer
Bibliothek: Zustand, Reihenfolge und Einstellungen der Plugin-Instanzen, ISIL
und ELN, den Standard-Workflow und Dateien, die mitgeliefert werden (etwa die
lokale Sacherschließungsdatenbank für `geo_sachdb` und die
Aufstellungssystematik für `local_scheme`). Es kann LLM-Zugänge (Provider) und
die Standardmodelle mitgeben (siehe „`providers`“, „`default_models`“).
Dieselbe Datei belegt den Einrichtungsassistenten vor (LLM-Adresse, Modelle,
Datenbank, Libero-Adressen). Ein Paket kann außerdem Teile des Caches mitgeben
(siehe „`cache`“).

Module: [`src/utils/preset_apply.py`](../src/utils/preset_apply.py) (finden,
lesen, Vorschau, anwenden, zurücknehmen),
[`src/utils/preset_export.py`](../src/utils/preset_export.py) (exportieren),
[`src/utils/preset_cache.py`](../src/utils/preset_cache.py) und
[`src/core/_ukm_cache_merge.py`](../src/core/_ukm_cache_merge.py) (Cache),
[`src/utils/preset_loader.py`](../src/utils/preset_loader.py) (Vorbelegung des
Assistenten). CLI: `python3 src/alima_cli.py preset`. GUI: Einstellungen ▸ Datenquellen, Gruppe
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
    cache/catalog_dk_cache.jsonl.gz
    cache/search_mappings.jsonl.gz
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
Gruppe im Plugin-Reiter und von `python3 src/alima_cli.py preset show` und `python3 src/alima_cli.py preset apply`
ohne Pfad.

## Schlüssel

Alle Schlüssel sind optional. Schlüssel, die mit `_` beginnen, sind Kommentare.

| Schlüssel | Wirkung |
|---|---|
| `format` | `2`. Ein höherer Wert ergibt einen Hinweis in der Vorschau. |
| `preset` | `{"id", "version", "label"}`: erscheinen in der Vorschau und im Verlauf. Ohne `id` gilt `institution_name` (als Kennung geschrieben) oder der Ordnername. |
| `institution_name` | Name im Begrüßungsbanner des Assistenten (sonst `preset.label`). |
| `providers` | Liste der LLM-Provider, siehe unten. Anwenden legt sie an oder ersetzt den gleichnamigen Eintrag; der Assistent liest daraus seine Vorbelegung. |
| `default_models` | Standard-Provider und -Modell je Bereich (`general`, `pipeline`, `vision`, `decision`), siehe unten. Überschreibt beim Anwenden; abwählbar. |
| `database` | Vorbelegung des Assistenten: `type` (`sqlite`, `mariadb`), `sqlite_path` oder `host`, `port`, `database`, `username`, `password`, `charset`, `gnd_preloaded`. |
| `catalog` | Vorbelegung der Katalogseite: `soap_search_url`, `soap_details_url`, `web_search_url`, `web_record_url`. Ohne diesen Block nimmt der Assistent die Adressen aus dem Eintrag `catalog` unter `plugins`. |
| `institution` | `{"isil": "...", "eln": "..."}` setzt `SystemConfig.institution_isil`/`institution_eln` (siehe [plugin_system.md](plugin_system.md), „Institution“). |
| `workflow` | setzt `SystemConfig.default_workflow`, wenn es den Workflow gibt (`__classic__` für die klassische Pipeline); sonst ein Hinweis, der Wert bleibt. |
| `plugins` | Liste der Instanzen, siehe unten. |
| `files` | Liste der mitgelieferten Dateien, siehe unten. |
| `secrets` | `[{"plugin", "key", "hint"}]`: nur Deklaration. Die Vorschau listet sie als „noch einzutragen“ mit der Umgebungsvariable `ALIMA_PLUGIN_<INSTANZ>_<SCHLÜSSEL>`; geschrieben wird nichts. |
| `cache` | Teile des Caches, die beim Anwenden in den Cache der Installation übernommen werden, siehe unten. |

Der Block `llm` früherer Pakete (Vorbelegung des Assistenten mit
`provider_type`, `provider_name`, `base_url`, `default_model`, `task_models`)
wird nicht mehr gelesen. Steht er in einer Datei, nennt die Vorschau einmal den
Hinweis „Veralteter Block 'llm' wird ignoriert“; sonst ändert sich nichts. Ein
Paket ohne `providers` wirkt wie bisher. `format` bleibt `2`: ein älteres ALIMA
übergeht `providers` und `default_models` mit dem Hinweis „Unbekannter
Schlüssel“.

**Schlüssel und Tokens.** Felder mit dem Namen `api_key` oder `token` (auch
`*_api_key`, `*_token`) werden beim Lesen verworfen, überall in der Datei, mit
**einer** Ausnahme: `api_key` direkt in einem Eintrag der Liste `providers`
(`providers[n].api_key`). Schon eine Ebene tiefer in einem Provider-Eintrag,
in `plugins`, `catalog`, `default_models` oder einer anderen Liste namens
`providers` wird auch `api_key` verworfen. Hatte ein verworfenes Feld einen
Wert, steht einmal die Warnung „Im Preset gefundene Schlüssel werden nicht
verwendet“ (mit den Feldnamen, nie den Werten) im Log und als Hinweis in der
Vorschau. Das Datenbankpasswort (`database.password`) fällt nicht unter diese
Regel (siehe „Export“). Ein Provider-Schlüssel kommt nur auf ausdrücklichen
Wunsch in ein Paket und steht dann **im Klartext** in `alima_presets.json`:
das Paket ist dann vertraulich, gehört nur an berechtigte Personen und nicht in
ein Repository oder eine offene Netzablage. Vorschau, Export-Zusammenfassung,
Log-Zeilen und Fehlermeldungen der Preset-Module und die Ausgabe von
`python3 src/alima_cli.py preset` zeigen den Schlüssel nie, nur ob einer
gesetzt wird oder im Paket steht (Test
`tests/test_preset_providers.py::TestKeyNeverShown`).

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

### `providers`

```json
{"name": "ub-ollama", "provider_type": "ollama", "enabled": true,
 "preferred_model": "modell:32b", "description": "LLM-Server der UB",
 "host": "llm.ub.example", "port": 11434, "use_ssl": false, "connection_type": "native_client"}
{"name": "ub-chat", "provider_type": "openai_compatible", "enabled": true,
 "base_url": "https://chat.ub.example/v1", "preferred_model": "modell-chat"}
```

Die Felder sind die Zugangsfelder von `UnifiedProvider`
(`src/utils/config_models.py`): `name` und `provider_type` (Pflicht),
`enabled`, `base_url`, `preferred_model`, `description`; nur für `ollama`
außerdem `host`, `port`, `use_ssl`, `connection_type`. Nicht im Paket: `id`
(je Installation) und `available_models` (zur Laufzeit gefüllt). `api_key` nur
aus einem Export mit Schlüssel (siehe „Export“). Fehlende Felder bekommen die
Standardwerte von `UnifiedProvider`.

- **Typen:** `ollama` (nativ, auch Ollama Cloud über Host oder `base_url`),
  `openai_compatible`, `gemini`, `anthropic` (`PROVIDER_TYPES` in
  `config_models.py`; ein Test hält die Liste gleich mit den Typen, die
  `LlmService` bedient).
- **Prüfung (Fremdeingabe):** bekannter Typ; `name` nicht leer, höchstens 100
  Zeichen, ohne die Zeichen `/`, `\`, `` ` ``, `$`, `;`, `|`, `&`, `<`, `>`
  und ohne Steuerzeichen; `base_url` leer oder
  `http`/`https` mit Rechnername, ohne Benutzer und Passwort in der Adresse und
  ohne Leerzeichen; `host` nur Rechnername oder IP-Adresse (ohne Schema und
  Pfad); `port` 1 bis 65535; `connection_type` `native_client` oder
  `openai_compatible`; `preferred_model` ohne Leer-, Steuer- und
  Shell-Sonderzeichen; `api_key` Text ohne Leerraum. Ein Eintrag, der das nicht
  erfüllt, steht in der Vorschau als „✗ abgelehnt“ mit Grund und als Hinweis und
  wird übergangen; der Rest des Presets bleibt anwendbar (wie bei `plugins`).
  Zweimal derselbe Name (ohne Groß-/Kleinschreibung): der zweite wird abgelehnt.
  `host`/`port`/`use_ssl`/`connection_type` bei einem anderen Typ und
  unbekannte Felder: Hinweis, übergangen.
- **Anwenden:** Gibt es einen Provider mit diesem Namen (ohne Groß-/
  Kleinschreibung, wie `get_provider_by_name` und der Einrichtungsassistent),
  wird er **an seiner Stelle ersetzt**: alle Zugangsfelder kommen aus dem Paket,
  `id` und `available_models` bleiben. Sonst wird er hinten angefügt. Alle
  anderen Provider bleiben unberührt.
- **Schlüssel:** gesetzt nur, wenn der Eintrag einen trägt. Ohne Schlüssel im
  Paket bleibt der Schlüssel des ersetzten Eintrags **erhalten**. Ein Schlüssel
  aus dem Paket steht danach im Klartext in der `config.json`, wie ein im
  Anbieter-Dialog eingetragener. Ein `gemini`/`anthropic`-Provider ohne
  Schlüssel ist nicht nutzbar; die Vorschau sagt das als Hinweis.
- **Vorschau:** je Provider „neu“, „ersetzt“ (mit den geänderten Feldern, alt →
  neu) oder „unverändert“, dazu „Schlüssel wird gesetzt“, „Schlüssel
  unverändert“ oder „vorhandener Schlüssel bleibt“.

### `default_models`

```json
"default_models": {
  "general":  {"provider": "ub-chat",   "model": "modell-chat"},
  "pipeline": {"provider": "ub-ollama", "model": "modell:32b"},
  "vision":   {"provider": "ub-ollama", "model": "modell-vision"},
  "decision": {"provider": "ub-ollama", "model": "entscheider"}
}
```

Je Bereich die Felder von `UnifiedProviderConfig`: `general` =
`preferred_provider`/`preferred_model` (Allgemein, zugleich Agentik und Chat),
`pipeline_default_*`, `vision_default_*`, `decision_default_*` (siehe
`explain_default_provider_model`). Der Provider muss nach dem Block
`providers` vorhanden sein, sonst Hinweis und übergangen. Ohne `model` gilt das
bevorzugte Modell des Providers (`resolve_provider_model`); `decision` braucht
ein Modell (es hat keinen Rückfall). Ist der Provider ausgeschaltet oder ohne
Schlüssel nicht nutzbar, steht ein Hinweis in der Vorschau; der Wert wird
trotzdem gesetzt und greift, sobald der Provider nutzbar ist.

Standard: die Werte **überschreiben** die eingestellten; die Vorschau nennt je
Bereich alt → neu. Abwählbar beim Anwenden: `--no-default-models` (CLI), der
Haken „Standardmodelle aus dem Preset übernehmen“ in der Vorschau (GUI),
`build_plan(..., default_models=False)` bzw. `apply_plan(..., default_models=False)`.
Abgewählt listet die Vorschau sie als „abgewählt: bleiben, wie sie sind“;
Provider und alles andere werden trotzdem angewendet.

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

### `cache`

Ein Paket kann Teile des Such-Caches mitgeben, die beim Anwenden in den Cache
der Zielinstallation **zusammengeführt** werden. Anlass: sind Libero und finc
lange nicht erreichbar, liest der Zustand „Nur Cache“ den DK-Cache
(`catalog_dk_cache`); mit dem Paket muss ihn nicht jede Person selbst aufbauen.

```json
"cache": {"isil": "DE-105", "created": "2026-11-10T14:02:00+00:00",
          "tables": [
            {"id": "dk_cache", "table": "catalog_dk_cache",
             "source": "files/cache/catalog_dk_cache.jsonl.gz",
             "sha256": "…64 Hexziffern…", "rows": 102, "merge": "newer_wins"},
            {"id": "raw_lobid", "table": "search_response_cache", "source_filter": "lobid",
             "source": "files/cache/search_response_cache_lobid.jsonl.gz",
             "sha256": "…", "rows": 40, "merge": "newer_wins"}]}
```

**Umfang.** Erlaubt sind nur drei Tabellen:

- `catalog_dk_cache` (Export-Teil `dk`): nur Einträge mit `search_status =
  'success'` und `result_count > 0`. Einträge „keine Ergebnisse“ und Fehler
  bleiben draußen, weil „Nur Cache“ ohne Altersgrenze liest und sie dort als
  dauerhafte Antwort gälten. Die Fehlerfelder (`error_message`, `retry_after`,
  `consecutive_failures`) stehen nicht in der Datei und werden beim
  Zusammenführen zurückgesetzt (leer, leer, 0).
- `search_mappings` (Teil `mappings`): alle Zeilen.
- `search_response_cache` (Teil `raw:<quelle>`, eine Datei je Quelle): nur auf
  Wunsch, nie `kvk`. Exportiert werden Antworten mit HTTP-Status 200 oder ohne
  gespeicherten Status, ohne Schlüssel `error`/`errors` und nicht leer. Leer
  heißt: `totalItems`, `total` oder `numFound` ist 0, oder `member`, `records`
  bzw. `docs` ist leer ohne Zähler über 0, oder `subjects` ist leer (swb).
  `result_count` schreiben nur kvk und sru, bei allen anderen Quellen steht 0;
  es allein kennzeichnet keine leere Antwort. Antworten anderer Form werden
  exportiert.

Nie im Paket: die GND-Kopie (`gnd_entries`, `gnd_local.db`) und jede andere
Tabelle.

**Format.** Je Tabelle eine Datei `files/cache/<tabelle>.jsonl.gz`
(Rohantworten `search_response_cache_<quelle>.jsonl.gz`), gzip, UTF-8. Die erste
Zeile ist der Kopf `{"table": …, "columns": […], "rows": N}` (bei Rohantworten
zusätzlich `"source_filter"`), jede weitere Zeile eine JSON-Liste mit den Werten
in der Reihenfolge von `columns`. Zeitstempel stehen als `JJJJ-MM-TT hh:mm:ss`.
Die Datei besteht aus zwei gzip-Abschnitten (Kopf, Zeilen) ohne Zeitstempel;
gleiche Zeilen ergeben gleiche Bytes. `zcat` liest sie als eine Datei.

**Prüfung beim Lesen.** Vor dem Zusammenführen liest die Vorschau jede Datei
vollständig und prüft: Prüfsumme (`sha256` Pflicht), Pfad innerhalb des Pakets,
Tabelle aus der Liste der drei, Spalten nur aus dem Schema der Tabelle
(dieselben Namen wie die CREATE-Anweisungen in `_ukm_schema.py`; ein Test
vergleicht sie mit einer neu angelegten Datenbank), Pflichtspalten (Schlüssel,
NOT NULL, `last_updated`), Typen (Text, ganze Zahl, Zeitstempel, Text mit JSON),
dieselbe Regel wie beim Export (etwa kein `no_results`, kein `kvk`), jeder
Schlüssel nur einmal, Zeilenzahl wie im Kopf und im Preset. Gelesen wird
zeilenweise, mit Obergrenzen gegen Entpackungsbomben: höchstens 300 MB entpackt
und 2 Millionen Zeilen je Datei, eine Zeile höchstens 16 MB
(`MAX_UNPACKED_BYTES`, `MAX_ROWS_PER_FILE`, `MAX_LINE_BYTES` in
`preset_cache.py`). Der Export hält dieselben Grenzen ein und bricht sonst mit
einer Meldung ab.

**Zusammenführen.** Je Primärschlüssel gewinnt der Eintrag mit dem neueren
`last_updated`: fehlt er lokal, wird er eingefügt („neu“); ist der Eintrag im
Paket neuer, werden seine Spalten übernommen („aktualisiert“); sonst bleibt der
lokale („unverändert, lokal neuer oder gleich“; ebenso, wenn der lokale
Zeitstempel nicht lesbar ist). Lokale Einträge werden nie gelöscht. `created_at`
und `last_updated` bleiben die des Pakets. Ein lokaler DK-Eintrag ohne Treffer
(Fehler, keine Ergebnisse), der neuer ist als der Eintrag im Paket, bleibt
ebenfalls; die Vorschau zählt diese Einträge eigens. Geschrieben wird mit
einfachen `SELECT`/`INSERT`/`UPDATE` (kein `INSERT OR REPLACE`, kein `ON
CONFLICT`, kein `ON DUPLICATE KEY`), in Blöcken zu 500 Zeilen mit je einer
Transaktion, über `UnifiedKnowledgeManager.merge_cache_rows`. Keine
Schemaänderung.

**Welche Datenbank.** Die Cache-Datenbank der gespeicherten Konfiguration
(`database_config`). Gibt es im Prozess schon den Wissensmanager (laufende
GUI), wird über ihn geschrieben, nicht über eine zweite Verbindung daneben.
Zeigt er auf eine andere Datenbank als die Einstellung (etwa nach einem Wechsel
ohne Neustart) oder läuft er auf dem SQLite-Rückfall, wird der Cache
übersprungen, und die Vorschau nennt den Grund. Gibt es ihn nicht, entsteht er
mit dieser Einstellung (bei SQLite legt er die Datei mit dem Schema an). Die
Vorschau liest nur: über den laufenden Wissensmanager oder eine eigene
Verbindung, die danach geschlossen wird; eine fehlende SQLite-Datei wird nicht
angelegt, alle Einträge zählen dann als neu.

**ISIL-Schutz.** Das Paket trägt unter `cache.isil` die ISIL der exportierenden
Installation. Eingespielt wird nur, wenn sie zur ISIL der Zielinstallation passt,
verglichen ohne Groß-/Kleinschreibung und Leerraum. Die ISIL der
Zielinstallation ist die nach dem Anwenden des Presets: setzt das Preset
`institution.isil`, gilt dieser Wert (auch bei einer frischen Installation),
sonst die vorhandene `institution_isil`. Passt sie nicht, fehlt sie im Paket oder
im Ziel, wird der Cache übersprungen; die Vorschau nennt den Grund als Zeile und
als Hinweis. Ein exportiertes Paket trägt seine ISIL auch unter `institution`
und setzt sie beim Anwenden (die Vorschau zeigt die Änderung der ISIL); der
Cache passt dann. Die Prüfung greift, wenn ein Paket keine oder eine andere
`institution.isil` trägt (etwa ein Paket nur mit Cache).

Grenzen des Schutzes: Er verhindert versehentliches Einspielen, er ist **keine
Sicherheitsgrenze**. Die Zeilen des Caches tragen keine ISIL; wer die Werte im
Paket ändert oder Cache-Dateien ohne Paket in eine Datenbank kopiert, umgeht ihn.

**Fehler blockieren nur den Cache.** Eine falsche Prüfsumme, eine fehlende oder
abgelehnte Datei oder eine nicht nutzbare Cache-Datenbank überspringen nur den
betroffenen Cache-Teil, mit Grund in der Vorschau; Einstellungen, Instanzen und
Dateien des Presets werden trotzdem angewendet. Bei `files` halten Fehler
dagegen das ganze Preset an. Grund für den Unterschied: eine Datei unter
`files` bestimmt, was die Plugins lesen; ein Cache-Teil ist ein vorgefüllter
Zwischenspeicher, den ALIMA auch selbst aufbaut. Ein Preset nicht anzuwenden,
weil ein Cache-Teil fehlt, nähme der Einrichtung mehr, als der Cache ihr gibt.

**Rechte.** Der Cache enthält Daten aus dem Katalog und den Suchen der
Einrichtung. Das Paket nur intern verteilen, nicht in ein öffentliches
Repository.

**Nicht rückgängig zu machen.** Ein Cache ist neu aufbaubar; das Zusammenführen
wird deshalb nicht zurückgenommen. `python3 src/alima_cli.py preset revert` und „Rückgängig“ nehmen
Einstellungen und Dateien zurück wie bisher und lassen den Cache, wie er ist.
Der Verlauf (`applied_presets`) hält je Teil nur die Zahlen fest
(`cache: [{id, table, status, new, updated, unchanged, …}]`).

Nicht geprüft: MariaDB/MySQL als Cache-Datenbank (die Abfragen sind in einer
Form geschrieben, die beide Datenbanken kennen, liefen aber nicht gegen einen
Server; MariaDB vergleicht Texte in der Regel ohne Groß-/Kleinschreibung, und
`last_updated` steht dort in der Zeit des Servers, SQLite schreibt UTC), das
Einspielen in eine laufende GUI mit gleichzeitigen Suchen, Windows.

## Vorschau, Anwenden, Rückgängig

Die Vorschau (`build_plan`) schreibt nichts. Sie listet je Änderung den Wert
vorher und nachher, je Provider den Status (siehe „`providers`“), die
Standardmodelle (alt → neu, überschrieben oder abgewählt), neue, unveränderte
und ersetzte Dateien, je Cache-Teil
„neu: a, aktualisiert: b, unverändert (lokal neuer oder gleich): c“ oder
„übersprungen: <Grund>“, noch einzutragende Secrets, Hinweise und Fehler. Ein Plan mit Fehlern wird nicht angewendet, auch
nicht teilweise; Hinweise halten das Anwenden nicht auf.

Anwenden (`apply_plan`):

- setzt nur, was das Preset nennt; alle anderen Werte bleiben, Secrets nie;
- prüft vorher, dass die Konfiguration noch die Werte hat, auf denen die
  Vorschau beruht (sonst: Vorschau neu erstellen);
- installiert zuerst die Dateien, führt dann den Cache zusammen (mit erneuter
  Prüfsumme und der ISIL der Vorschau) und speichert über
  `ConfigManager.save_config`; scheitert das Installieren oder Speichern,
  werden die installierten Dateien zurückgenommen, ein schon eingespielter
  Cache bleibt. Mit Provider- oder Standardmodell-Änderungen speichert es mit
  `preserve_unified=False` (wie `save_unified_config`), weil das bewahrende
  Speichern eine leere Providerliste und einen leeren allgemeinen Standard nie
  schreibt (Rückgängig braucht beides);
- trägt in `AlimaConfig.applied_presets` ein: ID, Version, Name, Quelle,
  Zeitpunkt, je Änderung der vorherige und der neue Wert, installierte Dateien
  (mit `.bak`), je Cache-Teil die Zahlen. Ein leerer Verlauf steht nicht in der
  `config.json`. Für Provider sind das die Zugangsfelder vorher/nachher und,
  wenn das Paket einen Schlüssel trägt, der vorherige und der neue
  **Schlüssel** im Klartext: der Verlauf steht in derselben `config.json` wie
  die Schlüssel selbst, und ohne den vorherigen Schlüssel könnte Rückgängig ihn
  nicht wiederherstellen. Abgewählte Standardmodelle stehen nicht im Verlauf.

Ein zweites Anwenden desselben Presets ist ein leerer Plan (beim Cache: 0 neu,
0 aktualisiert) und speichert nichts.

Rückgängig (`revert_last`) nimmt den letzten Eintrag des Verlaufs zurück: jeder
Wert, den das Preset gesetzt hat, bekommt seinen vorherigen Wert, solange er
noch den Wert des Presets hat. Seitdem geänderte Werte bleiben und werden
genannt. Eine vom Preset angelegte Instanz wird entfernt, wenn sie seitdem
nicht geändert wurde. Ein ersetzter Provider bekommt seine vorherigen
Zugangsfelder und seinen vorherigen Schlüssel zurück (jedes nur, solange es den
Wert des Presets hat), die Standardmodelle ihre vorherigen Werte. Ein vom
Preset angelegter Provider wird entfernt, außer er wurde seitdem geändert oder
hat einen Schlüssel, den nicht das Preset gesetzt hat. Wird dabei ein Provider
`gemini`/`anthropic` entfernt oder sein Schlüssel geleert, wird auch der
Spiegel `gemini_api_key`/`anthropic_api_key` geleert, wenn er denselben
Schlüssel trägt; sonst legte das nächste Laden den Provider aus dem Spiegel
wieder an. Dateien: neue werden gelöscht, ersetzte
kommen aus der `.bak` zurück, seitdem geänderte bleiben. Der Cache bleibt.

```bash
python3 src/alima_cli.py preset show                                   # gefundenes und zuletzt angewendetes Preset
python3 src/alima_cli.py preset apply ub-musterstadt/ --dry-run        # nur Vorschau
python3 src/alima_cli.py preset apply ub-musterstadt/                  # Vorschau, Rückfrage, anwenden
python3 src/alima_cli.py preset apply ub-musterstadt/ --no-default-models  # Standardmodelle bleiben
python3 src/alima_cli.py preset apply --yes                            # gefundenes Preset ohne Rückfrage
python3 src/alima_cli.py preset revert                                 # letztes Preset zurücknehmen
```

`python3 src/alima_cli.py preset` läuft ohne vorherige Einrichtung (wie `setup`). Ohne `--yes`
fragen `apply` und `revert` nach. Mit Fehlern endet `apply` mit Exit-Code 1.

**GUI.** Die Gruppe „Einrichtung (Preset)“ zeigt das gefundene Preset (Name,
Version, Quelle, Pfad) und das zuletzt angewendete. „Preset anwenden…“ bietet
das gefundene Preset, eine Datei oder einen Ordner; die Vorschau zeigt den Plan,
„Anwenden“ gibt es nur ohne Fehler. Hat das Preset Standardmodelle, steht unter
dem Plan der Haken „Standardmodelle aus dem Preset übernehmen“ (an); ohne Haken
zeigt der Plan sie als abgewählt. Vorschau, Kopieren, Rückgängig und Export
laufen im Hintergrund, Fehler erscheinen als Meldung und in der Statuszeile der
Gruppe. Angewendet wird sofort, unabhängig von OK oder Abbrechen des
Einstellungsdialogs; der Dialog übernimmt die gesetzten Werte (Plugin-Listen,
ISIL, ELN, Standard-Workflow, Provider, Standardmodelle) in seine Felder, nicht
gespeicherte Eingaben in den Plugin-Listen bleiben erhalten. „Abbrechen“ nimmt
danach nur die im Dialog selbst gemachten Provider-Änderungen zurück, nicht die
des Presets (`UnifiedProviderTab.adopt_preset_steps`).

**Erster Start.** Der Einrichtungsassistent (GUI und
`python3 src/alima_cli.py setup`) belegt seine Felder aus dem gefundenen Preset
vor, ohne Schlüssel und Token. Seine LLM-Seite nimmt einen Provider aus
`providers`: den, den `default_models.general` nennt, sonst den ersten
eingeschalteten; daraus Typ, Name und Adresse (bei Ollama ohne `base_url` aus
`host`, `port`, `use_ssl`), auf der Modellseite die Modelle von
`default_models.pipeline`/`vision`, wenn sie diesen Provider nennen. Einen
Schlüssel aus `providers` übernimmt er nicht. Nach „Fertig stellen“ speichert er
seine Datenbankeinstellung und wendet danach `institution`, `workflow`,
`plugins`, `files` und `cache` an (derselbe Plan, ohne weiteren Dialog; der
Cache geht damit in die eben eingestellte Datenbank). `providers` und
`default_models` wendet er **nicht** an (`apply_after_setup` baut den Plan mit
`llm=False`): Provider und Modelle hat er gerade selbst abgefragt. Die übrigen
Provider des Presets legt „Preset anwenden…“ bzw.
`python3 src/alima_cli.py preset apply` an; die Zusammenfassungsseite nennt
sie. Die Zusammenfassungsseite listet vorher, was das Preset setzt; das Ergebnis
steht im Log, in der GUI-Statuszeile nach einem Lauf aus dem Menü, im Terminal
bei `python3 src/alima_cli.py setup`. Ein fehlerhaftes Preset bricht den
Assistenten nicht ab; es wird dann nicht angewendet. Was das Preset unter
`plugins` nennt, gilt nach dem Assistenten auch dann, wenn auf der Katalogseite
etwas anderes eingetragen wurde.

## Export

`python3 src/alima_cli.py preset export <ordner> [--plugin <instanz-id> …] [--id …] [--version …] [--label …] [--with-database-password] [--no-verify-database] [--with-cache[=TEILE]] [--with-provider-key NAME …]`
oder „Preset exportieren…“ (Gruppe im Plugin-Reiter und „Als Preset
exportieren“ im Einstellungsdialog; eine Checkliste wählt die Instanzen, das
Paket entsteht als Unterordner `<Preset-ID>` im gewählten Ordner). Der Ordner
muss neu oder leer sein. Exportiert wird der gespeicherte Stand.

**Provider.** Jeder eingeschaltete Provider kommt in Listenreihenfolge mit
seinen Zugangsfeldern ins Paket (`providers`), **ohne** Schlüssel; ein Eintrag,
den das Anwenden ablehnen würde (etwa eine `base_url` mit Passwort), bleibt mit
Hinweis draußen. Den Schlüssel eines Providers nimmt der Export nur auf
ausdrückliche Nennung mit: `--with-provider-key NAME` (wiederholbar; ein
unbekannter oder ausgeschalteter Name bricht den Export ab, ein Provider ohne
Schlüssel ergibt einen Hinweis) bzw. im Export-Dialog die Gruppe „LLM-Provider“
mit einem Kästchen „Schlüssel von „NAME“ im Klartext in das Paket schreiben“ je
Provider, der einen Schlüssel hat (voreingestellt aus).
`export_preset(..., provider_keys=[NAME, …])`. Der Schlüssel steht dann
**unverschlüsselt** in `alima_presets.json`; die Datei wird mit
Besitzerrechten (`0600`) angelegt (wo das System keine solchen Rechte kennt,
etwa unter Windows, wird sie trotzdem geschrieben; nicht geprüft), `_comment`
beginnt mit „ACHTUNG: enthält API-Schlüssel im Klartext (Provider …)“, und die
Zusammenfassung nennt „Schlüssel im Klartext enthalten für: …“. `default_models`
schreibt der Export aus den gespeicherten Standards (nicht den aufgelösten),
nur für Provider im Paket; ein Standard, der auf einen ausgeschalteten oder
fehlenden Provider zeigt, bleibt mit Hinweis draußen. Den Block `llm` schreibt
der Export nicht mehr.

**Cache.** Ohne Angabe enthält das Paket keinen Cache. `--with-cache` ohne Wert
nimmt `dk,mappings`, mit Wert die genannten Teile, durch Komma getrennt (`dk`,
`mappings`, `raw:<quelle>`; `raw:kvk` wird abgelehnt). Den Schalter hinter den
Ordner setzen oder `--with-cache=TEILE` schreiben, sonst liest argparse den
Ordner als Wert. Der Export-Dialog hat die Gruppe „Cache einschließen“: je Teil
mit Einträgen ein Kästchen mit Zahl und ungefährer Größe (Länge der
Textspalten, ungepackt), „DK-Cache (Libero/finc)“ und „GND-Zuordnungen“
voreingestellt an, Rohantworten je Quelle aus; gezählt wird im Hintergrund,
bevor der Dialog erscheint. Ohne gesetzte ISIL sind die Kästchen gesperrt, und
ein Export mit Cache bricht mit einer Meldung ab (das Paket wäre beim Anwenden
wirkungslos). Das Ergebnis nennt je Teil Zeilen und Größe (gepackt, entpackt),
nicht exportierte Einträge mit Grund und den Hinweis auf die Rechte. Gelesen
wird über den laufenden Wissensmanager oder eine eigene Verbindung der
Konfiguration, die danach geschlossen wird.

**Prüfung der Server-Datenbank.** Ist eine MariaDB/MySQL-Datenbank eingestellt,
prüft der Export zuerst, ob sie auf diesem Rechner funktioniert: lädt der
Qt-SQL-Treiber (ohne Verbindung), und gelingt eine Verbindung
(`ConfigManager.test_database_connection` mit dem eingestellten
`connection_timeout`; ein Rückfall auf SQLite zählt als gescheitert). Die
Prüfung baut dazu einmal eine Verbindung zum eingestellten Datenbankserver auf.
Scheitert eines, steht im Block `database` nur `{"type": "sqlite"}` (ohne Pfad,
Server und Passwort, auch mit `--with-database-password`), und der Export
meldet „Datenbank: MARIADB nicht nutzbar (<Grund>); exportiert wird SQLite“.
Der GUI-Assistent belegt dann SQLite vor, das Pfadfeld bleibt leer, und es gilt
der Standardpfad. Abschalten: `--no-verify-database` bzw. das Kästchen
„Datenbankverbindung vor dem Export prüfen“ (voreingestellt an, nur bei einer
Server-Datenbank sichtbar); dann bleibt es beim Server-Block wie ohne Prüfung.
Bei SQLite gibt es keinen Block `database`. Der Assistent der CLI
(`python3 src/alima_cli.py setup`) fragt keine Datenbank ab und richtet immer
SQLite mit dem Standardpfad ein.

Das Paket enthält:

- die gewählten Instanzen (ohne Auswahl alle) in Listenreihenfolge mit
  Kategorie, Typ, Name, Zustand und den Einstellungen, die sie speichern;
- die Datenbank von `geo_sachdb` und das Vokabular von `local_scheme` (auch
  das Standardvokabular, wenn keines eingestellt ist), kopiert nach
  `files/<instanz>/`, mit Prüfsumme, die Einstellung als `@file:<id>`;
- ISIL und ELN, den Standard-Workflow;
- die eingeschalteten Provider (`providers`, Schlüssel nur auf Nennung) und die
  Standardmodelle (`default_models`), siehe oben;
- für den Assistenten eine Server-Datenbank (MariaDB/MySQL); deren Passwort
  nur auf ausdrücklichen Wunsch (siehe unten);
- `secrets` für jedes verwendete geheime Feld;
- auf Wunsch Teile des Caches (siehe oben und „`cache`“).

Das Datenbankpasswort nimmt der Export nur mit `--with-database-password`
bzw. dem Kästchen „Datenbank-Passwort im Klartext in das Paket schreiben“ im
Export-Dialog (das Kästchen erscheint nur, wenn eine Server-Datenbank mit
Passwort eingerichtet ist) mit. Es steht dann **unverschlüsselt** in
`alima_presets.json`, die Datei bekommt nur Besitzerrechte (`0600`), der Export
meldet es als Hinweis, und das Paket gehört nur an berechtigte Personen. Der
Assistent beim ersten Start belegt das Passwortfeld daraus vor. `python3 src/alima_cli.py preset
apply` ändert die Datenbankverbindung einer bestehenden Installation nicht.

Nicht im Paket, jeweils mit Hinweis: geheime Felder und Felder namens
`api_key`/`token` der Plugins, ohne ausdrücklichen Wunsch das Datenbankpasswort
und die Provider-Schlüssel, ausgeschaltete Provider, der Pfad einer lokalen
SQLite-Datenbank, jede andere Einstellung mit einem lokalen Pfad (etwa
`k10plus.cache_dir`), Einstellungen, die der Typ nicht deklariert, die
Einstellungen einer Instanz, deren Typ nicht geladen ist (nur ihr Zustand).
Ein exportiertes Paket, auf einer anderen Installation angewendet, ergibt
dieselben Zustände, Einstellungen und Reihenfolgen und Dateien mit derselben
Prüfsumme (Test `tests/test_institution_presets.py::TestExport`), dieselben
eingeschalteten Provider (mit den genannten Schlüsseln) und Standardmodelle
(Test `tests/test_preset_providers.py::TestProviderExport`).

## Grenzen

- **Beratend.** Nutzer können nach dem Anwenden alles ändern; ein Preset sperrt
  nichts.
- **API-Schlüssel und Tokens nur als Ausnahme.** Plugin-Schlüssel und -Tokens
  trägt ein Preset nie; jede Person trägt sie im Plugin-Reiter oder als
  Umgebungsvariable ein. Einen Provider-Schlüssel trägt es nur, wenn er beim
  Export ausdrücklich genannt wurde, dann im Klartext, ohne Verschlüsselung:
  ein solches Paket ist so vertraulich wie die Schlüssel selbst. Das
  Datenbankpasswort im Block `database` liest der Assistent als Vorbelegung;
  der Export schreibt es nur auf ausdrücklichen Wunsch, im Klartext (siehe
  „Export“).
- **Provider: nur die Konfiguration.** Das Anwenden prüft die Felder eines
  Provider-Eintrags, nicht, ob der Dienst erreichbar ist oder der Schlüssel
  gilt; das zeigt erst der erste Aufruf oder „Verbindung testen“ im
  Anbieter-Dialog. Gegen echte LLM-Dienste ist der Weg über ein Preset nicht
  geprüft.
- **Server ohne GUI:** `preset apply` setzt `first_run_completed` nicht; die
  übrigen CLI-Befehle verlangen weiter einmal `python3 src/alima_cli.py setup`
  oder `skip_first_run_check` in der `config.json`.
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
- **Cache:** Der ISIL-Schutz ist keine Sicherheitsgrenze, das Zusammenführen
  ist nicht rückgängig zu machen, MariaDB/MySQL als Cache-Datenbank ist nicht
  gegen einen Server geprüft (siehe „`cache`“).

## Beispielablauf (UB)

1. Auf einem eingerichteten Arbeitsplatz (Plugins, ISIL/ELN, Standard-Workflow,
   Geo-Datenbank, Aufstellungssystematik) speichern, dann
   `python3 src/alima_cli.py preset export /srv/alima/ub-paket --id ub-musterstadt --version 2026.1 --label "UB Musterstadt"`.
   Mit `--with-cache` dahinter kommen DK-Cache und GND-Zuordnungen mit (die
   ISIL muss eingetragen sein). Die eingeschalteten Provider kommen ohne
   Schlüssel mit; `--with-provider-key NAME` nimmt den Schlüssel eines
   Providers mit (etwa für einen Server ohne GUI).
2. Den Ordner prüfen (Endpunkte, Dateien; ohne `--with-provider-key` keine
   Schlüssel enthalten) und intern verteilen, etwa auf ein Netzlaufwerk. Nicht
   in ein öffentliches Repository. Ein Paket mit Schlüssel nur an die Personen
   oder Rechner, für die der Schlüssel bestimmt ist.
3. Auf jedem Arbeitsplatz entweder
   `python3 src/alima_cli.py preset apply /srv/alima/ub-paket` (mit Vorschau und Rückfrage) oder
   `ALIMA_PRESET=/srv/alima/ub-paket` setzen: dann findet der
   Einrichtungsassistent das Paket beim ersten Start und wendet es an, und die
   GUI-Gruppe bietet es unter „Gefundenes Preset“ an.
4. Den Libero-Token (und andere Secrets) trägt jede Person selbst ein, ebenso
   Provider-Schlüssel, die nicht im Paket stehen.
5. Neue Version des Pakets: erneut anwenden; nur Abweichungen werden geändert.

Tests: [`tests/test_institution_presets.py`](../tests/test_institution_presets.py),
[`tests/test_preset_cache.py`](../tests/test_preset_cache.py),
[`tests/test_preset_providers.py`](../tests/test_preset_providers.py).
