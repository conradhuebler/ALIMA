# Beispielpaket: Einrichtungs-Preset

Ein Einrichtungspaket ist ein Ordner mit `alima_presets.json` und den Dateien,
die das Preset mitbringt. Alle Werte hier sind erfunden (Demo-Bibliothek,
ISIL `ZZ-DEMO`, Adressen unter `*.example`).

```
demo_institution/
  alima_presets.json        # Preset (Format 2)
  files/
    geo_demo.sqlite         # winzige Datenbank im Schema geo_sachdb v1 (3 Schlagwörter, 2 Notationen)
    demo_systematik.yaml    # Aufstellungssystematik für local_scheme (5 Einträge)
```

## Ausprobieren

```bash
# Vorschau, ändert nichts
python3 src/alima_cli.py preset apply examples/presets/demo_institution --dry-run

# anwenden (fragt nach), zurücknehmen
python3 src/alima_cli.py preset apply examples/presets/demo_institution
python3 src/alima_cli.py preset revert
```

Das Anwenden setzt ISIL und ELN, den Standard-Workflow, Zustand und
Einstellungen von `catalog`, `swb`, `geo_sachdb` und `local_scheme`, legt die
SRU-Instanz `k10plus_demo` an und kopiert die beiden Dateien nach
`<Konfigurationsverzeichnis>/data/`. Der Libero-Token wird nur als „noch
einzutragen“ gemeldet.

## Eigene Dateien

- Die Datenbank im Schema v1 erzeugt der Importer des Hauses, für die Freiberger
  Access-Datenbank `scripts/access_to_sql.py import-geo`; das Schema steht in
  `src/utils/lookups/geo_sachdb/schema.py` (`create_schema`).
- Das Vokabular hat die Form von `vocabularies/freihand_fg.yaml`.
- Nach jedem Austausch einer Datei die Prüfsumme in `alima_presets.json`
  erneuern (`sha256sum files/…`); mit falscher Prüfsumme wird nichts
  installiert. Einfacher: die eingerichtete Installation mit
  `alima preset export <ordner>` als neues Paket schreiben.

Beschreibung aller Schlüssel: [`docs/institutional_presets.md`](../../../docs/institutional_presets.md).
