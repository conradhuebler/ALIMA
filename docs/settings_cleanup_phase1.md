# Einstellungsdialog — Phase 1 (September 10, 2026)

Befund und Umbau des `ComprehensiveSettingsDialog`. Phase 1 hat entfernt, was
nichts tut, repariert, was etwas tun sollte, und die Fläche so hinterlassen,
dass jedes Bedienelement darauf eine Wirkung hat. Die Neugliederung ist
Phase 2 (§3).

## 1. Was der Befund ergab

Konfiguration verteilte sich auf **neun Oberflächen**, die alle dieselbe
`~/.config/alima/config.json` schreiben. Der Hauptdialog war 1624 Zeilen groß,
tiefste Verschachtelung sechs Ebenen, ohne einen einzigen Test.

Drei Klassen von Problemen, unabhängig voneinander:

| Klasse | Beispiele |
|---|---|
| **Wirkungslos** | Repetition-Gruppe (6 Regler), Scope-Auswahl (3 Checkboxen), `debug`/`data_dir`/`temp_dir`, `PromptConfig` (3 Felder), `system_prompt_override`, `catalog_type` |
| **Defekt** | fünf Felder bei jedem Speichern zurückgesetzt, Reset-Knopf mit `AttributeError`, Singleton-Leck über „Test Connection", Cancel ohne Wirkung für Provider, `save_config()` ohne Argument |
| **Struktur** | 32 % toter Code, fünf konkurrierende Modell-Defaults, unerreichbarer Wizard |

## 2. Dauerhafte Erkenntnisse

Diese gelten über den Anlass hinaus.

- **Dataclasses beim Speichern mutieren, nie neu bauen.** Ein Konstruktoraufruf
  zwingt dazu, jedes Feld ohne Bedienelement von Hand durchzureichen; wer eines
  vergisst, setzt es bei jedem Speichern still zurück. So gingen fünf Felder
  verloren, zwei davon sicherheitsrelevant (`url_fetch_allowlist`).
- **Ein Flag, dessen Konsumenten alle einen Fallback haben, entscheidet nichts.**
  `is_primary` war auf 16 von 16 Instanzen gesetzt, bei keinem Typ mit zwei
  Instanzen. Die Prüffrage ist nicht „wird es gelesen?", sondern „gibt es einen
  Fall, in dem es das Ergebnis ändert?".
- **Ein Testfile ist kein Beleg dafür, dass der getestete Code benutzt wird.**
  `test_model_selection_dialog.py` deckte eine tote Kopie ab, während die
  lebende Klasse ungetestet war — beide hießen fast gleich.
- **Fixtures müssen den Pfad erreichen, den sie prüfen sollen.** Minimale
  Konfigurationen ohne Provider-Block lösen in `_parse_config` eine Ausnahme
  aus, deren eigenes `except` eine Default-Konfiguration liefert. Ein Test darauf
  prüft den Fallback und wird aus dem falschen Grund grün.
- **Ein Mutationstest, der nicht rot wird, ist ein Befund.** Der Guard in
  `migrate_catalog_type` überlebte seine Mutation, weil die Migration nur je auf
  `True` setzt — er konnte nichts abkürzen und war redundant.

## 3. Offen (Phase 2)

**Der eigentliche DAU-Blocker:** fünf Ebenen entscheiden über das Modell, das
läuft — General/Pipeline/Agentic Default, die Spalte „Preferred Model" je
Provider, die Task Preferences, dazu `pipeline_tab.global_override_selector` und
die Standard-Combos im `PipelineConfigDialog`. Aufgelöst wird das in
`UnifiedProviderConfig.resolve_default_provider_model`
(`config_models.py`); **keine Oberfläche zeigt, welche Ebene gewinnt.**

Dazu zwei unsichtbare Achsen:

- `UnifiedProviderConfig.disabled_providers` (`config_models.py`, gelesen in
  `smart_provider_selector`) überstimmt `UnifiedProvider.enabled`, hat aber kein
  Bedienelement — nur per Handedit oder Legacy-Migration befüllbar.
- `provider_priority` wird von `pipeline_config_dialog` und `pipeline_cmd`
  gepflegt, obwohl `resolve_default_provider_model` es nie liest.

Erst wenn das auf **eine** sichtbare Achse zusammengeführt ist, lohnt die
Gliederung der Tabs nach Aufgaben („Modelle", „Datenquellen", „Erschließung",
„Programm") statt nach Technik.

**Kleinere offene Punkte:**

- `TaskPreference.chunking_threshold` hat einen Leser
  (`task_preferences_widget`), wird aber von keinem Parser wiederhergestellt —
  die Bewahrung über einen Neustart hinweg ist wirkungslos. Bewusst nicht in
  Phase 1 angefasst.
- `UnifiedProvider.id` (uuid4) hat keinen Leser, ist aber ein persistierter
  Identitätsschlüssel. Eigene Durchsicht wert.
- `RepetitionDetectionConfig.show_suggestions` hat auch nach der Reparatur
  keinen Leser.
- Die Wiederholungserkennung hängt an `AlimaManager.analyze_abstract` und deckt
  agentische Läufe nicht ab. Der Gruppentitel sagt das jetzt; ob sie es abdecken
  *sollte*, ist offen.
