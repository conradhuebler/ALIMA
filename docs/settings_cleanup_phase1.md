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

**Schritt 1 ✅ erledigt (Sep 21):** `explain_default_provider_model` nennt die
gewinnende Einstellung, `resolve_default_provider_model` delegiert daran (eine
Implementierung, damit die Erklärung nicht abdriftet). Die Gruppe „Aufgelöster
Standard" im Provider-Tab zeigt sie je Bereich. Dabei gefunden: die
Task-Preferences-Auswahl zur Laufzeit (`_resolve_provider_smart`, PRIORITY 2)
war durch ein `TaskType.TEXT` tot, das es nicht gibt — der breite `except`
meldete den `AttributeError` als `warning`.

**Der eigentliche DAU-Blocker:** fünf Ebenen entschieden über das Modell, das
läuft — General/Pipeline/Agentic/Vision Default, die Spalte „Preferred Model"
je Provider, die Task Preferences, dazu `pipeline_tab.global_override_selector`
und die Standard-Combos im `PipelineConfigDialog`. Aufgelöst wird das in
`UnifiedProviderConfig.resolve_default_provider_model` (`config_models.py`).

✅ **Zwei der fünf Ebenen sind weg (Oct 2026):** Task Preferences sind als
aktive Konfigurationsfläche zurückgezogen (kein Schreiber, kein Leser mehr
außer der einmaligen Vision-Migration `adopt_vision_task_preference`) —
sechs Wizard-Fragen pro Pipeline-Schritt wurden zu zweien (Pipeline, Vision),
`AlimaManager.execute_task` löst jetzt über `resolve_default_provider_model`
auf statt über eine Task-Preferences-Fallback-Kette.
`PipelineConfigDialog` (Smart/Advanced/Expert-Pro-Schritt-Override) ist
komplett entfernt — die klassische Pipeline lief ohnehin nur je auf dem einen
Pipeline-Standard, ein Pro-Schritt-Override hatte nichts zu unterscheiden.
`pipeline_tab.global_override_selector` bleibt als einzige verbliebene
Sonderebene (expliziter Toolbar-Override, nicht Teil der
Resolution-Hierarchie) — eigene Durchsicht wert, nicht Teil dieser Runde.

Dazu zwei unsichtbare Achsen:

- ~~`disabled_providers`~~ ✅ **erledigt (Sep 21).** Die Liste überstimmte
  `UnifiedProvider.enabled` an drei Stellen, hatte kein Bedienelement und wurde
  von keinem Code-Pfad befüllt. `apply_disabled_providers` führt sie beim Laden
  einmalig in `enabled` über und leert sie; das Feld bleibt als
  Migrations-Eingabe. Ein Schalter statt zweier.
- ~~Auto-Lerner~~ ✅ **erledigt (Sep 21).** Zwei Stellen leiteten
  `preferred_provider` aus Häufigkeitszählungen ab, statt ihn setzen zu lassen.
  Beide entfernt. Befund bei der Prüfung: der CLI-Block hing an
  `args.save_preferences`, einer Flagge, die `cli/main.py` nie definiert —
  unerreichbar, mit einer Statuszeile, die dauerhaft „❌ No" druckte. Der
  GUI-Knopf lief nur auf Klick und war zu drei Vierteln Gerüst (vier Blöcke mit
  `pass  # Disabled until proper implementation`). `preferred_provider` schreibt
  jetzt nur noch der sichtbare Selektor, `provider_priority` nur noch der
  Einrichtungsassistent.
- ~~`provider_priority`~~ ✅ **gegenstandslos.** Das Feld und
  `task_preferences_widget.py`, das es las, wurden bereits am 21. Sep 2026
  entfernt (`f77cf6d`, vor dieser Dokumentationskorrektur) — beide Verweise
  hier waren seitdem stale.

**Schritte 3 und 4 ✅ erledigt (Sep 21).** Die abgeleiteten Standards sind weg,
und die Tabs sind nach Aufgabe gegliedert statt nach Subsystem: 🤖 Modelle,
🔗 Datenquellen, 📚 Erschließung, ⚙️ Programm. Jede Gruppe ist ein eigener
Builder, die Tabs sind Komposition. Eine gebaute, aber nicht eingehängte Gruppe
fällt nur im Lade-/Speicherpfad auf (dort werden Widgets über `self.<name>`
adressiert) — dafür gibt es jetzt einen Test.

**Kleinere offene Punkte:**

- ~~`TaskPreference.chunking_threshold` hat einen Leser...~~ ✅ **gegenstandslos
  (Oct 2026).** Der letzte Leser (`_pipeline_classic_steps.py`, ein
  Keywords-Schritt-Override) ist mit der Task-Preferences-Zurückziehung
  entfernt — nichts schrieb das Feld je, also war der Leser tot, nicht nur
  unvollständig bewahrt.
- `UnifiedProvider.id` (uuid4) hat keinen Leser, ist aber ein persistierter
  Identitätsschlüssel. Eigene Durchsicht wert.
- `RepetitionDetectionConfig.show_suggestions` hat auch nach der Reparatur
  keinen Leser.
- Die Wiederholungserkennung hängt an `AlimaManager.analyze_abstract` und deckt
  agentische Läufe nicht ab. Der Gruppentitel sagt das jetzt; ob sie es abdecken
  *sollte*, ist offen.
