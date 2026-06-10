# WP13 — Aufräumen: toter Code, Backup-Dateien, CLAUDE.md-Hygiene

**Status**: ⬜ geplant (2026-06-10). Formuliert aus „WP E" des Maßnahmenplans
der Basis-Bewertung (Plan `ich-h-tte-gerne-eine-snappy-backus.md`), erweitert
um Befunde aus WP B (Debugbarkeits-Audit). Alle Dead-Code-Behauptungen unten
sind **per grep verifiziert** (Stand: commit `dc05125`), nicht aus Doku
übernommen — vor dem Löschen trotzdem je Datei erneut prüfen (Imports können
sich geändert haben).

**Einordnung**: schließt an die WP1–WP12-Serie an
([`agentic_ui_workpackages.md`](agentic_ui_workpackages.md)). Reines
Hygiene-Paket: **kein** Verhaltenswechsel, keine Restrukturierung (die
utils/core/suggesters-Konsolidierung aus der Vision ist ein eigenes,
größeres Thema und gehört NICHT hierher).

**Voraussetzung**: WP12 committet (sonst kollidieren Löschungen mit dem
offenen Worktree-Stand; siehe `wp12_unified_render_layer.md` §9.1).

---

## 1. Motivation

Die Basis-Bewertung (Juni 2026) fand toten Code, getrackte Backup-Dateien und
CLAUDE.md-Dateien, die die eigenen Hygiene-Regeln verletzen. Kosten: jede
Exploration (Mensch wie KI) verschwendet Zeit auf Module, die nichts tun;
stale Doku erzeugt falsche Annahmen (z. B. listet `src/ui/CLAUDE.md` den
CrossrefTab als Komponente, obwohl er bereits entfernt ist).

---

## 2. Arbeitsliste

### 2.1 Tote Module entfernen (verifiziert)

| Modul | Befund | Aktion |
|---|---|---|
| `src/core/search_engine.py` | Einzige Referenzen: `main_window.py:44` (Import) + `:343` (Instanz) — die Instanz `self.search_engine` wird **nirgends benutzt**. Abgelöst durch `SearchCLI`. | Import + Instanziierung in `main_window.py` entfernen, Modul löschen. Erwähnungen in `src/core/CLAUDE.md` mitziehen. |
| `src/core/lobid_subjects.py` | Von keinem Modul importiert; Standalone-Skript mit eigenem `__main__`; 13 `print()`. | Löschen (git-History reicht). |
| `src/core/katalog_subject.py` | Von keinem Modul importiert; 10 `print()`; `__main__` auskommentiert. | Löschen. |
| `src/ui/tablewidget_new.py` | Von nichts importiert (aktiv ist nur `tablewidget.py`, Import in `main_window.py:62`). | Löschen. |
| `src/ui/tablewidget_original.py` | Von nichts importiert; trägt „DO NOT USE"-Marker. | Löschen. |

**Vorgehen pro Modul**: `grep -rn "<modulname>" src/ tests/ docs/ *.py` →
0 Code-Treffer → löschen → Suite + GUI-Start. Doku-Treffer (CLAUDE.md, docs/)
im selben Commit bereinigen.

### 2.2 Getrackte Backup-Dateien aus dem Repo (verifiziert via `git ls-files`)

- `CLAUDE.md.bak`
- `src/ui/comprehensive_settings_dialog.py.backup`
- `src/ui/tablewidget.py.backup`
- `src/utils/config_models.py.bak`
- `src/alima_cli.py.backup` ist **nicht** getrackt → nur lokal löschen.

Aktion: `git rm`, plus `.gitignore`-Einträge `*.bak`, `*.backup`, `*~`
(Repo-Wurzel enthält weitere ungetrackte Artefakte — JPGs, Export-JSONs,
`alima_pipeline.{tex,pdf,aux,log}` —, die durch `.gitignore` gar nicht erst
anbieten werden sollten).

### 2.3 `workflows/legacy/` (v3-YAMLs) — Entscheidung

`meta_agent_default.yaml`, `default_alima.yaml`, `extended.yaml`,
`minimal.yaml`: zur Laufzeit nicht discovered, v3-Architektur ist seit
April 2026 entfernt, Referenz-Doku existiert
(`docs/legacy/agentic_workflow_v3.md`). **Empfehlung: löschen** —
git-History + die Legacy-Doku reichen. (Operator-Entscheidung, da
„kept for reference" einmal bewusst gewählt war.)

### 2.4 CLAUDE.md-Hygiene (Regelverstöße der eigenen Regeln)

Regel: „section >20 lines → place elsewhere", „Remove completed/resolved
items after 2-3 updates". Ist-Zustand:

- `src/ui/CLAUDE.md`: listet CrossrefTab (Tab existiert nicht mehr; nur
  `src/core/crossref_worker.py` lebt weiter, genutzt vom DOI-Resolver —
  **nicht** löschen); seitenlange „Recently ADDED / MAJOR NEW FEATURES"-
  Listen → nach `AIChangelog.md` bzw. streichen.
- `src/core/CLAUDE.md`: „Recently ADDED Features"- und „PRODUCTION
  STATUS"-Blöcke (✓-Listen längst gelebter Features) → streichen;
  SearchEngine-Erwähnung entfernen (siehe 2.1).
- `src/utils/CLAUDE.md`: dito („Recently ADDED", CLI-Beispielblöcke,
  „PRODUCTION READY"-Abschnitt).

Ziel pro Datei: Preserved-Teil (Architektur, knapp) + Variable-Teil
(nur aktuell Offenes) + Instructions-Block. Faustregel < 100 Zeilen.

### 2.5 Einmalige Datenpflege: SWB-Cache

Vor WP A (commit `7850222`) cachte der SWB-Suggester Netzwerkausfälle dauerhaft
als „kein Treffer" (`swb_gnd_cache.json` in den Suggester-Datenverzeichnissen,
Default `…/alima_data/swbsuggester/`). Bestehende Caches können solche leeren
Falsch-Einträge enthalten → Datei(en) einmalig löschen (Cache baut sich neu
auf). Kein Code nötig.

### 2.6 Dokumentieren, nicht lösen: API-Keys im Klartext

`~/.config/alima/config.json` speichert Provider-API-Keys unverschlüsselt.
Verschlüsselung/Keyring ist ein **eigenes** Paket (Security, plattform-
abhängig) — hier nur als bekannter Zustand in `src/utils/CLAUDE.md`
festhalten, damit es nicht als Versehen gilt.

---

## 3. Nicht-Ziele

- Keine Code-Restrukturierung (utils/core/suggesters-Konsolidierung = Vision,
  eigenes Paket).
- Keine Tab-Zusammenlegung (Abstract+Verifikation mergen, Review-Tab
  verbessern — operator-bestätigt gewünscht, aber UI-Arbeit → gehört in die
  WP1/WP10-Linie, nicht in ein Lösch-Paket).
- Kein Anfassen von `workflows/*.yaml` außerhalb von `legacy/`.

---

## 4. Verifikation / Akzeptanzkriterien

1. `grep -rn "search_engine\|lobid_subjects\|katalog_subject\|tablewidget_new\|tablewidget_original" src/ tests/` → 0 Treffer.
2. `git ls-files | grep -E '\.bak$|\.backup$|~$'` → leer.
3. `python -m pytest tests/` grün; GUI startet (`python src/alima_gui.py`),
   Webapp startet, CLI `--help` läuft.
4. Jede Sub-CLAUDE.md < ~100 Zeilen, keine „Recently ADDED"-Friedhöfe,
   keine Erwähnung gelöschter Module.
5. Ein Commit pro Block (2.1 / 2.2+2.3 / 2.4), Messages mit `Remove`/`Clean`.

## 5. Aufwand

~1–2 PT, rein mechanisch; größter Einzelposten ist 2.4 (Umformulieren statt
Löschen).
