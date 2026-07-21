# ALIMA AI Changelog

> **Developer log.** Detailed dated entries per feature: file lists,
> phase plans, internal refactors. For user-facing release notes
> (topic-grouped), see [`CHANGELOG.md`](CHANGELOG.md).

## 2026

### Aufräumen A+C: tote Aufrufe, ungetestete Kernlogik (July 21, 2026)

Drei Commits (`a3bb317`, `11f77c8`, `06c4417`), Suite 1432 → 1525. Ausgelöst
durch die Frage „was muss code-seitig noch angefasst werden" — beantwortet durch
Messen statt durch Abschreiben des Registers, weil die sieben Defekte der
Vorrunde in **keinem** Registereintrag standen.

**A — Aufrufe nicht existierender Methoden (`a3bb317`).** Ein AST-Scan über die
zwei zentralen Manager fand zwei weitere Fälle der `update_gnd_entry`-Klasse:
- `_validate_catalog_subjects` rief `get_all_gnd_ids_for_keyword()` — der Zweig
  für Katalog-Schlagwörter **ohne** GND-ID konnte also nie laufen. Methode jetzt
  implementiert (Union der `found_gnd_ids` über alle Quellen, Treffer auch auf
  den normalisierten Term).
- `search_cmd` druckte eine Cache-Notiz via `gnd_keyword_exists()`. Entfernt
  statt repariert: die Prüfung lief *nach* der Suche, die das Mapping gerade
  selbst geschrieben hatte.

Wurzel war eine halbe „CacheManager compatibility"-Fassade: Aufrufer schrieben
gegen eine API aus dem Kopf, und weil die Hälfte existierte, wirkte der Rest
plausibel. Toter Rest gelöscht (−216 Z., 4 Methoden ohne Aufrufer, darunter
`store_classification_results` mit 166 Z.). Scan danach: **0 Fälle**.

**C — ungetestete Kernlogik (`11f77c8`, `06c4417`).** 84 Module ohne
Test-Erwähnung; drei davon reine Logik: `_pipeline_rvk_scoring` (1992 Z.),
`_pipeline_dk_steps` (1186), `batch_processor` (895). Bezeichnend: **der
Batch-Save-Crash der Vorrunde lag genau in dem Modul, das die Suite nie
berührt**, und der einzige Test, der RVK-Scoring nennt, mockt es weg.

84 Charakterisierungstests — Ist-Verhalten festgenagelt, damit es absichtlich
geändert werden kann. Auffälliges markiert, nicht stillschweigend korrigiert:
- DK-Regex-Fallback findet Codes mit >3 Vorkomma-Ziffern nicht (`\d{1,3}`).
- `final_list`-Extraktion emittiert gemischte Groß-/Kleinschreibung.
- `_is_parent_like` verlangt **mehr als ein** Zusatzzeichen (`WI 47001` gilt
  nicht als Kind von `WI 4700`).
- `_branch_key` ist case-sensitiv — eine kleingeschriebene Notation umgeht die
  Branch-Begrenzung.

Dazu eine verhaltenserhaltende Extraktion: 5 reine Helfer aus Methodenkörpern
auf Modulebene (`_is_parent_like` entscheidet, ob eine Notation ihre Eltern
verdrängt — vorher von keinem Test erreichbar), `_source_rank`/`_status_rank`
existierten byte-identisch **doppelt** und sind zusammengeführt. Verifiziert per
Opcode-Vergleich gegen HEAD + `LOAD_GLOBAL`-Scan; Suite vor/nach unverändert.

**Methodik-Notiz:** eine Mutationsprobe griff zunächst nicht (Einrückungsmuster
passte nach der Extraktion nicht), der Test blieb grün. Nachgezogen statt als
bestanden verbucht — dieselbe Klasse wie die Defekte selbst: eine Prüfung, die
aussieht, als hätte sie stattgefunden.

**Offen (B):** 336 breite `except`-Blöcke schlucken still. Politikfrage, keine
Refaktorierung.

### WP-D2: Klassifikationen tragen Daten — P0-Revision + lobid-Ernte (July 20–21, 2026)

Vier Commits (`2f34e8c`…`8670d6c`), Suite 1383 → 1405. Anlass war eine Messung,
nicht ein Plan: der kanonische `classifications`-Block war in **allen** echten
Läufen leer (0 von 5128 / 2364 / 1134 Pool-Einträgen), weil `lobid` und `swb`
beide hart `{}` schrieben. Das Vokabular, das WP-D1 P0 über fünf Commits
vereinheitlicht hatte, transportierte im Normalbetrieb nichts.

- **P0-Revision (`2f34e8c`)**: P0 pinnte `{system: [codes]}` und behandelte eine
  Klassifikation damit als **Fakt**. Die Ernte liefert aber **gewichtete
  Evidenz** — und dasselbe Feld hätte künftig beides ununterscheidbar getragen
  (gnd_locals DDC aus der Normdatei neben Ko-Vorkommen aus dem Katalog). Ein
  Eintrag ist jetzt `{code, count?, origin}`; `origin ∈ {authority,
  cooccurrence}`. Einträge sind **sortiert** (Autorität, dann Stärke), damit
  „das erste" das beste ist. `count` wird per **max** gemergt, nie summiert —
  dieselbe Landmine wie der Pool-`count`. Beide Regeln per Mutation belegt.
  Nebenwirkungen bewusst: die gepinnte Set/Listen-Dualität entfällt für dieses
  Feld (Dicts sind nicht hashbar), `SET_FIELDS` verliert die System-Keys.
- **Ernte (`3f9772d`)**: lobid liefert die Klassifikationen die ganze Zeit mit,
  auf den `member`-Records (4627 Notationen über 248 gecachte Antworten: 2788
  RVK, 761 DDC, 48 BK). Zuordnung als **Ko-Vorkommens-Heuristik**, im Code so
  benannt: eine RVK-Notation klassifiziert den Titel, nicht jedes seiner
  Schlagwörter. Top-3 je System statt Schwelle (eine Schwelle ≥2 hätte die
  Abdeckung von 1319 auf 346 Keywords gekappt). BK ergänzt;
  bibliotheksspezifische Systematiken werden verworfen statt mis-abgelegt, auch
  „DDC-Sachgruppen der ZDB" (grobe Gruppen neben vollen DDC-Nummern wären zwei
  Granularitäten als gleichwertig).
- **`page_size` (`8670d6c`)**: Default 30 statt lobids 15, als Plugin-Feld. Die
  Seitengröße ist die Obergrenze der Abdeckung; ab ~50 reißt der schlechteste
  Fall die 1-MB-Grenze des Raw-Caches, und ungecachte Antworten verlieren die
  Ernte wieder. Nicht Teil des Cache-Schlüssels — vorhandene Zeilen laufen über
  die 24-h-Frische aus, die Umstellung wirkt graduell.
- **Konsumenten (Phase 4)**: `initial_gnd_classes` wird nach Evidenz sortiert
  statt in Fundreihenfolge; Dedup per max, nicht Summe.

⚠️ **Korrektur einer eigenen Angabe:** ich hatte **61 % Abdeckung** angekündigt,
real sind es **6 %**. Der Nenner war falsch — gemessen über distinkte GND-IDs in
`componentList`, während der Pool aus dem Aggregation-Facet kommt (15
ausgelieferte Records gegen ~100 Facet-Subjects je Antwort). Strukturelle Grenze
der lobid-API, kein Implementierungsfehler, aber eine ungeprüfte Zahl.

**Offen:** swb liefert weiterhin keine Notationen; `initial_gnd_classes` meint
klassisch (GND-Systematik aus dem LLM-`<class>`-Tag) und agentisch (geerntete
DDC) Verschiedenes — dieselbe Namenskollision wie F-1, unentschieden.

### WP-D1: P0-Verifikation, System-Keys, BibRecord, F-2 (July 20, 2026)

Sechs Commits (`2e2b647`…`87b6eb4`). Suite 1324 → 1357, jeder Zwischenstand
einzeln grün verifiziert. Ausgangspunkt: P0 war code-seitig fertig, aber
**unverifiziert** — die einzige offene Zusage war ein Vergleichslauf
klassisch↔agentisch.

- **P0-Gate (`bd2e87b`)**: deterministischer Konvergenztest in
  `test_e2e_smoke.py`. Beide Pfade ingestieren dieselbe Fixture durch echten
  Produktionscode (klassisch `execute_complete_pipeline`, agentisch
  `pool_entry_from_reduced` + `to_keyword_analysis_state`) — nichts nachgebaut,
  also nicht zirkulär. Verglichen werden die Werte der vier kanonischen Felder;
  Container-Unterschiede (Dict vs `List[SearchResult]`, Set vs Liste) sind
  vor-P0 bzw. gepinnt und werden normalisiert. Fixture bewusst nicht-trivial
  (zwei Systeme, `display_count != count`, zwei `gnd_ids`); ein eigener Test
  hält diese Eigenschaft fest. Zwei Mutationsproben bestanden.
  **Ergebnis: P0s Collapse ist auf Payload-Ebene verifiziert.**
  *Der vorhandene `compare_runs.sh` vergleicht 4× agentisch und beantwortete
  die Frage nie.*
- **Zwei Bugs, gefunden weil die Fixture realistisch wurde (`a509967`)** — die
  alte trug leere `classifications` und verdeckte beide:
  - **Batch-Speichern crashte.** `task_state_to_dict` lieferte rohes `asdict`,
    also erreichten die Sets der nested Sicht `json.dump` unkonvertiert →
    `TypeError: Object of type set is not JSON serializable` bei jedem State mit
    GND-Treffer samt Klassifikation. `BatchProcessor._save_result` fängt das pro
    Item ab → Batch-Läufe meldeten Fehlschläge statt Ergebnisse. **Vorbestehend**,
    keine P0-Regression (auch vor P0 waren `gndid`/`ddc`/`dk` Sets).
  - **RVK war beim Reload nicht gleichrangig** (s. Korrektur unten).
- **Ein System-Vokabular (`2e2b647`)**: der Daten-Layer schrieb seit P0 klein,
  `classification_systems.py` kannte nur GROSS und bediente 6 Display-Konsumenten
  — zwei Vokabulare für eine Sache. Aufgelöst zugunsten GROSS;
  `classification_systems` ist alleiniger Owner (`SYSTEM_KEYS`,
  `normalize_system`). 5 Produktions-, ~40 Teststellen. **Nicht angefasst**:
  `"dk"` als *Feldname für den Notationscode* in DK-Trefferzeilen und die
  Ausgabe-Form `dk_classifications` — andere Konzepte, WP-D2.
- **swb pre-v2-Fallback entfernt (`e04f7f2`)**: P0s harter Schnitt war
  unvollständig. Ersatzlos entfernen ging nicht — 63 von 194 Raw-Cache-Zeilen in
  der produktiven DB trugen die pre-v2-Form, und der Cache altert nach Zeilenzahl,
  nicht nach Zeit. Mit v2-Keys gelesen hätten sie ein leeres `gnd_ids` ergeben
  (still schlagwortlos statt Cache-Miss). Daher Fallback weg **und** Zeilen
  einmalig verworfen (`_purge_pre_v2_swb_raw_rows`, idempotent). Blueprint-READMEs
  auf Vertrag v2.
- **BibRecord (`1d92714`)**: `src/core/bib_record.py` + `to_bibrecord()` für
  finc/catalog/sru/k10plus, Fixtures aus den echten Client-Tests. Drei Fallen:
  finc `authors` ist verschachtelt (Namen sind die *inneren* Keys); SRU
  `decimal_classifications` ist verlustbehaftet (`(?:DDC|DK)`-Regex) → stattdessen
  die präfigierten Strings parsen; k10plus `url` ist `fulltext`, nicht `catalog`
  (F-7). **Bewusst nicht verdrahtet** — die `ResultItem`-Nähte bleiben unberührt,
  bis P1 einen Konsumenten hat.
- **F-2 (`87b6eb4`)**: DOI-Record-Keys klein, 5 Emissions- + 8 Lesestellen;
  Crossrefs eigene `message.get("DOI"/"URL")` bleiben groß (kommentiert). Der
  Fallback-Gate in `_resolve_doi_with_fallback` hatte **keine** Abdeckung und ist
  die gefährlichste Stelle: falscher Key → Abstract immer leer gelesen → die Kette
  geht immer an einem guten Crossref-Ergebnis vorbei, ohne Absturz. Jetzt in
  beiden Richtungen getestet.

**Live-Verifikation (Operator-Läufe, July 20 abends).** Der Operator hat beide
Pipelines auf demselben Input (SupraFit-Paper) laufen lassen — der Vergleich, der
seit P0 offenstand:
- **1037 gemeinsame Pool-Titel, 0 Abweichungen** bei `gnd_ids`/`classifications`;
  Payload-Felder identisch. Mengendifferenz (2364 klassisch / 1134 agentisch)
  kommt aus der Suchbreite (50 vs. 20 initiale Keywords), nicht aus der Form.
  **P0 ist damit auch live bestätigt.**
- Dabei fiel auf, dass der Modus im Export nicht steht und aus Nebenwirkungen
  geraten werden musste → `pipeline_mode`/`workflow_name` (`d5855fc`).
- Und: **`classifications` ist in allen Läufen zu 100 % leer** — 0 von 5128 /
  2364 / 1134. Strukturell, nicht zufällig (lobid+swb schreiben hart `{}`).
  Als WP-D2-Eintrag mit Zahlen hinterlegt.
- Derselbe Anlass deckte den toten Raw-Cache-Purge auf (`2b073f7`).

**Offen in D1**: die Konsumenten P1–P4. P1 (Record→Analyse-Input) hat eine eigene
Parity-Landmine — das `input_type`-Vokabular driftet über GUI/CLI/Webapp/Batch,
und DOI läuft heute an `execute_input_extraction` vorbei.
**Notiert**: `examples/classic_result.json` trägt 1981× `gndid` (pre-P0), ist als
Referenz irreführend; Neuerzeugung braucht einen Live-Lauf.

### WP-D1 P0: F-1-Collapse — ein GND-Pool-Vokabular + Notation-Datenform (July 19, 2026)

Fünf Commits (`6991d57`…), Plan aus Explore+Plan-Runde + vier Operator-Entscheidungen
(Spec-Pins wie empfohlen; **Plugin-Vertrag v2 mitziehen**; **harter Schnitt** — die
alte Form war nie produktiv im Umlauf; **D2-Datenform vorgezogen**: statt
`ddc`/`dk`-Rename direkt EIN Feld `classifications: {system: codes}` mit dk/ddc/rvk
als gleichrangigen Systemen — WP-D2 behält nur die Logik-Generalisierung).

- **Vorher vier Vokabulare** für dieselben Konzepte: Suggester-nested
  `{gndid,ddc,dk}`, Pool `{gnd_ids,ddc_codes,dk_codes}`, `ResultItem`
  (`gnd_ids/ddc/dk`), persistierte KAS-Hybridform (`gndid`+`ddc_codes`) — plus
  Live-Roundtrip-Rename `_entry_from_kw_data` ↔ `nested_from_aggregate` (F-1).
- **Jetzt eins**: `{count, gnd_ids, classifications: {system: codes},
  display_count?}` überall — Suggester-Vertrag v2, nested, Pool, agentische
  `gnd_entries`, persistierte `SearchResult.results`. Container: Set (nested) /
  Liste (Pool), pro System; `gnd_id` = erstes Element bleibt Pool-Konvenienz.
- **Merge-Atom**: `merge_code_entry` + `classifications_field` (per-System-Union,
  Copy-on-Write gegen Pool-Aliasing); `_merge_codes` erhält bei sparsem Ziel den
  Quell-Containertyp (Set degradierte sonst zur ungeordneten Liste — von den
  Tests gefangen). Count-Landmine unangetastet (max-merge, count=1 bei
  Cache-Hits, `display_count` display-only).
- **Ingestion** heißt jetzt `pool_entry_from_reduced` — konvertiert nur noch
  Repräsentation, keine Namen; `nested_from_aggregate` ist reine
  terms_map-Inversion + Fresh-Sets.
- **Persistenz**: `SET_FIELDS` = `{ddc, dk, missing_concepts}` (greifen via
  Rekursion in `classifications`); `gnd_ids` bleibt Liste → der latente
  Set-Roundtrip-Nichtdeterminismus (`gnd_id` wechselte nach Save/Load) ist weg.
  Alte Saves laden ohne Codes (harter Schnitt, akzeptiert).
  ⚠️ **Korrigiert July 20**: `rvk` fehlte in `SET_FIELDS` — RVK blieb nach dem
  Roundtrip Liste, während DK/DDC Sets wurden. Jetzt aus `SYSTEM_KEYS` abgeleitet.
- **SWB-Sonderfall**: der Raw-Cache-Blob ist die *reduzierte* Form → der
  `transform` behält einen dokumentierten Storage-Format-Fallback für
  pre-v2-Zeilen (altern binnen `max_age` aus); Datei-Session-Cache per
  v2-Dateinamen hart geschnitten. Einzige verbleibenden `gndid`-Zeilen in `src/`.
  ⚠️ **Korrigiert July 20**: die Zeilen altern *nicht* aus — `_prune_raw_responses`
  kappt nach Zeilenzahl pro Quelle, nicht nach Alter; in der produktiven DB lagen
  63 von 194 swb-Zeilen noch in pre-v2-Form. Fallback jetzt entfernt, Zeilen
  einmalig verworfen.
- **MCP-sichtbar**: serialisierte Tool-Antworten tragen `gnd_ids` +
  `classifications` (geteilte `_serialize_result_row`, serialisiert auch nested
  Sets); Tool-*Namen*/-Parameter unverändert.
- **Out of scope** (→ WP-D2 bzw. später): Katalog-Title-Record-Keys
  (`dk_codes`/`rvk_codes`/`ddc_codes` in `biblio_client`/`tool_providers`
  :301-337/`title_list`), DOI-Casing (F-2), DDC-Harvest + Mixin-Generalisierung.
- **Verifikation**: Suite 1324 grün pro Phase; Grep-Gates (null Alt-Keys außer
  dokumentiertem SWB-Fallback + Record-Shape); Spec gepinnt in
  `docs/wp_records_as_first_class.md` (authors `List[str]`, `urls{}`-Map,
  count-Kontrakt). Offen: Vergleichslauf klassisch↔agentisch auf demselben
  Input (Harness), GUI-Klick-Tests on the fly.

### WP Chat-UX-Aufräumen: GUI + Webapp (July 19, 2026)

Neun Commits (`07537d1`…`6985c8b`), Plan aus drei Explorationsberichten + vier
Operator-Entscheidungen (Vollpaket; §9.4 beides-ein-Stil; echte i18n; inkrementelles
Markdown). Kernergebnisse:

- **1/9 Totcode**: `static/index.html`, `toggle_tool_call`+`tool://`-Kette,
  `render_tool_marker`-Shim raus.
- **2/9 i18n-Fundament**: `src/utils/i18n.py` + `locales/de,en.json` (flache Dot-Keys,
  nie raisen), `UIConfig.ui_language` (Hook im Config-Load deckt GUI/Webapp/CLI),
  `alimaT()`/`window.__alimaI18n` fürs geteilte JS; LLM-Antwortsprache bleibt eigene Achse.
- **3/9 Fehler-Rendering (wp12 §9.3)**: additives `kind="error"` auf Collapsibles →
  rotes `rc-error`-Chrome; beide Bus-Konsumenten reichen den bisher verworfenen
  `payload["error"]`-Text durch; Chat-Fehler beider Frontends via `render_error_block`.
- **4/9 Live-Markdown**: Assistant-Bubble rendert den Stream-Puffer throttled (120 ms)
  durch den vorhandenen JS-Formatter (Escape-sicher, unclosed Konstrukte = Literal),
  Cursor-Blink, 200-kB-Cap; Python-markdown-it-Finalize bleibt autoritativ
  (Link-Trust bewusst nur dort). Node-Contract-Test.
- **5/9 Webapp-Einflächigkeit**: Legacy-`#stream-text` komplett raus (~25 Call-Sites);
  klassische LLM-Tokens als geteilte Stream-Blöcke (klassisch-gated — derselbe
  `stream_callback` feuert auch agentisch!), Client-Notices via `appendLocalNotice`,
  DK-Progress als In-Place-Element in der Pipeline-Bar, `displayResults`-Freitext-Recap
  entfällt (Server-Karten + Summary tragen es).
- **6/9 Theming (§9.4 „ein Stil")**: `--alima-*`-Variablen im geteilten CSS (Dark-Default
  = GUI unverändert), Webapp-Light-Overrides — der Theme-Toggle erreicht endlich das Log;
  Renderer emittiert Klassen statt Inline-Hex; Summary-Styles unter `#results-summary`
  gescoped, verwaiste Duplikat-Blöcke gelöscht.
- **7/9 GUI-Politur**: Status-Strip statt Emoji-Spam im Verlauf (🧭/💾/🤖/✅),
  Assistant-Bubble = visueller Peer der User-Bubble (Header in die Bubble),
  Pipeline-Reset kündigt sich an (war stumm), Placeholder vereinheitlicht,
  Worker-Status-Leak → Logger.
- **8/9 i18n-Sweep**: Bus-Systemzeilen, gemischtsprachige JS-Alerts, restliche
  Chat-Messages → Katalog (de+en synchron, Parity-Test).
- **9/9 Guard + Doku**: Chat-Endpoint subscribt mit `tool_events=False`
  (Doppelrender strukturell unmöglich); Ownership-Notizen für die drei
  Bus-Konsumenten (volle `BusRenderBridge`-Extraktion bewusst Follow-up —
  `renderer.subscribe` hat zwei echte Mini-Log-Caller); wp12 §3-Tabelle
  vervollständigt, §9.3 ✅, §9.4 entschieden, §9.5 als stale geschlossen
  (agentisches Tool-Chrome erreicht die Webapp längst, Test als Beleg).

**Nachtrag `e4101fd`:** GUI-Links waren tot (WP12-Regression) — das geteilte
`_ensureLinksNewTab` stempelte `target="_blank"` auf alle Links; im
QWebEngineView fordert das ein nie erzeugtes Popup an, der Klick verpuffte.
Fix: Scaffold-Flag `window.__alimaSameWindowLinks` → No-op im GUI (Klicks
laufen wieder über `acceptNavigationRequest` → `QDesktopServices`), Webapp
behält Neu-Tab. +2 Tests.

Suite 1305 → **1323** (neue Tests: i18n 9, Fehler-Chrome 5, JS-Contract 1,
Klassik-Stream-Integration 1, Guard 1, Link-Flag 2; 4 Totcode-Tests entfernt). Offene
Operator-Sign-offs: Klick-/Browser-Checkliste (Live-Markdown, Fehlerkarte,
Status-Strip, Bubble-Optik, Reset-Hinweis, Light/Dark im `#log`,
Sprachumschaltung, WS-Reconnect) + die wp12-§9.2-Liste.

### WP Struktur-Aufräumen: Audit-Funde + Klassifikations-Extraktion (July 19, 2026)

Struktur-Audit nach den Konvergenz-WPs: **keine toten Module** (Import-Scan über alle
Verdachtskandidaten in `clients/`, `lookups/`, `core/` — jedes Modul hat Nutzer); die
Restschuld war Konzentration, Duplikat-Datenpfade und Doku-Drift. Fünf Commits:

- **`04d9eb0` Strukturreste** — `src/utils/suggesters/` aufgelöst (`base_suggester.py` →
  `src/core/search/`, alle 4 Importer sind die Plugin-Suggester dort); finc-Client-Shims
  (`clients/finc_*.py`) gelöscht, Tests auf die kanonischen `providers/finc/`-Pfade;
  leeres `src/core/tests/` weg; `.gitignore`-Bug repariert (Zeile
  `my_backup.json.claude/` war ein verschmolzener fehlender Newline — `.claude/` wurde
  nie ignoriert).
- **`b8f25ca` Doku-Sweep** — 20 überholte Mai/Juni-Planungsdocs → `docs/legacy/`
  (provider_strategy-Quartett, UI/Chat-Design-Cluster, Audits, Handoff), eingehende
  Links umgeschrieben; CLAUDE.md-Drift abgeräumt (Future-Tasks #6/#9/#10, stale
  Residual-Notizen); `chat_agent_roadmap.md` korrigiert: **Anthropic streamt mit Tools**
  (code-verifiziert, per-Chunk `should_stop`), nur Gemini bleibt deferred.
- **`9f726ca` Status-Korrektur Counter-Bug** — der C1-Fix (agentische GND-Häufigkeit
  persistierte als 0) war **schon am July 16 gelandet** (`038738e`, 4 Edits + Tests);
  Spec + CLAUDE.md behaupteten noch „kein Code geschrieben". Offen nur der
  Operator-Vergleichslauf (GUI).
- **`3e4d0cd` LobidSuggester-Download entschärft** — `prepare(False)` raus aus
  `__init__` (Provider-Bau ist jetzt I/O-frei, GND-Label-Tabelle lazy+memoisiert beim
  ersten `transform()`); `default_data_dir` von tempdir → persistent
  `~/.config/alima/suggesters/<name>/` (tempdir erzwang 25-MB-Re-Download nach jedem
  Temp-Clean; die noch ältere script-relative Ära hatte Kopien in
  `src/{,cli/,webapp/}data/` hinterlassen — lokal gelöscht). +3 Tests
  (`test_lobid_lazy_init.py`).
- **`32670d5` Klassifikations-Extraktion** — `pipeline_utils.py` 5166 → 2041 Zeilen:
  der zusammenhängende DK/RVK-Block (~3100 Zeilen) verbatim in `DkStepsMixin`
  (`_pipeline_dk_steps.py`) + `RvkScoringMixin` (`_pipeline_rvk_scoring.py`), F-5-Technik,
  null Call-Site-Änderungen. Verifiziert: Diff-byte-identisch gegen HEAD-Blöcke,
  Bytecode-`LOAD_GLOBAL`-Scan 0 unresolved, MRO-Smoke, Suite 1308. **Keine**
  Notation-Generalisierung — die Extraktion schafft nur die Naht dafür.

Suite durchgängig grün (1305 → 1308 mit den neuen Tests). Offene Operator-Sign-offs:
Counter-Bug-Vergleichslauf, Lobid-Erst-Download-Klick (neuer Cache-Pfad), P7-Restklicks.

### APPROVED: Kern-Konvergenz klassisch↔agentisch WP-K1–K4 (July 19, 2026)

Aus dem CLAUDE.md-Variable-Block abgeräumt (Status war TESTED seit dem
Operator-Vergleichslauf June 29, 2026 — beide Pfade laufen durch, reasonable
results, klassisch≠agentisch by design): mapping-first GND-Suche in MCP-Tools,
`verify_keywords`-Step, geteilte DK-Vorfilterung, Klassifikations-Gating,
source_count-Ranking. Implementierungsdetails: Eintrag June 10, 2026.

### WP Plugin-Konvergenz P7: Config-Mirror abgebaut — `CatalogConfig` + `SearchProviderConfig` gelöscht (July 17, 2026)

`AlimaConfig.plugins` war schon autoritativ, aber zwei Dataclasses spiegelten es bei jedem
Save zurück (`derive_search_mirrors`), damit Legacy-Leser weiterlaufen. Der Mirror ist weg;
gelesen wird über `factory.primary_settings(config, provider_id, *, enabled_only)`,
geschrieben über `set_primary_settings`. Fünf Commits:

- **`9ac148a` (Vorarbeit)** — die Suite las die echte `~/.config/alima/config.json`, war also
  grün oder rot je nach letztem GUI-Klick: 7 Fehler bei deaktiviertem `rvk_api` + aktivem
  `catalog`, 13 ohne Config. Kein Produktionsfehler — P6a (Quellen aus den *aktivierten*
  Providern) und P5 (deaktiviertes Lookup gated jeden Pfad) sind korrekt, die Tests waren
  unisoliert. Zehn Tests hermetisch gemacht (`_pin_sources`, `_lookup_registry`), verifiziert
  über drei Config-Zustände.
- **`8596d97` (A)** — `execute_dk_search`s fünf `catalog_*`-Parameter waren im 510-Zeilen-Rumpf
  **nirgends** referenziert (AST-geprüft; der eigene Kommentar sagte „vestigial … dropped with
  the mirror in WP P7"). Mit ihnen die Ketten, die sie füllten: 16 der 29 Lesungen, die drei
  toten `pipeline --catalog-*`-Flags, `_load_catalog_config`. Suite ohne Testanpassung grün.
- **`fb2bf75` (B)** — die 13 echten Leser auf Instanzen. `enabled_only` macht explizit, was der
  Mirror implizit ließ: Quellen-Gates lesen gegated (P5-Parität), Policy-Felder ignorieren den
  Enable-Status (sonst fiele die DK-Policy still auf Default zurück, wenn der Katalog aus ist
  und finc das DK-Backend liefert). `SearchProviderConfig`-Fallback in `_search_instances`
  gelöscht — in Produktion unerreichbar, existierte nur für Test-Stubs.
- **`354a39d` (C)** — die Wizards schreiben Instanzen statt Mirror. `ensure_search_instances`
  als Zwilling zu `ensure_lookup_instances`: der Synthese-Guard greift *pro Kategorie*, eine
  einzelne handgemachte Katalog-Instanz hätte die anderen fünf Built-ins gestrandet.
- **(D)** — Löschung + Load-Migration auf die rohen JSON-Sektionen.

**Drei Planannahmen fielen bei der Ausführung:**

1. Der größte Leser-Cluster war *tot*, nicht migrationsbedürftig (s. A). Dazu: `PipelineStepConfig`
   hat keinen `__getattr__`-Proxy → `getattr(step_config,'catalog_token','')` lieferte **immer**
   `''`; die „Step-Config schlägt globale Config"-Vorrangkette hat nie funktioniert.
2. **„`synthesize_search_instances` nimmt ein plain dict" war eine Falle.** `getattr(cc, attr, None)`
   lieferte für ungesetzte Felder die *Dataclass-Defaults* (`catalog_type='libero_soap'`,
   `strict_gnd…=True`, `finc_default_limit=20`). Ein naives `dict.get` hätte `None` an
   `cls(**settings)` gereicht → stille Fehlkonfiguration beim Upgrade. Absente Keys werden
   **weggelassen**, der `ConfigField`-Default greift. Guard:
   `MigrationTest.test_absent_legacy_keys_are_omitted_not_none`.
3. **Der Mirror hatte einen Live-Bug.** `catalog_web_record_url` ist Ziel *zweier* Mappings
   (catalog + finc); `derive_search_mirrors` iterierte in Dict-Ordnung → finc gewann. Katalog-URL
   gesetzt + finc-Feld leer ⇒ Mirror `''` ⇒ **keine OPAC-Links**. Am alten Code demonstriert;
   jetzt explizite Präzedenz katalog-vor-finc (`catalog_web_bases`).

**Zwei gewollte Verhaltensänderungen** (Operator-Entscheid): OPAC-Link-Präzedenz (s.o.) und
`finc harvest_enabled` liest enabled-gated — eine deaktivierte finc-Quelle stoppt jetzt auch
`finc_subject_harvest`.

**Upgrade:** ein `catalog_config`-Key wird beim Load noch als Migrations-Eingabe gelesen und
beim nächsten Save nicht mehr geschrieben. Für Configs *mit* `plugins` ein No-op (die Instanzen
führen die Werte seit der Synthese). Downgrade nach dem ersten Save verliert die
Katalog-Einstellungen.

**Verifikation:** Suite 1305. Differenzvergleich gegen die echte Operator-Config — alle sechs
migrierten Leser liefern identische Werte wie der Mirror. Wizard-A/B (alt vs. neu) im selben
Prozess: identisches Instanz-Set, null Settings-Diffs. Beide Upgrade-Pfade durchgefahren
(Pre-Plugin-Config → Werte migriert + Defaults statt `None` + Mirror-Keys weg; reale Config mit
stalem Key → lädt unverändert). **Offen: Operator-Klick-Tests** — DK-Suche (Pipeline +
UB-Katalog-Tab), OPAC-Links mit gesetzter Katalog- und leerer finc-URL, agentischer Lauf mit
deaktiviertem finc, First-Start-Wizard + `alima wizard`, Bundle export→install.

**Findings (nicht gefixt, notiert):** `execute_gnd_search` trägt dieselben drei
`catalog_*`-Parameter, kein Produktionsaufrufer übergibt sie; `get_effective_config`
(`pipeline_manager.py:448`) hat null Aufrufer; `ub_catalog_tab` übergibt `strict_gnd_validation=False`
hart, während die Pipeline die Einstellung liest; `AlimaConfig.repetition_config` wird von
`asdict` geschrieben, aber nie geparst.

### Chat-Agent: Prompt folgt der Plugin-Config; leere Quellenmenge ist kein Nulltreffer (July 15, 2026)

Operator-Report: mit **allen** Such-Plugins deaktiviert rief der Agent `search_finc` und
`search_catalog_titles` (beide nicht registriert → Fehler, 2 von 30 Iterationen verbrannt)
und antwortete dann *„Es wurden keine Treffer für Quantenchemie gefunden"* — eine
plausible, aber falsche Auskunft über einen Begriff, der selbstverständlich in der GND
steht. Zwei unabhängige Defekte, beide gefixt. Suite `1259 → 1273 passed`.

- **Prompt war config-blind.** `build_system_prompt` kannte die ToolRegistry nie und
  nannte 23 Toolnamen hart — u.a. „Katalogsuche: `search_finc` (besser als
  `search_catalog`/Libero)". Die `search_*`-Tools werden aber **pro aktivierter Instanz**
  generiert. Neu: `available_tools`-Parameter (`ToolRegistry.get_tool_names()`); der
  Tool-Wahl-Block wird über einen Platzhalter `<<CATALOG_TOOL_RULES>>` in **beide**
  Regelsätze eingesetzt und nennt nur registrierte Tools. `None` → statischer Text
  (Back-Compat für `DEFAULT_SYSTEM_PROMPT`). Verdrahtet in GUI-Chat
  (`_chat_panel_chat_agent`) **und** headless/CLI/HTTP (`headless_agent`).
  ⚠️ Der auslösende Satz stand im **Compact**-Regelsatz — `mistral-small` matcht den
  „small"-Marker in `_COMPACT_MODEL_MARKERS`. Ein Fix nur in `SHARED_RULES` hätte den
  gemeldeten Fall nicht berührt.
- **`search_gnd` ist keine Live-Quelle.** Es liest `search_local_gnd` (lokaler Bestand)
  und bleibt registriert, auch wenn jeder Provider aus ist. Ohne Live-Quelle sagt der
  Prompt jetzt explizit: keine Suchquelle konfiguriert, `search_gnd` liest nur lokal,
  0 Treffer dort heißt „lokal nicht vorhanden", **nicht** „existiert nicht" — und
  „keine Treffer gefunden" ist verboten.
- **`aggregate_gnd_results` bei leerer Quellenmenge**: liefert statt des stillen
  Leerergebnisses ein `error` („…This is NOT a zero-hit result"), unterschieden nach
  „keine Quelle aktiviert" vs. „angeforderte Quellen lösen nicht auf". `pool`/`sources`/
  `missing`/`terms_map` bleiben vorhanden (Strukturleser wie `gnd_batch_search`
  unberührt).
- **Tests +14**: neu `test_chat_prompts.py` (erste Tests für die Prompt-Assembly
  überhaupt) inkl. Rekonstruktion des gemeldeten Laufs + Slot-Mechanik; `test_aggregate`
  um die drei Leer-Quellen-Fälle erweitert. Dabei aufgefallen: `cfg.plugins = []` heißt
  für `resolve_gnd_instances` „kein Config-Abschnitt" (→ synthetisiert aktivierte
  Defaults), **nicht** „alle deaktiviert" — die Tests bilden jetzt den realen Zustand
  (vorhanden + `enabled=False`) ab. Dieselbe Zweideutigkeit ist der offene
  P5-Entscheidungspunkt.
- **Nicht behoben (ehrlich):** `MODE_SUCHE` nennt Toolnamen weiterhin *beschreibend*
  („nicht allein aus Katalog-Tools … beantworten"); das sind negative Hinweise, keine
  Aufruf-Direktiven, und sie haben den Fehler nicht ausgelöst.

### WP Plugin-Konvergenz P1: Built-in-Namen entkoppelt (July 15, 2026)

Erster Schritt des Konvergenz-WP ([`docs/wp_plugin_convergence.md`](docs/wp_plugin_convergence.md)):
ein kopiertes/externes Provider-Plugin bekommt jetzt `agent_view`, eigene Raw-Cache-Keys
und Default-Provenienz — ohne Core-Edit. Alle vier Kopplungen laufen über **vorhandene**
Deklarationskanäle (Leitregel: was `deploy_poc.py` beim Kopieren mitnimmt, muss im
Provider-Dir liegen). Suite `1245 → 1259 passed` (0 fail).

- **`_SOURCE_PARAM_KEYS` → Klassenattribut `raw_cache_param_keys`** (`provider_base.py`
  Default `("search_type",)`; swb `+max_pages`, finc `+facets` — die Map war zu 60%
  redundant) + Accessor `registry.raw_cache_param_keys(source)`; `raw_cache_params_for`
  löst darüber auf (lazy import, sonst Zyklus). Byte-identisch für alle sechs Quellen +
  unbekannte Labels. Geschlossen: `poc_swb` keyte `max_pages=3` und `=5` auf denselben
  Eintrag.
- **`_agent_view_deriver`**: `if source == "lobid"` → `getattr(underlying_suggester(p),
  "transform_agent_view", None)`, exakt gespiegelt von `_source_transform`.
- **Aggregate-Default**: `sources or ["lobid","swb","catalog"]` →
  `enabled_gnd_provider_ids(config=self._alima_config())` mit `is not None`-Check.
  ⚠️ Kein Provenienz-Bruch: der Default liefert bei Standard-Config **vier** Ids
  (`gnd_local` ist GND-fähig), aber `gnd_local` hat keinen Suggester → der bestehende
  Transform-Filter verwirft ihn → identische Menge/Ordnung. Als Test festgenagelt.
  `tool_schemas.py` entsprechend umformuliert (prompt-sichtbar; benennt jetzt
  `list_plugins` statt einer Id-Liste, und warnt vor `source_label ≠ provider_id`).
- **`hand_wired` gelöscht**: canonical → `_make_search_handler` für alle Typen; das eine
  `"finc"`-Literal wandert *hinein* (P2 = Löschung). `raise ValueError` bei unbekannter
  `result_shape` entfernt — war unerreichbar, wurde durch den Edit erreichbar und hätte
  die ganze Tool-Liste gesprengt.
- **Bugs mit erledigt**: `_attach_agent_view` baute den Cache-Key von Hand (passte nur
  für lobid zufällig); `find_keywords` kollabierte `None`/`[]` (suchte gegen ein
  explizites Disable) und erfand an zwei weiteren Stellen `"lobid"` ohne aktive Quelle.
- **Tests +14**: Kopie erbt Key-Shape; `agent_view` für Kopie + max_pages-gekeyte
  Quelle; Vergleichslauf Default==Literal (netzfrei, ersetzt den im WP geforderten
  manuellen Lauf); None/`[]` beidseitig; canonical Code-Plugin mit umbenanntem Tool
  bekommt den nuancierten Handler; unsinnige `result_shape` sprengt die Generierung
  nicht; erste Tests überhaupt für `find_keywords`-Logik.
  **`test_all_external_plugins_poc` war nach dem Edit still falsch-grün** (`assertIn
  ("error")` traf auch den AttributeError-Absturz, `assertTrue(built_ids)` schon durch
  `poc_finc` allein) → verschärft: beide Patch-Targets, `len(built_ids)`, Fehlertext
  geprüft.
- **Doc-Korrektur**: P1s Versprechen „POC-E2E deckt dann Aggregation ab" gilt **nur**
  direkt gerufen — `gnd_batch_search` übergibt `sources` aus der hartkodierten
  `source_tools`-Map (P6). In `plugin_authoring.md` + WP-Doc richtiggestellt.
- **Offen (Operator):** Click-Test `find_keywords` (Quellen-Checkboxen, „keine Quelle
  aktiv"-Pfad) — headless nicht verifizierbar.

### Plugin-System-Audit: Quick-Wins + Konvergenz-WP (July 14, 2026)

Drei-Agenten-Audit der Plugin-Umstellung (kritische Bilanz): Konstruktions-Schicht +
Trust-Modell echt vereinheitlicht; Orchestrierung dreifach, Built-in-Namen-Kopplung
konzentriert in WP2-Aggregation/finc/RVK/DK-Resolver. Mittlere Brocken als
priorisiertes WP dokumentiert: [`docs/wp_plugin_convergence.md`](docs/wp_plugin_convergence.md)
(P1 WP2-Entkopplung, P2 finc, P3 RVK-3-Pfade, P4 DK-Resolver, P5 Lookup-Vertrag inkl.
**vertagtem Disable-Entscheid**, P6/P7 Backlog). Quick-Wins umgesetzt:

- **Toter Code**: `fetch_dois_for_siegel` (`k10plus_resolver.py`, 0 Caller) gelöscht;
  falscher „stay untouched"-Docstring in `lookups/k10plus.py` korrigiert.
- **Doc-Drift**: `plugin_authoring.md` (MetaSuggester-Claim), `wp_tool_data_passthrough.md`
  (stale Suggester-Pfade + Status), `wp_search_tool_plugin_potential.md` (Zeilenref,
  F3-Status, Residual-Liste) berichtigt.
- **Lookup-Build-Parität**: `warn_operator_urls` nach `core/plugins/schema.py`
  extrahiert — Search-Factory **und** `LookupCategory.build` warnen jetzt bei
  Operator-URLs (webindex `base_url`); Lookup-Plugins im ToolRegistry per Instanz
  memoisiert (`_lookup_for`, Clear in `refresh()`) — webindex öffnete sonst pro
  Tool-Call eine neue SQLite-Connection.
- **GUI-Bypässe**: `SiegelCacheLoadWorker` über neues `K10PlusLookup.load_cached`
  (Instanz-`cache_dir`-Fallback) statt Direkt-Import; `import_lobid_dnb_data` baut
  über Factory/`underlying_suggester` — der alte Direktbau schrieb nach `./data/lobid`,
  ein Verzeichnis, das der Factory-Suggester (`$TMP/alima_data/lobidsuggester`,
  jetzt `BaseSuggester.default_data_dir()`) nie las; `find_keywords`-Quellen-Checkboxen
  dynamisch aus `enabled_gnd_provider_ids(available_only=True)` (neuer Parameter,
  filtert Availability-Gates) — externe Provider erscheinen automatisch, toter
  `catalog_token`-Loader entfernt.
- Tests: `test_lookup_plugins.py` +3 (URL-Warnung, Handler-Memoization, `load_cached`).
  **Offen (Operator):** Click-Test find_keywords-Checkboxen, Lobid-Import, Siegel-Cache-Load.

### Lookup-Plugins: Einbindung in Pipeline + Agent (WP Phase D, July 10, 2026)

Die Lookup-Plugins (rvk_api/k10plus/dnb — aus WP Phasen B/C, dort + im WP-Doc
`docs/wp_search_tool_plugin_potential.md` dokumentiert; die Phasen A–C + webindex
sind unten nachgezogen) waren nur vom Chat-Agent automatisch erreichbar. Phase D schließt die Einbindung:
**ein Aufrufpfad je Quelle** für Pipeline, CLI, GUI und Workflow-Agent. Suite
`1231 → 1241 passed` (0 fail).

- **Ein Konstruktionspunkt** — neu `src/utils/lookups/resolve.py` `build_lookup(config,id)`:
  baut dasselbe konfigurierte Plugin wie der Tool-Handler (`get_category("lookup").build`),
  Instanz-Auswahl gespiegelt von `ToolRegistry._lookup_instances`, `config=None` → Auto-
  Load. Re-exportiert über `lookups/__init__.py`.
- **Workflow-Agent** — neues Preset `lookup` (+ rvk in `classification`) in
  `src/mcp/default_presets.yaml` (+ Fallback in `llm_agent_step.py`); dieselben Tools,
  die der Chat-Agent automatisch hat. Presets filtern nur die bereits registrierten Tools.
- **Pipeline-RVK** — `_build_rvk_api_fallback_results` + `_validate_catalog_rvk_candidates`
  (`pipeline_utils.py`) übers `rvk_api`-Plugin statt `RvkApiClient()` direkt; WP2-Cache-
  Keys unverändert. Validierungs-Timeout war hart 4s → jetzt per-Instanz-konfigurierbar.
- **k10plus** — `K10PlusLookup.fetch_records()` (ungedeckelt, `List[K10PlusRecord]`) als
  einziger Harvest-Kern; `fetch_package` = JSON+Cap-Wrapper. CLI `batch --siegel`
  (`pipeline_cmd.py`) + GUI `SiegelFetchWorker` (`batch_processing_dialog.py`) routen
  darüber (GUI ungedeckelt, CLI zieht `.doi`); kehrt „direct batch usages stay" um. CLI
  ohne `--siegel-cache-dir` nutzt jetzt den Plugin-`cache_dir`.
- **DNB-GUI** — `DNBSyncWorker` + `find_keywords.update_entry`: `DnbLookup()` →
  `build_lookup(None,"dnb")` (Konstruktions-Parität).
- Tests: `test_lookup_plugins.py` +10; netzfreie Laufzeit-Verifikation aller Nahtstellen
  (RVK-Pipeline-Methode, k10plus-DOI, Preset→Registry, DNB-GUI).
- Residual: `fetch_dois_for_siegel` bleibt ungenutzter Compat-Wrapper; `RvkMarcIndex`
  weiter direkt; deaktivierte Instanz → synthetische Default-Instanz. **Offen:** Commit
  + GUI-Sign-off.

### webindex: Website-RAG-Chatbot als Lookup-Plugin (July 9, 2026)

ALIMA als Chatbot für Webseiteninhalte (commit `6094d1a`). Suite `→ 1231 passed`.

- **Lookup-Plugin `src/utils/lookups/webindex/`**: eigene `webindex.db` (`store.py`
  nach `LocalGndStore`-Muster), BeautifulSoup-Crawler (`indexer.py`) mit injiziertem
  Keyword-Extractor, `provider.py` Tools `search_webindex` / `fetch_page` /
  `list_webindex_keywords`. Retrieval: Frage → Keyword-Match gegen `page_keywords` →
  gerankte Trefferseiten (Cache oder Live-Fetch) → Text → Antwort. Reuse der geteilten
  `fetch_guarded_response`- (Phase B) + `pdf_extractor.extract_text`-Primitive.
- Indizieren: GUI-Button „Seite indizieren …" (`_TYPE_ACTIONS`-Registry +
  `WebIndexCrawlWorker` in `src/ui/webindex_crawl.py`) ODER CLI
  `alima webindex crawl/stats/list-keywords/search` (`cli/commands/webindex_cmd.py`,
  treibt store/indexer/provider direkt, nicht über die ToolRegistry).
- Prompts als Workflows: Keyword-Standprompt `workflows/webindex_keywords.yaml`
  (tool-less `llm_agent`, **nicht** prompts.json), Antwort-Prompt
  `workflows/website_rag.yaml` — der einzige Workflow, der die webindex-Tools listet.
- Tests netzfrei (`test_webindex_{store,indexer,lookup,keywords,crawl_ui}.py`, 51).
- Sub-CLAUDE: `src/utils/lookups/webindex/CLAUDE.md`. **Offen:** Operator-E2E gegen
  echte Biblio-URL.

### Lokale GND-DB vom Cache entkoppelt + Lookups geseedet + DNB-Plugin (WP Phase C, July 9, 2026)

Operator-Review nach Phase B: drei Residuen adressiert. Commit `e7eb824` (24 Dateien).
Suite `→ 1177/1178 passed`.

- **C1 — lokale GND-Kopie ist jetzt eine plugin-eigene DB.** Neuer `LocalGndStore`
  (`src/core/search/providers/gnd_local/store.py`) besitzt die `gnd_entries`-Tabelle in
  eigener Datei `gnd_local.db` (Pfad `DatabaseConfig.gnd_local_path`); der
  `UnifiedKnowledgeManager` behält seine GND-Fact-API, routet aber jede `gnd_entries`-
  Query über den Store (eigener `connection_name` → keine Per-Thread-Kollision).
  `search_mappings` bekam eine denormalisierte `titles`-Spalte;
  `CachingProvider._items_from_cache` baut Cache-Treffer daraus statt aus `get_gnd_fact`,
  und `warm_gnd_entries` wurde **entfernt** → F1 bleibt gefixt, ohne die Autoritäts-Kopie
  zu berühren. Einmalige non-destruktive ATTACH-Migration der Legacy-Same-File-Tabelle.
  **Verhaltensänderung:** `search_gnd`/`gnd_local` zeigen keine nur-online-gesuchten
  Terme mehr (bewusster Operator-Tradeoff).
- **C2 — Lookup-Instanzen geseedet** (`synthesize_lookup_instances` in
  `plugin_migration.py` + `ensure_lookup_instances`-Guard): die Plugins-Tab-Liste zeigt
  `rvk_api`/`k10plus`/`dnb` out-of-the-box (war leer, weil nur die Typ-Combobox gefüllt war).
- **C3 — restliche externe API-Fetcher migriert:** DNB → Lookup-Plugin
  (`src/utils/lookups/dnb.py` `DnbLookup`, Tool `dnb_classification`, raw-cached; GUI-DNB-
  Sync routet durch); RVK erbt den Raw-Cache über den neuen Helper
  `src/utils/lookups/cache.py` (`lookup_cache_enabled` + `cached_call`, shared
  `rvk_search`/`rvk_validate`-Keys); k10plus bekam ein `cache_dir`-Verzeichnis-Cache-
  Setting. Dead code entfernt (`crossref_worker.py`, `print_abstracts.py`, tote
  Resolver-Refs); stale crossref-Notizen in `core/CLAUDE.md` + `ui/CLAUDE.md` gefixt.

### Lookup-Kategorie + RVK/k10plus-Plugins + URL-Fetch-Kern (WP Phase B, July 8, 2026)

„Alte Zöpfe abschneiden": externe-API-Zugriffe werden Plugins. Additiv (kein
agent-facing Tool umbenannt). Suite `→ 1167 passed`.

- **Neue dritte Plugin-Kategorie `lookup`** (`src/utils/lookups/`, commit `a773a2c`) neben
  search_provider + input_source: `registry.py` (`@register_lookup` + `LookupToolSpec`),
  `category.py` (`LookupCategory`-Adapter, self-registers + injiziert das `cache_field`).
  Grenze formalisiert: externe-API-Interaktion = Plugin (cached, per-Plugin-toggle);
  lokale DB / Pipeline / Export + komponierte Tools (`rvk_lookup`, `resolve_doi`) = Core.
- **RVK-API-Plugin** (`rvk.py` `RvkLookup`, id `rvk_api`) wrappt `RvkApiClient` → Tools
  `rvk_search` (Schlagwort→gerankte Notationen) + `rvk_validate` (Notation→Label+Ahnen),
  generiert via `ToolRegistry._generated_lookup_tools`, raw-gecacht über den per-Plugin-
  Gate. Der komponierte `rvk_lookup`-Core-Tool bleibt unverändert. Live: Biologie→AN 94700.
- **Ein geführter URL-Fetch-Kern** (commit `29b4ce4`): `url_fetch.fetch_guarded_response()`
  als einziger SSRF-geschützter Fetch-Einstieg (net_guard + Guard-Settings-Auflösung);
  sowohl `url_fetch.scrape_url` (Main-Content) als auch der MCP-`scrape_url`-Tool
  (Full-Page + PDF-Detektion) rufen ihn; die zwei Content-Shapings bleiben.
- **k10plus als Lookup-Plugin** (commit `a205e79`): `K10PlusLookup` → Tool
  `k10plus_package` (Siegel→Records, one→many-Query, deshalb `lookup` statt
  `input_source`), raw-cached; die direkten Batch-Aufrufer blieben zunächst (erst
  Phase D über das Plugin geroutet). Tests: `test_lookup_plugins.py`.

### Plugin-Cache-Fundament: GND-Warming + per-Plugin cache_responses (WP Phase A, July 8, 2026)

Non-breaking Fundament für „cache jede Suche, per-Plugin schaltbar" (commit `4dfd5f2`).
Suite `→ 1159 passed`.

- **F1 gefixt (eine Suche vergiftete ihren eigenen Cache):**
  `UnifiedKnowledgeManager.warm_gnd_entries` am geteilten Write-Seam
  (`CachingProvider._live_search`) füllt die lokale GND-Wissens-DB aus jeder GND-Suche
  (`INSERT OR IGNORE`, überschreibt keine reichere Enrichment-Fact) → Cache-Treffer
  0→49 verifiziert. *(Phase C1 ersetzt dieses Warming später durch eine denormalisierte
  `titles`-Spalte und entkoppelt den lokalen GND-Store — siehe oben.)*
- **F2 gefixt:** `search_local_gnd` liefert Teil-Treffer statt `[]` bei < min_results.
- **Per-Plugin `cache_responses`** — Tri-State-`ConfigField` (auto/on/off,
  `schema.cache_field()` + `cache_pref_enabled()`) in beide Kategorie-Formulare injiziert,
  bei Ausführung gelesen (Search: `SuggesterBackedProvider._cache_raw_enabled`; Input:
  `_make_input_handler`); `auto` folgt dem globalen `enable_response_cache` (per Default
  aus). Tests: `test_gnd_cache_warming.py`, `test_cache_setting.py`.

### GND-Suche vereinheitlicht: MetaSuggester retired (July 8, 2026)

Operator-Auftrag: die fragmentierten Such-Anbindungen zusammenführen — Pipeline
*und* Agentik holen ihre Daten über *einen* Provider-Weg (die Factory), nicht mehr
über die alte MetaSuggester-Infrastruktur. Suite `1145 → 1152 passed` (0 fail).

- **Neuer Single-Entry `src/core/search/service.py`** — `search_gnd_keywords(terms,
  instances, *, cache, aggregate_from_raw, …)` + `resolve_gnd_instances(ids)` +
  `underlying_suggester()`. Baut Provider über `factory.build_provider` aus der
  autoritativen `PluginInstanceConfig`, merged quellenübergreifend, erhält den
  WP2-Raw-Seam (live/merge **und** raw-first, byte-kompatibel zu
  `SearchCLI.search_from_raw`). `resolve_gnd_instances` ist das *eine* Enable/
  Disable-Gate (respektiert deaktivierte Instanzen, synthetisiert nur unbekannte
  ids, Overlay für den Klassik-Catalog-Token).
- **Merge-Atom vereinheitlicht** — `gnd_search_core.merge_code_entry` bekommt
  `display_count_field` (max-Merge, F-4); MetaSuggesters Spezial-Merge gefaltet.
- **Klassik (`SearchCLI`)** delegiert an den Service (dünner Adapter, behält
  Catalog-Token/URL-Wiring + Context-Manager). Live verifiziert gegen lobid:
  live/merge `count=139`, raw-first `count=1`+`display_count=139`, Raw-Cache 71 940 B.
- **MCP (`ToolRegistry`)** — `_init_suggesters`-Primaries (lobid/swb = MetaSuggester,
  catalog = BiblioSuggester-aus-`CatalogConfig`) ersetzt durch factory-gebaute
  Provider (`_provider_for`, memoisiert). `_source_transform` (agentisches
  `aggregate_gnd_results`) liest die Transforms von denselben Providern. Live
  verifiziert: search_lobid (100 kw + gnd_urls + errors), title-Passthrough,
  aggregate_gnd_results (Pool 100, count-landmine + provenance). **Residual:** finc
  (`_handle_search_finc` + `_init_suggesters._finc`) bleibt `CatalogConfig`-basiert
  (institutionsspezifisch, `test_finc_client` pinnt es) — nutzte nie MetaSuggester.
- **GUI (`find_keywords`)** Standalone- + Manuell-Suche gehen jetzt über den Service
  → schließt den einzigen Pfad, der bisher den Raw-Cache umging. (Operator-Click-Test
  offen — GUI nicht headless verifizierbar.)
- **`src/utils/suggesters/meta_suggester.py` gelöscht.** `grep "MetaSuggester("` → 0.
  `BaseSuggester` bleibt (Per-Source-Contract).
- **Defaults/Blueprints** (bestätigt, kein Code nötig): lobid + swb zero-config
  Default; libero/catalog + finc sind enabled-but-`is_available()`-gated Blueprints;
  gnd_local offline. Service-Fallback (Config unlesbar) = `["lobid","swb"]`.
- Tests: neu `test_gnd_search_service.py` (7); angepasst `test_aggregate.py`,
  `test_provider_tool_generation.py` (Mocks am Factory-Seam statt an
  `_lobid`/`_swb`/`_biblio`).

### Plugin-System: Self-contained Blueprint-Dirs + Security-Härtung (July 6, 2026)

Operator-Auftrag: robustes, sicheres Plugin-System — Built-ins als kopierbare
Blaupausen, Sicherheitsevaluation + Härtung. Suite `965 → 1009 passed` (0 fail).
Specs: [`docs/plugin_system.md`](docs/plugin_system.md) (aktualisiert), neu:
[`docs/plugin_authoring.md`](docs/plugin_authoring.md).

- **Loader (`src/core/plugins/loader.py`):** Multi-File-Code-Plugins via
  synthetischem Package `alima_plugin_<id>` (`__path__`-Mount, nur entry-Modul
  wird ausgeführt, `sys.modules`-Cleanup bei Fehlimport); entry-Datei-Containment
  (kein Symlink, resolved im Plugin-Dir); Konsistenz-Check Klassen-`id` ==
  Manifest-`id`; id-Kollision → freundlicher Rename-Hinweis.
- **Security (`security.py`):** `iter_plugin_files` (folgt nie Symlinks, skip
  `__pycache__`/hidden/`*.pyc`); `hash_dir` über **alle** regulären Dateien
  (⚠️ invalidiert bestehende Approvals einmalig → Re-Approval-Prompt); Symlink =
  High-Finding; `requests.*` ohne `timeout` = Medium-Finding.
- **Manifest (`manifest.py`):** `entry.module` genau eine Top-Level-`NAME.py`
  (nicht `__init__.py`), `entry.class` muss Identifier sein.
- **net_guard (neu, `src/utils/net_guard.py`):** Zwei-Posture-URL-Validierung —
  Operator-URLs: Schema-Gate + Warnungen (Settings-Save-Dialog +
  Factory-Log, Intranet erlaubt); Laufzeit-/LLM-URLs: `fetch_guarded`
  (public-only per Redirect-Hop, Size-Cap, `SystemConfig.url_fetch_allowlist`/
  `url_fetch_max_bytes`). Verdrahtet: `url_fetch.scrape_url`, MCP `scrape_url`,
  finc-/marcxml-Client (`require_http_url`). Timeouts: swb `requests.get`
  (15 s), lobid `urlopen`/Dump-Download.
- **Secrets:** `ALIMA_PLUGIN_<INSTANCE_ID>_<KEY>`-Env-Override für
  `ConfigField(secret)` — nur zur Konstruktionszeit (`factory.build_provider`,
  `InputSourceCategory.build`, `MetaSuggester.__init__`, `_init_suggesters`-
  Catalog-Token), nie persistiert; GUI-Placeholder zeigt aktiven Override;
  `list_plugins` maskiert schemabasiert. Lücke dokumentiert: Legacy-Mirror-Leser.
- **Restructure:** jede Built-in-Anbindung ist ein self-contained Plugin-Dir
  `src/core/search/providers/{lobid,swb,catalog,finc,sru,gnd_local}/` mit
  `plugin.toml` (echtes Code-Manifest, testvalidiert) + `README.md` (Copy-
  Anleitung) + `provider.py` [+ `suggester.py` = ehem. `lobid_suggester`/
  `swb_suggester`/`biblio_suggester`/`finc_suggester` aus `src/utils/suggesters/`].
  Import-Regel: Framework absolut, intra-Plugin relativ. `_base.py` →
  `src/core/search/provider_base.py` (öffentliche API). Shared Clients bleiben
  in `src/utils/clients/` (Multi-Consumer). Nebenbefund gefixt: toter Import
  `_main_window_data.py:462`.
- **Tests (+44):** `test_plugins.py` erweitert (Multi-File, Symlinks, Hash-
  Abdeckung, entry-Validierung, Kollision, sys.modules-Cleanup, headless-deny);
  neu `test_net_guard.py`, `test_plugin_secrets.py`,
  `test_builtin_plugin_manifests.py` (alle 6 Manifeste konsistent),
  `test_plugin_blueprint_e2e.py` (copytree → rename → discover → search =
  der Operator-Workflow). `ConfigField.coerce` typsicher (str-Cast, CHOICE-
  Validierung).
- **Operator-Click-Test-Fixes (July 6, nachmittags; Suite → 1014 passed):**
  (1) GUI: `enable_code_plugins`-Checkbox + Scan-Button + Approval-Dialog im
  Plugins-Tab (existierten nicht; config-load bleibt headless=deny).
  (2) Loader prüft Klassen-id **statisch per AST vor dem Import** — „nur
  plugin.toml umbenannt" bricht jetzt mit präzisem Hinweis ab, ohne Code
  auszuführen. (3) Loader **seedet für Code-Plugins eine Instanz** (vorher nur
  Typ-Registrierung → Plugin unsichtbar, keine Tools); `[settings]` im Manifest
  jetzt auch für Tier 2. (4) Tool-Generierung: Kopien mit unverändertem
  Spec-Namen werden suffigiert statt das Built-in zu überschatten; kanonische
  Handler fremder Typen laufen über den generischen Factory-Pfad
  (hand-wired nur lobid/swb/catalog/finc).


### Raw-First Response Cache (WP2, P1–P5) (July 2, 2026)

Cache the source response **verbatim**, derive the reduced pool view on read
("Fetch ≠ Transform"). Suite `919 → 956 passed`, 10 skipped, 0 failures. Spec:
[`docs/wp_raw_response_cache.md`](docs/wp_raw_response_cache.md).

- **Infra (`UnifiedKnowledgeManager`):** `search_response_cache` table (dialect-safe
  composite PK) + `params_hash`/`store_raw_response`/`get_raw_response` (size cap 1 MB,
  soft row cap, 24 h TTL). `SystemConfig.enable_response_cache` master switch
  (+ per-instance `settings['cache_responses']`).
- **Capture seam:** `SuggesterBackedProvider._gnd_search` dual-writes each source's
  `last_raw`; `factory.build_provider` injects the policy. lobid/swb/catalog + finc +
  catalog-titles all populate raw.
- **Fetch/transform split:** `LobidSuggester`/`SWBSuggester`/`BiblioSuggester` expose a
  pure `transform(raw)`; lobid also `transform_agent_view` → `search_lobid` gains an
  additive `agent_view` (member/totalItems).
- **Aggregation (`src/core/search/aggregate.py`) + `aggregate_gnd_results` MCP tool:**
  counter (`display_count`) + provenance (`sources`/`source_count`) derived from raw,
  **raw-first with mapping fallback**. Both pipelines converged onto it
  (`gnd_batch_search`, `SearchCLI.search_from_raw`), rollback via `aggregate_from_raw`.
  Count-landmine preserved (pool count = 1, real count in `display_count`).
- **Input tools:** `InputToolSpec.cacheable` + DOI read-through cache.
- **Caveat:** classic convergence is default-on but only test-green — GUI/Webapp visual
  verification + a comparison lauf are still pending.

### Generic Plugin System — framework + Search & Input categories (July 1, 2026)

Turned ad-hoc extension points into one category-agnostic plugin system. Suite
`863 → 907 passed` (44 new tests, 0 failures). Spec: [`docs/plugin_system.md`](docs/plugin_system.md).

**Framework (`src/core/plugins/`, Qt-free):** `ConfigField` schema (single source for
settings form + availability gating), `PluginCategory` adapter registry, `plugin.toml`
manifest parser, AST security scanner + trust-on-first-use hashing, two-tier directory
loader. `PluginInstanceConfig` + `AlimaConfig.plugins` are the authoritative per-instance
store.

**Search category (`src/core/search/`):** every provider now declares `config_fields`
(so `is_available` is derived from a gating field, killing bespoke overrides);
`factory.py` `build_provider`/`build_enabled` is the single config→provider site
(D-1/D-4); new first-class `sru` provider type (D-5); MCP tools generated **per enabled
instance** (multiple finc endpoints → distinct tools, `usage_hint` in the description),
the primary keeping the canonical tool name.

**Input category (`src/utils/input_sources/`):** new `INPUT_SOURCE_REGISTRY`;
`execute_input_extraction` is now a registry dispatcher (text/file/pdf/image byte-parity,
D-11); the BeautifulSoup scraper extracted from `batch_processor` into `url_fetch`
(D-9); the DOI resolver split into three separately-configurable plugins
`doi_crossref`/`doi_openalex`/`doi_datacite` wrapping the shared `UnifiedResolver` (D-10).

**Config migration (facade-preserving):** instances authoritative; `CatalogConfig` +
DOI `SystemConfig` fields kept as derived mirrors so the ~298 legacy readers are
untouched. Synthesise-on-load + derive-on-save + reverse-sync for the legacy tabs;
`load→save→load` is diff-free (`test_plugin_config_roundtrip.py`).

**UI:** `PluginSettingsTab` (category-grouped, per-instance form auto-built from
`config_fields`, add/duplicate/remove, primary + usage_hint) replaces the checkbox-only
`SearchProviderSelectorWidget` (removed). It is now the *single* editor for all
provider/source config: the **Catalog tab** and the **DOI-resolution entries in the System
tab** were removed (−321 LoC) and their fields folded into the `catalog` plugin
(token/URLs/`catalog_type`/`strict`) and the three DOI plugins (`contact_email` + toggles);
values are derived back into the `CatalogConfig`/`SystemConfig` mirrors on save (verified
build→save→reload). Operator click-test outstanding.

**Directory plugins + security:** Tier-1 declarative (no code) covers all current
strategies; Tier-2 code plugins gated by `enable_code_plugins` + AST scan + hash-pin +
approval (`approved_plugins` ledger). Honest limit: consent + tamper-detection, not a
sandbox.

**Follow-ups (same day):**
- **Classic-pipeline enable-gate** — `execute_gnd_search` filters its provider list via
  `enabled_gnd_provider_ids()`, so disabling a provider in the Plugins tab now also
  drops it from the classic keyword step (was a fixed list). Agentic already gated via
  per-instance tools; DOI via the `doi_use_*` mirror.
- **Self-documentation contract** — `PluginDoc(description, input, output)`; every
  provider/source declares `doc()` (tests enforce completeness). Shown in the settings
  form + fed to the agent. Directory plugins document themselves in `plugin.toml`.
- **`list_plugins` MCP tool** — the agent can introspect the real active plugins (with
  self-docs) instead of conflating them with workflows (`list_workflows`).
- **Input-source MCP tools per instance** — the three DOI resolvers are now individually
  callable (`resolve_doi_crossref/openalex/datacite`). They query each source's API
  **directly** and return the *complete raw metadata record* (success = record found,
  independent of abstract) — fixes OpenAlex/DataCite dropping their metadata when no
  abstract was present. `resolve_doi` (merged, abstract-oriented) is unchanged.
- **Runtime plugin toggle** — `ToolRegistry.refresh()` (clear + reload config +
  re-register) wired to the settings save, so the chat agent picks up enable/disable +
  config changes without a restart.
- **Tool data-passthrough (cut old braids)** — agent tools now forward the *complete*
  source data instead of the old-pipeline subset: `resolve_doi` aggregates all enabled
  DOI sources' full records (+ convenience abstract); `scrape_url` returns the full page
  text (only script/style stripped, `max_chars=0` default); `read_pdf` defaults to no
  truncation. The GND/catalog search tools still reduce to `{count,gndid,ddc,dk}`
  (pipeline-pool-coupled) — audited as a WP before change: [`docs/wp_tool_data_passthrough.md`](docs/wp_tool_data_passthrough.md).

Debt register D-1…D-13 recorded in [`docs/cleanup_findings.md`](docs/cleanup_findings.md).

### Webapp `app.py` God-File-Split → APIRouter (F-6) (June 30, 2026)

`src/webapp/app.py` von **2537 → 240 LoC (−90%)** zerlegt. 8 Commits (je ein
Modul/Router, Suite nach jedem grün: 841 passed, 10 skipped). Sandbox-verifiziert
(kein GUI-Gate — anders als F-5).

**Phase A — Infrastruktur (re-export, keine Test-Änderung):**
- `session_state.py` — `sessions`-Registry, `Session`-Modell, lazy `AppContext`,
  Autosave/WebSocket-Konstanten.
- `render_bridge.py` — WP12-Transport + `_SessionBusSubscriber` (verbatim; `Session`
  als TYPE_CHECKING-Forward-Ref → kein Zyklus).

**Phase B — `APIRouter`-Module (je Commit, Test-Patch-Ziele mitwandern):**
- `routers/{workflows,models,sessions,export,websocket,analysis,agent}.py`,
  gemountet via `app.include_router(...)`.
- Geteilte Helfer in `session_io.py` (`make_json_serializable`, `sanitize_filename`,
  Autosave, `_parse_think_override`) statt in einem Router — Router importieren nie
  `app` (azyklische DAG `session_state ← render_bridge/session_io ← routers ← app`).
- `app.py` bleibt: Factory, Lifespan, Middleware/Static/Templates, Includes, 3 Seiten
  (`/`, `/webapp`, `/health`), Re-Export-Shims.

**Test-Kontrakt-Technik (der knifflige Teil):** Tests patchen/importieren via
`src.webapp.app.*`. Re-Export hält Direktimporte + *Klassen-Methoden*-Patches
(`patch.object(appmod.AppContext, …)`) am Leben, weil dieselbe Klassen-Objekt-Identität
erhalten bleibt. Nur *Modul-Attribut-Ersetzungen* (`patch("src.webapp.app.PipelineManager")`,
`appmod.X = …`) müssen auf den neuen Router umziehen — denn der bewegte Consumer löst
den Namen jetzt im Router-Namespace auf. Pro Router migriert; Klassen-Methoden-Patches
blieben unverändert.

**Verifikation pro Commit:** AST-Undefined-Name-Scan (fing einen echten
Funktionskörper-`NameError` — `_parse_think_override` — den Import + grüne Suite beide
verfehlten), Route-Tabelle byte-identisch (25 Routen), Live-Endpoint-Proben
(`/api/workflows`, Session-Roundtrip, WS-`complete`, `/api/analyze` durch `run_analysis`),
volle Suite grün. Eine vorbestehend flakige WS-Statebus-Test (Memory) unverändert.

### Search-Provider-Plugin-System (F-3) + „Häufigkeit zeigt 1" (F-4) (June 29, 2026)

Umsetzung des capability-basierten Search-Provider-Standards (CLAUDE.md-Vision)
plus des gekoppelten Anzeige-Bugs F-4. Drei Phasen, je ein Commit, Suite grün
(835 passed, 1 vorbestehender DK-Title-Fail).

**P1 — Standard + Registry** (`src/core/search/`, additiv/facade-erhaltend):
- Qt-freier `SearchProvider`-Protocol + getypter `ProviderResult` (capability-
  getaggt) + `SearchCapability`; eine `@register_provider`-Registry analog
  `@register_step`/`@register_tool_fn`.
- 5 Provider (lobid/swb/catalog/finc/gnd_local) umhüllen die bestehenden Suggester
  und registrieren sich beim Import. **finc ist jetzt im Standard** (TITLE_RECORDS
  + SUBJECT_FACETS) statt MCP-Sonderfall. Verlustfreie Legacy-Shape-Konverter.
- `tests/test_providers.py` (Round-Trip + Registry-Guards).

**P2 — Caching-Wrapper, `SuggesterType` entfernt, F-4-Fix:**
- Mapping-first-Caching aus `MetaSuggester` in den `CachingProvider`-Decorator
  ausgelagert; `MetaSuggester` enumeriert jetzt die Registry (kein Enum/if-elif).
  `SuggesterType` entfernt, alle 8 Importer auf Provider-Id-Strings migriert.
- **F-4:** Mapping-Cache speichert jetzt Per-GND-ID-Counts (additive Spalte
  `gnd_counts` + abgesicherte Migration). Cache-Treffer behalten Pool-`count = 1`
  (Ranking/Chunking unverändert — Count-Landmine), tragen aber ein separates
  `display_count` mit der echten Häufigkeit; fließt über `gnd_search_core` →
  `flatten_gnd_hits` + agentische Anzeige; `rank_pool` liest es nie.
- `tests/test_caching_provider.py`.

**P3 — Registry-getriebene MCP-Tools + Provider-Config + GUI:**
- Die 5 Library-Such-Tools werden aus `ProviderToolSpec`-Deklarationen generiert
  (`ToolRegistry._generated_search_tools()`); 5 Schemas + 4 Handler entfernt
  (`_handle_search_finc` bleibt, via Spec verdrahtet). **Bytegleich** zu den alten
  Handlern bewiesen (Schemas + Outputs über kw/non-kw/default/non-default).
- `SearchProviderConfig` (per-Provider enable/disable) gated die Tool-Exposition;
  GUI-Selektor (`src/ui/provider_selector.py`) als Tab in den Settings.
- `tests/test_provider_tool_generation.py`.

Spec: [`docs/search_provider_plugins.md`](docs/search_provider_plugins.md).
Offen: GUI-Selektor operator-Klicktest; finaler agentischer Lauf zur F-4/Landmine-
Bestätigung (braucht LLM).

### Kern-Aufräumung III: llm_service Per-Provider-Entdopplung (June 29, 2026)

Untersuchung der vermuteten „~80% Per-Provider-Duplikation" in
`src/llm/llm_service.py`. **Befund (verifiziert):** Die Behauptung hält nicht —
die *lebenden* Generatoren sind genuin provider-spezifisch (eigene SDKs,
Streaming-Protokolle, Tool-Schema-Formate, Response-Parsing) und teilen ihr
Gerüst bereits: Dispatch-Registry (`supported_providers[p]['generator']` für den
Text-Pfad, Dispatch-by-`provider_type` für den Tool-Pfad), `_convert_messages_for_*`,
`_retry_on_rate_limit`, `_apply_openai_think`. Die „Duplikation" war in Wahrheit
**abgelöster Dead-Code**.

**Entfernt (436 Zeilen, alle mit 0 Referenzen — keine Calls/Strings/getattr/
Registry/Tests):**
- `_generate_ollama` (HTTP) — abgelöst durch `_generate_ollama_native` (Registry
  nutzt nur den Native-Generator, vgl. „BUGFIX"-Kommentar).
- `_generate_github` + `_generate_azure_inference` — abgelöst durch
  `_generate_openai_compatible` (GitHub/Azure laufen als `openai_compatible`).
- `_init_ollama` (HTTP) + `_init_azure_inference` — zugehörige tote Initializer.

Die 4 lebenden Text-Generatoren (gemini/anthropic/ollama-native/openai_compatible)
+ 5 Tool-Generatoren + alle 4 `_cancel_*`-Helfer (via `cancel_generation`) bleiben
unangetastet. `llm_service.py`: **3493 → 3057 Zeilen** (−12,5%).

**Bewusst NICHT gemacht:** Strategy-Pattern-Rewrite der lebenden Generatoren —
hohes Risiko an der kritischsten Schicht für minimalen echten Dedup-Gewinn.

**Tests.** `test_llm_service_seed`, `test_streaming_with_tools`, `test_rate_limit_retry`
grün (47). File-isolierte Gesamtsuite: 60 clean, unverändert die 2 bekannten
Pre-existing-Issues. Keine neuen Fehler; keine Restreferenzen auf entfernte Methoden.

### Kern-Aufräumung II: pipeline_utils Modul-Split (June 29, 2026)

Der 7615-Zeilen-Gott-Modul `src/utils/pipeline_utils.py` wurde in fokussierte
Module zerlegt. **Strategie: Facade.** Code wird in neue Module verschoben und in
`pipeline_utils` per `from .<modul> import …` **re-exportiert** — kein einziger
externer Importer (`from …pipeline_utils import X` in UI/CLI/Webapp/Tests) muss
geändert werden. Inkrementell, Modul für Modul, Tests nach jedem Schritt.

**Neue Module (`src/utils/`):**
- `pipeline_input.py` (465) — `execute_input_extraction` + PDF/Image/OCR-Helfer.
- `gnd_keyword_utils.py` (508) — `verify_keywords_against_gnd_pool`,
  `extract_keywords_from_descriptive_text*`, `canonicalize_*`, `extract_gnd_id`,
  `deduplicate_canonical_keywords` (Leaf, kein Executor-Import).
- `pipeline_text_utils.py` (243) — reine Text/Display/Title-Helfer
  (`repair_display_text`, `sanitize_for_filename`, `build_working_title`,
  `extract_source_identifier`, `flatten_keyword_centric_results`); Leaf, von
  Executor **und** Formatter genutzt.
- `pipeline_formatters.py` (992) — `PipelineResultFormatter` (importiert nur die
  Text-Leaf-Helfer, einseitig).
- `pipeline_persistence.py` (400) — `PipelineJsonManager`,
  `export_analysis_state_to_file`, `AnalysisPersistence`.

`pipeline_utils.py`: **7615 → 5098 Zeilen** (−33%), enthält jetzt fokussiert die
Classic-Step-Helfer (`_emit_classic_*`, `_run_classic_step`) + die Klasse
`PipelineStepExecutor` (inkl. `execute_complete_pipeline`).

**Fallen beim Split (alle gefixt + verifiziert):** Modul-globale Namen wandern
nicht automatisch mit — `logger = logging.getLogger(__name__)` musste in
`pipeline_text_utils`/`pipeline_formatters`/`pipeline_persistence` neu gesetzt
werden; Annotationen werden bei `def`-Zeit ausgewertet → fehlende Typing-Namen
(`Set`) und Datamodel-Klassen (`TaskState` & Co. in `pipeline_persistence`) mussten
importiert werden. Import-Test fängt Annotation-NameErrors zuverlässig.

**Bewusst NICHT zerlegt:** `PipelineStepExecutor` bleibt eine Klasse (~4900 Z.) —
ein Split via Mixins wäre riskant/unleserlich ohne echten Nutzen.

**Tests.** File-isolierte Gesamtsuite: 59–60 Dateien clean; unverändert die 2
bekannten Pre-existing-Issues (DK-Title-Konvergenz, Qt-Abort
`test_analysis_review_tab`). `test_image_analysis_tab` zeigte einen **flaky** Qt-
Teardown-Abort am Interpreter-Exit (auf Wiederholung grün), keine Logik-Regression.

**Caveat.** Keine Verhaltensänderung beabsichtigt; abgesichert über die bestehende
Suite + Import-Checks, nicht über einen Live-Pipeline-Lauf.

### Kern-Aufräumung I: geteilter GND-Such-Kern + Chunking-Dedup (June 29, 2026)

Erste Aufräum-Runde an den beiden Kernen (klassische Pipeline ↔ Agentik v4). Ziel:
über Zeit „vibisch"/agentisch gewachsene Doppelungen auflösen, **ohne** Verhalten zu
ändern. Leitprinzip: schmal & sicher, byte-identische Outputs, abgesichert durch
Charakterisierungs-Tests.

**Verifizierter Befund (wichtiger als der Umbau):** Die vermutete Duplikation ist
deutlich kleiner als ein Oberflächen-Scan nahelegt. Klassik und Agentik teilen die
GND-Such-**Engine** bereits (`MetaSuggester` mapping-first — klassisch direkt via
`SearchCLI`, agentisch via `tool_registry`-Tools, die intern denselben `MetaSuggester`
nutzen). Die Abhängigkeit ist **einseitig** agentisch→klassisch (kein Zirkel:
`deterministic_functions` importiert `verify_keywords_against_gnd_pool` +
`PipelineStepExecutor.execute_dk_search` aus `pipeline_utils`, nicht umgekehrt).
`PipelineManager` *komponiert* `AlimaManager` (kein Subclass, keine Doppel-Orchestrierung).
Der Seed wird in `shared_context.to_keyword_analysis_state` bereits durchgereicht.

**Neu — `src/core/gnd_search_core.py`** (reine Funktionen, kein Import aus
pipeline_utils/deterministic_functions/search_cli → kein Zyklus):
- `merge_code_entry` — geteilter Merge-Atom (Max-Count + Union, containertyp-erhaltend:
  set→`update`, list→order-preserving dedup). Genutzt von klassisch
  `SearchCLI.merge_results` **und** agentisch `merge_into_pool`.
- `merge_into_pool` / `parse_batch_response` / `parse_batch_response_with_terms` —
  aus `deterministic_functions` extrahiert; backen jetzt `gnd_batch_search`
  **und** `catalog_multi_search` (3 Aufrufstellen entdoppelt).
- `rank_pool` — `source_count`-Attachierung + Ranking `(source_count, count)` desc.
  Docstring dokumentiert die **Count-Landmine** (Pool-`count` steuert
  `selection_chunks`→`selection`; nie summieren, nur `max`).

**Neu — `src/utils/chunking.py`** `split_into_equal_chunks`: das gespiegelte
Equal-Chunk-Splitting aus `pipeline_utils._execute_chunked_keyword_analysis` und
`llm_agent_step` (dort als `_split_chunks_classic` re-exportiert) — eine Quelle der
Wahrheit gegen künftige Drift.

**Bewusst NICHT angefasst (Befund):** Keyword-Extraktion ist keine sichere
Konsolidierung — `extract_keywords_from_response` (String) und
`extract_keywords_from_descriptive_text` (Tupel + GND-Validierung) haben verschiedene
Verträge; `extract_keywords_from_descriptive_text_simple` ist Dead-Code (→ WP13).

**Tests.** Neu `tests/test_gnd_search_core.py` (14). Bestehende Safety-Nets grün:
`TestGndBatchSearchConvergence`, `TestClassicChunkSplitting`. Voller Lauf
(file-isoliert): 60 Dateien clean; unverändert die 2 bekannten Pre-existing-Fails
(Qt-Abort `test_analysis_review_tab`, DK-Title-Konvergenz). Keine neuen Fehler.

**Caveats.** Verifiziert via Charakterisierungs-Tests + isolierter Suite.
Operator-Vergleichslauf klassisch↔agentisch (June 29, 2026): beide Pfade laufen
durch, reasonable results — keine Regression (klassisch≠agentisch ist erwartet,
verschiedene Pfade by design). Byte-
Identität gilt für die getesteten Pfade; die `list`-Merge-Reihenfolge ist nun
deterministisch (vorher via `set()` nicht-deterministisch) — funktional äquivalent,
da Selektion/Ranking nicht von Code-Reihenfolge abhängt.

### Rate-Limit-Retry für Tool-Calling (HTTP 429) (June 25, 2026)

Provider-Rate-Limits (429) brachen bisher den ganzen agentischen Workflow ab:
der Provider-Call wirft → `AgentLoop` fängt → `LLMAgentStep` macht `RuntimeError`
→ Workflow-Fehler. Jetzt wird gewartet + wiederholt statt abgebrochen.

**Implementierung (`src/llm/llm_service.py`).** Drei Modul-Funktionen + Einhängung
an `generate_with_tools` (ein Dispatch-Punkt → gilt für alle Provider:
OpenAI/Mistral, Anthropic, Gemini, Ollama, Fallback):
- `_is_rate_limit_error(exc)` — erkennt 429 providerübergreifend (Status-Attribute
  `status_code`/`http_status`/`code`/`status`, `response.status_code`,
  Exception-Klassenname, Meldungstext) ohne SDK-Import. Bewusst breit.
- `_rate_limit_retry_after(exc)` — liest die vom Server vorgegebene Wartezeit:
  HTTP-`Retry-After` (Sekunden **oder** HTTP-Datum) → Geminis `retry_delay`
  (`.seconds`) → Zahl aus dem Meldungstext. `None` ⇒ Backoff.
- `_retry_on_rate_limit(fn, label, status_cb, should_stop)` — respektiert
  `Retry-After` (Hard-Ceiling 300 s), sonst exponentielles Backoff (2→4→8…s,
  Cap 60 s, + Jitter). Nicht-429-Fehler sofort re-raise; Aufgeben nach 5
  Versuchen. Wartezeit ist unterbrechbar (`should_stop` jede Sekunde geprüft).
  Status `⏳ Rate-Limit erreicht – warte Xs (Versuch n/5)` fließt in GUI/CLI-Log.

Modul-Funktionen (nicht Methoden), damit die `MagicMock(spec=LlmService)`-
Dispatch-Tests (`test_llm_service_seed`, `test_streaming_with_tools`) sie nicht
wegmocken. Env-Tuning: `ALIMA_RATE_LIMIT_MAX_RETRIES`, `_BASE_DELAY_S`,
`_MAX_DELAY_S`, `_RETRY_AFTER_CEILING_S`.

**Tests.** `tests/test_rate_limit_retry.py` (15) — Erkennung, Retry-After-Parsing
(Header/Datum/Gemini/Text), Retry-dann-Erfolg, Aufgeben nach Max, Passthrough,
`should_stop`-Abbruch, End-to-End über `generate_with_tools`.

**Scope/Caveats.** Nur der agentische `generate_with_tools`-Pfad; klassischer
`generate_response`-Stream hat bereits eigenes (gröberes) Retry in
`pipeline_utils.py`. Getestet mit simulierten 429ern, nicht gegen ein Live-Limit.

### Webapp-Redesign: vertikaler Stack + Pipeline-Leiste mit Live-Stepper (June 23, 2026)

Restructured the `/webapp` layout from a fixed 3-column grid (`input | editor |
stream`) into a vertical stack so the chat/log becomes the focal element. Webapp
UI only — no pipeline-logic change, CLI/GUI parity untouched. Operator decisions:
bottom bar + live step-stepper.

**Layout (`templates/webapp.html`, `static/styles.css`).** Two stacked zones:
- `#input-zone` (top) — a framed widget with: header (chevron toggle) + a
  **collapsible** `.input-zone-body` (input sources + text editor + extracted
  text + `#results-panel` summary; sources/editor side-by-side ≥768px) + the
  `.pipeline-bar` as a **persistent footer inside the zone**. The body
  auto-collapses to the header on run start; reopened manually.
- `.pipeline-bar` (footer of `#input-zone`, never collapses) — `#pipeline-stepper`
  row + controls row (workflow/provider/model/thinking + refresh, then
  Analyse/Abbrechen/Schritt-abbrechen, then save/load Speichern/Laden/Neue
  Analyse). Because it sits below the collapsible body but inside the zone, the
  run status + abort + save/load stay visible while the body is collapsed during
  a run. Styled as a deck (top border + `--clr-surface-2`, no standalone card
  chrome; bottom corners clipped by the zone radius). `#export-btn` `disabled`
  until results exist. Removed the old `position:fixed` `.editor-footer` and
  `grid-template-areas`.
- `.panel-stream` (bottom, `flex:1`) — chat/log, now dominant; grows as the input
  body collapses. Desktop fills viewport via `.workspace { display:flex;
  flex-direction:column; height:calc(100vh - header) }`.
- New CSS: `.input-zone*`, `.pipeline-bar`, `.bar-group`, `.pipeline-stepper`,
  `.step-node`/`.step-dot` (done/active/pending states, `step-pulse` animation,
  theme-token colors). Mobile: zones stack, bar wraps, stepper scrolls X.

**Stepper data (`src/webapp/app.py`).** `/api/workflows` items gain an ordered
`steps:[{id,label}]`: agentic from each YAML `steps:` (`_extract_workflow_steps`),
`__classic__` hardcoded (`_CLASSIC_STEPS`, mirrors `PipelineManager.step_definitions`).
Agentic step progress now reaches the session via a new `agentic_context`
callback wired into `set_callbacks` (per-completed-step; classic still uses
`step_started/completed`).

**Frontend (`static/app.js`).** Caches `workflowSteps` from `/api/workflows`;
`renderStepper`/`renderStepperForSelected` build nodes on load + workflow change;
`updateStepper(currentStep,status)` highlights from `current_step` in both the
WS and polling paths (`updatePipelineStatus`); `markStepperComplete` on finish.
`setInputZoneCollapsed`/`toggleInputZone` drive the collapse (auto on run via
`updateButtonState`, reset on "Neue Analyse"). Element IDs preserved → existing
handlers unchanged.

**Verified** (Playwright, headless): vertical zone order + geometry, collapse
435→52px with chat expanding to fill, stepper render (classic 6 / `alima_v51` 8)
and live state transitions (running→active, completed→next-active, last-step
clean), unknown-id graceful ignore, mobile stack + horizontal stepper scroll,
`/api/workflows` `steps` payload, all static assets 200. Note: a full live
pipeline run (needs provider/API key, consumes tokens) was not executed; stepper
progression was driven through the exact functions the WS/polling handlers call.

### Workflow-YAML-Editor im Qt6-GUI (June 23, 2026)

Structured GUI editor to view, edit and create agentic v4 workflow YAML files
(`workflows/*.yaml`) — previously only selectable, not editable. The workflow
engine (loader, executor, registry, steps) is consumed **read-only**; nothing
in it changed.

**New: `src/ui/workflow_editor_dialog.py` — `WorkflowEditorDialog`.**
- Left nav: vertical splitter — a „⚙ Workflow-Einstellungen" toggle button on
  top over a `QListWidget` of steps (`id · type`) with Add/Remove + Move Up/Down
  (order = execution order); button and list are mutually exclusive.
- Settings panel: `name`/`version` + a multi-line `description` (`QTextEdit`) +
  a vertical splitter of `settings`/`meta_agent` key-value tables.
- Step editor is a **`QTabWidget`** (Allgemein / Ein-/Ausgaben / LLM / Prompts /
  Funktion) so each concern stays uncluttered and the prompts get a full tab
  with a resizable splitter (monospace `system_prompt`/`user_prompt`). Tab
  visibility follows `type`: `llm_agent` → LLM + Prompts; `deterministic` →
  Funktion (`function` from `list_tool_fns()` + `config`). `type` choices come
  from `STEP_REGISTRY`; common fields (`id`, `description`, `enabled`,
  `depends_on`, `when`) + `inputs`/`outputs` tables live in the first two tabs.
- **ruamel.yaml round-trip** (new dep `ruamel.yaml==0.18.10`): only edited
  leaves are mutated in place, so comments/section headers survive. Verified
  zero-diff no-op round-trip on all 5 shipped workflows. Edited multi-line
  prompts kept as `|` block scalars (`LiteralScalarString`); `None` rendered as
  explicit `null`.
- **Validate-before-write**: dumped YAML is loaded with the real execution
  loader `load_workflow(strict=True)` (same call `pipeline_manager` makes); an
  invalid workflow is never written (unknown type / missing id / duplicate id
  all blocked).
- **Save target** `~/.config/alima/workflows/` (already in
  `DEFAULT_SEARCH_PATHS`). Shadow guard: warns when a same-named file exists in
  project `workflows/` (which wins `find_workflow_file`), since the user copy
  would otherwise be silently ignored at execution.

**Access points** (engine untouched): Bearbeiten-Menü „📋 Workflow-Editor"
(`main_window.show_workflow_editor`) and a „✏️" button next to the workflow
combo in `PipelineConfigDialog` (refreshes the combo via the existing
`_populate_workflow_combo` after close).

### finc / VuFind-JSON catalog backend (June 11, 2026)

Established finc (TU Freiberg finc solrproxy) as a LOCAL catalog backend.
Endpoint/URLs are config-driven (`CatalogConfig.finc_*`, default off); no
institution URL is hard-coded.

**Phase A — review fixes** (`finc_client.py`, `finc_suggester.py`,
`tool_schemas.py`): the proxy returns HTTP 200 with `{"status":"ERROR"}` on bad
queries — now surfaced instead of silently reporting 0 results. Corrected the
institution facet key (`institution`, not `institution_facet`) and the phrase-
quote guidance (literal `"`, not `%22` which double-encodes via requests).

**Phase B — facets** (`finc_client.py`, `finc_suggester.py`, `tool_registry.py`):
`FincClient.search(facets=…)` requests `facet[]=` and parses the facet block;
`limit=0` for facet-only; `normalize_dk_value("dk 530.145")→"DK 530.145"`.
`search_finc` MCP tool gains `facets` + clarified one/many-title, subject,
author modes.

**Phase C — pipeline (opt-in, gated)**:
- `FincCatalogClient` (`finc_catalog_client.py`) — BiblioClient-compatible
  extractor: finc Subject search → titles, then per-title `udk_raw_de105`/
  `rvk_facet` via single-record isolation (`lookfor=id:"…"`), parallelized;
  funnels through `extract_classifications_from_titles` for shape-identical
  output. Wired into `execute_dk_search` behind `finc_dk_enabled` (Libero/SRU
  fallback). Live benchmark ~0.9s/keyword at 8 workers.
- `finc_subject_harvest` deterministic step (`deterministic_functions.py`,
  `alima_classic.yaml`) behind `finc_harvest_enabled` — harvests finc titles +
  reconciles subjects against the local GND cache into the selection pool.
- finc DK source = `udk_raw_de105` (numeric DK, matches Libero's scraped field);
  `rvk_facet` for RVK; only `dk `-prefixed values emitted (drops `fg`/`fgaut`
  artifacts). Flags exposed in settings dialog + CLI wizard.

Tests: `test_finc_client.py`, `test_finc_catalog_client.py`,
`test_finc_subject_harvest.py` + gated live integration/benchmark
(`RUN_INTEGRATION_TESTS=1 FINC_TEST_BASE_URL=…`).

### Kern-Konvergenz klassisch ↔ agentisch (WP-K1–K4) (June 10, 2026)

Befund: der agentische v5x-Workflow füllte den Klassifikations-/Keyword-Kontext
unzuverlässiger als die klassische Pipeline — 5 strukturelle Divergenzen im
Kern, kein Prompt-/UX-Thema. Suite: 591 passed / 5 skipped.

**WP-K1 — GND-Suche mapping-first** (`src/mcp/tool_registry.py`,
`deterministic_functions.py`):
- MCP `search_swb`/`search_lobid` instanziieren jetzt `MetaSuggester` statt
  roher Suggester → mapping-first-Cache (Read + Write-back via
  `UKM.search_with_mappings_first`) identisch zum klassischen Pfad. Non-kw
  `search_type`/abweichende `max_pages` gehen weiter an den rohen
  Kind-Suggester (Mapping-Cache ist nach Term+Suggester gekeyt, nicht nach
  Suchtyp). Handler-Antworten tragen jetzt `errors` (per-Term-Fehler).
- `gnd_batch_search`: Quellausfälle werden gestreamt + als `source_errors`
  zurückgegeben; **alle Quellen tot → RuntimeError**; leerer Pool + Teilausfall
  → RuntimeError (Ergebnis nicht vertrauenswürdig); leerer Pool ohne Fehler →
  laute Warnung, kein Abbruch (echte Nulltreffer).

**WP-K2 — Selection-Verifikation** (`verify_final_keywords` in
`deterministic_functions.py`, neuer Step `verify_keywords` in
`alima_v51.yaml` + `alima_classic.yaml`):
- LLM-Auswahl wird gegen `gnd_entries`-Pool verifiziert (GND-ID-Match →
  Titel-Match → DB-Fallback `search_gnd_by_title`), falsche/fehlende GND-IDs
  werden korrigiert/ergänzt, Unverifizierbares geloggt statt still von der
  strict-Validation der DK-Suche verworfen. Wiederverwendet
  `verify_keywords_against_gnd_pool` aus pipeline_utils (klassischer Code).

**WP-K3 — DK-Klassifikations-Parität** (`pipeline_utils.py`,
`deterministic_functions.py`, beide YAMLs):
- Frequenz-Filter + Titel-losen-Filter + Institution-RVK-Filter +
  RVK-Guardrail aus `execute_dk_classification` in
  `PipelineStepExecutor.prepare_dk_classification_context()` extrahiert;
  klassischer Pfad ruft sie unverändert, `dk_search_agentic` baut
  `formatted_prompt` jetzt darüber (vorher: rohe Top-60 ohne Filter).
- `dk_search_agentic` übergibt `rvk_anchor_keywords`
  (`_derive_rvk_anchor_keywords`, heuristischer Pfad ohne LLM).
- Klassifikations-Step beider Workflows: `when: "${extra.dk_prompt_text} != ''"`
  — läuft nicht mehr mit leerem Katalog-Kontext (klassische Pipeline
  überspringt dann ebenfalls). `dk_frequency_threshold` als
  `dk_collect`-Config (Default 1 = `DEFAULT_DK_FREQUENCY_THRESHOLD`;
  Plan-Annahme „Default 10" war falsch).

**WP-K4 — Prompt↔Daten-Mismatch**: `gnd_batch_search` berechnet jetzt
`sources`/`source_count` pro Entry und sortiert nach `(source_count, count)`
— das im selection_chunks-Step beschriebene Multi-Source-Ranking existiert
damit wirklich; YAML-Beschreibung des search-Steps korrigiert.

**Tests**: `tests/test_core_convergence.py` (12 neue Tests: Quellfehler-
Propagation, Ranking, Verifikation inkl. ID-Korrektur/DB-Fallback,
geteilte DK-Filter); Step-Listen-Erwartungen in `test_agents_v2.py`,
`test_e2e_smoke.py`, `test_step_form_builder.py` aktualisiert.

**WP-K5 — Chunking angeglichen** (Nachtrag, Operator-Anweisung):
- `LLMAgentStep._run_chunked`: `chunk_size: 0`/fehlend → Auto-Detection via
  `model_capabilities.get_chunking_threshold` (per-model Config > Pattern >
  Default 500) — dieselbe Quelle wie der klassische
  `keyword_chunking_threshold`; `_auto_chunk_size()` neu.
- Split-Semantik klassisch (`_split_chunks_classic`): ≤ Threshold = EIN Call;
  darüber gleichmäßige Chunks (2 bis 1,5×, sonst ⌈n/Threshold⌉) statt fester
  Slices mit Mini-Restchunk.
- Alle 4 Workflow-YAMLs: `chunk_size: 350` → `0` (auto).

**WP-K6 — Pipeline-Tab: fehlende/geleerte Anzeigen** (Operator-Befund:
„DK-Inhalte im Katalog-Recherche-Tab werden beim Beenden der agentischen
Pipeline geleert", „manchmal fehlen Infos"; drei Mechanismen gefunden):
- **Snapshot-Cap 50**: `WorkflowExecutor._emit`-Snapshots kappten ALLE Listen
  auf 50 Einträge — GUI sah nur 50 von ~1200 Pool-/392 DK-Einträgen; der
  `{"_truncated": N}`-Sentinel crashte zudem `"\n".join()` bei String-Listen
  (still geschluckt → leeres Widget). Jetzt feldspezifische Caps
  (gnd_entries 5000, dk_search_results 2000, …; execution_history bleibt 50)
  + sentinel-toleranter Join im Extraction-Handler.
- **Blank-Overwrite**: `_display_dk_search_results` setzte bei nicht-leerer
  Eingabe kommentarlos den Formatter-Output — war der leer (keyword-zentrisches
  Format, titel-lose Einträge), wurde das Widget beim Abschluss-Sync geleert.
  Formatter (`format_dk_search_results_text`, `get_titles_for_dk_code`)
  flatten jetzt keyword-zentrische Eingaben (via neuem modulglobalem
  `flatten_keyword_centric_results`), zeigen `matched_keywords`, und die
  Anzeige wird nie mehr stumm geblankt (Placeholder + Warning statt "").
  End-of-run-Sync überschreibt nur noch, wenn die neuen Daten darstellbar sind.
- **Sync-Kaskade**: ein Fehler im GND-Teil von `_sync_classical_tabs_from_state`
  brach per breitem try/except den ganzen Sync ab → DK-Teil nie befüllt.
  Blöcke jetzt unabhängig geguarded.

**WP-K7 — GND-Recherche 3-stufig** (`pipeline_tab.py`): Pool (alle Treffer)
→ ☑ Chunk-Auswahl (selection_chunks-Überlebende, blau) → ✅ Final
(verifizierte Keywords, grün/fett); Textfilter (Begriff/GND-ID) +
Stufen-Combo ersetzen die alte Checkbox; Zähler-Label
(`Pool: N · Chunk: M · Final: K`); Snapshot-Handler für `selection_chunks`
und `verify_keywords` (zeigt verifizierte Liste inkl. korrigierter GND-IDs).
Offscreen-Smoke-Test: 1200 Zeilen, Tier-/Textfilter korrekt.

**Bewusst NICHT angefasst** (Operator-Entscheidung bzw. Folge-Items):
v5.1-Prompts bleiben inline (nur Mechanik konvergiert),
Konsolidierungslauf nach Chunk-Merge, RVK-Nachselektion im agentischen Pfad
(braucht LLM in `dk_search_agentic`). Gemma-Befund „nur rudimentäre
DK-Auswahl trotz vollem Kontext": modellsensitiv — v5.1-Prompts haben (anders
als prompts.json) keine modellspezifischen Varianten; ggf. Folge-Item.

### WP B+C — Debugbarkeit + E2E-Sicherheitsnetz (June 10, 2026)

Fortsetzung des Maßnahmenplans (nach WP A). Suite: 579 passed / 5 skipped.

**WP B — Debugbarkeit:**
- **Zentrale Logging-Konfiguration komplett**: GUI und CLI nutzten
  `logging_utils.setup_logging` bereits; die Webapp (vorher nur
  `basicConfig`, Konsole) nutzt es jetzt auch → Konsole + `alima_webapp.log`,
  `LOG_LEVEL=DEBUG` env-Var wird auf Stufe 2 gemappt (`src/webapp/app.py`).
- **print() → Logger**: `swb_suggester.py` (18×) und `lobid_suggester.py` (4×)
  auf `self.logger.debug` umgestellt; die zwei „could not extract"-Fälle in
  SWB-Einzeltreffer-Seiten sind jetzt unbedingte `logger.warning` (Datenverlust).
  Nicht angefasst: `src/core/lobid_subjects.py` (13×) und
  `src/core/katalog_subject.py` (10×) — werden von nichts importiert,
  **Dead-Code-Kandidaten für WP E**; `registry.py`-Treffer sind Docstring-Beispiele.
- **except:pass-Audit** (~55 Stellen): nackte `except:` auf konkrete Typen
  eingegrenzt (`OSError` bei unlink-Cleanups ×6, `ValueError/TypeError` bei
  Datums-/JSON-Parsing ×3); Silent-Swallows mit Logging versehen
  (`unified_provider_tab` Modell-Lookup/-Persist → warning,
  `pipeline_config_dialog` Prompt-Fallback → debug, Bus-Emits in
  `pipeline_manager` → warning bzw. `llm_agent_step`/`pipeline_utils` → debug);
  übrige Best-Effort-Stellen mit Begründungskommentar. Übersprungen:
  `pipeline_chat_panel.py` (3 Stellen, WP12-Datei).

**WP C — E2E-Smoke-Tests** (`tests/test_e2e_smoke.py`, LLM an der
LlmService-Grenze gemockt, Netzwerk an SearchCLI-/Tool-Grenze gefakt):
- Klassische Pipeline: `execute_complete_pipeline` initialisation → search →
  keywords → `KeywordAnalysisState` mit Keywords, Suchergebnissen, Streaming.
- Agentisch: `alima_classic.yaml` (7 Steps) durch `WorkflowExecutor` mit
  `LLMAgentStep` + deterministischen Funktionen; Kontext trägt Ergebnisse
  durch die ganze Kette; plus Negativ-Test (LLM down → `report.success=False`).
- `AgentLoop` Multi-Turn: 2 Tool-Calls + finale Antwort über 3 LLM-Turns,
  Tool-Results landen in der Konversation, Hooks feuern.

**Dabei gefundener+behobener Silent-Fail** (vom Negativ-Test aufgedeckt):
`AgentLoop` wandelte LLM-Exceptions in `content="Error: …"` um und
`LLMAgentStep` wertete das als Erfolg → Workflow lief mit Müll weiter und
meldete `success=True`. Jetzt: `AgentResult.error`-Feld (rückwärtskompatibel),
`AgentLoop` setzt es, `LLMAgentStep` lässt den Step fehlschlagen
(`src/core/data_models.py`, `src/core/agent_loop.py`,
`src/core/agents/steps/llm_agent_step.py`).

### WP A — Fehler sichtbar machen / Silent-Fail-Härtung (June 10, 2026)

Erste Stufe des Maßnahmenplans aus der Basis-Bewertung (Plan-Datei
`ich-h-tte-gerne-eine-snappy-backus.md`): Fehler, die bisher geschluckt
wurden und leere Ergebnisse als Erfolg erscheinen ließen, werden jetzt
gemeldet. Tests: `tests/test_error_visibility.py` (10 Negativ-/Positiv-Tests);
Suite 575 passed / 5 skipped.

- **Worker**: `PipelineWorker` hat neues Signal `pipeline_error(str)` und
  emittiert es im bisher stummen `except`-Block (`src/ui/workers.py`);
  `PipelineTab.on_pipeline_error` zeigt Dialog, setzt Status, reaktiviert
  den Start-Button (`src/ui/pipeline_tab.py`).
- **Klassische Pipeline stoppt bei Schritt-Fehlschlag**: `_execute_next_step`
  hatte keinen `else`-Zweig für `success=False` — die Pipeline lief nach
  einem fehlgeschlagenen Schritt weiter (auto_advance), und Schritte, die
  „sauber" `False` zurückgaben (z. B. DK-Klassifikation), lösten gar keinen
  `step_error_callback` aus. Jetzt: Status `error`, Callback genau einmal,
  Bus-Event `state.pipeline_step` mit `status="error"` + `error`-Payload,
  kein Auto-Advance (`src/core/pipeline_manager.py`).
- **WorkflowExecutor (agentisch)**: try/except um Step-Konstruktor,
  `step.execute()` und `ConditionalEngine.evaluate` → `StepResult(success=False)`
  statt Thread-Crash; kaputte `when:`-Bedingung ist Step-Fehler, kein
  stilles Überspringen (`src/core/agents/workflow_executor.py`).
  Hinweis: `BaseStep.execute` fing `run()`-Exceptions schon ab — ungeschützt
  waren Konstruktor, Condition und execute-Overrides.
- **Parse-Fehler ≠ leeres Ergebnis**: unparsebare LLM-Antwort bei der
  Initialisierung wirft jetzt `ValueError` mit Response-Preview statt mit
  0 Schlagwörtern „erfolgreich" weiterzulaufen (`src/utils/pipeline_utils.py`);
  `extract_keywords_from_response` loggt WARNING bei leerem Resultat aus
  nicht-leerer Antwort (`src/core/processing_utils.py`); generischer Pfad in
  `alima_manager._create_analysis_result` warnt (kein Raise, da
  `match_keywords_against_text`-Fallback legitime Teilergebnisse liefert).
- **Suggester: Quelle-down ≠ kein Treffer**: `BaseSuggester` bekommt
  `last_errors` + `_record_search_error` (immer `logger.warning`, nicht mehr
  `if self.debug: print`). SWB cached fehlerbehaftete Suchen **nicht** mehr
  (vorher wurde ein API-Ausfall dauerhaft als „kein Treffer" persistiert).
  Propagation: Suggester → `MetaSuggester` → `SearchCLI.last_errors` →
  `execute_gnd_search` streamt `⚠️ Quelle(n) fehlgeschlagen für '<term>'`
  und eine Abschluss-Warnung an GUI/CLI/Webapp.
- **Zurückgestellt** (WP12-Dateien, Vermischung vermeiden): Rendering des
  `status="error"`-Bus-Events im Chat-Panel/Webapp.

### WP12 — Unified Render Layer (GUI ↔ Webapp) (June 9, 2026)

GUI and webapp rendered the same pipeline data with separately-maintained
chrome (the WP2 DK/GND divergence). Now both render from **one** CSS + JS
render layer driven by a versioned JSON render-event protocol over two
transports. Spec: [`docs/wp12_unified_render_layer.md`](docs/wp12_unified_render_layer.md).

- **WP12.1 — Asset extraction**: the theme CSS + DOM-dispatcher JS were lifted
  out of the inline `_HTML_TEMPLATE` in `src/ui/web_log_view.py` into
  `src/webapp/static/alima_render.{css,js}` (single source). `WebLogView`
  inlines them at construction (lowest-risk QWebEngine load path); the webapp
  serves them as static assets. All content CSS is **scoped under `#log`** so it
  can load into the multi-element webapp page without clobbering its theme or
  page-level `<details>`/`<a>`/`<table>`. Font size moved to the `--alima-fs`
  custom property. GUI document chrome (page bg, scrollbars) stays in the
  scaffold.
- **WP12.2 — Event protocol + producer abstraction**: new Qt-free
  `src/core/render_events.py` (event builders 1:1 with the JS funcs +
  `RenderTransport` protocol + `MockTransport`); new `src/ui/render_transport.py`
  (`WebLogViewTransport`). `UnifiedMessageRenderer` now emits JSON render events
  to an injected transport instead of calling `WebLogView` directly; historical
  callers passing a `WebLogView` are auto-wrapped (back-compat, no call-site
  change). Events are append-only + idempotent per id; `block` events carry a
  semantic `kind` so Tier-3 frontends can drop GUI-only chrome (`proposal`).
- **WP12.3 — Webapp consumes shared chrome**: the webapp drives the *same*
  `UnifiedMessageRenderer` producer headless via a per-session
  `WebSocketRenderTransport`; events are buffered on the `Session` (monotonic
  `seq`) and broadcast over the WS (`render_events` field on `status`/`complete`,
  full replay on reconnect via a per-connection cursor; polling cursor for the
  fallback). `app.js` dispatches them into a `#log` region in the results panel
  via the shared funcs, deduping by `seq`. The webapp **keeps its 5-step widget**
  (WP9 Tier-3) and only adopts the DK/GND result-card chrome.
- **WP12.4 — Consolidation**: DK/GND card HTML is now produced by shared
  `PipelineResultFormatter.format_dk_search_card_html` /
  `format_dk_classifications_card_html`, called by **both** the GUI panel
  (`pipeline_chat_panel.py`) and the webapp — one maintenance location. No
  duplicate chrome CSS to remove (the `#log` scoping is non-overlapping with the
  webapp's `.classification-*` summary cards, which are kept).
- **Follow-up (reverse port)**: the webapp's nicer **structured DK/RVK badge
  cards** were lifted into the shared layer — new
  `PipelineResultFormatter.normalize_classifications` +
  `format_classification_badge_card_html`, with the `.classification-*` CSS
  ported into `alima_render.css` (scoped `#log`, recoloured for the dark
  surface). `format_dk_classifications_card_html` (GUI agentic-chat log + webapp
  `#log`) now renders the badge card with system badges (DK/RVK), RVK
  validation badges (standard / nicht standard / API-Fehler), a hit-count
  confidence badge, and per-code catalog titles. The Pipeline-Tab keeps its own
  `format_dk_classifications_html` confidence card (untouched; `test_pipeline_utils`
  green). This is the symmetry payoff of WP12: the GUI being a QWebEngineView
  means webapp render components flow back into it through the same shared layer.

Tests: `tests/test_unified_message_renderer.py` gains `MockTransport`
event-emission + `WebLogViewTransport`-mapping classes; new
`tests/test_webapp_render_events.py` covers the session buffer, cursors,
headless producer, and an end-to-end WS broadcast + reconnect-replay
(`fastapi.testclient`). Full suite: 556 passed, 5 skipped.

**Caveats (conservative self-assessment).** Verified via headless tests
(`QT_QPA_PLATFORM=offscreen`, `TestClient`) and JS `node --check` — **not**
visually confirmed in a running GUI or browser. The webapp now shows DK/GND
classifications in both its compact summary panel **and** the new shared `#log`
cards (complementary, like the GUI, but not yet de-duplicated by an operator UX
review). Streaming/assistant/collapsible events are wired on the webapp client
but only exercised in the classic pipeline's DK/GND path server-side; the
agentic tool-bus chrome is not emitted to the webapp.

### Chat/log rendering moved to QWebEngineView — reliable collapse + live streaming (June 9, 2026)

The chat/pipeline log rendered everything into a single `QTextBrowser` via
`QTextCursor` surgery (`UnifiedMessageRenderer`). Two regressions followed the
June 8 "declutter" change: (1) collapsible blocks were unreliable — "once
expanded, won't close" — because `_rerender_tool_call_block` re-rendered a block
in place by `setUserState` marker, which broke when the expanded body spanned
more than one `QTextBlock` or when concurrent streaming shifted block positions;
(2) intermediate LLM reasoning no longer streamed live.

**Redesign** (operator chose QWebEngineView; collapse-first):
- New `src/ui/web_log_view.py` — `WebLogView(QWidget)` wrapping a `QWebEngineView`.
  Collapsible blocks are native `<details>/<summary>` (toggle is 100% browser-side
  → no Python re-render, reliable even mid-stream). Streaming appends text nodes to
  an isolated `<div>`; markdown is rendered once on finalize. JS calls are queued
  until `loadFinished`; link clicks (`mutation://`, `http(s)://`) route back via
  `acceptNavigationRequest` → `link_clicked` (replaces `QTextBrowser.anchorClicked`).
- `UnifiedMessageRenderer` keeps its public API + `history` contract; internals now
  emit HTML strings into the `WebLogView` instead of cursor surgery. Deleted the
  cursor machinery (`_rerender_tool_call_block`, `_tool_call_blocks`, `setUserState`);
  `toggle_tool_call` is now a server-side mirror only.
- Panel + both mini-logs (`pipeline_chat_panel.py`, `analysis_review_tab.py`,
  `image_analysis_tab.py`) construct `WebLogView` instead of `QTextBrowser`.
- **Import-order constraint:** `QtWebEngineWidgets` must be imported before the
  `QApplication` — explicit early import added to `alima_gui.py`.

**Backend streaming-with-tools** (`llm_service.py`, partial P-δ.5/#7): Anthropic
`_generate_anthropic_with_tools` now uses `messages.stream()` + `get_final_message()`
to stream text deltas when a `stream_callback` is set (Ollama/OpenAI already did);
Gemini still completes-then-delivers (noted in-code).

**Live LLM stream → collapsible block** (follow-up): the flat inline streaming
line is replaced by an expanded `<details>` block. `start_streaming_line` opens it
open, `render_streaming_token` appends to its body live, `end_streaming_line`
collapses it and writes a one-line text preview into the summary. Both classic
(`step_id=""`) and agentic (`step_id="agentic"`) LLM output already route through
these three methods (`workers.py` → `on_llm_stream_token` → panel), so streamed
content — including the agent's initial keywords — is now visible live and then
folded away with a preview, consistent with the deterministic step summaries.
Caveat: agentic prose still passes `_AgenticStreamFilter` (raw-JSON suppression,
off when `ChatConfig.agentic_verbose`); content emitted as tool-call JSON rather
than prose is still filtered.

**Dependency:** `PyQt6-WebEngine==6.10.0` (+ `PyQt6-WebEngine-Qt6==6.10.2`) added to
`requirements.txt` — pulls in a Chromium runtime.

**Tests:** `test_unified_message_renderer.py` rewritten against a mock `WebLogView`
(captured HTML strings) — native collapse means the body is always in the DOM and
toggling is a mirror. Suite bootstrap (`tests/__init__.py` + `tests/conftest.py`)
imports WebEngine before any `QApplication`, creates the app with a non-empty argv,
and swaps a lightweight `WebLogView` stub so headless Chromium isn't constructed in
unit tests. **531 passed, 5 skipped.**

**Caveat (per self-assessment rules):** verified that native `<details>` toggling is
reliable while streaming (expand → re-close → re-expand, stream intact) and that the
suite is green — this does not prove correctness across all providers/inputs. Markdown
is still rendered post-stream (unchanged). The QWebEngine route adds a heavyweight
Chromium dependency and three render processes in the running app.

### Unified DK/GND result rendering + agentic-log declutter (June 8, 2026)

Commit `0cfba1a`. Pipeline-Tab and the agentic chat panel rendered the same
pipeline data differently (catalog research, final DK/RVK notations, GND hits).
Root cause: divergent ad-hoc formatters per surface. Consolidated into shared
formatters and fixed several agentic-mode display bugs.

**Shared formatters** (`src/utils/pipeline_utils.py` → `PipelineResultFormatter`,
single source of truth, pure-Python, unit-tested):
- `format_dk_classifications_html` (HTML fragment, confidence colours + title list),
  `format_dk_search_results_text`, `split_classification_code`,
  `get_titles_for_dk_code`.
- `select_dk_title_source` — picks the title-carrying source regardless of mode
  (classic stores the rich list in `dk_search_results_flattened`, agentic in
  `dk_search_results`; the other field is keyword-centric / thin). **This field
  inversion between modes is the recurring trap behind the agentic display bugs.**
- `flatten_gnd_hits` (dict / List[SearchResult] / flat `gnd_entries` → dedup rows),
  `extract_selected_gnd_keys` (final keywords → gnd-id + label sets).

**Fixes**:
- Agentic completion (`pipeline_tab._sync_classical_tabs_from_state`) cleared the
  Katalog-Recherche view and dropped titles on final notations — now uses
  `select_dk_title_source`.
- GND-Recherche tab: flat text → sortable `QTableWidget` (Begriff / GND-ID /
  Häufigkeit / Auswahl) + "nur ausgewählte" filter; completion no longer collapses
  to bare search terms (`_populate_gnd_hits` / `_render_gnd_hits_table` / `_filter_gnd_hits`).

**Agentic GUI polish**:
- Input prompt → collapsible, timestamped 📥 block via
  `UnifiedMessageRenderer.render_collapsible` + `state.pipeline_prompt` /
  `state.pipeline_prompt_done` bus events (emitted in `llm_agent_step._emit_prompts`
  / `_emit_prompt_done`, reflection tagged `kind="reflection"` → 🔍). Prompt no
  longer streamed inline (killed the duplicate dump). Added `render_html_block`.
- Decluttered the agentic log: compact MetaAgent/LLMAgent banners, hidden empty
  `[]` stream tag, dropped duplicate "Pipeline gestartet".

**Open follow-ups / findings** (not yet done):
1. **Agentic GND `Häufigkeit` column = 0** — the agentic `search_results` structure
   (`SharedContext.to_analysis_state`) carries no per-entry count; thread it through
   `gnd_entries` to populate the column.
2. **GND "only free keywords" — cache-vs-live hypothesis unverified**: the display
   fix is done, but whether the mapping-first cache narrows results to the exact
   GND mapping (vs the broad live Lobid aggregation) needs a runtime check.
3. **"LLM Antwort:" prefix on agentic orchestration**: orchestration text and the
   real LLM response share one streaming line / step_id `agentic`, so orchestration
   inherits the misleading prefix. Clean separation (orchestration as discrete log
   lines) needs a small stream-routing refactor.
4. **Duplicate selection logic** in `analysis_review_tab.py:~595-639`
   (`_split_classification_code` + title lookup) — consolidate onto the shared
   `PipelineResultFormatter` helpers.
5. **GUI runtime verification** — all changes are unit-tested (499 green) but not
   GUI-verified end-to-end; confirm in the running app.

### Chat-Agent P-η + P-θ: Input-Beschaffung + Export & Reporting (May 26, 2026)

Closes both open chat-agent roadmap phases (`docs/chat_agent_roadmap.md`).
The agent can now drive the full DOI/URL/PDF/Image → Pipeline → Export/Report
workflow without operator GUI interaction.

**New helper modules** (`src/utils/`, pure-Python, no Qt):
- `pdf_extractor.py` — PyPDF2 text extraction + quality heuristic
  (`_assess_text_quality`) + optional Vision-LLM OCR fallback via pdf2image.
  Extracted from `unified_input_widget.py:93-158`.
- `image_analyzer.py` — sync wrapper over `LlmService.generate_response(image=...)`
  with generator coalescing. Default `DEFAULT_PROMPT` = OCR. Extracted from
  `ImageAnalysisWorker`.
- `exporters.py` — `export_json/csv/tex/marc` + `load_state('latest'|file|abspath)`
  + `default_output_path`. Reuses `webapp.result_serialization.build_export_payload`
  as JSON schema source. K10+/WinIBW tags (5550/6700) via `generate_k10plus_lines`.
- `report_renderer.py` + `report_templates/{ub_freiberg,short}.tex.j2` — Jinja2 LaTeX
  with custom delimiters `(((  )))` / `((* *))` to avoid LaTeX brace collision.
  Optional pdflatex two-pass build; missing binary is non-fatal.

**New MCP tools** (`src/mcp/`):
- `read_pdf(path, max_chars, ocr_fallback, provider, model)`
- `analyze_image(path, prompt, provider, model, temperature)`
- `export_results(source, format, output_path, validate_rvk)`
- `generate_report(source, template, output_path, build_pdf)`
- `scrape_url` extended with Content-Type / .pdf-suffix auto-detect →
  temp download → `pdf_extractor.extract_text`.

**ToolRegistry**: gains optional `llm_service` constructor arg; `_get_llm_service()`
lazy-inits from config if not injected. New `export` tool-set + `EXPORT_TOOLS` list
in `tool_schemas.py`.

**Tests**: `tests/test_input_export_tools.py` (30 tests, all pass) covers
extractor, analyzer, all 4 exporter formats, both templates, MCP dispatch +
scrape PDF branch.

**Doku**: `docs/chat_agent_roadmap.md` (P-η/P-θ marked done, tool matrix updated),
`src/mcp/CLAUDE.md` + `src/utils/CLAUDE.md` mention new modules. Plan file:
`~/.claude/plans/p-input-beschaffung-immutable-spring.md`.

**Operator decisions** baked in: kept `resolve_doi` name (no rename to
`fetch_doi_metadata`); Jinja2 + `paper/`-style templates for report; e-mail
delivery deliberately deferred.

### P-η: Provider-Variants + Seed-Retrofit (May 18, 2026)

WP10 Foundation Phase 3/3. Closes the agentic reproducibility blocker
(WP2 Sek 3) and seeds the family-aware prompt-routing.
Pre-tag: `wp10-pη-pre`.

**Seed Retrofit** (WP11 Sek 8 — 7+1 sites):
- `LlmService.generate_with_tools()` gains `seed: Optional[int] = None`.
  Dispatch forwards seed to all sub-handlers except Anthropic.
- `_generate_ollama_native_with_tools`, `_generate_openai_with_tools`,
  `_generate_gemini_with_tools`, `_generate_text_fallback_with_tools`
  accept seed and propagate to provider API.
- `_generate_anthropic_with_tools` **deliberately skipped** —
  Anthropic SDK has no `seed` parameter and operator config is empty.
  Dispatch omits seed entirely when routing to Anthropic; text-path
  Anthropic seed setting at `llm_service.py:1851` is unchanged
  (silently ignored by SDK). See operator decision in P-η plan.
- `AgentLoop.run()` gains seed param; forwards to both main and
  force-final `generate_with_tools()` calls.
- `BaseSharedContext` + `SharedContext` add `seed` field with
  serde symmetry in `to_dict`/`from_dict`.
- `LLMAgentStep._llm_params()` resolves
  `step.llm.seed > context.seed > None` and forwards via
  `_invoke_loop()` to `AgentLoop.run(seed=...)`.
- `shared_context.py` 3 hardcoded `seed=None` in `LlmKeywordAnalysis`
  factories replaced with `seed=self.seed`.

**Workflow YAML seed schema** (Track C):
- All 6 workflows (`alima_classic`, `alima`, `catalog_search`,
  `synonym_expansion`, `title_list_search`, `batch_metadata`) gain
  optional `settings.seed: null` field.
- `WorkflowExecutor.run()` propagates `settings.seed` to
  `context.seed` when the latter is unset (caller wins otherwise).
- Per-step override remains via `steps[].llm.seed`.

**Capability YAML** (WP11 Sek 3, Track A):
- New file: `config/model_capabilities.yaml` covering 3 providers
  (openai_compatible, ollama, gemini) × 12 model patterns × 10 flags
  (json_mode, tool_use, vision, max_context_tokens, seed_support,
  streaming, thinking_tokens, parallel_tool_calls, system_prompt,
  family). Anthropic excluded by operator decision.
- New helpers in `src/utils/model_capabilities.py`:
  `load_capabilities_yaml(path)`, `get_capability(provider, model, flag,
  default)`, `reset_capability_cache()`. 3-tier lookup: exact →
  fnmatch wildcard → caller default. Cached per-path.
- Existing `KNOWN_CAPABILITIES` regex registry untouched (chunking
  threshold lookup unaffected).

**Prompt Variants** (WP11 Sek 5, Track D):
- 9 new family-specific variants added to `prompts.json`:
  - `keywords` × {thinking, instruct-open, openai-chat} (+3)
  - `dk_classification` × {thinking, instruct-open, openai-chat} (+3)
  - `initialisation` × {thinking, instruct-open} (+2)
  - `dk_list` × instruct-open (+1, on top of existing 2)
- All existing 5-tuple variants canonicalized to 6-tuple with
  `seed="0"`. PromptService 3-tier selector unchanged.
- Backup at `prompts.json.pre-pη.bak`.

**Tests** (Track E, +22 tests):
- New `tests/test_llm_service_seed.py` (12 tests): handler dispatch,
  ollama options pass-through, AgentLoop forward, SharedContext
  roundtrip.
- New `tests/test_model_capabilities_yaml.py` (10 tests): YAML load,
  3-tier resolution, default fallback, shipped-YAML smoke.
- `tests/test_agents_v2.py` +4: settings/context/step seed resolution.
- Full suite: 188 passed / 6 pre-existing failures in
  `test_pipeline_utils.py` (unrelated to P-η, verified via stash).

**Verification**:
- 22 new tests green; 0 regressions.
- PromptService picks correct family variant for `llama3.1:8b`
  (instruct-open), `qwen2.5:32b` (thinking), `gpt-4o-mini` (openai-chat),
  `exotic-model:1b` (default fallback).
- End-to-end seed reproducibility smoke test deferred to manual run
  (requires Ollama runtime).

**Out of scope**: Anthropic family + claude variants, test matrix
(WP11 Sek 9), per-step provider-mix UI (WP11 Sek 10),
`KNOWN_CAPABILITIES` → YAML migration of existing consumers.

**Next phase**: P-γ — SingleStepDialog (4 PT, first user-visible win).

### v4 Agent Workflow System (April 22, 2026)
- **Replaces MetaAgent + SubAgents**: The hardcoded 4-SubAgent pipeline (`KeywordExtractionAgent`, `SearchAgent`, `KeywordSelectionAgent`, `ClassificationAgent`) was deleted. Agent dispatch now runs through the generic v4 `WorkflowExecutor`.
- **Plan**: Option B from Agent-System-Restructuring plan — Generic LLMAgentStep + DeterministicStep + plugin registry.
- **Phase 1-2 (Foundation + Migration)**:
  - New files: `registry.py`, `workflow_loader.py`, `workflow_executor.py`, `context_path.py`, `steps/{base_step,llm_agent_step,deterministic_step}.py`, `deterministic_functions.py`
  - `SharedContext.extra: Dict` added for non-ALIMA fields + `${steps.X.Y}` / `${extra.Y}` context-path resolver
  - `workflows/alima_classic.yaml` reproduces the classic 4-step pipeline in v4 schema
- **Phase 3 (PoC workflows)**:
  - `workflows/catalog_search.yaml` — multi-source catalog lookup (SWB + Lobid + catalog) with optional LLM ranking
  - `workflows/synonym_expansion.yaml` — single keyword → GND entry → LLM expansion → validated GND candidates
  - `workflows/batch_metadata.yaml` — bulk GND-ID metadata fetch with optional Lobid fallback
- **Phase 4 (CLI/GUI integration)**:
  - New CLI: `alima workflow <name> [--input|--input-file|--output|--only-step]` + `alima workflows list`
  - `PipelineConfigDialog` gained a workflow-selection `QComboBox` populated from discovered v4 YAMLs
  - Fixed pre-existing argparse conflict: CLI `--step` (provider override, `append`) vs single-step agentic `--step`; renamed the second to `--only-step`
- **Phase 5 (cleanup)**:
  - Deleted: `meta_agent.py`, `base_sub_agent.py`, `keyword_extraction_agent.py`, `search_agent.py`, `keyword_selection_agent.py`, `classification_agent.py`
  - Archived: `workflows/{meta_agent_default,default_alima,extended,minimal}.yaml` → `workflows/legacy/` (no longer discovered)
  - Removed MetaAgent fallback branch from `PipelineManager._start_agentic_pipeline()`
  - Default `workflow_name` changed from `meta_agent_default` → `alima_classic`
  - `tests/test_agents.py` reduced to `SharedContext` + `ToolResultCache` + `CachingToolRegistry` coverage; MetaAgent/SubAgent tests removed (replacement coverage in `tests/test_agents_v2.py`, 59 tests total)
- **Kept unchanged**: `CachingToolRegistry`, MCP tool layer, `agent_loop.py`, `LlmService`, rigid `pipeline_utils.py` path

### WebApp Auto-Save & Recovery System (January 6, 2026)
- **Complete reliability upgrade** for long-running pipeline analyses in web interface
- **Auto-Save Infrastructure**: Incremental JSON saving after each pipeline step
  - Auto-save directory: `/tmp/alima_webapp_autosave/` with session-specific files
  - Metadata tracking: session_id, timestamp, last_step, status
  - Uses existing `PipelineJsonManager` for consistent serialization
- **Extended WebSocket Timeout**: Increased from 5 minutes to 30 minutes
  - Heartbeat mechanism: Sends heartbeat every 5 seconds to maintain connection
  - Prevents timeout during long DK searches (100+ keywords)
  - Frontend filters heartbeat messages (no console spam)
- **Recovery Mechanism**: Complete result restoration after connection loss
  - New API endpoint: `GET /api/session/{id}/recover`
  - Auto-detection of WebSocket errors (code 1006, 1011)
  - Recovery UI: Orange "🔄 Ergebnisse wiederherstellen" button with status messages
  - Full result reconstruction using shared `_extract_results_from_analysis_state()` helper
- **Auto-Cleanup**: Automatic deletion of old auto-save files (>24h) on webapp startup
- **Progress Enhancement**: DK search now shows percentage progress `[idx/total] (pct%)`
- **Code Quality**: DRY principle - shared result extraction logic between callback and recovery
- **Backward Compatibility**: Old sessions without auto-save continue to work
- **Files Modified**:
  - `src/webapp/app.py`: +4 functions, +1 endpoint, auto-save infrastructure
  - `src/webapp/static/index.html`: Recovery button + message span
  - `src/webapp/static/app.js`: +2 recovery functions, WebSocket handler enhancements
  - `src/utils/pipeline_utils.py`: Percentage display in DK search
  - `src/webapp/CLAUDE.md`: Documentation update

### DK Deduplication Statistics Display (January 2026)
- **Phase 2 Complete**: Comprehensive statistics visualization for DK classification deduplication
- **CLI Statistics Display**: New `format_dk_statistics()` in `show-protocol` detailed mode
  - Shows deduplication metrics: original→deduplicated count, duplicates removed, rate, token savings
  - Top 10 most frequent classifications with keyword provenance and title counts
  - Keyword coverage summary showing keywords→DK codes mapping
- **GUI Statistics Tab**: New "📊 DK-Statistik" tab in AnalysisReviewTab (index 9)
  - Deduplication Summary box with 5 key metrics
  - Top 10 table with rank, DK code, type, count, keywords, and color-coded confidence
  - Keyword Coverage table showing keyword→DK codes relationships
  - Color-coded confidence indicators: Green (>50 titles), Teal (>20), Yellow (>5), Red (<5)
- **Critical Bug Fixes**:
  - Fixed `dk_statistics` not being loaded from JSON in CLI display functions (3 locations)
  - Fixed incorrect tab navigation indices in GUI `on_step_selected()` method
  - Added missing navigation cases: chunk_details, k10plus, dk_statistics
- **Backward Compatibility**: Old JSON files without statistics handled gracefully with fallback messages
- **Files Modified**: `src/alima_cli.py`, `src/ui/analysis_review_tab.py`, `CLAUDE.md`

## 2025

### Unified Database Configuration (November 2025)
- Eliminated duplicate `SystemConfig.database_path` + `DatabaseConfig.sqlite_path` → single source of truth
- Implemented OS-specific default paths (Windows, macOS, Linux) via `get_default_db_path()`
- Singleton pattern for UnifiedKnowledgeManager with thread-safe `__new__()` override
- Automatic backward compatibility migration for old configs
- All 12 UnifiedKnowledgeManager instantiations now use singleton automatically

### K10+/WinIBW Catalog Export (October 2025)
- Direct export in K10+/WinIBW format for seamless catalog integration
- GUI: New "K10+ Export" Tab with Copy-Button
- CLI: `--format k10plus` for direct Copy-Paste
- Configuration: K10PLUS_KEYWORD_TAG, K10PLUS_CLASSIFICATION_TAG

### DK Classification Transparency (October 2025)
- Automatic display of which catalog titles led to each DK classification
- GUI: PipelineStreamWidget shows sample titles during DK search, AnalysisReviewTab with color coding
- CLI: show-protocol with DK titles in detailed/compact/k10plus format

### Protocol Display CLI Command (October 2025)
- `show-protocol` command for displaying pipeline results from JSON files
- Three modes: `--format detailed` (readable), `--format compact` (CSV), `--format k10plus` (catalog export)

### Batch Processing System (August 2025)
- Complete batch processing engine using PipelineManager for full pipeline execution
- ALL Source Types Supported: DOI (via doi_resolver), PDF (PyPDF2 + LLM-OCR fallback), TXT, IMG (vision model), URL (BeautifulSoup4)
- Batch Review UI: Toggle mode for batch overview vs. detail view, table with Status/Source/Keywords/Date/Actions
- Continue-on-error vs. stop-on-error modes with detailed error reporting
- Resume functionality for interrupted batches via JSON persistence
- Pipeline configuration inheritance from global settings

### Unified Logging System (August 2025)
- Central logging infrastructure with 4-level verbosity system (0=Quiet, 1=Normal, 2=Debug, 3=Verbose)
- CLI: `--log-level` argument (0-3, default=1)
- GUI: Uses level 1 (Normal) by default
- Setup function: `setup_logging(level)` with automatic third-party suppression
- Result output respecting quiet mode: `print_result()` function

### Three-Mode CLI System (July 2025)
- Smart Mode: Uses task preferences from config.json automatically
- Advanced Mode: Manual provider|model override with `|` separator
- Expert Mode: Full parameter control (temperature, top-p, seed)

### Vertical Pipeline UI (June 2025)
- Chat-like vertical workflow with 5 pipeline steps
- Visual status indicators: ▷ (Pending), ▶ (Running), ✓ (Completed), ✗ (Error)
- Auto-Pipeline button for one-click complete analysis
- Integrated input tabs (DOI, Image, PDF, Text) in first step
- Real-time result display in each step
- Direct integration with PipelineManager for workflow orchestration

### Global Status Bar (June 2025)
- Unified provider information display across all tabs
- Real-time cache statistics (entries count, database size)
- Pipeline progress tracking with color-coded status
- Auto-updating every 5 seconds for live monitoring
- Integration with LlmService and CacheManager

### Pipeline Manager (May 2025)
- Orchestrates complete ALIMA workflow using existing AlimaManager logic
- 5-step pipeline: Input → Keywords → Search → Verification → Classification
- Uses proven `KeywordAnalysisState` for data management
- UI callback system for real-time progress updates
- Auto-advance functionality for seamless workflow
- Refactored to use shared `PipelineStepExecutor` from utils

### Automated Data Flow (May 2025)
- AbstractTab automatically sends results to AnalysisReviewTab
- New `analysis_completed` signal in AbstractTab
- `receive_analysis_data()` method in AnalysisReviewTab
- Seamless workflow progression without manual data transfer
