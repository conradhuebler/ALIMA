# Entscheidungsmodelle (Ollama `/v1/systemone`)

Ein Entscheidungsmodell erzeugt keinen Text. Es bekommt einen `state` und benannte,
typisierte Fragen und liefert je Frage eine Wahrscheinlichkeit aus einem
Forward-Pass. ALIMA nutzt das in der Pipeline an einer Stelle (Vorfilter für große
GND-Kandidatenlisten) und als Werkzeug des Agenten (Abschnitt „Agent-Werkzeuge“). Alles hier ist **opt-in**: ohne ein gesetztes
Entscheidungsmodell läuft jeder Pfad unverändert.

## Was es gibt

- **Rolle in den Einstellungen:** vierte Zeile der Tabelle unter Sprachmodelle,
  „Entscheidungsmodell (optional)". Felder `decision_default_provider/model` in
  `unified_config`. Anders als Allgemein/Pipeline/Bild **ohne Rückfall**: ein
  leerer, abgeschalteter oder modellloser Eintrag heißt „aus", nie „erstes Modell
  des ersten Anbieters" (`ProviderScope.DECISION`, `config_models.py`).
- **Modellauswahl:** nur Ollama-Anbieter, nur Modelle, deren `/api/tags`-Eintrag die
  Fähigkeit `decision` trägt (`utils/decision_client.list_decision_models`,
  `ui/settings/decision_models.py`). `/v1/models` unterscheidet nicht und listet
  Entscheidungsmodelle neben Chat-Modellen; `/v1/chat/completions` lehnt sie ab.
- **Client:** `src/utils/decision_client.py`, Qt-frei, direkt per HTTP (der
  gepinnte `ollama==0.6.1` kennt `systemone` nicht). Batches zu höchstens 64
  Fragen (mehr ergibt HTTP 400), 429/503 und Verbindungsfehler werden bis zu
  dreimal wiederholt, `decision_max_parallel` gleichzeitige Anfragen. Der Host zählt
  den State bei lokalen Modellen für jede Frage mit; sprengt ein Batch den Kontext
  („Context size has been exceeded“), halbiert der Client ihn, bis er passt.
- **Vorfilter:** `src/utils/decision_prefilter.py`. Je Kandidat eine deutsche
  `noul`-Frage („Ist das Schlagwort … ein geeignetes Sacherschließungs-Schlagwort
  für das Dokument im State?"), die `top_k` besten bleiben in ursprünglicher
  Reihenfolge, Gleichstand entscheidet die ursprüngliche Position. Der State ist
  auf 1500 Zeichen gekürzt (der Endpunkt zählt ihn für jede Frage mit).
- **Agentisch:** Block `decision_prefilter` unter `chunking:` des Schritts
  `selection_chunks` (`enabled`, `top_k`, `text_field`, `min_pool`). Er läuft nur, wenn der Pool
  mehr als einen Chunk braucht (`min_pool`, Standard die Chunkgröße des Schritts): bei
  einem Chunk spart er keinen LLM-Aufruf. Bei einem Beispiel (Pool 619, `beam:cloud`)
  brauchte die Auswahl auf 80 Kandidaten 44 s statt 94 s auf allen, plus etwa 10 s Vorfilter. Die fünf mitgelieferten
  Workflows mit diesem Schritt tragen `enabled: true, top_k: 80`; wirksam wird er
  erst mit gesetzter Rolle.
- **Klassisch (GUI, CLI, Batch):** greift im Schlüsselwort-Schritt
  (`execute_final_keyword_analysis`), wenn die Rolle gesetzt ist **und** der Pool
  ohnehin gechunkt würde, mit `top_k` 80.
- **Schalter je Lauf (A/B):** In der GUI steht in der Werkzeugleiste des Pipeline-Tabs das
  Kontrollkästchen „mit Decision“ (nur aktiv, wenn ein Entscheidungsmodell eingerichtet ist),
  in der CLI `alima pipeline … --no-decision-prefilter`. Beides setzt
  `PipelineConfig.use_decision_prefilter` für diesen Lauf und ändert keine Einstellung; so
  läuft derselbe Text mit und ohne Vorfilter, in beiden Pfaden (agentisch über den Kontext,
  klassisch über `decision_prefilter_top_k=0`). Die Webapp hat keinen Schalter.
- **Fallback:** Rolle aus, Pool nicht größer als `top_k`, Host nicht erreichbar
  oder Fehlerantwort: die Liste bleibt unverändert, der Lauf meldet es im Stream
  („Weiter mit der bisherigen Auswahl") und im Log.
- **Messprotokoll:** `decision_prefilter` im Ergebnis (`KeywordAnalysisState`, geht
  mit `asdict` in die JSON-Ausgabe): Anbieter, Modell, `n_in`/`n_out`, Sekunden,
  `state_chars`, `input_tokens`, `max_parallel` und der Score **jedes** Kandidaten. Damit lässt
  sich der Recall gegen die späteren `final_keywords` aus der Datei nachrechnen.

## Agent-Werkzeuge

Mit gesetzter Rolle erscheinen im MCP-Katalog (`src/mcp/tool_registry.py`) zwei
Werkzeuge; ohne Rolle gibt es sie nicht (`ToolRegistry.refresh()` wertet das nach
einer Änderung der Einstellungen neu aus). Der Chat in GUI, CLI und HTTP bekommt
sie über den MCP-Adapter; Workflow-Schritte nehmen sie mit `tools: {preset: decision}`.
Logik in `src/utils/decision_tools.py`.

- **`decide(state, questions)`**: typisierte Fragen (`noul`, `choice`) an einen
  Text oder ein JSON-Objekt, Rückgabe sind die Wahrscheinlichkeiten des Modells.
  Höchstens 64 Fragen je Aufruf, State höchstens 6000 Zeichen (er wird je Frage
  mitgezählt). Eingaben werden vor dem Aufruf geprüft und mit lesbarer Meldung
  abgewiesen.
- **`rank_by_relevance(text, candidates, top_k)`**: Kandidatenliste (Texte oder
  Objekte mit `keyword`/`title`, Zusatzfelder wie `gnd_id` bleiben erhalten) nach
  Passung zum Text ordnen, die `top_k` besten (Standard 30, höchstens 200) mit
  Score; höchstens 3000 Kandidaten. Statt `candidates` kann `from_search_terms`
  (die Suchbegriffe der vorherigen `search_*`-Aufrufe) angegeben werden: der Pool
  kommt dann aus `aggregate_gnd_results` (Rohantwort-Cache), und der Agent muss ihn
  nicht abtippen; mit `balanced` nimmt es die besten reihum je Suchbegriff statt global
  (Zeilen tragen dann `terms`), damit Facetten nicht von Allgemeinbegriffen verdrängt
  werden; `question_variant` (`geeignet` als Voreinstellung, `behandelt`, `spezifisch`)
  wählt den Fragetext. Beides sind Versuchsparameter ohne Messnachweis; die Pipeline
  nutzt weiter `geeignet` und die global besten. Das Ergebnis trägt `keyword`, `gnd_id` und `count`. Die Katalog-Häufigkeit
  (`display_count`) steht bewusst nicht darin: `select_from_gnd_pool` sortiert nach
  `count` und bevorzugt hohe Werte, ein dorthin kopierter Häufigkeitswert stellt
  Allgemeinbegriffe wie „Krankheit“ nach vorn. Pool-Einträge, deren Bezeichnung nur die GND-ID ist, werden
  nicht bewertet und als `skipped_unlabeled` gezählt. Das Ergebnis nennt außerdem
  `input_tokens`, die der Host gezählt hat (bei bezahlten Anbietern der Aufwand). Der Chat-Systemprompt nennt das Werkzeug als
  Schritt vor `select_from_gnd_pool`.
- **Pipeline starten:** `run_pipeline` und `execute_workflow` brauchen nichts davon;
  mit gesetzter Rolle kürzt der Vorfilter große Pools selbst. `get_workflow` zeigt
  dafür je Schritt `decision_prefilter {top_k, active}`; die Beschreibungen beider
  Werkzeuge und der Chat-Systemprompt sagen es dem Agenten.

## Blockgröße und Parallelität (Messung vom 4.11.2026)

Ein Werk, 128 Kandidaten, Text 1500 Zeichen, Fragetext `geeignet`, ein Lauf je Zelle, Sekunden bei
einer Anfrage zur Zeit (`scripts/bench_decision_batching.py`):

| Fragen je Anfrage | LLMachine d1 | LLMachine d3 | Cloud clef | Laptop d1 |
|---|---|---|---|---|
| 1 | 7,2 | 13,6 | 66,1 | 46,5 |
| 8 | 2,6 | 5,3 | 10,5 | 18,3 |
| 16 | 2,4 | 6,2 | 6,3 | 16,9 |
| 32 | 2,3 | 5,8 | 3,3 | 15,8 |
| 64 | 2,2 | 5,7 | 1,8 | 22,7 |

- Größere Blöcke sind schneller; bei `d1` flach ab etwa 16, bei der Cloud bis 64 (36-fach schneller als 1
  Frage je Anfrage), auf dem Laptop am besten bei 32. Gleichzeitige Anfragen (2 oder 4) brachten bei
  Blöcken ab 16 keinen Gewinn und auf dem Laptop eine Verschlechterung.
- Tokens: Lokal (`d1`, `d3`) zählt der Text je Frage, die Summe ist von der Blockgröße unabhängig
  (53 371 für `d1`). Die Cloud zählt den Text je Anfrage: 64 708 bei 1 Frage, 17 298 bei 64 Fragen.
- **Cloud-Scores hängen von der Nachbarschaft in der Anfrage ab:** gegenüber Block 64 weichen sie bei
  kleineren Blöcken bis 0,38 ab; dieselben Kandidaten in umgekehrter Reihenfolge ändern sie um bis zu
  0,29 (Top-40-Überlappung 36 von 40), dieselbe Reihenfolge wiederholt nur um 0,03. Bei `d1` bleibt die
  Abweichung unter 0,07.
- Auf die Trefferquote (11 Soll-Begriffe im Pool, Schnitt 80) schlägt das nicht durch: `clef` Block 1
  10, Block 8 11, Block 64 11 Treffer; `d1` in allen Fällen 10. Block 64 bleibt deshalb die
  Voreinstellung. Ein Mittel aus zwei Reihenfolgen brachte nur bei Schnitt 40 einen Treffer mehr.
- Passt ein Block nicht in den Kontext des Modells ("Context size has been exceeded", `d3`), halbiert
  der Client ihn und merkt sich die Größe für den Rest des Laufs.
- Grenzen: ein Werk, eine Soll-Liste (vorläufig, nicht fachreferatlich), ein Lauf je Zelle. Die
  Rohdaten und das Protokoll stehen im Laborjournal "ALIMA Entscheidungsmodelle" des Obsidian-Vaults.

## Messskript

`scripts/eval_decision_rank.py` vergleicht Modelle und Fragetexte an einer Soll-Liste.
Pro Modell und Fragetext wird jeder Kandidat einmal bewertet und der Score in
`--out` gespeichert; Recall@K (`--ks`) und `balanced` rechnet das Skript danach aus
denselben Scores, ohne weitere Abfragen (`--reuse` rechnet gespeicherte Läufe neu aus).
Pool aus dem Rohantwort-Cache (`--from-cache --terms …`, nach den `search_*`-Aufrufen)
oder aus einer Datei im Format von `aggregate_gnd_results` (`--pool`). Ohne `--yes`
druckt es nur den Plan (Anzahl Anfragen) und fragt nichts an. Es liefert eine
Übereinstimmung mit der Soll-Liste je Dokument, keine allgemeine Aussage.

`scripts/bench_decision_batching.py` misst Blockgröße und Parallelität (Zeit, Anfragen, Tokens und
Abweichung der Scores gegenüber dem größten Block); `--max-requests` begrenzt die Last je Modell.
`scripts/eval_decision_rank.py` kennt `--batch-size`, `--save-pool` (Pool als Datei, unabhängig vom
24-Stunden-Rohantwort-Cache) und `--ignore-cache-age`.

## Grenzen (nicht vorhanden)

- Kein Bedienelement für `decision_max_parallel`: nur `config.json`. Standard 1.
- Klassischer Pfad: `top_k` 80 fest; der Parameter `decision_prefilter_top_k`
  (0 schaltet ab) existiert im Executor, hat aber keinen CLI- oder GUI-Schalter.
- Die Webapp-Ergebnisansicht (`webapp/result_serialization.py`) übernimmt das
  Feld `decision_prefilter` nicht; in der JSON-Datei des Laufs steht es.
- Nur Ollama-Anbieter. Kein Anschluss an den MetaAgent-Planer (der Planer wählt
  Schritte, er stellt keine Entscheidungsfragen; die Werkzeuge gehören dem Agenten), keine Bild- oder
  Audio-Eingabe, in der Pipeline kein weiterer Anwendungsfall als der Vorfilter.
- Kein Abbruch im klassischen Pfad (`should_stop` ist nur agentisch verdrahtet);
  die Agent-Werkzeuge sind ebenfalls nicht abbrechbar.
- Typ `score` (laut Modellseite bei `clef` vorhanden) wird nicht angeboten.

## Messwerte (Orientierung, kein Nachweis)

Ein Dokument (Cadmium), 979 Kandidaten, Referenz = die 20 `final_keywords` des
Pipeline-Laufs, aus dem auch der Pool stammt (also **kein** unabhängiger
Goldstandard). Zufall erwartet 0,6 / 1,6 / 4,1 Treffer bei Top 30 / 80 / 200.
Der Prompt wurde an demselben Dokument gewählt.

| Modell | Top 30 | Top 80 | Top 200 | Zeit je Dokument |
|---|---|---|---|---|
| `d1` | 16 | 18 | 20 | 12 bis 18 s (Server) |
| `d3` | 11 | 16 | 19 | 16 bis 19 s (Server) |
| `clef:27b-cloud` | 14 | 15 | 20 | 16 s, 130 000 Eingabetoken |
| `d1-omni:600m` | 0 bis 5 | 2 bis 10 | 5 bis 14 | 32 bis 48 s (Laptop) |

- Laptop (GPU) ca. 105 s für `d1`, ohne Gewinn durch Parallelität; Server
  (4 × RTX A4500) ca. 25 s seriell, ca. 14 s mit zwei gleichzeitigen Anfragen,
  vier bringen nichts mehr.
- Die Eingabetokens wachsen linear mit der Fragenzahl **und** der Länge des
  States (979 Fragen: 117 000 Token bei 165 Zeichen, 351 000 bei 1451 Zeichen).
  Bei bezahlten Anbietern zählt das.
- Weitere Dokumente werden in Praxisläufen über das Messprotokoll ausgewertet.
