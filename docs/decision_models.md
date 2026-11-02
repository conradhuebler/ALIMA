# Entscheidungsmodelle (Ollama `/v1/systemone`)

Ein Entscheidungsmodell erzeugt keinen Text. Es bekommt einen `state` und benannte,
typisierte Fragen und liefert je Frage eine Wahrscheinlichkeit aus einem
Forward-Pass. ALIMA nutzt das an einer Stelle: den Vorfilter für große
GND-Kandidatenlisten. Alles hier ist **opt-in**: ohne ein gesetztes
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
  dreimal wiederholt, `decision_max_parallel` gleichzeitige Anfragen.
- **Vorfilter:** `src/utils/decision_prefilter.py`. Je Kandidat eine deutsche
  `noul`-Frage („Ist das Schlagwort … ein geeignetes Sacherschließungs-Schlagwort
  für das Dokument im State?"), die `top_k` besten bleiben in ursprünglicher
  Reihenfolge, Gleichstand entscheidet die ursprüngliche Position. Der State ist
  auf 1500 Zeichen gekürzt (der Endpunkt zählt ihn für jede Frage mit).
- **Agentisch:** Block `decision_prefilter` unter `chunking:` des Schritts
  `selection_chunks` (`enabled`, `top_k`, `text_field`). Die fünf mitgelieferten
  Workflows mit diesem Schritt tragen `enabled: true, top_k: 80`; wirksam wird er
  erst mit gesetzter Rolle.
- **Klassisch (GUI, CLI, Batch):** greift im Schlüsselwort-Schritt
  (`execute_final_keyword_analysis`), wenn die Rolle gesetzt ist **und** der Pool
  ohnehin gechunkt würde, mit `top_k` 80.
- **Fallback:** Rolle aus, Pool nicht größer als `top_k`, Host nicht erreichbar
  oder Fehlerantwort: die Liste bleibt unverändert, der Lauf meldet es im Stream
  („Weiter mit der bisherigen Auswahl") und im Log.
- **Messprotokoll:** `decision_prefilter` im Ergebnis (`KeywordAnalysisState`, geht
  mit `asdict` in die JSON-Ausgabe): Anbieter, Modell, `n_in`/`n_out`, Sekunden,
  `state_chars`, `max_parallel` und der Score **jedes** Kandidaten. Damit lässt
  sich der Recall gegen die späteren `final_keywords` aus der Datei nachrechnen.

## Grenzen (nicht vorhanden)

- Kein Bedienelement für `decision_max_parallel`: nur `config.json`. Standard 1.
- Klassischer Pfad: `top_k` 80 fest; der Parameter `decision_prefilter_top_k`
  (0 schaltet ab) existiert im Executor, hat aber keinen CLI- oder GUI-Schalter.
- Die Webapp-Ergebnisansicht (`webapp/result_serialization.py`) übernimmt das
  Feld `decision_prefilter` nicht; in der JSON-Datei des Laufs steht es.
- Nur Ollama-Anbieter. Kein Anschluss an den MetaAgent-Planer, keine Bild- oder
  Audio-Eingabe, kein weiterer Anwendungsfall als der Vorfilter.
- Kein Abbruch im klassischen Pfad (`should_stop` ist nur agentisch verdrahtet).

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
