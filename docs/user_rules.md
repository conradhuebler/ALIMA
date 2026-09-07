# Persönliche Zusatzregeln

Regeln, die der Operator selbst formuliert und die an die Systemprompts der
agentischen Schritte, des MetaAgent-Planers und des Chat-Agenten angehängt
werden. Sie liegen in der Benutzerkonfiguration, nicht im Repository.

Code: [`src/core/user_rules.py`](../src/core/user_rules.py).

## Ablage

`~/.config/alima/rules.yaml`, Geschwister von `config.json`
(`ConfigManager.rules_file`; unter Windows `%APPDATA%\ALIMA`, unter macOS
`~/Library/Application Support/ALIMA`). Die Umgebungsvariable
`ALIMA_RULES_FILE` überschreibt den Pfad; die Testsuite setzt sie, damit kein
Testlauf die echten Regeln des Operators liest.

```yaml
version: 1
rules:
  - id: r-20260903-01
    text: "Formschlagwörter gehören nicht in core_keywords."
    applies_when: "bei Lehrbüchern"
    scope:
      workflows: ["*"]
      steps: ["selection*"]
    enabled: true
    origin:
      source: chat          # chat | manual | cli
      created: "2026-09-03T14:12:00"
      author: ""
      session_id: "…"
      note: "aus dem Lauf zu <Titel>"
```

Geschrieben wird atomar (temporäre Datei im selben Verzeichnis, dann
`os.replace`). Eine fehlende Datei ist eine leere Regelmenge; eine kaputte wird
protokolliert und ebenfalls als leer behandelt — eine Regeldatei kann keinen
Lauf verhindern.

## Bedingung und Geltungsbereich

- **`applies_when` ist Prosa und wird nicht ausgewertet.** Sie wird der Regel als
  „Nur …: " vorangestellt; ob sie zutrifft, beurteilt das Modell. Es gibt keine
  Bedingungssprache. Fachliche Bedingungen wie Fachgebiet oder Dokumentart sind
  nichts, was ein Ausdruck über den `SharedContext` entscheiden könnte.
- **`scope` ist strukturell** (Glob-Muster). `workflows` prüft gegen
  `WorkflowDef.name`, `steps` gegen die Step-Id. Zusätzlich zu den Step-Ids der
  Workflows gibt es drei Pseudo-Ids: `planner`, `reflection`, `chat`.
  `*` heißt „überall". Der Geltungsbereich ist das Mittel gegen Tokenkosten:
  eine Klassifikationsregel hat im Extraktions-Prompt nichts zu suchen.
- **Die wählbaren Schritte kommen aus der Workflow-YAML**
  (`available_scope_steps`), nicht aus einer gepflegten Liste — eine Regel auf
  einer Step-Id, die es nicht gibt, greift nie, und nichts würde das melden.
  Dieselbe Liste speist den Regeldialog (Häkchen statt Freitext) und die
  `steps`-Beschreibung von `propose_rule`, damit das Modell die echten Ids kennt
  und nicht mangels Wissen auf `*` ausweicht.
- **Eine Regel über die fertige Ausgabe gehört zu `reflection`** — dem letzten
  LLM-Turn eines Laufs. Nur dort kann sie etwas erzeugen.
- Ein leerer Step-Name (ein Prompt, dessen Schritt sich nicht bestimmen lässt)
  erhält nur Regeln mit `steps: ["*"]`.

## Wo die Regeln landen

| Prompt | Stelle | Step-Id |
| --- | --- | --- |
| jeder agentische LLM-Step | `prompt_resolver.resolve_prompts` | die Step-Id aus der YAML |
| Reflexion | derselbe Weg über `ReflectionStep` | `reflection` |
| MetaAgent-Planer | `MetaAgent._append_user_rules` | `planner` |
| Chat (GUI, Webapp, `alima agent`) | `chat_prompts.build_system_prompt(user_rules=…)` | `chat` |

Der Block wird **nach** dem `{name}`-Rendering angehängt. Geschweifte Klammern
im Regeltext bleiben dadurch stehen und werden nie als Platzhalter gelesen.

Ohne passende aktive Regel ist der Block leer und jeder Prompt byte-identisch
wie ohne diese Funktion; ein Test hält das fest.

## Regeln aus dem Chat

Der Chat-Agent erkennt im Gespräch, wenn eine Aussage über den Einzelfall hinaus
gelten soll, und bietet sie mit `propose_rule` an. Das Werkzeug fragt über
denselben Weg zurück wie die Mutations-Tools: anklickbare Bubble in der GUI,
y/N auf stdin in der CLI. Erst nach Zustimmung wird gespeichert, dann ist die
Regel ab dem nächsten Lauf aktiv.

| Tool | Bestätigung |
| --- | --- |
| `list_rules` | nein (liest nur) |
| `propose_rule` | ja |
| `set_rule_enabled` | nein — umkehrbar und im Regeldialog sichtbar |
| `set_rule_scope` | nein — ändert nur, wo die Regel gelesen wird, nicht den Wortlaut |
| `delete_rule` | ja — löscht die Herkunft mit |

**`autonomous_pipeline` deckt Regeln nicht ab.** Für die übrigen
Mutations-Tools heißt der autonome Modus „ohne y/N durchlaufen" — richtig, denn
sie ändern den Lauf, den der Nutzer gerade gestartet hat. Eine Regel ändert
stattdessen jeden künftigen Lauf, und zwar unbemerkt. Deshalb fragen
`propose_rule` und `delete_rule` auch im autonomen Modus. Ohne Kanal für die
Rückfrage wird abgelehnt, nicht gespeichert.

Bestätigte und abgelehnte Aufrufe hinterlassen eine Zeile in der Audit-Tabelle
`chat_mutations` (`operation: rule_save` / `rule_delete`), sofern ein
`UnifiedKnowledgeManager` verfügbar ist.

## Oberflächen

- **GUI**: Regeldialog (`src/ui/dialogs/rules_dialog.py`) — erreichbar über
  Einstellungen → System → Chat-Agent → „Zusatzregeln verwalten…" und über 📌 in
  der Kopfzeile des Chat-Panels.
- **CLI**: `alima rules list | show | add | enable | disable | remove | export | import`.
- **Webapp**: `GET /api/rules` zeigt die geltenden Regeln. Angelegt wird dort
  nicht: die Webapp-Session läuft mit `AutoRejectGateway` und hat keinen Kanal
  für eine Rückfrage.

## Austausch

`alima rules export [--out DATEI] [--ids …] [--enabled-only]` schreibt eine
teilbare Datei derselben Struktur. Der `origin`-Block wird unverändert
übernommen — wer eine Regel wann und warum formuliert hat, ist der Grund, sie
weiterzugeben.

`alima rules import DATEI [--activate]` übernimmt sie, behält das ursprüngliche
`origin` und ergänzt `imported_from` und `imported_at`. Kollidierende Ids
bekommen eine neue Id, die alte bleibt als `origin.original_id`. Importierte
Regeln sind **inaktiv**, bis der Operator sie einzeln scharf schaltet: eine
fremde Datei kann ein Dutzend Regeln mitbringen, und keine davon soll ungesehen
einen Lauf verändern.

## Regeln, die am Ende greifen sollen

Manche Regeln beschreiben keine Erschließungsentscheidung, sondern etwas, das
am Schluss zu tun ist („am Ende soll der Katalogeintrag im Format … erzeugt
werden"). Dafür gibt es keinen eigenen Abschluss-Schritt; zuständig ist die
**Reflexion** des MetaAgent, der letzte LLM-Turn eines agentischen Laufs.

- Der Reflexions-Prompt bekommt die geltenden Regeln in einem eigenen Block
  (`USER_RULES_INTRO`) — sie sind Prüfkriterien wie jedes andere. Der generische
  Regelblock wird für diesen Schritt übersprungen, sonst stünde jede Regel
  zweimal im Prompt.
- **Der Auftrag, die Ausgabe zu erzeugen (`USER_RULES_FINAL_GATE`), kommt nur
  bei der letzten Reflexion dazu.** Die Reflexion läuft einmal pro Zyklus; ein
  Modell, das früh `complete` meldet, erzeugte den Block sonst in jedem
  verbleibenden Zyklus neu. Wann die letzte ist, entscheidet der MetaAgent
  deterministisch über `_pending_step` — er kennt den Schrittgraphen, das Modell
  nicht.
- Die Ausgabe steht **nach** dem JSON in `<final_output>…</final_output>`. Der
  Prompt sagt ausdrücklich, dass danach kein weiterer Schritt folgt — ein Modell
  kündigte sie sonst nur an.
- **Warum nicht als JSON-Feld:** ein mehrzeiliger Eintrag in einem JSON-String
  braucht escapte Zeilenumbrüche. Ein Modell schreibt dort rohe Umbrüche, das
  JSON wird ungültig, und dann ist nicht nur die Ausgabe weg, sondern die ganze
  Antwort — Status, Action und Begründung. Der Lauf endete daraufhin still auf
  dem Standardwert `finish`. `repair_json_newlines` (`json_repair.py`) rettet
  seither wenigstens das Urteil, wenn ein Modell die Ausgabe doch ins JSON legt.
- `MetaAgent._capture_rule_output` legt sie auf dem Kontext ab, sie landet als
  `KeywordAnalysisState.rule_output` im Ergebnis und wird in GUI und Webapp
  unter „Aus einer Zusatzregel" angezeigt.
- Eine Regel, die nur die Ausgabe betrifft, ist ausdrücklich **kein** Grund,
  einen Schritt zu wiederholen.
- Ohne passende Regel bleibt der Block leer und der Reflexions-Prompt ist
  unverändert.

Das Format entsteht dabei jedes Mal neu vom Modell und ist damit nicht
reproduzierbar. Für einen deterministischen Katalogexport siehe WP-K6 in
[`open_workpackages.md`](open_workpackages.md).

## Nachvollziehbarkeit

Jeder agentische Lauf schreibt die tatsächlich injizierten Regeln als
`{id, text}` nach `KeywordAnalysisState.applied_rules`; die Ergebnisansicht in
GUI und Webapp zeigt sie unter „Zusatzregeln in diesem Lauf". Ohne diese Spalte
wäre ein Vergleichslauf zwischen zwei Rechnern nicht interpretierbar — dieselbe
Workflow-YAML, andere Regeln.

Die Liste sagt, welche Regeln **in einen Prompt gegangen** sind. Sie sagt nicht,
dass ein Modell sie befolgt hat, und für eine Regel, die eine Aktion am Ende des
Laufs verlangt, sagt sie nichts über deren Ausführung (siehe Grenze 0).

## Grenzen

0. **Eine Regel wirkt nur dort, wo ein Prompt gebaut wird.** Sie ist Text in
   einem Systemprompt und löst von sich aus keine Aktion aus. Für Regeln, die
   etwas **am Ende** des Laufs verlangen, gibt es genau eine Stelle: die
   Reflexion (siehe unten). Sie ist der letzte LLM-Turn — danach folgen in
   `alima_v51` nur noch `rvk_guard` und `dk_postprocess`, beide deterministisch.
   Ein Workflow **ohne** `meta_agent.reflection:`-Block hat diese Stelle nicht;
   dort bleibt eine solche Regel wirkungslos.
1. **Die klassische Pipeline liest diese Regeln nicht.** Ihre Prompts werden mit
   `str.format(**variables)` gebaut (`alima_manager.py`); eine nackte Klammer im
   Regeltext bräche dort den Lauf. Der klassische Zweig ist bewusst ausgenommen.
2. **Eine Prosa-Regel ist keine Garantie.** Beim Vergleichslauf vom 3. September
   2026 hielt sich Mistral nicht an die Mengenregel im Prompt; deshalb trägt die
   Klassifikation seither ein strukturiertes `rank`-Feld. Für die Bedingung gilt
   dasselbe doppelt: ob ein Werk eine Gesamtdarstellung ist, beurteilt das
   Modell, und eine Fehleinschätzung schaltet die Regel still zu oder ab.
3. **Jede aktive Regel kostet Token in jedem passenden Step.**
4. **Regeln sind maschinenlokal.** Sie wandern nur über Export/Import.
