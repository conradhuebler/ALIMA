# Design-Auffrischung: Farben und Bedienelemente aus dem ALIMA-Logo

Statusdatei und Referenz für die Palette von Webapp, geteiltem Log (`alima_render.css`) und
Qt-GUI. Die Werte stehen in den Quelldateien; dieses Dokument erklärt sie und hält fest,
was geprüft ist und was nicht.

## Stand

| Bereich | Stand |
|---|---|
| Palette aus `alima.png` gesampelt, Tokens festgelegt | erledigt |
| Webapp: Tokens, Bedienelemente, Marke (Kopfzeile, Favicon, Leerzustand, Stepper) | erledigt |
| Geteiltes Log `alima_render.css` (GUI + Webapp), hell und dunkel | erledigt |
| Qt: `COLORS`/`DARK_COLORS`, Haupt-Stylesheet, Paletten, Fenster-Icon, Splash | erledigt |
| Qt: fest verdrahtete Farben der Widgets auf Tokens umgestellt | erledigt, eine begründete Ausnahme |
| Qt: Themenwechsel erreicht alle Widgets unter dem Hauptfenster, GUI-Log hell/dunkel | erledigt (per Skript geprüft) |
| Kontrast gemessen (`tests/test_design_tokens.py`, 239 Paare) | erledigt |
| Wächter gegen neue Farbliterale (`tests/test_no_hardcoded_colors.py`) | erledigt |
| Gerenderte Prüfung Webapp und Qt (alle Tabs, Dialoge), hell und dunkel | erledigt (offscreen) |
| Farbschemata ALIMA / Neutral / Desktop (Qt und Webapp), siehe „Farbschemata“ | erledigt |
| Schemawechsel erreicht alle Widgets unter dem Hauptfenster; ALIMA pixelgleich zum Stand davor (Hauptfenster, Log, Anbieter-Dialog) | erledigt (per Skript und Test geprüft) |
| HTML-Inhalte und Tabellenzellen nach einem Themen- oder Schemawechsel neu einfärben | offen, siehe „Grenzen“ |

## Farben im Logo

Gemessen mit PIL an `alima.png` (1024 × 1024): Mittelwert aller hellen, gesättigten Pixel
(HLS: L > 0,45, S > 0,35) je 40-px-Band zwischen y = 200 und 640, Hintergrund als Mittel
kleiner Felder am Rand.

| Stelle im Logo | Wert | Pixel |
|---|---|---|
| Hintergrund oben Mitte (Bordeaux-Schimmer) | `#5b1c26` | 100 × 50 |
| Hintergrund oben Rand (Aubergine) | `#1e0e1a` | 50 × 50 |
| Hintergrund unten (Blauschwarz) | `#00000b` … `#000816` | 60 × 60 |
| Linien oben (Creme/Pfirsich) | `#f8e7d5`, `#f9d9c0` | n = 71, 517 |
| Knoten außen (Apricot) | `#fca482` | n = 1406 |
| Knoten Mitte (Koralle) | `#fa7778` | n = 931 |
| Buchseiten (Rosa-Rot) | `#f14e67`, unten `#e63565` | n = 6702, 407 |
| Stamm (Blau) | `#658dcb`, gesättigtster Pixel `#3091e9` | n = 2031 |
| Seitenäste unten (Violett) | `#7b81c6` | n = 652 |
| Wortbild | `#f3f5f8` | n = 6404 |

## Tokens

### Webapp (`src/webapp/static/styles.css`)

Hell steht in `:root`, Dunkel in zwei gleichlautenden Blöcken (`prefers-color-scheme: dark`
und `[data-theme="dark"]`). Kurzfassung der tragenden Tokens:

| Token | Hell | Dunkel | Rolle |
|---|---|---|---|
| `--clr-bg` | `#f6f2ef` | `#0c0a14` | Seitengrund, Eingabefelder |
| `--clr-surface` | `#fffcfa` | `#15111e` | Karten |
| `--clr-surface-2` | `#f1ebe7` | `#1c1626` | Pipeline-Leiste, Tab-Leiste |
| `--clr-surface-3` | `#e8e0db` | `#251d31` | Hover |
| `--clr-stream-bg` | `#fffcfa` | `#120e1a` | Log-Karte (= `--alima-bg`) |
| `--clr-text` | `#231925` | `#f1ebf0` | Text |
| `--clr-muted` | `#6b5c68` | `#b8aab9` | Nebentext |
| `--clr-subtle` | `#72626d` | `#a3949f` | Hinweise, Platzhalter |
| `--clr-border` | `#e2d8d2` | `#2e2539` | Trennlinien |
| `--clr-border-2` | `#8c7c88` | `#7a6a88` | Ränder von Bedienelementen |
| `--clr-primary` | `#a8284c` (Rosa vertieft) | `#f7a07f` (Apricot) | Akzent als Text/Rand |
| `--clr-grad` | `#b8483a → #a8284c → #8c2e63` | `#f9cdb0 → #f7a07f → #f27a7a → #ea5b73` | Primärknopf, erledigte Schritte |
| `--clr-on-solid` | `#ffffff` | `#1b0c14` | Text auf Verlauf und Signalfüllungen |
| `--clr-grad-soft` | `#f6b493 → #f07c7a → #d8395f` | `#f9cdb0 → #f27a7a → #ea5b73` | Zierlinien (Kopfzeile, Log-Karte, Stepper-Verbinder) |
| `--clr-focus` | `#a8284c` | `#f9cdb0` | Fokusring |
| `--clr-cool` | `#3f4fb0` | `#9aa6f2` | Gegenakzent Blau/Violett (Stamm) |
| `--clr-success` / `-warning` / `-error` | `#1a7044` / `#8a5800` / `#b42318` | `#7ad6a0` / `#f3c56b` / `#ff7b7b` | Signalfarben |

### Geteiltes Log (`src/webapp/static/alima_render.css`)

Dunkle Werte sind die Vorgabe in `#log { … }`; die hellen stehen in derselben Datei in zwei
gleichlautenden Blöcken (`[data-theme="light"] #log` und `prefers-color-scheme: light` ohne
`data-theme="dark"`). Die Webapp setzt `data-theme` über ihren Umschalter, die GUI über
`WebLogView.set_theme` aus dem GUI-Thema (das Dokument wird dabei nicht neu aufgebaut).

| Token | Hell | Dunkel |
|---|---|---|
| `--alima-bg` / `--alima-fg` | `#fffcfa` / `#231925` | `#120e1a` / `#f1ebf0` |
| `--alima-surface` (Blasen, Karten) | `#f5efeb` | `#1c1626` |
| `--alima-user-bubble` | `#f6dde3` | `#4a1d30` (Bordeaux) |
| `--alima-accent` (Tool-Titel, Labels) | `#9c2a4b` | `#f8b896` (Pfirsich) |
| `--alima-purple` (Stream-Titel, RVK) | `#4a4fb0` | `#aab0f6` |
| `--alima-link` | `#3446b0` | `#9db8ff` |
| `--alima-grad` (Kante der Ergebniskarte) | `#b8483a → #a8284c → #8c2e63` | `#f9cdb0 → #f27a7a → #ea5b73` |

### Qt (`src/ui/styles.py`)

`COLORS` und `DARK_COLORS` haben dieselben Schlüssel (Test). Neben den Grundfarben gibt es
Schlüssel für Text auf Füllungen (`on_*`), Interaktion (`focus`, `indicator`, `hover_tint`,
`surface_raised`, `secondary_tint`, `scroll_handle`), Verläufe (`grad_*`, `progress_*`),
Textrollen (`accent_text`, `violet`) und getönte Gründe (`success_bg`, `warning_bg`,
`error_bg`, `info_bg`).

| Schlüssel | Hell | Dunkel |
|---|---|---|
| `primary` (Füllfarbe, immer mit weißer Schrift) | `#a8284c` | `#b8325a` |
| `secondary` (Gegenakzent) | `#3f4fb0` | `#9aa6f2` |
| `accent` (Apricot) | `#f7a07f` | `#f7a07f` |
| `background` / `background_light` / `background_dark` | `#fffcfa` / `#f6f2ef` / `#e8e0db` | `#120e1a` / `#1b1625` / `#0b0912` |
| `text` / `text_light` / `text_muted` | `#231925` / `#5e4f5b` / `#6f606b` | `#f1ebf0` / `#c9bdcb` / `#a497a8` |
| `border` / `border_light` | `#8c7c88` / `#e2d8d2` | `#7a6a88` / `#2e2539` |
| `focus` / `indicator` | `#a8284c` / `#a8284c` | `#f9cdb0` / `#f7a07f` |
| `accent_text` / `violet` | `#9c2a4b` / `#4a4fb0` | `#f8b896` / `#aab0f6` |
| `success_bg` / `warning_bg` / `error_bg` / `info_bg` | `#e7f3ec` / `#f8eedb` / `#f9e6e4` / `#eceefa` | `#16261e` / `#2a2214` / `#2c1619` / `#22233d` |

Dunkel `background` ist `--alima-bg`, damit das GUI-Log nahtlos im Fenster sitzt.

## Wie Qt-Widgets ihre Farben bekommen

Drei Wege, in dieser Reihenfolge der Vorliebe:

1. **Markierung statt Farbe.** Ein Widget bekommt eine Eigenschaft, das Haupt-Stylesheet
   (`styles._component_rules`) färbt sie:
   `set_tone(w, "muted"|"light"|"heading"|"accent"|"info"|"violet"|"success"|"warning"|"error"|"divider")`,
   `set_box(w, "success"|…)` (getönter Meldungsrahmen), `set_status(label, kind)`
   (Statuszeile wie `get_status_label_styles`), `set_variant(button, "primary"|"secondary"|…)`,
   `set_style_prop(w, "chip"|"role"|"state", …)` sowie Objektnamen für Einzelteile
   (`panelStrip`, `chatInput`, `chatSend`, `pipelineStep`, `dropZone`, …). Das Haupt-Stylesheet
   wird bei jedem Themenwechsel neu gesetzt, diese Widgets folgen ohne eigenen Code.
2. **Eigenes Stylesheet aus `get_colors()` plus `refresh_styles()`** für Teile mit vielen
   Zuständen (Wiederholungsleiste, Vorschlagsleiste, Schritt-Tabs der Pipeline, Agentic-Panels,
   Validierungsrahmen der Modellauswahl).
3. **`palette(...)` in QSS**, wo eine Palettenrolle passt (Dokumentationskasten der Plugins).

`MainWindow._reapply_tab_stylesheets` ruft nach einem Themen- oder Schriftwechsel
`refresh_styles()` an **jedem** Widget unter dem Hauptfenster auf, nicht mehr nur an den Tabs.
Widgets, die das Haupt-Stylesheet bei sich selbst festhalten (Chat-Panel, UB-Katalog-Panels),
setzen es dort neu.

Gezählt in `src/ui` ohne `styles.py` (Hex, `rgb()`, `QColor` aus Zahlen oder Text,
`Qt.GlobalColor`, CSS-Farbnamen; Kommentarzeilen ausgenommen): vorher 400 Farbliterale in 39
Dateien (315 Hex, 13 `rgb()`, 19 `QColor`, 2 `Qt.GlobalColor`, 51 Farbnamen), nachher eines.
`setStyleSheet`-Aufrufe: 251 vorher, 123 nachher.

**Bewusst geblieben** (Ausnahmeliste in `tests/test_no_hardcoded_colors.py`):
`QColor(0, 0, 0, 120)` in `webcam_capture_dialog.py` dunkelt das Kamerabild außerhalb des
Zuschnitts ab; es liegt auf einem Foto, nicht auf einer Themenfläche. Der Zuschnittrahmen
nutzt den hellen Logo-Pfirsich (`DARK_COLORS["focus"]`) in beiden Themen, aus demselben Grund.

## Verwendungsregeln

- **Der Logo-Verlauf markiert genau eine Hauptaktion je Ansicht** (Webapp `.btn-primary`,
  Qt `variant="primary"`, z. B. „Auto-Pipeline“, „Analyse starten“, „Suche starten“, der
  Senden-Knopf im Chat), dazu erledigte Stepper-Schritte, Fortschrittsbalken und Zierlinien.
  Alle übrigen Knöpfe sind neutral.
- **Auf dunklem Grund die hellen Verlaufsstufen, auf hellem Grund die tiefen**, wie im Logo
  selbst (helle Linien auf Nacht). Darum ist der Akzent dunkel Apricot, hell vertieftes Rosa.
- **Blau/Violett ist der kühle Gegenakzent**: Links, Info, RVK, Qt-Sekundärknöpfe.
- **Signalfarben tragen nie allein Bedeutung**: Fehler behalten ❌/„Fehler“, erledigte Schritte
  ein ✓, aktive einen offenen Ring, Qt-Checkboxen ein Häkchen, Pipeline-Schritte ihr Symbol
  (▷ ▶ ✓ ✗), Statuschips ihren Text.
- **Kein Farbliteral in `src/ui` außerhalb von `styles.py`**; eine Ausnahme braucht einen Eintrag
  mit Begründung in `tests/test_no_hardcoded_colors.py`.
- **Ein `QFrame { … }` ohne Objektnamen trifft auch alle Kinder** (QLabel und QTextEdit sind
  QFrames). Daher kamen die weißen Textfelder und die gestrichelten leeren Kästen in der
  Ablagezone; Rahmenregeln stehen deshalb nur noch mit `#objectName`.
- **QSS-Reihenfolge**: eine spätere Regel gewinnt bei gleicher Eigenschaft, darum
  `:hover:!selected` (Bäume, Tabs) (`tests/test_provider_list_selection_contrast.py`).
- **Qt-Stylesheets laden Bilder nur aus Dateien**, nicht aus `data:`-URIs (gemessen: 0 statt
  26 weiße Pixel im Häkchen). Darum liegen Häkchen und Pfeil als `ui-check.svg` und
  `ui-chevron.svg` in `src/webapp/static/`.
- **Nur lokale Schriften**: Systemschrift-Stack, das Wortbild nutzt eine geometrische Schrift,
  falls installiert (`Avenir Next`, `Century Gothic`, `URW Gothic`), sonst die Systemschrift.

## Entscheidungen

- **Neutrale Standardknöpfe statt farbiger.** Vorher war jeder Knopf blau gefüllt. Ein Logo-Rosa
  auf jedem Knopf wäre lauter als das Blau; der Verlauf trägt nur, wenn er selten ist. Die
  Alternative (Dialog-Standardknopf über `:default` farbig) wurde gerendert und verworfen: in
  Qt wandert `:default` mit dem Tastaturfokus, im Einstellungsdialog war dadurch „Hinzufügen…“
  hervorgehoben statt „Speichern“.
- **Knopfvarianten als Eigenschaft (`variant`) statt als Stylesheet-Text.** Die Tabs setzten
  `get_button_styles()[...]` einmal beim Aufbau; ein Themenwechsel erreichte diese Knöpfe nie.
  `get_button_styles()` bleibt für andere Aufrufer erhalten.
- **Qt `primary` bleibt eine Füllfarbe für weiße Schrift**, auch dunkel (`#b8325a`): Widgets
  kombinieren sie mit `on_primary`. Der helle Akzent für Text auf dunklem Grund heißt dort
  `focus`/`indicator`/`accent_text`.
- **Gesperrte Knöpfe auf `surface_raised`/`background_light`** statt `background_dark`: im
  dunklen Thema waren sie schwarze Kästen.
- **Das GUI-Log folgt dem GUI-Thema.** Vorher war es immer dunkel.
- **Wiederholungsleiste im Ruhezustand ganz unsichtbar.** Sie behält ihre Höhe (kein Springen
  des Logs) und war schon so gedacht („hidden“ malt alles transparent); fest gesetzte
  Einzelfarben machten aber Titel „Wiederholung erkannt“ und den roten Knopf „Abbrechen“
  dauerhaft sichtbar, auch ohne Wiederholung. Jetzt sind sie im Ruhezustand unsichtbar wie
  der Rest der Leiste. Die Knöpfe bleiben dabei an ihrer Stelle anklickbar, wie „Fortfahren“
  und „✕“ es vorher schon waren (siehe „Grenzen“).
- **Einrichtungsassistent mit eingesetzten Farben statt `tone`**: er läuft beim ersten Start vor
  dem Hauptfenster, also bevor das Haupt-Stylesheet gesetzt ist.
- **Die Log-Karte der Webapp folgt dem Thema.** Vorher war die Karte immer dunkel, das Log darin
  im hellen Thema aber hell; Karte und Log wirkten wie zwei Flächen.
- **Leerzustand ohne Text**: nur die Marke, gedämpft, mittig. Ein Satz dort bräuchte i18n;
  die Untertitelzeile der Log-Karte erklärt den Bereich bereits.

## Marke

- `src/webapp/static/alima-mark.svg` (1294 B): Buch, Stamm mit Ästen und acht Knoten,
  Verlaufsstriche (warm für Buch/Äste, kühl für den Stamm). Als `data:`-URI auch im
  Leerzustand von `#log:empty` (GUI und Webapp).
- `src/webapp/static/alima-icon.svg` (1625 B): dieselbe Zeichnung auf einer Aubergine-Kachel
  mit Bordeaux-Schimmer. Kopfzeile, Favicon, Qt-Fenster-Icon.
- Splash: `alima.png` aus dem Repository-Wurzelverzeichnis, auf 320 px skaliert. Der alte Pfad
  zeigte auf `src/alima.png`; dort liegt die Datei in diesem Repository nicht, der Splash
wurde nicht angezeigt.

## Kontrast

Gemessen als WCAG-Verhältnis an den Werten aus den Quelldateien (`measure_all()` in
`tests/test_design_tokens.py`, 239 Paare, keines unter der Schwelle). Schwellen: Text 4,5:1,
Ränder von Bedienelementen und Fokus 3:1. Durchscheinende Gründe (Badges: 14–16 % der
Textfarbe über der Karte) werden vor dem Messen über die Fläche gerechnet. Neu erzeugen:

```
QT_QPA_PLATFORM=offscreen .venv/bin/python -c "import sys; sys.path[:0]=['tests','.']; \
from test_design_tokens import measure_all; [print(r) for r in measure_all()]"
```

Kleinste gemessene Werte:

| Thema | Text (Schwelle 4,5) | Nicht-Text (Schwelle 3) |
|---|---|---|
| hell (n = 107 / 12) | 4,59: Log-Badge „Fehler“, `#b42318` auf 14 % über `#f5efeb` | 3,32: Bedienrand `#8c7c88` auf `#f1ebe7` |
| dunkel (n = 108 / 12) | 4,92: Qt-Fortschrittstext `#f1ebf0` auf `#ad4338` | 3,30: Qt-Checkbox gefüllt `#b8325a` auf `#120e1a` |

Nicht im Test, einzeln gemessen: Qt-Pfeil `ui-chevron.svg` (`#8c7c88`) 3,52–3,84 auf hellen,
4,51–4,83 auf dunklen Gründen; Häkchen weiß auf `primary` 6,83 (hell) / 5,77 (dunkel).
Rollbalken-Daumen liegen bei 1,89–2,3:1 zur Spur (GUI-Log vorher 1,85:1) und sind nicht an
eine Schwelle gebunden.

Vollständige Tabelle (hell / dunkel; mehrere Werte = je Verlaufsstopp):

#### Webapp

| Paar | Schwelle | hell | dunkel |
|---|---|---|---|
| Text / Seite | 4.5 | 15.25 | 16.73 |
| Text / Fläche | 4.5 | 16.61 | 15.81 |
| Text / Fläche 2 | 4.5 | 14.37 | 15.00 |
| Text / Fläche 3 (Hover) | 4.5 | 13.03 | 13.77 |
| Gedämpft / Fläche | 4.5 | 6.13 | 8.39 |
| Gedämpft / Fläche 2 | 4.5 | 5.30 | 7.96 |
| Gedämpft / Seite | 4.5 | 5.62 | 8.87 |
| Hinweis+Platzhalter / Fläche | 4.5 | 5.58 | 6.45 |
| Hinweis+Platzhalter / Fläche 2 | 4.5 | 4.83 | 6.12 |
| Platzhalter / Eingabegrund | 4.5 | 5.12 | 6.82 |
| Primär (Text) / Fläche | 4.5 | 6.69 | 9.10 |
| Primär (Text) / Fläche 2 | 4.5 | 5.79 | 8.63 |
| Kühler Akzent / Fläche 2 | 4.5 | 6.01 | 7.64 |
| DK-Fortschritt / Leiste | 4.5 | 6.01 | 7.64 |
| Log-Untertitel / Log-Karte | 4.5 | 6.13 | 8.20 |
| Erfolg / Fläche | 4.5 | 5.97 | 10.57 |
| Warnung / Fläche 2 | 4.5 | 5.11 | 10.91 |
| Fehler / Fläche | 4.5 | 6.44 | 7.40 |
| Info / Fläche | 4.5 | 6.95 | 8.06 |
| Knopftext / Fehler-Füllung | 4.5 | 6.57 | 7.55 |
| Knopftext / Erfolg-Füllung | 4.5 | 6.10 | 10.78 |
| Knopftext / Warnung-Füllung | 4.5 | 6.04 | 11.73 |
| Knopftext / Info-Füllung | 4.5 | 7.10 | 8.21 |
| Banner-Text / Banner | 4.5 | 7.01 | 11.09 |
| Bedienrand / Fläche | 3 | 3.84 | 3.75 |
| Bedienrand / Fläche 2 | 3 | 3.32 | 3.56 |
| Bedienrand / Seite | 3 | 3.52 | 3.97 |
| Fokusring / Fläche | 3 | 6.69 | 12.72 |
| Fokusring / Seite | 3 | 6.14 | 13.46 |
| Knopftext / Verlauf-Stopp | 4.5 | 5.22 / 6.83 / 7.82 | 12.97 / 9.28 / 7.07 / 5.63 |

#### Log

| Paar | Schwelle | hell | dunkel |
|---|---|---|---|
| Log-Text / Log | 4.5 | 16.61 | 16.21 |
| Fließtext / Blase | 4.5 | 13.32 | 13.53 |
| Fließtext / Nutzerblase | 4.5 | 11.84 | 10.68 |
| Fließtext / Tabellenkopf | 4.5 | 12.10 | 14.18 |
| Gedämpft / Log | 4.5 | 6.43 | 8.20 |
| Gedämpft / Blase | 4.5 | 5.76 | 7.59 |
| Tool-Zeile / Log | 4.5 | 6.65 | 6.89 |
| Meta / Karte | 4.5 | 5.49 | 6.79 |
| Zeitstempel / Log | 4.5 | 6.24 | 6.87 |
| Akzent / Log | 4.5 | 7.22 | 11.14 |
| Akzent / Tabellenkopf | 4.5 | 5.88 | 10.80 |
| Violett / Log | 4.5 | 6.77 | 9.29 |
| Violett / Karte | 4.5 | 6.07 | 8.60 |
| Grün / Log | 4.5 | 6.94 | 10.83 |
| Gelb / Log | 4.5 | 6.81 | 11.79 |
| Orange / Karte | 4.5 | 6.64 | 7.91 |
| Fehler / Log | 4.5 | 6.44 | 7.59 |
| Fehler-Text / Log | 4.5 | 8.73 | 11.60 |
| Link / Log | 4.5 | 7.76 | 9.73 |
| Link / Blase | 4.5 | 6.96 | 9.00 |
| Fremd-Link / Log | 4.5 | 7.40 | 11.79 |
| Systemzeile / Log | 4.5 | 6.94 | 10.83 |
| Live-Stream / Log | 4.5 | 9.64 | 12.66 |
| Info-Zeile / Info-Grund | 4.5 | 6.95 | 10.05 |
| Vorschlag-ID / Karte | 4.5 | 5.96 | 6.37 |
| Badge DK | 4.5 | 5.00 | 7.27 |
| Badge DDC | 4.5 | 5.21 | 5.94 |
| Badge RVK | 4.5 | 4.81 | 6.26 |
| Badge standard | 4.5 | 5.08 | 7.52 |
| Badge nicht standard | 4.5 | 4.99 | 8.06 |
| Badge Fehler | 4.5 | 4.59 | 5.62 |

#### Qt

| Paar | Schwelle | hell | dunkel |
|---|---|---|---|
| Text / Eingabe | 4.5 | 16.61 | 16.21 |
| Text / Gruppe | 4.5 | 15.25 | 15.05 |
| Text / neutraler Knopf | 4.5 | 14.37 | 13.84 |
| Text / Hover | 4.5 | 14.05 | 13.77 |
| Nebentext / Gruppe | 4.5 | 6.87 | 9.78 |
| Gedämpft / Gruppe | 4.5 | 5.30 | 6.37 |
| Gedämpft / Eingabe | 4.5 | 5.77 | 6.86 |
| Weiß / Primär | 4.5 | 6.83 | 5.77 |
| Weiß / Primär Hover | 4.5 | 8.69 | 4.93 |
| Weiß / Primär gedrückt | 4.5 | 10.81 | 7.15 |
| Statustext Info / Eingabe | 4.5 | 6.95 | 8.26 |
| Statustext Info / Gruppe | 4.5 | 6.38 | 7.67 |
| Knopftext / Sekundär | 4.5 | 7.10 | 8.21 |
| Knopftext / Sekundär Hover | 4.5 | 8.78 | 10.01 |
| Text / gedrückter Umschalter | 4.5 | 14.69 | 13.00 |
| Knopftext / Akzent | 4.5 | 8.32 | 9.28 |
| Knopftext / Akzent Hover | 4.5 | 7.16 | 10.82 |
| Knopftext / Erfolg | 4.5 | 6.10 | 10.78 |
| Knopftext / Erfolg Hover | 4.5 | 8.02 | 12.19 |
| Knopftext / Warnung | 4.5 | 6.04 | 11.73 |
| Knopftext / Fehler | 4.5 | 6.57 | 7.55 |
| Knopftext / Fehler Hover | 4.5 | 8.35 | 8.94 |
| Verlaufstext / Verlauf Anfang | 4.5 | 5.22 | 12.97 |
| Verlaufstext / Verlauf Mitte | 4.5 | 6.83 | 8.33 |
| Verlaufstext / Verlauf Ende | 4.5 | 7.82 | 5.63 |
| Fortschrittstext / Balken Anfang | 4.5 | 9.59 | 4.92 |
| Fortschrittstext / Balken Mitte | 4.5 | 6.98 | 5.82 |
| Fortschrittstext / Balken Ende | 4.5 | 5.04 | 6.66 |
| Fortschrittstext / Balkengrund | 4.5 | 13.03 | 16.84 |
| Statustext Erfolg / Gruppe | 4.5 | 5.48 | 10.06 |
| Statustext Warnung / Gruppe | 4.5 | 5.43 | 10.95 |
| Statustext Fehler / Gruppe | 4.5 | 5.91 | 7.05 |
| Bedienrand / Eingabe | 3 | 3.84 | 3.85 |
| Bedienrand / Gruppe | 3 | 3.52 | 3.57 |
| Fokusrand / Eingabe | 3 | 6.69 | 13.04 |
| Fokusrand / Gruppe | 3 | 6.14 | 12.11 |
| Tab-Markierung / Eingabe | 3 | 6.69 | 9.32 |
| Checkbox gefüllt / Eingabe | 3 | 6.69 | 3.30 |
| Akzenttext / Eingabe | 4.5 | 7.22 | 11.14 |
| Akzenttext / Gruppe | 4.5 | 6.63 | 10.34 |
| Violett / Eingabe | 4.5 | 6.77 | 9.29 |
| Violett / Gruppe | 4.5 | 6.21 | 8.63 |
| Erfolg / Erfolgsgrund | 4.5 | 5.35 | 8.99 |
| Warnung / Warngrund | 4.5 | 5.25 | 9.73 |
| Fehler / Fehlergrund | 4.5 | 5.47 | 6.77 |
| Info / Infogrund | 4.5 | 6.15 | 6.62 |
| Text / Erfolgsgrund (Tabellenzeile) | 4.5 | 14.90 | 13.44 |
| Text / Warngrund (Leiste) | 4.5 | 14.75 | 13.37 |
| Text / Fehlergrund (Schritt) | 4.5 | 14.12 | 14.47 |
| Text / Infogrund (Schritt, Zeile) | 4.5 | 14.69 | 13.00 |
| Gedämpft / Warngrund (Vorschlag) | 4.5 | 5.12 | 5.66 |
| Gedämpft / Infogrund (Kamera) | 4.5 | 5.10 | 5.50 |
| Erfolg / Eingabe (Tabellenzeile) | 4.5 | 5.97 | 10.83 |
| Fehler / Eingabe | 4.5 | 6.44 | 7.59 |
| Warnung / Eingabe | 4.5 | 5.91 | 11.79 |
| Trennlinie / Eingabe | 3 | 3.84 | 3.85 |

## Farbschemata

Drei Schemata, gewählt über `UIConfig.color_scheme` (Qt) bzw. die Auswahl in der Kopfzeile
(Webapp, je Betrachter in `localStorage`). Standard ist „ALIMA“; bestehende Konfigurationen
ohne den Schlüssel bleiben dabei. Ein unbekannter Wert in `config.json` fällt mit einem
Log-Eintrag auf „alima“ zurück.

| Schema | Qt | Webapp |
|---|---|---|
| **ALIMA** | `COLORS`/`DARK_COLORS` wie oben, unverändert | wie oben |
| **Neutral** | `NEUTRAL_COLORS`/`NEUTRAL_DARK_COLORS` | `:root[data-scheme="neutral"]` in `styles.css`, Log in `alima_render.css` |
| **Desktop** | aus der Systempalette abgeleitet (`derive_desktop_colors`) | CSS-Systemfarben (`Canvas`, `CanvasText`, `LinkText`, `Highlight`, `color-scheme: light dark`) |

Alle Tabellen haben dieselben Schlüssel wie `COLORS` (Test). Umschalten: **Extras ▸ 🎨
Farbschema** (drei Einträge, einer mit Häkchen) und **Einstellungen ▸ Anwendung ▸
Farbschema**; beide schreiben dieselbe Einstellung und folgen einander. Ein Wechsel läuft wie
der Hell/Dunkel-Wechsel durch alle Widgets (`apply_color_scheme` → `apply_theme` →
`_reapply_tab_stylesheets`), ohne Neustart. Beim Start setzt `load_settings` das Schema vor
dem einen Stylesheet-Durchgang; gemessen `load_settings` 1,11 s gegenüber 1,12 s am Stand
davor (Median aus je 3 Läufen, offscreen, leere Konfiguration), also kein messbarer Zuwachs.

### Neutral

Abgestufte Grautöne mit leichtem Blaugrau-Stich, ein ruhiges Blau als einziger Akzent,
flache Füllungen statt Verlauf, die Marke im leeren Log in Grau. Erfolg/Warnung/Fehler
behalten eigene Farbtöne und ihre Symbole.

| Token (Qt) | Hell | Dunkel |
|---|---|---|
| `background` / `background_light` / `background_dark` | `#fbfcfd` / `#f3f5f7` / `#e3e7ec` | `#14171c` / `#1b1f26` / `#0e1014` |
| `text` / `text_light` / `text_muted` | `#1c2330` / `#465161` / `#596473` | `#e7eaef` / `#bcc4ce` / `#9ca5b0` |
| `border` / `border_light` | `#7f8995` / `#dce1e7` | `#6c7582` / `#2b313a` |
| `primary` (Füllung, Schrift `on_primary`) | `#3a5f8f` / weiß | `#3c6aa6` / weiß |
| `focus` / `indicator` / `accent_text` | `#3a5f8f` / `#3a5f8f` / `#34557f` | `#93b9e8` / `#93b9e8` / `#a3c2e8` |
| `secondary` (Schiefer) | `#4f5d6e` | `#93abc8` |
| Verlauf (`grad_*`) | flach `#3a5f8f`, Schrift weiß | flach `#93b9e8`, Schrift `#10141a` |
| `success` / `warning` / `error` | `#256b45` / `#7a5600` / `#a82a2a` | `#7cc89d` / `#e3bd68` / `#f28b8b` |

Webapp (`--clr-*`): hell Seite `#f3f5f7`, Fläche `#fbfcfd`, Text `#1c2330`, Akzent `#3a5f8f`;
dunkel Seite `#0f1216`, Fläche `#171a20`, Text `#e7eaef`, Akzent `#93b9e8`.

### Desktop

**Qt.** `alima_gui.main` nimmt direkt nach dem Anlegen der `QApplication`, vor
`setStyle("Fusion")` und vor jeder eigenen Palette, einen Schnappschuss der Systempalette
(`take_desktop_snapshot`). Hell oder dunkel liefert `QStyleHints.colorScheme()`, sonst die
Helligkeit der Fensterfarbe; der Menüeintrag „Dunkles Design“ ist in diesem Schema gesperrt
(Tooltip nennt den Grund), die gespeicherte Wahl gilt wieder für die anderen Schemata. Das
Schema stellt die Systempalette wieder her; das Stylesheet behält alle Layout- und
Größenregeln, nur die Farben kommen aus `derive_desktop_colors`:

- Gründe: `background` = Base, `background_light` = Window, `surface_raised` = Button,
  Trennlinien und Hover als Mischung aus Window/Text bzw. Button/Highlight.
- Text: `text` = Text, `text_light` = 75 % WindowText über Window, `text_muted` =
  PlaceholderText, `border` = Mid.
- Akzent: `primary`, `focus`, `indicator`, `accent_text` = Highlight; `secondary`, `violet` =
  Link; der Verlauf ist flach in der Highlight-Farbe.
- Erfolg/Warnung/Fehler: die Neutral-Farbtöne (das System hat dafür keine Rolle), getönte
  Gründe als 12 % davon über Base.
- **Kontrastreparatur**: danach wird jedes Paar aus `DESKTOP_CONTRAST_RULES` (alle Qt-Paare
  dieser Tabelle und die Log-Paare) geprüft; was die Schwelle verfehlt, wird schrittweise zu
  Schwarz oder Weiß hin verschoben, bis es passt (ein lesbares Paar geht vor dem exakten
  Systemton). Text auf Füllungen (`on_*`) übernimmt HighlightedText nur, wenn es 4,5:1
  erreicht, sonst Schwarz oder Weiß (eines von beiden erreicht immer ≥ 4,58:1). Badges im Log
  (Farbe auf 14–16 % ihrer selbst) werden mit eigenen Schritten nachgezogen; wo selbst
  Schwarz nicht reicht (mittelgraue Fenster), rückt der Kartengrund ein Stück ab.
- Das GUI-Log bekommt im Desktop-Schema seine `--alima-*`-Variablen zur Laufzeit aus diesen
  Tokens (`styles.log_css_vars`, `WebLogView.set_scheme`), ohne Neuaufbau.

Ohne Schnappschuss (Tests, Werkzeuge ohne `alima_gui.main`) nutzt „Desktop“ die
Neutral-Tabellen.

**Webapp.** CSS-Systemfarben mit `color-scheme: light dark`; hell/dunkel folgt
`prefers-color-scheme`, der Schalter in der Kopfzeile ist in diesem Schema gesperrt.
Erfolg/Warnung/Fehler über `light-dark()` (wo der Browser es kennt, sonst die hellen
Neutral-Werte). **Die Kontraste der Systemfarben lassen sich nicht vorab messen**: sie hängen
von Browser, Betriebssystem und Einstellung ab. Für die Webapp im Desktop-Schema gibt es
deshalb keine Zahl.

### Kontrast je Schema

Gemessen wie oben (`tests/test_design_tokens.py`; `measure_all("neutral")`,
`measure_desktop()`). Keines der Paare liegt unter der Schwelle.

| Schema | Paare | kleinster Text-Wert (4,5) | kleinster Nicht-Text-Wert (3) |
|---|---|---|---|
| Neutral hell | 117 | 4,85: Webapp-Hinweis `#5f6a78` auf `#eef1f4` | 3,13: Webapp-Bedienrand `#7f8995` auf `#eef1f4` |
| Neutral dunkel | 117 | 4,66: Qt weiß auf `primary_hover` `#4776b3` | 3,25: Qt-Checkbox `#3c6aa6` auf `#14171c` |
| Desktop (Qt + GUI-Log), 7 Testpaletten | 609 | 4,52: „mittleres Grau“, Fortschrittstext `#161616` auf `#7f7f7f` | 3,04: „mittleres Grau“, Tab-Markierung `#414141` auf `#8c8c8c` |
| Desktop (Webapp) | – | nicht messbar (Systemfarben) | nicht messbar |

Testpaletten für Desktop: Breeze hell und dunkel, Adwaita hell, hell mit sehr hellem Akzent
(`#fff3a0` mit weißer Schrift), dunkel mit fast schwarzem Akzent (`#0b0b2a` mit schwarzer
Schrift), alles mittelgrau, hell mit kräftigem Magenta-Akzent. Ohne die Reparatur verfehlen
Paare in diesen Paletten die Schwelle (eigener Test hält das fest), ebenso wenn die
Systemschrift auf dem Akzent ungeprüft übernommen wird oder die Badge-Anpassung fehlt
(Mutationsprüfung, je ein Test schlägt fehl).

## Prüfung (Methode)

Offscreen gerendert (`QT_QPA_PLATFORM=offscreen`, QtWebEngine mit `--no-sandbox
--disable-gpu`), in einem Netz-Namensraum nur mit Loopback (`unshare -rn`) und mit einem
leeren temporären `HOME`, also ohne Anfragen an äußere Dienste:

- **Webapp**: die echte Vorlage (`webapp.html`, Jinja-gerendert) und die echten statischen
  Dateien hinter einem kleinen lokalen Stub-Server mit festen Antworten für `/api/workflows`
  und `/api/models`; Log-Inhalt aus dem echten `UnifiedMessageRenderer` (Pipeline-Zeilen,
  Stream-Block, Tool-Aufruf, Fehlerblock, Badge-Karte, Ergebniskarte, Chat-Blasen).
  1400 × 900 und 390 px breit, hell und dunkel, leer und mit Lauf. Horizontales Überlaufen
  per `scrollWidth` geprüft: in allen neun Aufnahmen `scrollWidth ≤ innerWidth`.
- **Qt**: echtes `MainWindow` mit einem synthetischen localhost-Anbieter, gebaut im hellen
  Thema und dann umgeschaltet (wie beim Start mit gespeichertem dunklem Thema). Alle sieben
  Tabs (mit „Vergleich“), der Pipeline-Tab mit aktiver Wiederholungsleiste und offenem Vorschlag, drei Seiten
  des Einstellungsdialogs, Anbieter-, Stapel- und Regel-Dialog, `WebLogView` leer und gefüllt,
  je hell und dunkel. Zum Vergleich derselbe Lauf auf einer Kopie des Quellbaums mit den alten
  Stildateien.
- **Themenwechsel per Skript**: Hauptfenster plus offener Einstellungs- und Anbieter-Dialog,
  `apply_theme` dunkel → hell → dunkel. Nach jedem Wechsel trug keines der 66 Widgets mit
  eigenem Stylesheet eine Farbe, die nur im anderen Thema vorkommt, und keine Farbe außerhalb
  beider Paletten; alle drei Log-Ansichten meldeten das passende `data-theme`.

- **Farbschemata**: Hauptfenster (Pipeline-Tab mit Eingabebereich und gefülltem Log),
  Einstellungen (Seite „Anwendung“ mit dem Feld „Farbschema“) und Anbieter-Dialog in ALIMA und
  Neutral je hell und dunkel sowie Desktop mit zwei künstlich gesetzten Systempaletten (Breeze
  dunkel, hell mit kräftigem Magenta-Akzent), alle über `apply_color_scheme` zur Laufzeit
  umgeschaltet. Nach jedem Wechsel trug kein Widget-Stylesheet eine Farbe, die nur in der
  vorherigen Tabelle vorkommt. ALIMA hell und dunkel sind pixelgleich zum Stand vor den
  Schemata (Hauptfenster 1539 × 900, Log, Anbieter-Dialog; 0 abweichende Pixel). Webapp je
  Schema hell und dunkel bei 1400 px und 390 px, Desktop zusätzlich mit erzwungenem
  System-Dunkelmodus (`--force-dark-mode`); kein horizontales Überlaufen in allen Aufnahmen.

Gefunden und behoben dabei: abgeschnittener Tab „Kamera“ in der Webapp, Listenpunkte außerhalb
der Assistentenblase, Rollbalken im leeren Log, auf 390 px unlesbar gestauchte Auswahllisten,
fehlende Häkchen und Pfeile in Qt (`data:`-URIs), weiße Textfelder und Flächen im dunklen
Pipeline-Tab (`QFrame`-Regeln ohne Objektnamen), unsichtbarer Text in der Ablagezone, die
dauerhaft sichtbare Wiederholungsleiste, Stilreste des Startthemas in Chat-Panel, Tab-Knöpfen
und UB-Katalog, schwarze gesperrte Knöpfe, fehlendes Fenster-Icon im Einrichtungsassistenten.

## Grenzen

- **HTML-Inhalte und Tabellenzellen** (GND-Details und Zuordnungstabelle in der Suche,
  Zusammenfassung im Review, Ketten und DK-Banner im Pipeline-Tab, Zeilenfarben der
  Treffertabellen) bekommen ihre Farben beim Füllen aus dem aktuellen Thema. Nach einem
  Themenwechsel behalten sie die alten, bis sie neu gefüllt werden. Die Agentic-Panels werden
  neu gerendert, die übrigen nicht.
- **Dialoge außerhalb des Hauptfensters** (Kamera, Libero-Anmeldung, Commit-Auswahl) übernehmen
  das Thema beim Öffnen; ein Wechsel, während sie offen sind, ist nicht geprüft.
- **Nicht geprüft**: echte Browser außer QtWebEngine (6.10.2, Chromium 134), echte Bildschirme
  und HiDPI, Hover- und Tastaturzustände in Qt per Pixel (offscreen nicht verlässlich, siehe
  `tests/test_provider_list_selection_contrast.py`), der Splash im echten Start, der
  Einrichtungsassistent gerendert.
- `color-mix()` (Badge-Gründe im Log) braucht Chromium ≥ 111; ältere Browser zeigen die Badges
  ohne getönten Grund.
- **Farbschemata**: Der Desktop-Schnappschuss wird beim Start genommen; ändert sich die
  Systempalette während ALIMA läuft, folgt die GUI erst nach einem Neustart. Ein HTML-Inhalt
  oder eine Tabellenzelle behält nach einem Schemawechsel die alten Farben bis zum nächsten
  Füllen (wie beim Themenwechsel). Der Einrichtungsassistent beim ersten Start erscheint
  immer in ALIMA, weil er vor dem Laden der Einstellungen läuft. Logo, Favicon und die
  Zuschnittmarke im Kameradialog bleiben in allen Schemata in den Logo-Farben. Die
  Kontraste der Webapp im Desktop-Schema sind nicht gemessen (Systemfarben); echte
  Desktop-Paletten (KDE, GNOME, Windows, macOS) sind nur über die synthetischen Testpaletten
  nachgebildet, nicht an echten Systemen geprüft.
