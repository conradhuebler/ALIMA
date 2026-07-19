# WP12 — Unified Render Layer (GUI ↔ Webapp)

**Status (2026-06-10)**: WP12.1–WP12.4 **implementiert und committed
(`9552d93`)**, (b) nur headless verifiziert — Abschluss-Definition siehe
[§9 „WP12.5 — Restarbeiten / Definition of Done"](#9-wp125--restarbeiten--definition-of-done).
Implementierungsdetails: [`../AIChangelog.md`](../AIChangelog.md) (2026-06-09,
„WP12 — Unified Render Layer"). Dieses Dokument ist Architektur-/Protokoll-
Referenz **und** Restarbeits-Tracker.

**Korrektur zur ursprünglichen Skizze**: der `PipelineResultFormatter` war
*nicht* bereits zwischen GUI und Webapp geteilt (nur GUI-Pipeline-Tab ↔
Agentic-Chat); die Webapp baute ihre Klassifikations-Karten in eigenem JS.
WP12.3 führt die geteilte Formatter-Nutzung in der Webapp erst ein.

Ermöglicht durch die QWebEngineView-Umstellung des GUI-Log/Chat-Renderers
(commit `7c68f67`): GUI und Webapp rendern jetzt beide HTML in einer Browser-
Engine, die Render-Chrome lässt sich erstmals teilen.

**Methode**: Code-Inspektion (`src/ui/web_log_view.py`,
`src/ui/unified_message_renderer.py`, `src/webapp/app.py` + `static/`) +
bestehende Konsolidierungs-Vorarbeit aus WP4/WP9.

**Querverweise**:
- [`renderer_registry_design.md`](legacy/renderer_registry_design.md) — WP4, Render-
  Slot-Architektur (Option C). WP12 ist der **Transport-/Frontend-Teil** dazu.
- [`frontend_tier_model.md`](legacy/frontend_tier_model.md) — WP9, Tier-Modell;
  Webapp = Tier-3 für die klassische Pipeline, Empfehlung „(B) Catch-Up".
- [`workflow_output_schemas.md`](workflow_output_schemas.md) — WP3, Slot-Vokabular.
- [`research_classic_vs_agentic.md`](legacy/research_classic_vs_agentic.md) — WP2, die
  DK/GND-Divergenz, die diese Doppel-Pflege motiviert.
- [`agentic_ui_workpackages.md`](legacy/agentic_ui_workpackages.md) — WP-Übersicht.

---

## 1. Context & Motivation

Heute rendern **zwei** Frontends dieselben Pipeline-Daten mit getrennter Chrome:

| | GUI | Webapp |
|---|---|---|
| Renderer | `WebLogView` (QWebEngineView) + `UnifiedMessageRenderer` | Vanilla-JS in `src/webapp/static/` |
| Transport | `page().runJavaScript(...)` | WebSocket (`/ws/{id}`), Token-Buffer 500 ms |
| Geteilt schon | `PipelineManager` + Callbacks, `PipelineResultFormatter` (DK/GND-HTML) | dito |
| Doppelt | Bubbles, Collapsibles, Stream-Blocks, Tool-Blocks, Theme-CSS | dito |

Die Render-Chrome ist an zwei Stellen gepflegt. Genau diese Divergenz hat in WP2
zu unterschiedlicher DK/GND-Darstellung geführt (behoben in `0cfba1a` durch den
gemeinsamen `PipelineResultFormatter`). Der nächste logische Schritt: auch die
**Chrome** (nicht nur die DK/GND-Fragmente) konsolidieren.

**Schlüssel-Beobachtung**: `WebLogView` ist faktisch ein HTML/CSS/JS-Renderer in
einer Browser-Engine. Die Webapp ist ebenfalls ein Browser. Damit ist ein
**gemeinsamer Render-Layer** möglich, der vorher (QTextBrowser-Cursor-Chirurgie)
ausgeschlossen war.

---

## 2. Ziel

Ein **gemeinsamer Render-Layer**: identisches CSS + identische JS-Render-
Funktionen + ein kleines, versioniertes **JSON-Render-Event-Protokoll**,
getrieben von zwei Transports.

```
                 ┌─────────────────────────────┐
   Pipeline ───▶ │ UnifiedMessageRenderer       │  (einziger Producer)
   Callbacks     │  → emittiert Render-Events   │
                 └──────────────┬──────────────┘
                                │  {type, …}  (JSON)
                ┌───────────────┴───────────────┐
        GUI-Transport                     Webapp-Transport
        WebLogView.runJavaScript          WebSocket-Broadcast
                │                                 │
                └────────► alima_render.js ◄───────┘   (geteilter DOM-Dispatcher)
                           alima_render.css           (geteiltes Theme)
```

Resultat: **pixelgleiches Rendering in GUI und Webapp, eine Pflegestelle.**

---

## 3. Architektur (3 Schichten)

**Schicht 1 — Shared Static Asset** (`src/webapp/static/alima_render.{js,css}`):
Die DOM-Render-Funktionen, die heute inline im `_HTML_TEMPLATE`-Scaffold von
`web_log_view.py` stehen — `appendBlock`, `appendCollapsible`,
`updateCollapsible`, `openAssistant`/`appendToken`/`finalizeAssistant`,
`openStreamBlock`/`appendStreamBlock`/`closeStreamBlock`, `maybeScroll` — plus das
Dark-Theme-CSS. Wird von **beiden** Frontends geladen.

**Schicht 2 — Render-Event-Protokoll** (versioniertes JSON): eine flache
Event-Liste, deren Typen 1:1 die Schicht-1-Funktionen treffen:

| `type` | Felder | Schicht-1-Funktion |
|---|---|---|
| `pipeline_log` | `html` | `appendBlock` |
| `block` / `html_block` | `html`, `kind?` | `appendBlock` |
| `collapsible` | `id`, `summary`, `body`, `open` | `appendCollapsible` |
| `collapsible_update` | `id`, `summary`, `body` | `updateCollapsible` |
| `assistant_open`/`_token`/`_finalize` | `header` / `text` / `html` | `openAssistant`/… |
| `stream_open`/`_token`/`_close` | `id`,`summary` / `text` / `id`,`summary`,`collapse` | `openStreamBlock`/… |
| `proposal` | `audit_id`, `tool`, `payload` | `appendBlock` (GUI) / ignorierbar (Webapp) |
| `system` | `text` | `appendBlock` |

Append-only + idempotent (per `id`), damit Webapp-Reconnect/Replay funktioniert
(die Webapp hat bereits Recovery + 30-min-WS-Timeout).

**Schicht 3 — Producer + Transport-Adapter**: `UnifiedMessageRenderer` baut die
HTML-Strings wie heute, sendet sie aber als Events an ein injiziertes
`transport.send(event)` statt direkt `self.web_view.*` aufzurufen.
- GUI-Transport: wrappt `WebLogView`, mappt Event → `runJavaScript(fn(...))`.
- Webapp-Transport: broadcastet das Event über den WebSocket der Session.
`PipelineResultFormatter` bleibt unverändert der Slot-Renderer für DK/GND und
landet als `html`-Feld in `html_block`-Events.

---

## 4. Schritte (Teil-Pakete)

- ✅ **WP12.1 — Asset-Extraktion** (Refactor, kein Verhaltenswechsel): CSS+JS aus
  dem `web_log_view.py`-Scaffold in `alima_render.{js,css}` gezogen; `WebLogView`
  inlined sie bei Konstruktion, die Webapp serviert sie statisch. Content-CSS
  unter `#log` gescoped, Schriftgröße via `--alima-fs`.
- ✅ **WP12.2 — Event-Protokoll + Producer-Abstraktion**: Qt-freies
  `src/core/render_events.py` (Event-Builder 1:1 zu den JS-Funktionen,
  `PROTOCOL_VERSION = 1`, `RenderTransport`-Protokoll + `MockTransport`);
  `src/ui/render_transport.py` (`WebLogViewTransport`). `UnifiedMessageRenderer`
  emittiert Events an injizierten Transport; Alt-Aufrufer mit `WebLogView`
  werden auto-gewrappt (rückwärtskompatibel).
- ✅ **WP12.3 — Webapp-Client**: per-Session `WebSocketRenderTransport`,
  Event-Puffer mit monotonem `seq` auf der `Session`, Replay bei Reconnect
  (per-Connection-Cursor), Polling-Cursor als Fallback; `app.js` dispatcht in
  eine `#log`-Region, dedupliziert per `seq`. Webapp behält ihr 5-Step-Widget
  (WP9 Tier-3) und übernimmt nur die DK/GND-Karten-Chrome.
- ✅ **WP12.4 — Konsolidierung**: DK/GND-Karten-HTML kommt aus geteilten
  `PipelineResultFormatter.format_dk_search_card_html` /
  `format_dk_classifications_card_html` (GUI-Panel + Webapp). **Reverse-Port**:
  die strukturierten Badge-Karten der Webapp wurden in den Shared Layer gehoben
  (`normalize_classifications` + `format_classification_badge_card_html`,
  `.classification-*`-CSS nach `alima_render.css`) — der Symmetrie-Gewinn von
  WP12: Webapp-Render-Komponenten fließen über denselben Layer in die GUI zurück.
- ⬜ **WP12.5 — Restarbeiten / Definition of Done**: siehe §9.

---

## 5. Scope

**In**: Log-/Chat-/Stream-/Collapsible-/Tool-/HTML-Block-Rendering (die
WebLogView-Chrome) + das geteilte Theme-CSS.

**Out**: Webapp-Eingabe-UI (Upload, Webcam, Session-Management), GUI-spezifische
Tabs/Widgets, **Auth des HTTP-Endpoints** (separates WP — siehe
[`chat_agent_roadmap.md`](chat_agent_roadmap.md) offene Frage 4 / P-ι).

---

## 6. Risiken

- **Transport-Semantik divergiert**: GUI `runJavaScript` (fire-and-forget,
  gequeued bis `loadFinished`) vs Webapp-WS (Reconnect, Multi-Client, Reihenfolge).
  → Protokoll append-only + per-`id` idempotent + replay-fähig halten.
- **Asset-Laden**: WebEngine (qrc/file/baseUrl) vs Webapp-Static-Serving — eine
  Pfad-Abstraktion nötig, damit dieselbe `alima_render.js` in beiden Kontexten lädt.
- **Tier-Mismatch**: Mutation-Proposals / Permission-Dialoge sind GUI-spezifisch
  (Tier-3). Protokoll muss **optionale/ignorierbare** Event-Typen erlauben.
- **Multi-Tab-Broadcast**: Webapp isoliert Sessions per Tab (eigenes HTML) — der
  Event-Strom muss strikt pro Session laufen.

---

## 7. Akzeptanzkriterien (Ist-Stand 2026-06-10)

1. 🟨 Ein Pipeline-Lauf erzeugt in GUI **und** Webapp visuell identische
   Log-/Result-Blöcke aus **demselben** Event-Strom.
   *Headless belegt (Tests vergleichen Event-Strom), visuell **nicht**
   verifiziert — siehe §9.2 (open).*
2. ✅ `UnifiedMessageRenderer` enthält keine GUI- oder Webapp-spezifische
   Render-Logik mehr — nur Event-Emission (Beleg: `MockTransport`-Tests in
   `tests/test_unified_message_renderer.py`).
3. ✅ Eine Pflegestelle (`alima_render.{js,css}`) für die Chrome; kein
   doppeltes Chrome-CSS. *Einschränkung: die Webapp-eigenen
   `.classification-*`-Summary-Karten existieren weiter neben den
   `#log`-Karten — bewusst (komplementär), aber UX-ungeprüft, siehe §9.4.*
4. ✅ Bestehende GUI-Tests grün; Protokoll-Tests (Mock-Transport) und
   Webapp-Tests (`tests/test_webapp_render_events.py`: Session-Puffer,
   Cursors, WS-Broadcast + Reconnect-Replay) grün. *Beleg: WP12-Commit
   `9552d93` mit 579 grünen / 5 übersprungenen Tests (headless).*

---

## 8. Dauer-Schätzung (grob, Skizze)

| Teil-Paket | Aufwand | Status |
|---|---|---|
| WP12.1 Asset-Extraktion | ~0.5 PT | ✅ |
| WP12.2 Event-Protokoll + Producer | ~1–1.5 PT | ✅ |
| WP12.3 Webapp-Client | ~2–3 PT | ✅ |
| WP12.4 Konsolidierung + Cleanup | ~1–2 PT | ✅ (+ Reverse-Port) |
| WP12.5 Restarbeiten (§9) | ~1–1.5 PT | ⬜ |

---

## 9. WP12.5 — Restarbeiten / Definition of Done

WP12 gilt erst als **abgeschlossen**, wenn alle fünf Punkte erledigt sind.
Reihenfolge = empfohlene Abarbeitung; 9.1 blockiert alles Weitere.

### 9.1 Commit der WP12-Arbeit ✅
Erledigt in Commit `9552d93` (2026-06-10, „WP12 WIP: unified render layer
12.1-12.4 + WP12.5 rest-work spec + WP13 cleanup workpackage") auf
Branch `agent`. Der Worktree-Stand ist eingefroren; abhängige Arbeiten
(§9.3, WP13) können sauber aufsetzen.

### 9.2 Visuelle Verifikation (GUI + Browser) ⬜
Bisher nur headless belegt (`QT_QPA_PLATFORM=offscreen`, `TestClient`,
`node --check`). Checkliste für einen echten Lauf (gleiches Dokument durch
beide Frontends):
- [ ] GUI: Collapsible während laufendem Stream öffnen → schließen → öffnen
      (Streaming intakt, Zustand bleibt).
- [ ] GUI ↔ Browser: DK/GND-Badge-Karten nebeneinander vergleichen
      (Badges, Farben auf dunkler Fläche, Titel-Listen, Konfidenz).
- [ ] Browser: WS-Disconnect erzwingen → Reconnect → vollständiges Replay
      ohne Duplikate (seq-Dedupe).
- [ ] Browser: zwei Tabs, zwei Sessions → keine Event-Leckage zwischen Sessions.
- [ ] GUI: Schriftgrößen-Wechsel (`--alima-fs`) wirkt auf Karten + Chrome.

### 9.3 Fehler-Events rendern (Anschluss an WP A) ⬜
Seit WP A (commit `7850222`) emittiert der `PipelineManager`
`state.pipeline_step` mit `status="error"` **und** `error`-Payload
(Fehlertext); ein fehlgeschlagener Schritt stoppt die Pipeline. Das Rendering
wurde bewusst zurückgestellt, um die uncommittete WP12-Arbeit nicht zu
vermischen. Zu tun (nach 9.1):
- `PipelineChatPanel`: `status="error"` als rot markiertes Collapsible
  rendern (Summary `❌ <step name>`, Body = `error`-Text), statt den
  Tool-Block offen/„running" stehen zu lassen.
- Webapp: dito über den Event-Strom (neuer Event-Builder oder `kind="error"`
  auf `collapsible`/`block` — Entscheidung beim Implementieren, Protokoll
  ggf. `PROTOCOL_VERSION` beachten; additives Feld = kein Bump).
- Die Quelle-down-Warnungen aus `execute_gnd_search` (`⚠️ Quelle(n)
  fehlgeschlagen …`) kommen bereits als Stream-Tokens durch beide Frontends —
  nur prüfen, nicht neu bauen.

### 9.4 Webapp-Doppelanzeige DK/GND — Operator-Review ⬜
Die Webapp zeigt Klassifikationen doppelt: kompaktes Summary-Panel **und**
geteilte `#log`-Karten. Gewollt komplementär (wie GUI), aber nie von einem
Operator UX-geprüft. Entscheidung: behalten / Summary eindampfen /
`#log`-Karten nur auf Anforderung.

### 9.5 Tier-Entscheidung: agentic Tool-Bus-Chrome → Webapp ⬜
Der agentische Tool-Bus (Tool-Calls/-Results als Collapsibles) wird derzeit
**nicht** an die Webapp emittiert; nur der klassische DK/GND-Pfad ist
serverseitig verdrahtet. Per WP9 ist die Webapp Tier-3 — explizit
entscheiden (und hier dokumentieren), ob das so bleibt oder agentische
Läufe im Browser sichtbar werden sollen. Kein Code-Zwang, aber eine
offene Festlegung.
