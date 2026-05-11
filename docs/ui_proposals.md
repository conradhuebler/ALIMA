# GUI-Vorschläge (3 Varianten)

**Status**: Konzept-Skizzen für Diskussion. ASCII-Wireframes,
keine Pixel-Designs. Pro Vorschlag: Layout, Vor-/Nachteile,
Migrations-Aufwand, Use-Case-Eignung.

Querverweise:
- Anforderungen: [`ui_requirements_catalog.md`](ui_requirements_catalog.md)
- Konsolidierung: [`consolidation_inventory.md`](consolidation_inventory.md)
- WP-Pläne: [`wp_detailed_plans.md`](wp_detailed_plans.md)

Alle Vorschläge teilen folgende Konstanten:
- **GlobalStatusBar** unten — bleibt (B existing).
- **AgenticContextWidget** als Dock — bleibt (B1).
- **ChatWidget** als Dock — bleibt (B8), Konzept aus WP8.

---

## Vorschlag A — Workflow-First

**Leitidee**: Workflow ist die zentrale Abstraktion. UI dreht sich um
„welcher Workflow läuft gerade, was sind seine Steps, wie sehe ich
deren Output". Single-Step-Tabs entstehen automatisch aus dem
aktiven Workflow.

### Layout

```
┌───────────────────────────────────────────────────────────────────────────────┐
│ [☰ ALIMA]  Workflow: [▼ ALIMA Pipeline (alima.yaml)]  Provider: [▼ Auto]  ⚙   │
├───────────────────────────────────────────────────────────────────────────────┤
│ ┌──────────────┐ ┌──────────────────────────────────────┐ ┌──────────────┐  │
│ │ Workflow-    │ │  Step-Output / Stream                │ │  Context     │  │
│ │ Browser      │ │  ┌────────────────────────────────┐  │ │  (Dock)      │  │
│ │              │ │  │ ▶ extraction       ✓ 1.2s      │  │ │ ┌─────────┐  │  │
│ │ Standard     │ │  │   → 18 keywords, title         │  │ │ │ extr.   │  │  │
│ │ ◉ ALIMA      │ │  │ ▶ search          ✓ 14s        │  │ │ │ 18kw    │  │  │
│ │ ○ Classic    │ │  │   → 1042 GND entries           │  │ │ ├─────────┤  │  │
│ │              │ │  │ ▶ selection_chunks ✓ 38s       │  │ │ │ search  │  │  │
│ │ Andere       │ │  │   → 80 selected (3 chunks)     │  │ │ │ 1042    │  │  │
│ │ ○ Catalog    │ │  │ ▶ selection       ⏵ running... │  │ │ ├─────────┤  │  │
│ │ ○ Title-List │ │  │   ╰─ Stream: token token...    │  │ │ │ ...     │  │  │
│ │ ○ Synonym    │ │  │ ░ classification   pending     │  │ │ └─────────┘  │  │
│ │ ○ Batch-Meta │ │  │ ░ dk_postprocess   pending     │  │ │              │  │
│ │              │ │  └────────────────────────────────┘  │ │              │  │
│ │ + Custom...  │ │                                      │ │              │  │
│ │              │ │  Steps:  [extr][search][...]         │ │              │  │
│ │              │ │  Tab pro Step (auto-generated)       │ │              │  │
│ │              │ │  └─ Klick öffnet Step-Detail/Single  │ │              │  │
│ │ ▶ Run        │ │                                      │ │              │  │
│ │ ⏵ Resume     │ │                                      │ │              │  │
│ └──────────────┘ └──────────────────────────────────────┘ └──────────────┘  │
│ ┌─────────────────────────────────────────────────────────────────────────┐ │
│ │ 💬 Chat (Dock, kollabierbar)                                            │ │
│ └─────────────────────────────────────────────────────────────────────────┘ │
├───────────────────────────────────────────────────────────────────────────────┤
│ Provider: ollama/qwen2.5:14b · Cache: 12345 entries · Pipeline: ✓ done · 47s │
└───────────────────────────────────────────────────────────────────────────────┘
```

**Sekundär-Tabs** (Top-Level neben „Workflow"):
- Review (JSON laden + multi-state-Diff)
- Tools (Crossref, Bild-OCR, GND-Suche, UB-Katalog — eigenständig)
- Batch
- Einstellungen

### Vorteile
- **Konsistenz**: jeder Workflow erscheint gleich, custom-Workflows
  out-of-the-box brauchbar.
- **Standard-Workflow Default-Highlighted** (REQ-W2 + M1).
- **Single-Step in Step-Detail-Tab** integriert (REQ-M2.x), keine
  separate Tab-Inflation.
- **Klassisch ist nur ein Workflow-Wechsel** weg (Forschungspfad
  bleibt, kein Sonderpfad in UI).
- **Renderer-Registry** voll genutzt (WP4) — pro Step Output-Renderer.

### Nachteile
- **Massiver UI-Umbau**. Heutige 10 Tabs → 4 Top-Level + Step-Auto-Tabs.
  Hoher Migrationsaufwand (WP10 P-γ + P-θ).
- **User-Re-Education** nötig: alte Tabs sind weg.
- **Auto-Tab-Generator** bei vielen Steps unübersichtlich (alima.yaml hat
  7 Steps).
- **Workflow-Browser braucht gute Beschreibung** pro Workflow, sonst
  User wählt blind.

### Migrations-Aufwand
**Hoch** (~3-4 Wochen). Abhängig WP4 + WP5 voll umgesetzt + WP10-Phase
γ + θ.

### Geeignet für
- Mittelfristige Vision (6+ Monate Horizont).
- Wenn Workflow-Vielfalt prioritär (M3 wichtiger als gedacht).

---

## Vorschlag B — Pipeline-First mit Workflow-Picker

**Leitidee**: Pipeline-Tab bleibt zentrales UI-Element wie heute, kriegt
nur einen prominenteren Workflow-Picker oben + bessere Output-Ansichten
durch Renderer-Registry. Single-Step-Tabs bleiben als Tools daneben.

### Layout

```
┌───────────────────────────────────────────────────────────────────────────────┐
│ [☰ ALIMA]                                          Provider: [▼ Auto]  ⚙      │
├───────────────────────────────────────────────────────────────────────────────┤
│ Tabs:                                                                         │
│ [🚀 Pipeline] [📝 Abstract] [🔍 GND] [📚 UB] [📊 DK] [📊 Review] [🔍 Vergl] │
├───────────────────────────────────────────────────────────────────────────────┤
│ Pipeline-Tab:                                                                 │
│  ┌─ Workflow ─────────────────────────────────────────────────────────────┐  │
│  │ [▼ ALIMA (alima.yaml)]   ⓘ Standard-Workflow für Erschließung         │  │
│  │  Modus: ◉ Agentic ○ Classic (Forschungspfad)                          │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
│  ┌─ Eingabe ──────────────────────────────────────────────────────────────┐  │
│  │ [Text] [DOI] [PDF] [Bild]                                              │  │
│  │  ┌───────────────────────────────────────────────────────────────┐    │  │
│  │  │ Abstract eingeben…                                             │    │  │
│  │  └───────────────────────────────────────────────────────────────┘    │  │
│  │              [▶ Run] [⏸ Stop]                                          │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
│  ┌─ Steps + Stream ───────────────────────────────────────────────────────┐  │
│  │ ✓ extraction      ─── 18 keywords                                      │  │
│  │ ✓ search          ─── 1042 GND entries                                 │  │
│  │ ✓ selection_chunks─── 80 selected                                      │  │
│  │ ⏵ selection       ─── Stream...                                        │  │
│  │ ░ classification                                                        │  │
│  │ ░ dk_postprocess                                                        │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
│  ┌─ Output ───────────────────────────────────────────────────────────────┐  │
│  │ Sub-Tabs (auto je nach Workflow-Output):                               │  │
│  │ [Final Keywords] [Schlagwortketten] [DK/RVK] [K10+] [Stats]            │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
│ ┌─────────────────────────────────────────────────────────────────────────┐ │
│ │ 💬 Chat                                            🧠 Agentic Context   │ │
│ └─────────────────────────────────────────────────────────────────────────┘ │
└───────────────────────────────────────────────────────────────────────────────┘
```

### Vorteile
- **Geringer Umbau**: bestehende Tabs bleiben.
- **Workflow-Picker prominent**, agentic/classic 1-Klick-Switch.
- **Sub-Tabs sind Renderer** (WP4) — können workflow-spezifisch sein,
  keine Hardcoding.
- **Tool-Tabs** (Abstract, GND-Suche etc.) bleiben verfügbar für
  spontane Einzelaktionen.
- **Bekannte UX**: User-Reibungsverlust minimal.

### Nachteile
- **Tab-Wildwuchs bleibt** — Konsolidierung nicht erzwungen (K6).
- **Custom-Workflows** mit ungewöhnlichen Outputs (z.B. duplicate-table)
  brauchen weiterhin Renderer (WP4) — aber Pipeline-Tab muss flexibel
  genug sein, sonst leer.
- **Single-Step über Tools-Tabs** nicht workflow-step-aligned (K1
  bleibt).
- **DkAnalysisUnifiedTab + UB-Katalog-Tab** überlappen weiter.

### Migrations-Aufwand
**Mittel** (~2 Wochen). Abhängig WP4 (Renderer) + WP-Hint-Erweiterung
für Workflow-Picker.

### Geeignet für
- Schritt 1 in einer Mehrphasen-Migration.
- Wenn User-Reibung minimieren prioritär.
- Wenn unklar wie weit Workflow-Pluralität wirklich gefordert ist.

---

## Vorschlag C — Modus-Schalter mit getrennten Workspaces

**Leitidee**: ALIMA bietet drei „Modi" als Top-Level-Wahl, jeder Modus
hat eigene UI-Optimierung. User wählt explizit was er tut.

### Layout

```
┌───────────────────────────────────────────────────────────────────────────────┐
│ [☰ ALIMA]   Modus: [⚡ Standard] [🔬 Forschung] [🔧 Tools] [🌐 Custom]      ⚙ │
├───────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│ MODUS: ⚡ Standard (alima.yaml agentic)                                       │
│                                                                               │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │  Eingabe: [Text/DOI/PDF/Bild]                                            │ │
│  │   ┌─────────────────────────────────────────┐                            │ │
│  │   │ Abstract...                              │  [▶ Erschließen]          │ │
│  │   └─────────────────────────────────────────┘                            │ │
│  │                                                                           │ │
│  │  ► extraction  ► search  ► selection  ► classification  ► postprocess    │ │
│  │  Stream / Result-Bereich (Renderer-Registry-basiert)                     │ │
│  │                                                                           │ │
│  │  ┌─ Ergebnis ────────────────────────────────────────────────────────┐  │ │
│  │  │ Schlagworte:         [chip] [chip] [chip] ...                     │  │ │
│  │  │ Schlagwortketten:    [chain1] [chain2] ...                        │  │ │
│  │  │ DK-Klassifikation:   DK 543.42 (12 Titel) [▶]                     │  │ │
│  │  │ K10+-Export:         [Copy]                                        │  │ │
│  │  └────────────────────────────────────────────────────────────────────┘  │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                                                               │
│  💬 Chat: "Frage zum Ergebnis…"                                              │
└───────────────────────────────────────────────────────────────────────────────┘

------- alternativ MODUS: 🔬 Forschung -------
┌───────────────────────────────────────────────────────────────────────────────┐
│ Forschungs-Modus (classic Pipeline für Reproduzierbarkeit + Vergleich)        │
│  ┌─ Run-Setup ────────────────────────────────────────────────────────────┐  │
│  │ Workflow: alima_classic (rigid)                                         │  │
│  │ Provider: [▼]  Modell: [▼]  Seed: [42]  Reproducible: ☑              │  │
│  │ Eingabe: ...                                                            │  │
│  │ [▶ Run] [▶ Run + Compare to last]                                       │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
│  ┌─ Vergleich ────────────────────────────────────────────────────────────┐  │
│  │ Lauf A (current) │ Lauf B (gespeichert) │ Diff                         │  │
│  │ ...              │ ...                  │ ▼ neue Keywords ...          │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────────────────────────────────┘

------- alternativ MODUS: 🔧 Tools -------
┌───────────────────────────────────────────────────────────────────────────────┐
│ Tools-Modus: Einzelfunktionen ohne Pipeline                                   │
│  Sidebar:                                                                     │
│  ▸ Crossref-DOI-Lookup                                                        │
│  ▸ Bildanalyse                                                                │
│  ▸ GND-Suche (Live)                                                           │
│  ▸ UB-Katalog-Suche                                                           │
│  ▸ Single-Step LLM-Aufruf (welcher Step? [▼])                                │
│  ▸ Synonym-Expansion (catalog_search.yaml)                                   │
│  ▸ Batch-Verarbeitung                                                         │
│                                                                               │
│  Hauptbereich: ausgewähltes Tool                                              │
└───────────────────────────────────────────────────────────────────────────────┘

------- alternativ MODUS: 🌐 Custom -------
┌───────────────────────────────────────────────────────────────────────────────┐
│ Custom-Workflow: jede YAML laden, ausführen, debuggen                         │
│  ┌─ Workflow ─────────────────────────────────────────────────────────────┐  │
│  │ [▼ catalog_search.yaml] [+ Eigene laden...] [✎ YAML editieren]        │  │
│  │ Description: Multi-source catalog lookup with LLM ranking              │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
│  Eingabe: dynamisches Form aus Workflow `inputs:`-Schema                     │
│  Steps + Stream + Output-Renderer (raw_json fallback)                        │
└───────────────────────────────────────────────────────────────────────────────┘
```

### Vorteile
- **Modus-Mentalität**: User weiß sofort was er tut (Standard ≠
  Forschung ≠ Custom).
- **Forschungspfad explizit unterstützt** mit Reproduzierbarkeits-
  Flags + integriertem Vergleich.
- **Tools-Modus konsolidiert** alle Einzeltools in einer Sidebar
  → Top-Level-Tab-Inflation reduziert.
- **Custom-Modus** macht Workflow-Vielfalt explizit-Tier-3-Feature
  ohne Standard-User zu verwirren.
- **Skaliert für Personas**: Bibliothekarin-Standard, Forscherin-
  Forschung, Power-User-Custom.

### Nachteile
- **Modus-Wechsel ist Friktion** wenn jemand zwischen Modi springt.
- **3-4 Wokspaces zu pflegen** (Code-Trennung).
- **Forschungs-Modus** hat eigene Reproduzierbarkeits-Anforderung
  → seed muss in agentic nachgerüstet (WP11) ODER classic-only.
- **„Standard" muss Top-Notch sein**, sonst Power-User springen
  zu Custom.

### Migrations-Aufwand
**Hoch-mittel** (~3 Wochen). Modus-Schalter neu, Workspaces können
existierende Tab-Komponenten bündeln. Nicht alles neu, aber
Bündelung + Custom-Mode-UI sind Neubau.

### Geeignet für
- Wenn User-Personas klar getrennt sind (Operator-Antwort auf offene
  Frage 4).
- Wenn Forschungspfad eigenständige UX braucht (Reproduzierbarkeits-
  Flags, Vergleichs-Workflow integriert).

---

## Vergleichs-Tabelle

| Kriterium | A: Workflow-First | B: Pipeline+Picker | C: Modus-Workspaces |
|---|---|---|---|
| Migrations-Aufwand | hoch (3-4 W) | mittel (2 W) | hoch-mittel (3 W) |
| User-Reibung | hoch (alles neu) | gering | mittel (neue Modi) |
| Workflow-Vielfalt-Support | exzellent | ok | gut (im Custom-Mode) |
| Klassischer-Pfad | normaler Workflow | 1-Klick-Toggle | eigener Modus |
| Single-Step | auto-Tabs pro Step | weiter via Tool-Tabs | im Tools-Modus |
| Konsolidierung | erzwungen | nicht | teil-erzwungen |
| Forschungs-Use-Case | nicht spezialisiert | nicht spezialisiert | first-class |
| Skalierbarkeit Workflows | sehr gut | mittel | gut |
| Lernkurve | hoch | minimal | mittel |
| Renderer-Abhängigkeit | hoch | hoch | hoch |
| Chat-Integration | universell | universell | pro-Modus |

## Empfehlung (Diskussionsgrundlage)

**Phasen-Mix-Vorschlag** statt Einzel-Wahl:

1. **Sofort (P-α/β/γ aus WP10)** → Vorschlag B: Renderer-Registry
   einbauen, Pipeline-Tab kriegt Workflow-Picker, sub-tabs werden
   Renderer. Geringe User-Reibung, Foundation für später.
2. **Mittelfristig (P-δ/ε/ζ)** → Migration zu Vorschlag C:
   Modus-Schalter einführen, alte Tabs nach Modus bündeln. Forschungs-
   Modus dediziert (Operator-Vorgabe respektiert).
3. **Langfristig (P-η/θ + Vision)** → optional zu Vorschlag A:
   wenn Workflow-Vielfalt sich als wichtig herausstellt, Workflow-
   First als Standard-Modus.

So ist B der Einstieg (geringe Reibung), C die Konsolidierung
(Persona-aware), A die Vision (workflow-pluralistisch).

## Offene Fragen für Operator-Entscheidung

1. **Personas klar?** Bibliothekarin / Forscherin / Power-User —
   getrennt oder einer? Beeinflusst A vs C.
2. **Workflow-Vielfalt-Realismus**: werden tatsächlich mehr als
   3-4 Workflows produktiv genutzt? Beeinflusst A vs B/C.
3. **User-Reibung-Toleranz**: Standard-User OK mit großer Änderung
   oder lieber unsichtbar?
4. **Forschungs-Modus Eigenständigkeit**: classic in eigenem Modus
   (C) oder als Workflow-Variante (A/B)?
5. **YAML-Editor in UI**: Custom-Modus von C wäre attraktiver mit
   YAML-Editor — out-of-scope oder dazu?
