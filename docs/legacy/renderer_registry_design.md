# Renderer-Registry-Design (WP4)

**Status**: Architektur-Skizze für T2-Entscheidung *„Renderer-Plugin-
Architektur ja/nein"*. Output von WP4 aus
[`wp_detailed_plans.md`](wp_detailed_plans.md) Sektion *WP4 — Renderer-
Registry-Skizze*. **Pseudo-Code only — keine Implementation.** Die
Migration in produktiven Code erfolgt in WP10.

**Methode**: Synthese von
- WP3-Slot-Vokabular ([`workflow_output_schemas.md`](../workflow_output_schemas.md))
- bestehendem Registry-Pattern in [`registry.py`](../src/core/agents/registry.py)
- bestehender Render-Logik (`AnalysisReviewTab`, `AgenticContextWidget`,
  diverse `setHtml`-Aufrufe).

**Querverweise**:
- [`workflow_output_schemas.md`](../workflow_output_schemas.md) — Slot-Vokabular (WP3).
- [`wp_detailed_plans.md`](wp_detailed_plans.md) WP4 — Soll-Definition.
- [`agentic_ui_workpackages.md`](agentic_ui_workpackages.md) WP4 — frühere Skizze.
- [`audit_tab_inventory.md`](audit_tab_inventory.md) — AnalysisReviewTab-Audit (WP1).
- [`research_classic_vs_agentic.md`](research_classic_vs_agentic.md) — Output-Feld-Diff (WP2).

## 1. Context & Motivation

### Warum überhaupt Renderer-Registry?
Heute ist `AnalysisReviewTab`
([`src/ui/analysis_review_tab.py:45`](../src/ui/analysis_review_tab.py))
der de-facto Output-Renderer mit 11 fest verdrahteten Sub-Tabs. Jede
Sub-Tab hat eigene `populate_*()`-Methode mit hardcoded HTML- oder
Tabellen-Generierung:

| Sub-Tab | Render-Methode | Stil |
|---|---|---|
| DK/RVK (Tab 7) | `populate_detail_tabs()` Z. 670-732 | `setHtml()` + Color-Divs |
| K10+ Export (Tab 8) | `populate_detail_tabs()` Z. 734-736 | `setPlainText()` |
| Statistiken (Tab 9) | `populate_statistics()` Z. 870+ | `setPlainText()` |
| DK-Statistik (Tab 10) | `populate_dk_statistics()` Z. 923+ | QTableWidget |
| Such-Ergebnisse (Tab 3) | `populate_search_results_table()` Z. 827 | QTableWidget |
| Iterationsverlauf (Tab 6) | `populate_iteration_history()` | QTableWidget |

Jeder neue Workflow erzwingt Code-Edit dieser Klasse. Ergebnis: 6
aktive Workflows × 5-7 Steps = potenziell 30-42 hardcoded Sub-Tabs.
Tab-Inflation in WP1 als Anti-Pattern markiert
([`audit_tab_inventory.md`](audit_tab_inventory.md)).

### Vorbild im Bestand
`AgenticContextWidget.AgenticStepPanel._render_body()`
([`src/ui/agentic_context_widget.py:210`](../src/ui/agentic_context_widget.py))
zeigt bereits ein workflow-agnostisches Render-Pattern:
1. Panel kennt nur die `output_paths` seines Steps (aus `WorkflowDef`).
2. Pro Feld dispatch nach Typ-Heuristik: `_chips()` (Z. 280),
   `_dict_list()` (Z. 307), `_chains()` (Z. 417), `_render_generic()`
   (Z. 443).
3. Truncation-Sentinel `_truncated` einheitlich behandelt.

Diese Heuristiken sind faktisch *Mini-Renderer*. WP4 macht sie
explizit + pluggable + frontend-übergreifend.

### Beziehung zu anderen WPs
- **Konsumiert WP3**: Slot-Vokabular (11 Slots) ist Input.
- **Liefert an WP5**: Single-Step-UI braucht Output-Renderer.
- **Liefert an WP8**: Chat-Tool-Outputs ggf. via Renderer.
- **Blockiert WP10**: Migration der hardcoded Tabs.

## 2. API-Sketch — `BaseRenderer`

### Vertrag
Pseudo-Code für `src/ui/renderers/base.py` (nicht zu erstellen in
WP4 — nur Spezifikation):

```python
# Pseudo-Code, nicht zur Implementation
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
# Qt-Import-Kommentar zeigt Render-Backend
# from PyQt6.QtWidgets import QWidget


class BaseRenderer(ABC):
    """Abstract base for slot renderers.

    Subclasses implement at least :meth:`render_html` (canonical) and
    SHOULD implement :meth:`render_qt` for native Qt-Widgets. CLI is
    optional and falls back to JSON dump in the base class.
    """

    #: Slot name matching WP3 vocabulary (e.g. "slot:dk_table").
    output_slot: str = ""

    #: Optional input-schema reference (TypedDict / Pydantic class).
    #: WP4 specifies the *contract* only; enforcement is a later WP.
    data_schema: Any = None

    @abstractmethod
    def render_html(self, data: Any, context: Optional[Dict] = None) -> str:
        """Return self-contained HTML (no external CSS/JS)."""

    def render_qt(self, data: Any, context: Optional[Dict] = None):
        """Return a QWidget. Default: QTextEdit with render_html()."""
        # Default: embed HTML in read-only QTextEdit (mirrors today's
        # `setHtml()` pattern in pipeline_tab.py / analysis_review_tab.py).
        ...

    def render_cli(self, data: Any, context: Optional[Dict] = None) -> str:
        """Return plain-text representation. Default: json.dumps."""
        import json
        return json.dumps(data, indent=2, ensure_ascii=False, default=str)
```

### Mirror-Verhältnis zu `BaseStep`
`BaseRenderer` spiegelt
[`BaseStep`](../src/core/agents/steps/base_step.py) (Z. 66):

| Aspekt | `BaseStep` | `BaseRenderer` |
|---|---|---|
| Abstract method | `run(context) -> Dict` | `render_html(data, context) -> str` |
| Registry-Eintrag | `STEP_REGISTRY: Dict[str, Type]` | `RENDERER_REGISTRY: Dict[str, Type]` |
| Decorator | `@register_step(name)` | `@register_renderer(slot)` |
| Look-up | `get_step_class(name)` | `get_renderer(slot, fallback="slot:raw_json")` |
| Identifier | `self.config.id` | `cls.output_slot` |

### Context-Parameter
`context` ist Dict mit optionalen Render-Hints (Theme-Color-Palette,
Truncation-Limit, Locale). Renderer dürfen `context=None` annehmen und
sinnvolle Defaults wählen. **Renderer DARF NICHT** auf
`SharedContext` zugreifen (Layering-Bruch).

## 3. Registry-Mechanik

### Datei
`src/ui/renderers/registry.py` (nicht zu erstellen in WP4).

### Implementierungs-Skizze
```python
# Pseudo-Code — gleiche Mechanik wie src/core/agents/registry.py:26-46
from typing import Dict, Type
from .base import BaseRenderer

RENDERER_REGISTRY: Dict[str, Type[BaseRenderer]] = {}


def register_renderer(slot: str):
    """Decorator: register a BaseRenderer subclass under ``slot``.

    Mirrors @register_step in src/core/agents/registry.py.
    Re-registering the same name raises ValueError.
    """
    def _wrap(cls):
        if slot in RENDERER_REGISTRY and RENDERER_REGISTRY[slot] is not cls:
            raise ValueError(
                f"Renderer slot '{slot}' already registered to "
                f"{RENDERER_REGISTRY[slot].__name__}"
            )
        RENDERER_REGISTRY[slot] = cls
        return cls
    return _wrap


def get_renderer(slot: str, fallback: str = "slot:raw_json") -> Type[BaseRenderer]:
    """Look up a renderer by slot, falling back to raw_json on miss."""
    if slot in RENDERER_REGISTRY:
        return RENDERER_REGISTRY[slot]
    return RENDERER_REGISTRY[fallback]   # raw_json MUST be registered


def list_renderers() -> list[str]:
    return sorted(RENDERER_REGISTRY)
```

### Konsistenz-Argument
- **Dieselbe Decorator-Signatur** wie `@register_step` / `@register_tool_fn`
  (collision-check inklusive).
- **Dieselbe Look-up-Konvention** (`get_*` für die Klasse, `list_*` für
  Inventar).
- **Plugin-Pfad gleich**: Discovery erfolgt durch Modul-Import in
  `src/ui/renderers/__init__.py` (analog zu
  `src/core/agents/__init__.py`).

## 4. YAML-Annotation + Slot-Resolution

### Default-Strategie: Code-side via register-fn-Hint
WP3 Sektion 7 schlägt vor, dass deterministic-fns ihren Output-Slot
über eine Decorator-Erweiterung deklarieren. WP4 übernimmt diesen
Pfad:

```python
# Erweiterung von src/core/agents/registry.py — Pseudo-Code
TOOL_FN_SLOT_HINT: Dict[str, str] = {}

def register_tool_fn(name: str, returns_slot: str = "slot:raw_json"):
    def _wrap(fn):
        ...  # bestehender collision-check
        TOOL_FN_REGISTRY[name] = fn
        TOOL_FN_SLOT_HINT[name] = returns_slot   # NEU
        return fn
    return _wrap
```

WP3-Tabelle (Sektion 7) liefert das initial-Mapping für die 9 heutigen
register-fns (`gnd_batch_search` → `slot:gnd_pool`, `dk_search_agentic`
→ `slot:dk_table`, …).

### Optional: YAML-Annotation pro Step
Für LLM-Agent-Steps (wo Felder nicht durch eine register-fn erzeugt
werden) bleibt YAML-Annotation als Ergänzung:

```yaml
# workflows/alima_classic.yaml (Pseudo-Sketch)
- id: selection
  type: llm_agent
  outputs:
    keyword_chains: response.keyword_chains
    missing_concepts: response.missing_concepts
  output_schema:                          # NEU, optional
    keyword_chains: slot:keyword_chains
    missing_concepts: slot:keyword_list
```

`output_schema:` ist **kein** required-Key — der Resolver fällt zurück.

### Slot-Resolution-Algorithmus
Ein Output-Feld wird auf einen Slot abgebildet via:

1. **Explicit YAML** `output_schema.<feld>` (falls vorhanden).
2. **TOOL_FN_SLOT_HINT** für deterministic-steps (Lookup über
   `step.config.raw["function"]`).
3. **Feldnamen-Heuristik**:
   - `*_keywords`, `*_concepts` → `slot:keyword_list`
   - `*_chains` → `slot:keyword_chains`
   - `gnd_entries`, `*_alternatives` → `slot:gnd_pool`
   - `dk_search_results`, `dk_*_entries` → `slot:dk_table`
   - `*_classifications` → `slot:classification_list`
   - `*_synonyms`, `*_candidates` → `slot:synonym_set`
   - `*_analysis`, `*_summary` → `slot:text_blob`
   - `duplicate_*` → `slot:duplicate_table`
4. **Fallback**: `slot:raw_json`.

Die Heuristiken decken die heutigen SharedContext-Felder (WP3 Sektion
3.1) ab. Neue Workflows können YAML-Annotation oder
TOOL_FN_SLOT_HINT explizit setzen.

### Code-Lokation für die Auflösung
`WorkflowExecutor` (oder ein neues Modul
`src/ui/renderers/slot_resolver.py`) wird Map
`field → slot` aus dem geladenen Workflow ableiten und dem UI-Layer
zur Verfügung stellen. WP4 spezifiziert nur die *Logik*, nicht den
Integrationspunkt; das klärt WP5/WP10.

## 5. Beispiel-Renderer (Pseudo-Code, 3 Demos + 1 Fallback)

### 5.1 `slot:dk_table` — DK-Klassifikations-Tabelle

**Quelle**:
- HTML-Variante extrahiert aus
  [`analysis_review_tab.py:670-732`](../src/ui/analysis_review_tab.py)
  (Color-Divs + `<ol>`-Titellisten).
- Qt-Variante extrahiert aus
  [`analysis_review_tab.py:923+`](../src/ui/analysis_review_tab.py)
  (`populate_dk_statistics` — 6-Spalten-`QTableWidget` mit `most_frequent`).

**Input-Schema** (aus WP3 Sektion 2):
```python
list[{
    "dk": str,                  # z.B. "57.62"
    "classification_type": str, # "DK" | "RVK"
    "titles": list[str],        # Kataloge-Treffer
    "count": int,               # total occurrences
}]
```

**Pseudo-Code**:
```python
@register_renderer("slot:dk_table")
class DkTableRenderer(BaseRenderer):
    output_slot = "slot:dk_table"

    def render_html(self, data, context=None) -> str:
        # 1:1 aus analysis_review_tab.py:670-732 extrahiert
        parts = ["<html><body style='font-family:Arial,sans-serif;'>"]
        for idx, row in enumerate(data, 1):
            count = row.get("count", 0)
            color, bg, _, _ = get_confidence_style(count)
            parts.append(
                f"<div style='background-color:{bg}; padding:12px; ...; "
                f"border-left:4px solid {color};'>"
                f"<h2 style='color:{color}'>#{idx} {row['dk']}</h2>"
                f"<span style='color:{color}'>{'🟩'*min(5,(count//10)+1)} {count}</span>"
                f"</div>"
            )
            if row.get("titles"):
                parts.append("<ol>")
                for t in row["titles"]:
                    parts.append(f"<li>{_esc(t)}</li>")
                parts.append("</ol>")
        parts.append("</body></html>")
        return "".join(parts)

    def render_qt(self, data, context=None):
        # Native QTableWidget — 6 Cols: Rank, DK, Type, Count, Titles-Preview, Bar
        # Pattern aus analysis_review_tab.py:957-987 extrahiert
        from PyQt6.QtWidgets import QTableWidget, QTableWidgetItem
        tbl = QTableWidget(len(data), 6)
        tbl.setHorizontalHeaderLabels(["#", "DK", "Type", "Count", "Top-Titel", "Confidence"])
        for row, item in enumerate(data):
            tbl.setItem(row, 0, QTableWidgetItem(str(row + 1)))
            tbl.setItem(row, 1, QTableWidgetItem(item["dk"]))
            tbl.setItem(row, 2, QTableWidgetItem(item.get("classification_type", "DK")))
            tbl.setItem(row, 3, QTableWidgetItem(str(item.get("count", 0))))
            tbl.setItem(row, 4, QTableWidgetItem("; ".join(item.get("titles", [])[:3])))
            # Confidence-Color analog Z. 957+
        return tbl

    def render_cli(self, data, context=None) -> str:
        # ASCII-Tabelle
        lines = ["DK     Type Count Titles"]
        for r in data:
            t = "; ".join(r.get("titles", [])[:2])
            lines.append(f"{r['dk']:<6} {r.get('classification_type','DK'):<4} "
                         f"{r.get('count',0):>5} {t[:40]}")
        return "\n".join(lines)
```

### 5.2 `slot:keyword_chains` — Schlagwortketten mit Begründung

**Quelle**: AnalysisReviewTab Sub-Tab + AgenticContextWidget
`_chains()` ([`agentic_context_widget.py:417`](../src/ui/agentic_context_widget.py)).

**Input-Schema** (WP3 Sektion 2):
```python
list[{"chain": list[str], "reason": str}]
```

**Pseudo-Code**:
```python
@register_renderer("slot:keyword_chains")
class KeywordChainsRenderer(BaseRenderer):
    output_slot = "slot:keyword_chains"

    def render_html(self, data, context=None) -> str:
        # 1:1 aus agentic_context_widget.py:417-441 extrahiert
        rows = [f"<div>keyword_chains ({len(data)})</div>"]
        for ch in data:
            chain = " → ".join(str(c) for c in ch.get("chain", []))
            reason = ch.get("reason", "")
            rows.append(
                f"<div><span>• {_esc(chain)}</span> "
                f"<span style='color:#6272a4'>{_esc(reason)}</span></div>"
            )
        return "".join(rows)

    def render_qt(self, data, context=None):
        # QTreeWidget: parent = chain-string, children = reason-Notes
        from PyQt6.QtWidgets import QTreeWidget, QTreeWidgetItem
        tree = QTreeWidget()
        tree.setHeaderLabels(["Kette", "Begründung"])
        for ch in data:
            chain = " → ".join(str(c) for c in ch.get("chain", []))
            QTreeWidgetItem(tree, [chain, ch.get("reason", "")])
        return tree

    def render_cli(self, data, context=None) -> str:
        return "\n".join(
            f"  • {' → '.join(c.get('chain', []))}    [{c.get('reason','')}]"
            for c in data
        )
```

### 5.3 `slot:duplicate_table` — Duplikatanalyse mit Status-Spalte

**Quelle**: `title_list_search`-Workflow, `extra.duplicate_analysis`.
Heute kein dediziertes UI-Rendering — Felder werden in
`AnalysisReviewTab` als JSON gedumpt. **Pure neuer Renderer.**

**Input-Schema** (WP3 Sektion 2):
```python
list[{
    "input_title": str,
    "status": "duplicate" | "likely_duplicate" | "different_edition" | "new" | "no_match",
    "matches": list[dict],
    "reasoning": str,
}]
```

**Pseudo-Code**:
```python
@register_renderer("slot:duplicate_table")
class DuplicateTableRenderer(BaseRenderer):
    output_slot = "slot:duplicate_table"

    _STATUS_COLOR = {
        "duplicate": "#dc3545",        # rot
        "likely_duplicate": "#fd7e14", # orange
        "different_edition": "#ffc107",# gelb
        "new": "#28a745",              # grün
        "no_match": "#6c757d",         # grau
    }

    def render_html(self, data, context=None) -> str:
        rows = ["<table style='border-collapse:collapse'>",
                "<tr><th>Status</th><th>Eingabe</th><th>Matches</th>"
                "<th>Begründung</th></tr>"]
        for r in data:
            color = self._STATUS_COLOR.get(r["status"], "#888")
            mcount = len(r.get("matches", []))
            rows.append(
                f"<tr><td style='color:{color};font-weight:bold'>{r['status']}</td>"
                f"<td>{_esc(r['input_title'])}</td>"
                f"<td>{mcount}</td>"
                f"<td style='font-size:9pt;color:#666'>{_esc(r.get('reasoning',''))}</td></tr>"
            )
        rows.append("</table>")
        return "".join(rows)

    def render_qt(self, data, context=None):
        from PyQt6.QtWidgets import QTableWidget, QTableWidgetItem
        from PyQt6.QtGui import QColor
        tbl = QTableWidget(len(data), 4)
        tbl.setHorizontalHeaderLabels(["Status", "Eingabe", "#Matches", "Begründung"])
        for row, r in enumerate(data):
            status_item = QTableWidgetItem(r["status"])
            status_item.setForeground(QColor(self._STATUS_COLOR.get(r["status"], "#000")))
            tbl.setItem(row, 0, status_item)
            tbl.setItem(row, 1, QTableWidgetItem(r["input_title"]))
            tbl.setItem(row, 2, QTableWidgetItem(str(len(r.get("matches", [])))))
            tbl.setItem(row, 3, QTableWidgetItem(r.get("reasoning", "")))
        return tbl

    def render_cli(self, data, context=None) -> str:
        lines = [f"{'STATUS':<20} TITLE"]
        for r in data:
            lines.append(f"{r['status']:<20} {r['input_title']}")
        return "\n".join(lines)
```

### 5.4 `slot:raw_json` — Fallback (REQUIRED)

**Rolle**: Default für jeden Slot ohne dedizierten Renderer. Muss
immer registriert sein (siehe `get_renderer(fallback=...)`).

```python
@register_renderer("slot:raw_json")
class RawJsonRenderer(BaseRenderer):
    output_slot = "slot:raw_json"

    def render_html(self, data, context=None) -> str:
        import json
        body = json.dumps(data, indent=2, ensure_ascii=False, default=str)
        return f"<pre style='background:#1e1e1e;color:#dcdcdc;padding:8px;'>{_esc(body)}</pre>"

    def render_qt(self, data, context=None):
        from PyQt6.QtWidgets import QTextEdit
        w = QTextEdit()
        w.setReadOnly(True)
        w.setHtml(self.render_html(data, context))
        return w

    def render_cli(self, data, context=None) -> str:
        import json
        return json.dumps(data, indent=2, ensure_ascii=False, default=str)
```

## 6. Frontend-Mapping (Per-Frontend-Split)

### Entscheidung
**Renderer haben drei unabhängige Render-Methoden** (`render_qt`,
`render_html`, `render_cli`). **Kein** geteilter HTML-Core zwischen
GUI und Webapp.

### Begründung
| Argument | Erläuterung |
|---|---|
| Native Qt-Widgets schlagen `QTextEdit.setHtml()` | `QTableWidget` erlaubt Sort/Click/Edit; HTML-in-QTextEdit ist statisch. Heutige `populate_dk_statistics()` nutzt QTableWidget aus genau diesem Grund. |
| Webapp braucht echtes HTML | Server-side-render-fähig, kein QWidget-Stub. Einbettung in Jinja-Template / Server-Komponente. |
| CLI braucht ASCII / TSV | Pipe-fähiges Format, keine HTML-Tags. |
| Per-Frontend-Aufwand mitigiert | `raw_json`-Fallback erlaubt schrittweise Migration; CLI darf nur den Default haben (siehe Sektion 9 Risiken). |

### Frontend-Konsumenten
```python
# GUI — analog zu AgenticStepPanel.setHtml-Pattern
cls = get_renderer(slot)
widget = cls().render_qt(data, ctx)
parent_layout.addWidget(widget)

# Webapp — Pseudo-Code
@app.route("/workflow/<wf>/<step>")
def step_view(wf, step):
    cls = get_renderer(slot_of(wf, step))
    return render_template("step.html", body=cls().render_html(data, ctx))

# CLI — Pseudo-Code
def cli_dump_step(slot, data):
    cls = get_renderer(slot)
    print(cls().render_cli(data))
```

### Was wird heute schon HTML-in-QTextEdit gerendert?
Inventar (zur Migration in WP10):
- [`analysis_review_tab.py:730`](../src/ui/analysis_review_tab.py) — DK-Klassifikationen
- [`pipeline_tab.py`](../src/ui/pipeline_tab.py) Z. 2071, 2333, 2352, 2396, 2667 — DK-Result-Panels
- [`crossref_tab.py`](../src/ui/crossref_tab.py) — DOI-Metadata
- [`ub_catalog_tab.py`](../src/ui/ub_catalog_tab.py) — DK-Catalog-Search-Detail
- [`comparison_tab.py`](../src/ui/comparison_tab.py) — Side-by-Side-Diff
- [`find_keywords.py`](../src/ui/find_keywords.py) — GND-Transparency-Tabelle
- [`agentic_context_widget.py:145`](../src/ui/agentic_context_widget.py) — Live-Snapshot

Die ersten drei Stellen rendern alle `dk_table`-artige Daten → erstes
Migrations-Ziel.

## 7. Migrations-Pattern (Backward-Compat)

### Phasen
| Phase | Schritt | Sichtbarkeit für User |
|---|---|---|
| 1 | Renderer-Klassen anlegen (`src/ui/renderers/`). | unsichtbar |
| 2 | `AnalysisReviewTab.populate_*()` ruft intern Renderer. HTML/Widget identisch zu heute. | unsichtbar |
| 3 | `pipeline_tab.py` Z. 2071 etc. dito (DK-Render-Stellen). | unsichtbar |
| 4 | Sub-Tab-Erzeugung in `AnalysisReviewTab` aus Workflow-`output_paths` ableiten + Renderer pro Slot mounten. | sichtbar: weniger Tab-Sprawl |
| 5 (WP10) | Komplette hardcoded Sub-Tab-Liste durch dynamisches „SlotPanel-pro-Workflow" ersetzen. | sichtbar: dynamische Tabs |

### Backward-Compat-Garantie
- Während Phasen 1-3 ändert sich **keine** Sichtbare UI.
- Renderer-Output (HTML-String + QWidget) wird Snapshot-getestet
  gegen heutigen Output (Spätere WP, vermutlich WP10-Begleitung).
- Wenn ein neuer Slot ohne Renderer auftaucht: `slot:raw_json`
  übernimmt — kein Crash, klar erkennbar im UI als JSON-Dump.

### Coexistenz mit `AgenticContextWidget`
`AgenticContextWidget` enthält schon Mini-Renderer (`_chips`,
`_dict_list`, `_chains`, `_render_record`, `_render_generic`). WP4-
Migrations-Pfad:
- Mini-Renderer-Funktionen bleiben (sind für die Live-Snapshot-Ansicht
  optimiert: kompakt, Truncation-Sentinel).
- Bei Bedarf rufen sie intern den registrierten Renderer auf
  (z.B. `_chains()` → `get_renderer("slot:keyword_chains")().render_html()`).
- **Nicht zwingend** für T2 — Optimierung in WP10.

## 8. Decision-Point T2: Option C trägt — Begründung

**T2 = „Renderer-Plugin-Architektur ja/nein"**
([`agentic_ui_workpackages.md`](agentic_ui_workpackages.md) Z. 222).

### Pro
- Registry-Pattern **2× bereits erfolgreich** (`STEP_REGISTRY`,
  `TOOL_FN_REGISTRY`) — Konsistenz für neue Entwickler, kein neues
  Pattern zu lernen.
- `AgenticContextWidget` **beweist workflow-agnostisches Rendern
  funktioniert** im Bestand.
- WP3-Slot-Vokabular minimiert Renderer-Inflation auf **11 Slots**,
  nicht „pro Workflow ein neuer Renderer".
- `slot:raw_json`-Fallback erlaubt graduelle Migration.

### Contra (und Mitigation)
- **Per-Frontend-Split = 3× Code pro Renderer.** Mitigation: nur dort
  splitten wo Mehrwert (`dk_table` → native QTableWidget lohnt;
  `text_blob` braucht keinen Qt-Sonderfall, kann sich auf
  `BaseRenderer.render_qt`-Default-HTML stützen).
- **Renderer-Inflation langfristig.** Mitigation: Slots sind in
  WP3 fixiert; neue Workflows müssen sich an Slot-Vokabular halten
  oder Slot-Aufnahme begründen.
- **Tests aufwändig.** Mitigation: Snapshot-Tests per Slot
  (HTML-String + QWidget-Tree-Dump); ein Test pro Renderer reicht
  für die T2-Entscheidung.

### Empfehlung
**Option C trägt** — Architektur ist bereits implizit im Bestand,
WP4 macht sie explizit. Implementierung gestaffelt (Sektion 7).

## 9. Risiken + offene Validierungen

| Risiko | Bewertung | Mitigation |
|---|---|---|
| Renderer-Inflation pro Workflow | mittel | WP3-Slot-Vokabular als Normalisierungs-Layer; `slot:raw_json` für Long-Tail. |
| Per-Frontend-Split-Aufwand | mittel | Optional: nur HTML zwingend, Qt/CLI können sich auf Default in `BaseRenderer` stützen. |
| Frontend-übergreifender Renderer (Qt vs HTML) | niedrig | Per-Frontend-Split macht es explizit — kein Übersetzungs-Layer nötig. |
| `catalog_hits`-Doppelschema (WP3 F1 offen) | offen | WP3 Sektion 6: Operator entscheidet zwischen Cluster-Split (zwei Slots) oder Einheitsschema (ein Slot mit Schema-Switch im Renderer). WP4 plant beide Optionen ein; Implementation wartet auf F1. |
| Renderer-Theme-Konsistenz (Color-Palette mehrfach hardcoded) | niedrig | `context`-Parameter erlaubt Palette-Injektion; Default-Palette zentral in `src/ui/renderers/theme.py` (späterer WP). |
| Snapshot-Tests für QWidget-Rendering nicht trivial | mittel | Snapshot-Test nur für HTML-String (deterministisch); QWidget-Tree per `qtbot.find*`-Pattern in pytest-qt (späterer WP). |

### Was WP4 nicht klärt (folge-WPs)
- Renderer-Daten-Validation (Pydantic / TypedDict): WP4 nennt
  `data_schema`-Slot in API, nimmt aber keine Library-Entscheidung.
- Interaktive Renderer (Click → drill-down): außerhalb T2-Scope.
- Renderer-Theming (Dark-Mode, Color-Palette): Hook über `context`,
  Implementierung später.

## 10. Cross-References / Folge-WPs

| Folge-WP | Konsumiert aus WP4 |
|---|---|
| **WP5** (Single-Step-UI) | Renderer-Registry + Slot-Resolver für Step-Output-Anzeige. |
| **WP8** (Chat-Tools workflow-aware) | Renderer für Tool-Call-Outputs (`slot:gnd_pool`, `slot:dk_table`, …). |
| **WP10** (Migration) | Migrations-Pattern (Sektion 7), HTML-in-QTextEdit-Inventar (Sektion 6). |
| **WP11** (Provider-Portabilität) | Indirekt: Provider-Capability-Hints können Slot-Anforderungen prüfen. |

## 11. Status

✅ WP4 abgeschlossen.
- API-Sketch (Sektion 2)
- Registry-Mechanik (Sektion 3)
- YAML-Hint + Slot-Resolution (Sektion 4)
- 3+1 Beispiel-Renderer (Sektion 5)
- Frontend-Mapping (Sektion 6)
- Migrations-Pattern + Backward-Compat (Sektion 7)
- T2-Decision (Sektion 8)
- Risiken (Sektion 9)

**Offene Fragen aus
[`agentic_ui_workpackages.md`](agentic_ui_workpackages.md) Z. 201-207
beantwortet**:
- Registry-Mechanik gleich wie bestehende → **ja** (Sektion 3).
- Renderer GUI/Webapp geteilt → **nein, per-Frontend-Split** (Sektion 6).
- Big Bang vs schrittweise → **schrittweise, 5 Phasen** (Sektion 7).

**Vorbedingung WP10 (Implementierung)**: Operator-Antwort auf WP3-F1
(catalog_hits-Schema) — entscheidet ob 11 oder 12 Slots zu rendern.
