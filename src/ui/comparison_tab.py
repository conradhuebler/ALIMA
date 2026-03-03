"""
Erschließungsvergleich - Comparison Tab for ALIMA pipeline results.
Compares two KeywordAnalysisState objects side-by-side with color-coded chips.
Claude Generated
"""
import difflib
import logging
from pathlib import Path
from typing import Optional, Dict, Set, Any

from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QScrollArea,
    QGroupBox,
    QTextEdit,
    QFileDialog,
    QMessageBox,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QFrame,
)
from PyQt6.QtCore import pyqtSignal, Qt
from PyQt6.QtGui import QFont

from ..core.data_models import KeywordAnalysisState
from ..utils.pipeline_utils import PipelineJsonManager
from .styles import get_colors, get_main_stylesheet, LAYOUT


logger = logging.getLogger(__name__)


class ComparisonTab(QWidget):
    """Tab for comparing two pipeline analysis states - Claude Generated"""

    comparison_loaded = pyqtSignal()

    def __init__(self, main_window=None, **kwargs):
        super().__init__()
        self.main_window = main_window
        self.state_a: Optional[KeywordAnalysisState] = None
        self.state_b: Optional[KeywordAnalysisState] = None
        self.file_a: Optional[str] = None
        self.file_b: Optional[str] = None
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(LAYOUT["margin"], LAYOUT["margin"],
                                  LAYOUT["margin"], LAYOUT["margin"])
        layout.setSpacing(LAYOUT["spacing"])

        # --- Toolbar ---
        toolbar = QHBoxLayout()
        toolbar.setSpacing(LAYOUT["spacing"])

        self.btn_load_a = QPushButton("Datei A laden")
        self.btn_load_a.clicked.connect(lambda: self.load_state("a"))
        toolbar.addWidget(self.btn_load_a)

        self.label_a = QLabel("—")
        self.label_a.setStyleSheet(f"color: {get_colors()['text_light']}; font-style: italic;")
        toolbar.addWidget(self.label_a, 1)

        self.btn_load_b = QPushButton("Datei B laden")
        self.btn_load_b.clicked.connect(lambda: self.load_state("b"))
        toolbar.addWidget(self.btn_load_b)

        self.label_b = QLabel("—")
        self.label_b.setStyleSheet(f"color: {get_colors()['text_light']}; font-style: italic;")
        toolbar.addWidget(self.label_b, 1)

        self.btn_current = QPushButton("Aktuelles Ergebnis als A")
        self.btn_current.clicked.connect(self._load_current_as_a)
        toolbar.addWidget(self.btn_current)

        self.btn_compare = QPushButton("Vergleichen")
        self.btn_compare.setStyleSheet(f"""
            QPushButton {{
                background-color: {get_colors()['primary']};
                color: white; font-weight: bold;
                padding: 8px 20px; border-radius: 6px;
            }}
            QPushButton:hover {{ background-color: {get_colors()['primary_hover']}; }}
            QPushButton:disabled {{ background-color: {get_colors()['background_dark']}; color: {get_colors()['text_muted']}; }}
        """)
        self.btn_compare.setEnabled(False)
        self.btn_compare.clicked.connect(self.run_comparison)
        toolbar.addWidget(self.btn_compare)

        layout.addLayout(toolbar)

        # --- Separator ---
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setFrameShadow(QFrame.Shadow.Sunken)
        layout.addWidget(sep)

        # --- Scroll area for comparison results ---
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_content = QWidget()
        self.scroll_layout = QVBoxLayout(self.scroll_content)
        self.scroll_layout.setSpacing(LAYOUT["spacing"])
        self.scroll_layout.setContentsMargins(4, 4, 4, 4)
        self.scroll_layout.addStretch()
        self.scroll_area.setWidget(self.scroll_content)
        layout.addWidget(self.scroll_area, 1)

        # Placeholder
        self.placeholder = QLabel(
            "Laden Sie zwei Analysis-States und klicken Sie auf \"Vergleichen\"."
        )
        self.placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.placeholder.setStyleSheet(f"color: {get_colors()['text_muted']}; font-size: 12pt; padding: 40px;")
        self.scroll_layout.insertWidget(0, self.placeholder)

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load_state(self, slot: str):
        """Open file dialog and load a state into slot 'a' or 'b' - Claude Generated"""
        path, _ = QFileDialog.getOpenFileName(
            self, f"Analyse-Datei {'A' if slot == 'a' else 'B'} laden",
            "", "JSON Files (*.json);;All Files (*)"
        )
        if not path:
            return
        try:
            state = PipelineJsonManager.load_analysis_state(path)
            name = Path(path).name
            if slot == "a":
                self.state_a = state
                self.file_a = name
                self.label_a.setText(name)
                self.label_a.setToolTip(path)
            else:
                self.state_b = state
                self.file_b = name
                self.label_b.setText(name)
                self.label_b.setToolTip(path)
            self.btn_compare.setEnabled(self.state_a is not None and self.state_b is not None)
        except Exception as e:
            logger.error(f"Failed to load state from {path}: {e}")
            QMessageBox.critical(self, "Ladefehler", f"Datei konnte nicht geladen werden:\n{e}")

    def load_from_current(self, state: KeywordAnalysisState):
        """Receive current pipeline result as State A - Claude Generated"""
        self.state_a = state
        title = state.working_title or "Aktuelles Ergebnis"
        self.file_a = title
        self.label_a.setText(title)
        self.label_a.setToolTip("Aus aktuellem Pipeline-Ergebnis")
        self.btn_compare.setEnabled(self.state_a is not None and self.state_b is not None)

    def _load_current_as_a(self):
        """Load current analysis from AnalysisReviewTab as State A - Claude Generated"""
        if not self.main_window:
            return
        review_tab = getattr(self.main_window, 'analysis_review_tab', None)
        if review_tab and hasattr(review_tab, 'current_analysis') and review_tab.current_analysis:
            self.load_from_current(review_tab.current_analysis)
        else:
            QMessageBox.information(
                self, "Kein Ergebnis",
                "Es ist kein aktuelles Analyse-Ergebnis verfügbar.\n"
                "Führen Sie zuerst eine Pipeline-Analyse durch."
            )

    # ------------------------------------------------------------------
    # Comparison logic
    # ------------------------------------------------------------------

    def _compare_sets(self, set_a: Set[str], set_b: Set[str]) -> Dict[str, Any]:
        """Reusable set comparison returning shared/only_a/only_b/overlap_pct - Claude Generated"""
        shared = set_a & set_b
        only_a = set_a - set_b
        only_b = set_b - set_a
        total = len(set_a | set_b)
        overlap_pct = (len(shared) / total * 100) if total > 0 else 100.0
        return {
            "shared": shared, "only_a": only_a, "only_b": only_b,
            "count_a": len(set_a), "count_b": len(set_b),
            "count_shared": len(shared), "overlap_pct": overlap_pct,
        }

    def _esc(self, text: str) -> str:
        """HTML-escape text - Claude Generated"""
        return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    def _build_keyword_chips_html(self, keywords, color_key: str) -> str:
        """Build inline HTML chips for a set of keywords - Claude Generated"""
        c = get_colors()
        bg = c[f"comparison_{color_key}_bg"]
        fg = c[f"comparison_{color_key}_text"]
        chips = []
        for kw in sorted(keywords):
            chips.append(
                f'<span style="background:{bg};color:{fg};border-radius:12px;'
                f'padding:4px 10px;margin:2px;display:inline-block;'
                f'font-size:10pt;">{self._esc(kw)}</span>'
            )
        return " ".join(chips)

    def _overlap_bar_html(self, pct: float) -> str:
        """Colored overlap percentage bar - Claude Generated"""
        c = get_colors()
        if pct >= 80:
            bar_color = c["comparison_shared_bg"]
            text_color = c["comparison_shared_text"]
        elif pct >= 50:
            bar_color = c["comparison_only_b_bg"]
            text_color = c["comparison_only_b_text"]
        else:
            bar_color = c["confidence_low_bg"]
            text_color = c["confidence_low_text"]
        return (
            f'<div style="background:{c["background_dark"]};border-radius:6px;height:22px;margin:4px 0;">'
            f'<div style="background:{bar_color};width:{pct:.0f}%;height:22px;border-radius:6px;'
            f'text-align:center;line-height:22px;color:{text_color};font-weight:bold;font-size:9pt;">'
            f'{pct:.1f}%</div></div>'
        )

    def _build_step_section(self, title: str, content_widget: QWidget) -> QGroupBox:
        """Wrap a content widget in a uniform QGroupBox section - Claude Generated"""
        box = QGroupBox(title)
        box_layout = QVBoxLayout()
        box_layout.setContentsMargins(8, 8, 8, 8)
        box_layout.addWidget(content_widget)
        box.setLayout(box_layout)
        return box

    def _make_text_display(self, html: str) -> QTextEdit:
        """Create a read-only QTextEdit for HTML content - Claude Generated"""
        te = QTextEdit()
        te.setReadOnly(True)
        te.setHtml(html)
        te.setMinimumHeight(60)
        return te

    # ------------------------------------------------------------------
    # Build comparison sections
    # ------------------------------------------------------------------

    def run_comparison(self):
        """Build all comparison sections and display them - Claude Generated"""
        if not self.state_a or not self.state_b:
            return

        # Clear previous content
        while self.scroll_layout.count():
            item = self.scroll_layout.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()

        c = get_colors()
        a, b = self.state_a, self.state_b

        # --- Summary cards ---
        self.scroll_layout.addWidget(self._build_summary_section(a, b, c))

        # --- Step 1: Input ---
        self.scroll_layout.addWidget(self._build_input_section(a, b, c))

        # --- Step 2: Initialisation (free keywords) ---
        self.scroll_layout.addWidget(self._build_init_section(a, b, c))

        # --- Step 3: Search ---
        self.scroll_layout.addWidget(self._build_search_section(a, b, c))

        # --- Step 4: Keywords (main section) ---
        self.scroll_layout.addWidget(self._build_keywords_section(a, b, c))

        # --- Step 5: Classification ---
        self.scroll_layout.addWidget(self._build_classification_section(a, b, c))

        # --- Meta section ---
        self.scroll_layout.addWidget(self._build_meta_section(a, b, c))

        self.scroll_layout.addStretch()
        self.comparison_loaded.emit()

    # ---------- Summary ----------

    def _build_summary_section(self, a, b, c) -> QGroupBox:
        """Two cards side-by-side with meta info + overall overlap - Claude Generated"""
        html_parts = [f"<html><body style='font-family:Arial,sans-serif;color:{c['text']};'>"]

        # Two columns
        html_parts.append("<table width='100%' cellpadding='8'><tr>")
        for label, state, fname in [("A", a, self.file_a), ("B", b, self.file_b)]:
            bg = c["comparison_only_a_bg"] if label == "A" else c["comparison_only_b_bg"]
            fg = c["comparison_only_a_text"] if label == "A" else c["comparison_only_b_text"]
            title = state.working_title or "—"
            ts = state.timestamp or "—"
            provider = model = "—"
            if state.final_llm_analysis:
                provider = state.final_llm_analysis.provider_used or "—"
                model = state.final_llm_analysis.model_used or "—"
            html_parts.append(
                f"<td width='50%' style='background:{bg};border-radius:8px;vertical-align:top;'>"
                f"<b style='color:{fg};font-size:12pt;'>State {label}</b><br>"
                f"<span style='color:{fg};'>{self._esc(fname or '—')}</span><br>"
                f"<span style='color:{fg};font-size:9pt;'>Titel: {self._esc(title)}</span><br>"
                f"<span style='color:{fg};font-size:9pt;'>Zeitstempel: {self._esc(ts)}</span><br>"
                f"<span style='color:{fg};font-size:9pt;'>Provider: {self._esc(provider)}</span><br>"
                f"<span style='color:{fg};font-size:9pt;'>Modell: {self._esc(model)}</span>"
                f"</td>"
            )
        html_parts.append("</tr></table>")

        # Overall overlap
        all_a = self._collect_all_keywords(a)
        all_b = self._collect_all_keywords(b)
        cmp = self._compare_sets(all_a, all_b)
        html_parts.append(f"<p style='margin-top:10px;'><b>Gesamt-Überlappung (alle Keywords):</b></p>")
        html_parts.append(self._overlap_bar_html(cmp["overlap_pct"]))
        html_parts.append(
            f"<p style='font-size:9pt;color:{c['text_light']};'>"
            f"Gemeinsam: {cmp['count_shared']} | Nur A: {len(cmp['only_a'])} | Nur B: {len(cmp['only_b'])}</p>"
        )
        html_parts.append("</body></html>")
        return self._build_step_section("Zusammenfassung", self._make_text_display("".join(html_parts)))

    def _collect_all_keywords(self, state: KeywordAnalysisState) -> Set[str]:
        """Collect all keyword strings from a state for overall overlap - Claude Generated"""
        kw = set()
        if state.initial_keywords:
            kw.update(state.initial_keywords)
        if state.final_llm_analysis and state.final_llm_analysis.extracted_gnd_keywords:
            kw.update(state.final_llm_analysis.extracted_gnd_keywords)
        return kw

    # ---------- Step 1: Input ----------

    def _build_input_section(self, a, b, c) -> QGroupBox:
        """Compare original abstracts - Claude Generated"""
        abs_a = a.original_abstract or ""
        abs_b = b.original_abstract or ""
        html = [f"<html><body style='font-family:Arial,sans-serif;color:{c['text']};'>"]

        if abs_a == abs_b:
            html.append(
                f"<div style='background:{c['comparison_shared_bg']};color:{c['comparison_shared_text']};"
                f"padding:10px;border-radius:6px;'>"
                f"<b>Identisch</b> ({len(abs_a)} Zeichen)</div>"
            )
        else:
            ratio = difflib.SequenceMatcher(None, abs_a, abs_b).ratio() * 100
            html.append(
                f"<p>Ähnlichkeit: <b>{ratio:.1f}%</b></p>"
            )
            html.append(self._overlap_bar_html(ratio))
            html.append(
                f"<p style='font-size:9pt;color:{c['text_light']};'>"
                f"A: {len(abs_a)} Zeichen | B: {len(abs_b)} Zeichen</p>"
            )
        html.append("</body></html>")
        return self._build_step_section("Schritt 1 — Input", self._make_text_display("".join(html)))

    # ---------- Step 2: Initialisation ----------

    def _build_init_section(self, a, b, c) -> QGroupBox:
        """Compare initial (free) keywords - Claude Generated"""
        set_a = set(a.initial_keywords or [])
        set_b = set(b.initial_keywords or [])
        cmp = self._compare_sets(set_a, set_b)
        html = self._keyword_chips_section_html(cmp, c, "Freie Schlagworte")
        return self._build_step_section("Schritt 2 — Initialisierung", self._make_text_display(html))

    # ---------- Step 3: Search ----------

    def _build_search_section(self, a, b, c) -> QGroupBox:
        """Compare search terms and result counts - Claude Generated"""
        terms_a = {r.keyword for r in (a.search_results or [])} if a.search_results else set()
        terms_b = {r.keyword for r in (b.search_results or [])} if b.search_results else set()

        # Handle dict format
        if a.search_results and isinstance(a.search_results, dict):
            terms_a = set(a.search_results.keys())
        if b.search_results and isinstance(b.search_results, dict):
            terms_b = set(b.search_results.keys())

        if not terms_a and not terms_b:
            html = (f"<html><body style='color:{c['text']};'>"
                    f"<p style='color:{c['text_muted']};'>Keine Suchergebnisse in beiden States.</p>"
                    f"</body></html>")
            return self._build_step_section("Schritt 3 — Suche", self._make_text_display(html))

        cmp = self._compare_sets(terms_a, terms_b)

        html = [f"<html><body style='font-family:Arial,sans-serif;color:{c['text']};'>"]
        html.append(f"<p><b>Suchbegriffe:</b> A={cmp['count_a']} | B={cmp['count_b']} | "
                    f"Gemeinsam={cmp['count_shared']}</p>")
        html.append(self._overlap_bar_html(cmp["overlap_pct"]))

        if cmp["shared"]:
            html.append(f"<p style='margin-top:8px;'><b>Gemeinsam:</b></p>")
            html.append(self._build_keyword_chips_html(cmp["shared"], "shared"))
        if cmp["only_a"]:
            html.append(f"<p style='margin-top:8px;'><b>Nur in A:</b></p>")
            html.append(self._build_keyword_chips_html(cmp["only_a"], "only_a"))
        if cmp["only_b"]:
            html.append(f"<p style='margin-top:8px;'><b>Nur in B:</b></p>")
            html.append(self._build_keyword_chips_html(cmp["only_b"], "only_b"))

        # Suggester comparison
        sugg_a = set(a.search_suggesters_used or [])
        sugg_b = set(b.search_suggesters_used or [])
        if sugg_a or sugg_b:
            html.append(f"<p style='margin-top:10px;font-size:9pt;color:{c['text_light']};'>"
                        f"Suggester A: {', '.join(sorted(sugg_a)) or '—'} | "
                        f"B: {', '.join(sorted(sugg_b)) or '—'}</p>")

        html.append("</body></html>")
        return self._build_step_section("Schritt 3 — Suche", self._make_text_display("".join(html)))

    # ---------- Step 4: Keywords (MAIN) ----------

    def _build_keywords_section(self, a, b, c) -> QGroupBox:
        """Compare final GND keywords — the most important section - Claude Generated"""
        final_a = set()
        final_b = set()
        llm_a = a.final_llm_analysis
        llm_b = b.final_llm_analysis

        if llm_a and llm_a.extracted_gnd_keywords:
            final_a = set(llm_a.extracted_gnd_keywords)
        if llm_b and llm_b.extracted_gnd_keywords:
            final_b = set(llm_b.extracted_gnd_keywords)

        cmp = self._compare_sets(final_a, final_b)
        html = [f"<html><body style='font-family:Arial,sans-serif;color:{c['text']};'>"]

        # Prominent header
        html.append(
            f"<div style='background:{c['background_light']};padding:12px;border-radius:8px;"
            f"border-left:4px solid {c['primary']};margin-bottom:10px;'>"
            f"<b style='font-size:13pt;'>Finale GND-Schlagworte</b><br>"
            f"<span style='font-size:10pt;'>A: {cmp['count_a']} | B: {cmp['count_b']} | "
            f"Gemeinsam: {cmp['count_shared']}</span></div>"
        )
        html.append(self._overlap_bar_html(cmp["overlap_pct"]))

        # Chips
        if cmp["shared"]:
            html.append(f"<p style='margin-top:10px;'><b>Gemeinsam ({cmp['count_shared']}):</b></p>")
            html.append(self._build_keyword_chips_html(cmp["shared"], "shared"))
        if cmp["only_a"]:
            html.append(f"<p style='margin-top:10px;'><b>Nur in A ({len(cmp['only_a'])}):</b></p>")
            html.append(self._build_keyword_chips_html(cmp["only_a"], "only_a"))
        if cmp["only_b"]:
            html.append(f"<p style='margin-top:10px;'><b>Nur in B ({len(cmp['only_b'])}):</b></p>")
            html.append(self._build_keyword_chips_html(cmp["only_b"], "only_b"))

        if not final_a and not final_b:
            html.append(f"<p style='color:{c['text_muted']};'>Keine finalen Keywords in beiden States.</p>")

        # LLM meta comparison
        if llm_a or llm_b:
            html.append(f"<p style='margin-top:14px;font-size:9pt;color:{c['text_light']};'><b>LLM-Details:</b></p>")
            html.append("<table style='font-size:9pt;width:100%;' cellpadding='4'>")
            rows = [
                ("Provider", getattr(llm_a, 'provider_used', '—') if llm_a else '—',
                 getattr(llm_b, 'provider_used', '—') if llm_b else '—'),
                ("Modell", getattr(llm_a, 'model_used', '—') if llm_a else '—',
                 getattr(llm_b, 'model_used', '—') if llm_b else '—'),
                ("Temperature", str(getattr(llm_a, 'temperature', '—')) if llm_a else '—',
                 str(getattr(llm_b, 'temperature', '—')) if llm_b else '—'),
                ("Seed", str(getattr(llm_a, 'seed', '—')) if llm_a else '—',
                 str(getattr(llm_b, 'seed', '—')) if llm_b else '—'),
                ("Task", getattr(llm_a, 'task_name', '—') if llm_a else '—',
                 getattr(llm_b, 'task_name', '—') if llm_b else '—'),
            ]
            for label, va, vb in rows:
                diff_style = f"color:{c['error']};" if va != vb else ""
                html.append(
                    f"<tr><td><b>{label}</b></td>"
                    f"<td style='{diff_style}'>{self._esc(str(va))}</td>"
                    f"<td style='{diff_style}'>{self._esc(str(vb))}</td></tr>"
                )
            html.append("</table>")

        html.append("</body></html>")

        te = self._make_text_display("".join(html))
        te.setMinimumHeight(200)
        return self._build_step_section("Schritt 4 — Keywords", te)

    # ---------- Step 5: Classification ----------

    def _build_classification_section(self, a, b, c) -> QGroupBox:
        """Compare DK classifications - Claude Generated"""
        dk_a = set(a.dk_classifications or [])
        dk_b = set(b.dk_classifications or [])

        if not dk_a and not dk_b:
            html = (f"<html><body style='color:{c['text']};'>"
                    f"<p style='color:{c['text_muted']};'>Keine DK-Klassifikationen in beiden States.</p>"
                    f"</body></html>")
            return self._build_step_section("Schritt 5 — Klassifikation", self._make_text_display(html))

        cmp = self._compare_sets(dk_a, dk_b)
        html = self._keyword_chips_section_html(cmp, c, "DK-Klassifikationen")
        return self._build_step_section("Schritt 5 — Klassifikation", self._make_text_display(html))

    # ---------- Meta ----------

    def _build_meta_section(self, a, b, c) -> QGroupBox:
        """Tabular meta comparison - Claude Generated"""
        html = [f"<html><body style='font-family:Arial,sans-serif;color:{c['text']};'>"]
        html.append("<table style='width:100%;font-size:10pt;' cellpadding='6' cellspacing='0'>")
        html.append(
            f"<tr style='background:{c['background_dark']};font-weight:bold;'>"
            f"<td>Eigenschaft</td><td>State A</td><td>State B</td></tr>"
        )

        llm_a = a.final_llm_analysis
        llm_b = b.final_llm_analysis
        init_a = a.initial_llm_call_details
        init_b = b.initial_llm_call_details

        rows = [
            ("Zeitstempel", a.timestamp or "—", b.timestamp or "—"),
            ("Arbeitstitel", a.working_title or "—", b.working_title or "—"),
            ("Abstrakt-Länge", f"{len(a.original_abstract or '')} Zeichen",
             f"{len(b.original_abstract or '')} Zeichen"),
            ("Initiale Keywords", str(len(a.initial_keywords or [])),
             str(len(b.initial_keywords or []))),
            ("Suchergebnisse", str(len(a.search_results or [])),
             str(len(b.search_results or []))),
            ("Suggester", ", ".join(a.search_suggesters_used or []) or "—",
             ", ".join(b.search_suggesters_used or []) or "—"),
            ("Finale Keywords", str(len(llm_a.extracted_gnd_keywords)) if llm_a else "—",
             str(len(llm_b.extracted_gnd_keywords)) if llm_b else "—"),
            ("DK-Klassifikationen", str(len(a.dk_classifications or [])),
             str(len(b.dk_classifications or []))),
            ("Provider (Init)", getattr(init_a, 'provider_used', '—') if init_a else '—',
             getattr(init_b, 'provider_used', '—') if init_b else '—'),
            ("Modell (Init)", getattr(init_a, 'model_used', '—') if init_a else '—',
             getattr(init_b, 'model_used', '—') if init_b else '—'),
            ("Provider (Final)", getattr(llm_a, 'provider_used', '—') if llm_a else '—',
             getattr(llm_b, 'provider_used', '—') if llm_b else '—'),
            ("Modell (Final)", getattr(llm_a, 'model_used', '—') if llm_a else '—',
             getattr(llm_b, 'model_used', '—') if llm_b else '—'),
            ("Temperature (Final)", str(getattr(llm_a, 'temperature', '—')) if llm_a else '—',
             str(getattr(llm_b, 'temperature', '—')) if llm_b else '—'),
            ("Seed (Final)", str(getattr(llm_a, 'seed', '—')) if llm_a else '—',
             str(getattr(llm_b, 'seed', '—')) if llm_b else '—'),
            ("Konvergenz", "Ja" if a.convergence_achieved else "Nein",
             "Ja" if b.convergence_achieved else "Nein"),
            ("Iterationen", str(len(a.refinement_iterations or [])),
             str(len(b.refinement_iterations or []))),
        ]

        for i, (label, va, vb) in enumerate(rows):
            bg = c["background_light"] if i % 2 == 0 else c["background"]
            diff_style = f"font-weight:bold;color:{c['error']};" if va != vb else ""
            html.append(
                f"<tr style='background:{bg};'>"
                f"<td><b>{self._esc(label)}</b></td>"
                f"<td style='{diff_style}'>{self._esc(str(va))}</td>"
                f"<td style='{diff_style}'>{self._esc(str(vb))}</td></tr>"
            )

        html.append("</table></body></html>")
        te = self._make_text_display("".join(html))
        te.setMinimumHeight(300)
        return self._build_step_section("Meta-Informationen", te)

    # ------------------------------------------------------------------
    # Shared HTML builders
    # ------------------------------------------------------------------

    def _keyword_chips_section_html(self, cmp: dict, c: dict, title: str) -> str:
        """Build a full HTML section with overlap bar and keyword chips - Claude Generated"""
        html = [f"<html><body style='font-family:Arial,sans-serif;color:{c['text']};'>"]
        html.append(
            f"<p><b>{self._esc(title)}:</b> A={cmp['count_a']} | B={cmp['count_b']} | "
            f"Gemeinsam={cmp['count_shared']}</p>"
        )
        html.append(self._overlap_bar_html(cmp["overlap_pct"]))

        if cmp["shared"]:
            html.append(f"<p style='margin-top:8px;'><b>Gemeinsam ({cmp['count_shared']}):</b></p>")
            html.append(self._build_keyword_chips_html(cmp["shared"], "shared"))
        if cmp["only_a"]:
            html.append(f"<p style='margin-top:8px;'><b>Nur in A ({len(cmp['only_a'])}):</b></p>")
            html.append(self._build_keyword_chips_html(cmp["only_a"], "only_a"))
        if cmp["only_b"]:
            html.append(f"<p style='margin-top:8px;'><b>Nur in B ({len(cmp['only_b'])}):</b></p>")
            html.append(self._build_keyword_chips_html(cmp["only_b"], "only_b"))

        if cmp["count_a"] == 0 and cmp["count_b"] == 0:
            html.append(f"<p style='color:{c['text_muted']};'>Keine Daten in beiden States.</p>")

        html.append("</body></html>")
        return "".join(html)

    # ------------------------------------------------------------------
    # Theme support
    # ------------------------------------------------------------------

    def refresh_styles(self):
        """Re-apply styles after theme change - Claude Generated"""
        self.setStyleSheet(get_main_stylesheet())
        c = get_colors()
        self.label_a.setStyleSheet(f"color: {c['text_light']}; font-style: italic;")
        self.label_b.setStyleSheet(f"color: {c['text_light']}; font-style: italic;")
        self.btn_compare.setStyleSheet(f"""
            QPushButton {{
                background-color: {c['primary']};
                color: white; font-weight: bold;
                padding: 8px 20px; border-radius: 6px;
            }}
            QPushButton:hover {{ background-color: {c['primary_hover']}; }}
            QPushButton:disabled {{ background-color: {c['background_dark']}; color: {c['text_muted']}; }}
        """)
        self.placeholder.setStyleSheet(f"color: {c['text_muted']}; font-size: 12pt; padding: 40px;")
        # Re-run comparison if data available
        if self.state_a and self.state_b:
            self.run_comparison()
