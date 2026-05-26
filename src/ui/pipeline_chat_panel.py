"""PipelineChatPanel — unified pipeline log + chat dock (WP10 P-δ.5a).

Claude Generated.

Replaces the previously separate ``PipelineStreamWidget`` (right-side
panel in ``PipelineTab``) and the floating ``ChatWidget`` (``chat_dock``).
Single widget hosting a shared log area that renders:

- Pipeline step events (▶/✅/❌, durations, GND-verification stats, DK
  catalog results, repetition warnings).
- LLM streaming tokens (purple) for pipeline LLM calls.
- User chat turns (WhatsApp-style right-aligned green bubble).
- Assistant chat turns (left-aligned grey bubble, streaming-capable
  cell-cursor).
- Tool-call markers — from both the chat-agent (direct signal) and the
  agentic-pipeline ``AgentLoop`` (via ``AlimaStateBus`` events).
- Status messages from the AgentLoop status channel.

The chat input lives at the bottom of the same panel. Operator can chat
about pipeline state in-place; future P-ζ lets the agent itself drive
the pipeline and "talk to itself" in the same log.

Public API preserved from ``PipelineStreamWidget`` so existing callers
in ``pipeline_tab.py`` / ``pipeline_manager.py`` need no signature
changes.
"""
from __future__ import annotations

import logging
import time
from datetime import datetime
from html import escape as html_escape
from typing import Any, Dict, List, Optional

from PyQt6.QtCore import Qt, QTimer, pyqtSignal, pyqtSlot
from PyQt6.QtGui import (
    QKeyEvent,
    QColor,
    QFont,
    QTextBlockFormat,
    QTextCursor,
    QTextLength,
    QTextTableCellFormat,
    QTextTableFormat,
)
from PyQt6.QtCore import QUrl
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QTextBrowser,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ..core.pipeline_manager import PipelineStep
from ..core.state_bus import AlimaStateBus
from .chat_agent_worker import ChatAgentWorker
from .chat_session import ChatSession
from .chat_tools import build_chat_toolset
from .styles import (
    LAYOUT,
    get_button_styles,
    get_main_stylesheet,
    get_scaled_font,
)


# ----------------------------------------------------------------------
# System-Prompt edit dialog (was previously in chat_widget.py).
# ----------------------------------------------------------------------


class SystemPromptDialog(QDialog):
    """Small dialog for editing the assistant system prompt."""

    def __init__(self, current_prompt: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("⚙️ System-Prompt bearbeiten")
        self.setMinimumSize(500, 300)
        self.setStyleSheet(get_main_stylesheet())

        layout = QVBoxLayout(self)
        m = LAYOUT.get("margin", 12)
        layout.setContentsMargins(m, m, m, m)
        layout.setSpacing(LAYOUT.get("spacing", 8))

        info = QLabel(
            "Dieser Prompt definiert die Rolle des Assistenten. "
            "Änderungen wirken sich sofort auf die nächste Nachricht aus."
        )
        info.setWordWrap(True)
        info.setStyleSheet("color: #666; font-size: 10px;")
        layout.addWidget(info)

        self.editor = QTextEdit()
        self.editor.setPlainText(current_prompt)
        self.editor.setFont(get_scaled_font(size_delta=-1, monospace=False))
        layout.addWidget(self.editor)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save
            | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_prompt(self) -> str:
        return self.editor.toPlainText().strip()


# ----------------------------------------------------------------------
# ChatInputEdit — multi-line QTextEdit, Enter sends, Shift+Enter newline.
# ----------------------------------------------------------------------


class ChatInputEdit(QTextEdit):
    """Multi-line chat input. Enter = submit, Shift+Enter = newline."""

    submit = pyqtSignal()

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
                super().keyPressEvent(event)
                return
            self.submit.emit()
            return
        super().keyPressEvent(event)


# ----------------------------------------------------------------------
# PipelineChatPanel
# ----------------------------------------------------------------------


class PipelineChatPanel(QWidget):
    """Unified pipeline-log + chat panel (WP10 P-δ.5a)."""

    # -- Pipeline signals (preserved from PipelineStreamWidget) ---------
    cancel_pipeline = pyqtSignal()
    pause_pipeline = pyqtSignal()
    retry_with_variations = pyqtSignal(dict)
    abort_generation_requested = pyqtSignal()

    # -- Chat signal ----------------------------------------------------
    message_sent = pyqtSignal(str)

    # ------------------------------------------------------------------
    # Default prompts (chat-agent).
    # ------------------------------------------------------------------

    DEFAULT_SYSTEM_PROMPT = (
        "Du bist Experte für Bibliothekswissenschaft und Sacherschließung "
        "(RSWK, GND, DDC/DK) und arbeitest als Assistent in der ALIMA-Pipeline. "
        "Antworte präzise, fachlich korrekt, auf Deutsch.\n\n"
        "WICHTIG — Tool-Use-Regeln (zwingend):\n"
        "- Du hast Tools für ALLE Pipeline-Daten: Keywords, Keyword-Ketten,\n"
        "  DK-Klassifikationen, GND-Einträge, fehlende Konzepte.\n"
        "- Rufe IMMER zuerst `list_available_data` auf, um zu sehen welche\n"
        "  Daten vorliegen — außer der Nutzer stellt nur eine Begrüßung\n"
        "  oder eine allgemeine Frage ohne Bezug zum konkreten Werk.\n"
        "- Nenne KEINE GND-ID, KEINEN DK-Code, KEINE Keyword-Anzahl,\n"
        "  KEINE Schlagwortkette aus dem Gedächtnis. Hole sie mit dem\n"
        "  passenden Tool (`get_keywords`, `get_keyword_chains`,\n"
        "  `get_dk_classifications`, `validate_gnd_term`).\n"
        "- WENN ein Tool 0 Treffer zurückgibt, sage das ehrlich. Gib DANN\n"
        "  KEINE eigenen 'Vorschläge' aus dem Training. Deine Aufgabe ist\n"
        "  NICHT, Schlagwörter oder DK-Codes zu erfinden — nur, was die\n"
        "  Tools liefern, darfst du nennen.\n"
        "- Halluzinationen kosten Vertrauen. Lieber kurz und korrekt als\n"
        "  ausführlich und erfunden.\n\n"
        "Katalog-, GND- und DK-Suchen (zwingend):\n"
        "- Wenn der Nutzer nach Titeln per Schlagwort oder Stichwort sucht,\n"
        "  nutze direkt `search_catalog` oder `search_catalog_titles`.\n"
        "- Wenn der Nutzer GND-Sachbegriffe oder -IDs sucht, nutze direkt\n"
        "  `search_gnd` oder `search_lobid`.\n"
        "- Wenn der Nutzer DK-Codes oder Klassifikationen sucht, nutze\n"
        "  direkt `get_classification` oder `get_dk_cache`.\n"
        "- Nutze NIEMALS nur `list_pipeline_results` oder `get_keywords`\n"
        "  als Antwort auf Katalog-/GND-/DK-Anfragen. Greife direkt auf\n"
        "  die Bibliotheks-Tools zu.\n"
        "- WENN keine Pipeline-Daten vorliegen UND der Nutzer nach GND/DK\n"
        "  fragt, dann SUCHE mit den Tools. Gib NIEMALS selbst erfundene\n"
        "  Codes oder Schlagwörter an.\n\n"
        "Offene Eingaben:\n"
        "- Wenn der Nutzer nur ein einzelnes Stichwort schreibt (z.B.\n"
        "  'Quantenchemie') OHNE vorherigen Kontext UND ohne Verb/Frage,\n"
        "  dann frage zurück, was zu tun ist.\n"
        "- WENN es einen vorherigen Kontext gibt (z.B. gerade über\n"
        "  'Quantenchemie' gesprochen) und der Nutzer schreibt dann kurze\n"
        "  Bezugswörter wie 'Bücher', 'Titel', 'Suchen', 'GND', 'DK',\n"
        "  dann ist das ein elliptischer Auftrag — führe die passende\n"
        "  Aktion direkt aus (z.B. search_catalog_titles mit dem vorherigen\n"
        "  Thema). Frage NICHT nochmal nach.\n"
        "- Rufe NIEMALS eigenmächtig Tools auf, wenn der Nutzer keinen\n"
        "  klaren Auftrag gegeben hat UND kein vorheriger Kontext existiert.\n\n"
        "Agentic Workflows (YAML-gesteuert):\n"
        "- ALIMA hat Workflows: `alima` (v5, self-contained, alle Prompts\n"
        "  inline), `alima_classic` (v4), `catalog_search`,\n"
        "  `synonym_expansion`, `batch_metadata`.\n"
        "- Nutze `list_workflows` um verfügbare Workflows zu sehen.\n"
        "- Nutze `get_workflow` um Schritte, Eingaben und Abhängigkeiten\n"
        "  eines Workflows anzusehen.\n"
        "- WENN der Nutzer eine vollständige Pipeline-Analyse will,\n"
        "  FÜHRE die Schritte SELBST aus:\n"
        "  1. `list_available_data` — prüfe ob Pipeline-Daten vorliegen.\n"
        "  2. Falls nein: `resolve_doi` oder Abstract vom Nutzer holen.\n"
        "  3. `search_gnd` / `search_lobid` für GND-Schlagwörter.\n"
        "  4. `search_catalog` / `search_catalog_titles` für DK-Daten.\n"
        "  5. `get_classification` für DK/RVK-Codes.\n"
        "  6. Ergebnisse zusammenfassen.\n"
        "  Der Chat-Agent KANN Workflows selbst ausführen — nutze die\n"
        "  verfügbaren Tools Schritt für Schritt.\n\n"
        "Wissen aus früheren Nachrichten:\n"
        "- Wenn du Daten brauchst, die in EARLIEREN Tool-Calls bereits\n"
        "  gewonnen wurden (z.B. ein DOI-Resolve oder eine Katalogsuche),\n"
        "  nutze `get_messages_history` mit passendem `offset` und `last_n`,\n"
        "  um die Ergebnisse zu finden. Du musst nicht nochmal die gleichen\n"
        "  Tools rufen — hole dir die Daten aus deiner eigenen Historie."
    )

    USER_PROMPT_TEMPLATE = (
        "Aktuelles Werk: {context}\n\n"
        "Die vollständigen Pipeline-Daten (Keywords, GND-Einträge, "
        "DK-Codes, Schlagwortketten, fehlende Konzepte) hole dir bei "
        "Bedarf via Tool-Calls. Beginne ggf. mit `list_available_data`.\n\n"
        "Nutzer-Frage: {user_message}"
    )

    # Status-message prefixes whose info is already rendered by the
    # tool-call/tool-result hook handlers — skip duplicates.
    _STATUS_SKIP_PREFIXES = ("🔧", "✓", "💭")

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        llm_service=None,
        prompt_service=None,
        pipeline_manager=None,
        mcp_registry=None,
        parent=None,
    ):
        super().__init__(parent)
        self.logger = logging.getLogger(__name__)

        # Chat-side dependencies
        self.llm_service = llm_service
        self.prompt_service = prompt_service
        self.pipeline_manager = pipeline_manager
        if mcp_registry is None:
            from src.mcp.tool_registry import ToolRegistry
            mcp_registry = ToolRegistry()
            mcp_registry.register_all_tools()
        self.mcp_registry = mcp_registry

        # Pipeline-side state
        self.current_step_id: Optional[str] = None
        self.is_streaming: bool = False
        self.step_start_times: Dict[str, datetime] = {}
        self.current_working_title: Optional[str] = None
        self.current_suggestions: List[Dict] = []
        self._last_scroll_time: float = 0.0

        # Chat-side state
        self.system_prompt: str = self.DEFAULT_SYSTEM_PROMPT
        self.session: ChatSession = ChatSession()
        self.current_worker: Optional[ChatAgentWorker] = None
        self.current_context: str = ""
        self.working_title: str = ""
        self._assistant_block_open: bool = False
        self._assistant_cell_cursor: Optional[QTextCursor] = None
        self._current_render_model: str = ""
        self._typing_dots: int = 0
        self._typing_model: str = ""

        # P-ε: cross-thread bridge for mutation tool confirmations.
        from src.ui.chat_tools.proposal_gateway import ProposalGateway
        self.proposal_gateway = ProposalGateway(self)
        self.proposal_gateway.proposal_requested.connect(
            self._render_proposal_bubble
        )

        self.setup_ui()

        # Auto-scroll throttle uses self._last_scroll_time.

        # Size policy: vertical Ignored to prevent sizeHint propagation
        # to window (multi-monitor safety, inherited from PipelineStreamWidget).
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Ignored,
        )

        # AlimaStateBus subscriptions:
        # - state.changed → refresh SharedContext snapshot for chat tools.
        # - tool.called / tool.result → render markers for agentic pipeline
        #   (chat-agent uses direct signals, not the bus).
        try:
            bus = AlimaStateBus()
            bus.subscribe("state.changed", self._on_state_changed)
            bus.subscribe("tool.called", self._on_bus_tool_called)
            bus.subscribe("tool.result", self._on_bus_tool_result)
            bus.subscribe("state.pipeline_step", self._on_bus_pipeline_step)
        except Exception:
            self.logger.exception("PipelineChatPanel: AlimaStateBus subscribe failed")

    # ------------------------------------------------------------------
    # UI assembly
    # ------------------------------------------------------------------

    def setup_ui(self):
        self.setStyleSheet(get_main_stylesheet())
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # --- Single consolidated header --------------------------------
        header = QFrame()
        header.setStyleSheet(
            "QFrame { background-color: #2d2d2d; border-bottom: 1px solid #444; }"
        )
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(8, 4, 8, 4)
        header_layout.setSpacing(6)

        title_label = QLabel("📝 Pipeline + 💬 Chat")
        title_label.setStyleSheet("color: #e0e0e0; font-weight: bold;")
        header_layout.addWidget(title_label)

        self.model_combo = QComboBox()
        self.model_combo.setMinimumWidth(180)
        self.model_combo.setMaximumWidth(280)
        self.model_combo.setStyleSheet(
            "QComboBox { font-size: 10px; padding: 2px 6px; border: 1px solid #555; "
            "border-radius: 3px; background-color: #3d3d3d; color: #ccc; }"
        )
        self._populate_model_combo()
        self.model_combo.currentIndexChanged.connect(self._on_model_combo_changed)
        header_layout.addWidget(self.model_combo)

        self.persist_combo_toggle = QCheckBox("💾")
        self.persist_combo_toggle.setChecked(False)
        self.persist_combo_toggle.setStyleSheet("color: #aaa; font-size: 10px;")
        self.persist_combo_toggle.setToolTip(
            "Bei Combo-Wechsel als ChatConfig-Default speichern."
        )
        header_layout.addWidget(self.persist_combo_toggle)

        self.model_status_label = QLabel("")
        self.model_status_label.setStyleSheet(
            "color: #8be9fd; font-size: 10px; padding-left: 4px;"
        )
        header_layout.addWidget(self.model_status_label)

        header_layout.addStretch()

        # Compact icon-style controls.
        icon_btn_style = (
            "QPushButton { background: transparent; border: 1px solid #555; "
            "border-radius: 3px; color: #ccc; font-size: 10px; padding: 2px 6px; }"
            "QPushButton:hover { background: #3d3d3d; }"
        )

        self.system_prompt_btn = QPushButton("⚙️")
        self.system_prompt_btn.setFixedSize(26, 22)
        self.system_prompt_btn.setStyleSheet(icon_btn_style)
        self.system_prompt_btn.setToolTip("System-Prompt bearbeiten")
        self.system_prompt_btn.clicked.connect(self.show_system_prompt_dialog)
        header_layout.addWidget(self.system_prompt_btn)

        self.reset_toggle = QCheckBox("🔄 Reset")
        self.reset_toggle.setChecked(True)
        self.reset_toggle.setStyleSheet("color: #aaa; font-size: 10px;")
        self.reset_toggle.setToolTip(
            "Chat-Verlauf bei jedem neuen Pipeline-Lauf leeren."
        )
        header_layout.addWidget(self.reset_toggle)

        self.autonomous_toggle = QCheckBox("🤖 Autonom")
        self.autonomous_toggle.setStyleSheet("color: #aaa; font-size: 10px;")
        self.autonomous_toggle.setToolTip(
            "Mutationen ohne Rückfrage anwenden (ChatConfig.autonomous_pipeline). "
            "Default off — Agent fragt vor jeder Änderung."
        )
        # Initial state from ChatConfig.
        try:
            cfg = self._get_chat_config()
            self.autonomous_toggle.setChecked(
                bool(getattr(cfg, "autonomous_pipeline", False))
            )
        except Exception:
            pass
        self.autonomous_toggle.stateChanged.connect(
            self._on_autonomous_toggle_changed
        )
        header_layout.addWidget(self.autonomous_toggle)

        self.auto_scroll_checkbox = QCheckBox("⬇")
        self.auto_scroll_checkbox.setChecked(True)
        self.auto_scroll_checkbox.setStyleSheet("color: #aaa; font-size: 10px;")
        self.auto_scroll_checkbox.setToolTip("Auto-scroll Log")
        header_layout.addWidget(self.auto_scroll_checkbox)

        self.clear_button = QPushButton("🗑️")
        self.clear_button.setFixedSize(26, 22)
        self.clear_button.setStyleSheet(icon_btn_style)
        self.clear_button.setToolTip("Log leeren")
        self.clear_button.clicked.connect(self.clear_stream)
        header_layout.addWidget(self.clear_button)

        self.save_log_button = QPushButton("💾")
        self.save_log_button.setFixedSize(26, 22)
        self.save_log_button.setStyleSheet(icon_btn_style)
        self.save_log_button.setToolTip("Log speichern")
        self.save_log_button.clicked.connect(self.save_stream_log)
        header_layout.addWidget(self.save_log_button)

        self.cancel_btn = QPushButton("⏹ Abbrechen")
        self.cancel_btn.setStyleSheet(get_button_styles().get("danger", ""))
        self.cancel_btn.setVisible(False)
        self.cancel_btn.setToolTip("Aktuelle Generierung abbrechen")
        self.cancel_btn.clicked.connect(self.cancel_generation)
        header_layout.addWidget(self.cancel_btn)

        outer.addWidget(header)

        # --- Vertical splitter between log + input (user-draggable) ----
        self.body_splitter = QSplitter(Qt.Orientation.Vertical)
        self.body_splitter.setChildrenCollapsible(False)
        self.body_splitter.setHandleWidth(4)
        self.body_splitter.setStyleSheet(
            "QSplitter::handle { background-color: #2d2d2d; }"
            "QSplitter::handle:hover { background-color: #555; }"
        )

        # Top half = log area container (stream_text + warning + typing).
        log_container = QWidget()
        log_layout = QVBoxLayout(log_container)
        log_layout.setContentsMargins(0, 0, 0, 0)
        log_layout.setSpacing(0)

        # --- Main log area (shared QTextBrowser for clickable bubbles) ---
        # P-ε: QTextBrowser exposes anchorClicked so inline mutation-proposal
        # bubbles can have ✓/✗ links the user clicks.
        self.stream_text = QTextBrowser()
        self.stream_text.setReadOnly(True)
        self.stream_text.setOpenLinks(False)
        self.stream_text.setOpenExternalLinks(False)
        self.stream_text.anchorClicked.connect(self._on_anchor_clicked)
        self.stream_text.setMinimumHeight(80)
        self.stream_text.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        font = get_scaled_font(monospace=True)
        font.setStyleHint(QFont.StyleHint.Monospace)
        self.stream_text.setFont(font)
        self.stream_text.setStyleSheet(
            """
            QTextBrowser {
                background-color: #1e1e1e;
                color: #f8f8f2;
                border: none;
                border-top: 1px solid #333;
                padding: 8px;
                font-family: 'Consolas', 'Monaco', monospace;
            }
            QScrollBar:vertical {
                background: #2d2d2d;
                width: 12px;
                border-radius: 6px;
            }
            QScrollBar::handle:vertical {
                background: #555;
                border-radius: 6px;
                min-height: 20px;
            }
            QScrollBar::handle:vertical:hover {
                background: #777;
            }
            """
        )
        log_layout.addWidget(self.stream_text)

        # --- Repetition-warning panel (pipeline-only feature) ---
        self.create_repetition_warning_panel(log_layout)

        # --- Typing indicator (chat-only) ---
        self.typing_label = QLabel("")
        self.typing_label.setStyleSheet(
            "color: #8be9fd; font-size: 9pt; font-style: italic; "
            "padding: 2px 12px; background-color: #1e1e1e; "
            "border-top: 1px solid #2a2a2a;"
        )
        self.typing_label.setVisible(False)
        log_layout.addWidget(self.typing_label)
        self._typing_timer = QTimer(self)
        self._typing_timer.setInterval(400)
        self._typing_timer.timeout.connect(self._tick_typing)

        self.body_splitter.addWidget(log_container)

        # --- Chat input frame (expandable QTextEdit) -------------------
        input_frame = QFrame()
        input_frame.setStyleSheet(
            "QFrame { background-color: #2d2d2d; border-top: 1px solid #444; }"
        )
        input_layout = QHBoxLayout(input_frame)
        input_layout.setContentsMargins(6, 4, 6, 4)
        input_layout.setSpacing(6)

        self.input_field = ChatInputEdit()
        self.input_field.setPlaceholderText(
            "Frage stellen — Enter = senden, Shift+Enter = neue Zeile"
        )
        self.input_field.setStyleSheet(
            "QTextEdit { background-color: #3d3d3d; color: #e0e0e0; "
            "border: 1px solid #555; border-radius: 4px; padding: 4px 8px; "
            "font-size: 11pt; }"
            "QTextEdit:focus { border: 1px solid #8be9fd; }"
        )
        self.input_field.setFont(get_scaled_font(size_delta=0))
        fm = self.input_field.fontMetrics()
        line_h = fm.lineSpacing()
        self.input_field.setMinimumHeight(line_h * 3 + 14)
        self.input_field.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        self.input_field.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        self.input_field.submit.connect(self.send_message)
        input_layout.addWidget(self.input_field, stretch=1)

        self.send_btn = QPushButton("Senden")
        self.send_btn.setStyleSheet(get_button_styles().get("primary", ""))
        self.send_btn.setDefault(True)
        self.send_btn.setSizePolicy(
            QSizePolicy.Policy.Fixed,
            QSizePolicy.Policy.Fixed,
        )
        self.send_btn.clicked.connect(self.send_message)
        input_layout.addWidget(
            self.send_btn,
            alignment=Qt.AlignmentFlag.AlignBottom,
        )

        self.body_splitter.addWidget(input_frame)

        # Default ratio: log gets ~80%, input ~20% — both grow with window.
        self.body_splitter.setStretchFactor(0, 8)
        self.body_splitter.setStretchFactor(1, 2)
        self.body_splitter.setSizes([800, 200])

        outer.addWidget(self.body_splitter, stretch=1)

        # Initial model-status paint.
        self._refresh_model_status()

    # ==================================================================
    # Repetition-warning panel (verbatim port from PipelineStreamWidget)
    # ==================================================================

    _STYLE_WARNING_GREEN = """
        QFrame {
            background-color: #1b3a1f;
            border: 1px solid #4caf50;
            border-radius: 3px;
            padding: 1px;
        }
        QLabel { color: #a5d6a7; }
        QPushButton {
            background-color: #2e7d32;
            color: #fff;
            border: none;
            border-radius: 3px;
            padding: 1px 5px;
            font-weight: bold;
        }
        QPushButton:hover { background-color: #43a047; }
    """

    _STYLE_WARNING_ORANGE = """
        QFrame {
            background-color: #3d2a00;
            border: 1px solid #ff9800;
            border-radius: 3px;
            padding: 1px;
        }
        QLabel { color: #ffcc80; }
        QPushButton {
            background-color: #ff9800;
            color: #1e1e1e;
            border: none;
            border-radius: 3px;
            padding: 1px 5px;
            font-weight: bold;
        }
        QPushButton:hover { background-color: #ffb74d; }
    """

    _STYLE_WARNING_HIDDEN = """
        QFrame { background: transparent; border: none; padding: 0; }
        QLabel { color: transparent; }
        QPushButton { background: transparent; border: none; color: transparent; }
    """

    def create_repetition_warning_panel(self, layout):
        self.repetition_warning_frame = QFrame()
        self.repetition_warning_frame.setFixedHeight(28)
        self._warning_style_state = "hidden"
        self._last_shown_detection_type = ""
        self.repetition_warning_frame.setStyleSheet(self._STYLE_WARNING_HIDDEN)

        bar = QHBoxLayout(self.repetition_warning_frame)
        bar.setContentsMargins(6, 1, 4, 1)
        bar.setSpacing(6)

        self.warning_icon_label = QLabel("⚠️")
        bar.addWidget(self.warning_icon_label)

        self.warning_title_label = QLabel("Wiederholung erkannt")
        self.warning_title_label.setStyleSheet("font-weight: bold; color: #ff9800;")
        bar.addWidget(self.warning_title_label)

        self.warning_details_label = QLabel("")
        self.warning_details_label.setWordWrap(False)
        self.warning_details_label.setStyleSheet("color: #ffe0b2;")
        bar.addWidget(self.warning_details_label, 1)

        self.countdown_label = QLabel("")
        self.countdown_label.setStyleSheet("color: #fff; font-weight: bold;")
        self.countdown_label.setVisible(False)
        bar.addWidget(self.countdown_label)

        self.suggestions_button_layout = QHBoxLayout()
        self.suggestions_button_layout.setSpacing(3)
        bar.addLayout(self.suggestions_button_layout)

        self.abort_now_button = QPushButton("🛑 Abbrechen")
        self.abort_now_button.setStyleSheet(
            "background-color: #d32f2f; color: white; font-weight: bold;"
            " border-radius: 3px; padding: 1px 5px;"
        )
        self.abort_now_button.clicked.connect(self._on_abort_requested)
        bar.addWidget(self.abort_now_button)

        self.continue_button = QPushButton("Fortfahren")
        self.continue_button.setStyleSheet(
            "background-color: #555; color: #ccc; padding: 1px 5px;"
        )
        self.continue_button.clicked.connect(self.hide_repetition_warning)
        bar.addWidget(self.continue_button)

        self.dismiss_warning_button = QPushButton("✕")
        self.dismiss_warning_button.setFixedSize(18, 18)
        self.dismiss_warning_button.setStyleSheet(
            "background-color: transparent; color: #ff9800; padding: 0;"
        )
        self.dismiss_warning_button.clicked.connect(self.hide_repetition_warning)
        bar.addWidget(self.dismiss_warning_button)

        self.grace_timer = QTimer(self)
        self.grace_timer.timeout.connect(self._update_countdown)
        self.grace_period_end = 0.0

        layout.addWidget(self.repetition_warning_frame)

    def show_repetition_warning(
        self,
        detection_type: str,
        details: str,
        suggestions: List[Dict],
        grace_period: bool = False,
        grace_seconds: float = 2.0,
    ):
        self.current_suggestions = suggestions
        already_showing = (
            self._warning_style_state == "orange"
            and self._last_shown_detection_type == detection_type
        )
        if self._warning_style_state != "orange":
            self.repetition_warning_frame.setStyleSheet(self._STYLE_WARNING_ORANGE)
            self._warning_style_state = "orange"
        self._last_shown_detection_type = detection_type

        if not already_showing:
            self.warning_icon_label.setText("⚠️")
            self.warning_title_label.setStyleSheet(
                "font-weight: bold; color: #ff9800;"
            )
            self.continue_button.setStyleSheet(
                "background-color: #555; color: #ccc; padding: 1px 5px;"
            )

            type_labels = {
                "char_pattern": "Zeichenwiederholung erkannt",
                "ngram": "Phrasenwiederholung erkannt",
                "window_similarity": "Textblock-Wiederholung erkannt",
            }
            self.warning_title_label.setText(
                type_labels.get(detection_type, "Wiederholung erkannt")
            )

            while self.suggestions_button_layout.count():
                item = self.suggestions_button_layout.takeAt(0)
                if item.widget():
                    item.widget().deleteLater()

            for i, suggestion in enumerate(suggestions[:3]):
                button = QPushButton(suggestion.get("label", f"Option {i+1}"))
                button.setToolTip(suggestion.get("description", ""))
                button.setStyleSheet("padding: 1px 4px;")
                params = suggestion.get("params", {})

                def make_handler(p):
                    return lambda: self._on_suggestion_clicked(p)

                button.clicked.connect(make_handler(params))
                self.suggestions_button_layout.addWidget(button)

        self.warning_details_label.setText(details)

        if grace_period and not already_showing:
            self.grace_period_end = time.time() + grace_seconds
            self.grace_timer.stop()
            self.grace_timer.start(200)
            self.countdown_label.setText(f"⏳ {grace_seconds:.1f}s")
        elif not grace_period:
            self.countdown_label.setText("")
            self.grace_timer.stop()

    def _update_countdown(self):
        remaining = self.grace_period_end - time.time()
        if remaining > 0:
            self.countdown_label.setText(f"⏳ {remaining:.1f}s")
        else:
            self.countdown_label.setText("⏳ …")
            self.grace_timer.stop()

    def hide_repetition_warning(self, resolved: bool = False):
        self.grace_timer.stop()
        self.countdown_label.setText("")
        self._last_shown_detection_type = ""

        if resolved:
            if self._warning_style_state != "green":
                self.repetition_warning_frame.setStyleSheet(self._STYLE_WARNING_GREEN)
                self._warning_style_state = "green"
            self.warning_icon_label.setText("✅")
            self.warning_title_label.setText(
                "Wiederholung behoben – Generation läuft weiter"
            )
            self.warning_title_label.setStyleSheet(
                "font-weight: bold; color: #4caf50;"
            )
            self.warning_details_label.setText("")
            while self.suggestions_button_layout.count():
                item = self.suggestions_button_layout.takeAt(0)
                if item.widget():
                    item.widget().deleteLater()
            self.continue_button.setStyleSheet(
                "background: transparent; border: none; color: transparent;"
            )
        else:
            if self._warning_style_state != "hidden":
                self.repetition_warning_frame.setStyleSheet(self._STYLE_WARNING_HIDDEN)
                self._warning_style_state = "hidden"
            self.warning_icon_label.setText("")
            self.warning_title_label.setText("")
            self.warning_details_label.setText("")
            self.countdown_label.setText("")
            self.continue_button.setStyleSheet(
                "background: transparent; border: none; color: transparent;"
            )
            while self.suggestions_button_layout.count():
                item = self.suggestions_button_layout.takeAt(0)
                if item.widget():
                    item.widget().deleteLater()

    def _on_abort_requested(self):
        self.hide_repetition_warning()
        self.abort_generation_requested.emit()

    def _on_suggestion_clicked(self, params: Dict):
        self.hide_repetition_warning()
        self.retry_with_variations.emit(params)
        self.add_pipeline_message(
            f"🔄 Retry mit Parametern: {params}",
            "info",
            self.current_step_id,
        )

    # ==================================================================
    # Pipeline rendering API (preserved from PipelineStreamWidget)
    # ==================================================================

    def add_pipeline_message(
        self,
        message: str,
        level: str = "info",
        step_id: Optional[str] = None,
    ):
        timestamp = datetime.now().strftime("%H:%M:%S")
        color_map = {
            "info": "#f8f8f2",
            "success": "#50fa7b",
            "warning": "#f1fa8c",
            "error": "#ff5555",
            "step": "#8be9fd",
            "stream": "#bd93f9",
            "debug": "#6272a4",
        }
        color = color_map.get(level, "#f8f8f2")

        if step_id:
            formatted = (
                f"<span style='color: #6272a4;'>[{timestamp}]</span> "
                f"<span style='color: {color}; font-weight: bold;'>[{step_id.upper()}]</span> "
                f"<span style='color: {color};'>{message}</span>"
            )
        else:
            formatted = (
                f"<span style='color: #6272a4;'>[{timestamp}]</span> "
                f"<span style='color: {color};'>{message}</span>"
            )

        self.stream_text.append(formatted)
        if self.auto_scroll_checkbox.isChecked():
            self.auto_scroll_to_bottom()

    def add_streaming_token(self, token: str, step_id: str):
        cursor = self.stream_text.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        escaped = html_escape(token).replace(" ", "&nbsp;").replace("\n", "<br>")
        cursor.insertHtml(f"<span style='color: #bd93f9;'>{escaped}</span>")
        if self.auto_scroll_checkbox.isChecked():
            self.auto_scroll_to_bottom()

    def start_streaming_line(self, step_id: str, prefix: str = ""):
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_prefix = (
            f"<span style='color: #6272a4;'>[{timestamp}]</span> "
            f"<span style='color: #8be9fd; font-weight: bold;'>[{step_id.upper()}]</span> "
            f"<span style='color: #bd93f9;'>{prefix}"
        )
        self.stream_text.append(formatted_prefix)
        self.is_streaming = True

    def end_streaming_line(self):
        if self.is_streaming:
            cursor = self.stream_text.textCursor()
            cursor.movePosition(QTextCursor.MoveOperation.End)
            cursor.insertHtml("</span>")
            self.is_streaming = False

    def auto_scroll_to_bottom(self):
        now = time.time()
        if now - self._last_scroll_time < 0.05:
            return
        self._last_scroll_time = now
        scrollbar = self.stream_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    @pyqtSlot(object)
    def on_pipeline_started(self, pipeline_id: str):
        self.add_pipeline_message("🚀 Pipeline gestartet", "step")
        self.add_pipeline_message(f"Pipeline ID: {pipeline_id}", "info")
        self.pipeline_start_time = datetime.now()

    @pyqtSlot(object)
    def on_step_started(self, step: PipelineStep):
        self.current_step_id = step.step_id
        self.step_start_times[step.step_id] = datetime.now()
        self.add_pipeline_message(
            f"▶ Starte Schritt: {step.name}", "step", step.step_id
        )
        if step.provider and step.model:
            self.add_pipeline_message(
                f"✓ Verwende: {step.provider} / {step.model}",
                "success",
                step.step_id,
            )

    @pyqtSlot(object)
    def on_step_completed(self, step: PipelineStep):
        duration = "unbekannt"
        if step.step_id in self.step_start_times:
            duration_seconds = (
                datetime.now() - self.step_start_times[step.step_id]
            ).total_seconds()
            duration = f"{duration_seconds:.1f}s"
        self.add_pipeline_message(
            f"✅ Schritt abgeschlossen in {duration}", "success", step.step_id
        )

        if step.output_data:
            if step.step_id == "keywords" and (
                "keywords" in step.output_data or "final_keywords" in step.output_data
            ):
                keywords = step.output_data.get(
                    "final_keywords", step.output_data.get("keywords", [])
                )
                self.add_pipeline_message(
                    f"Gefunden: {len(keywords)} Keywords", "info", step.step_id
                )
                self.add_pipeline_message(
                    f"Keywords: {', '.join(keywords[:5])}"
                    + ("..." if len(keywords) > 5 else ""),
                    "info",
                    step.step_id,
                )

                verification = step.output_data.get("verification")
                if verification and isinstance(verification, dict):
                    stats = verification.get("stats", {})
                    verified_count = stats.get("verified_count", 0)
                    total = stats.get("total_extracted", 0)
                    rejected = verification.get("rejected", [])
                    self.add_pipeline_message(
                        f"✅ {verified_count}/{total} Keywords GND-verifiziert",
                        "success",
                        step.step_id,
                    )
                    if rejected:
                        rejected_names = [r.split("(")[0].strip() for r in rejected]
                        self.add_pipeline_message(
                            f"⚠️ {len(rejected)} Keywords ohne GND-Pool-Treffer entfernt: "
                            + ", ".join(rejected_names),
                            "warning",
                            step.step_id,
                        )

            elif step.step_id == "search" and "search_results" in step.output_data:
                count = step.output_data["search_results"]
                self.add_pipeline_message(
                    f"Gefunden: {count} GND-Einträge", "info", step.step_id
                )

            elif (
                step.step_id == "verification"
                and "verified_keywords" in step.output_data
            ):
                verified = step.output_data["verified_keywords"]
                self.add_pipeline_message(
                    f"Verifiziert: {len(verified)} Keywords", "info", step.step_id
                )

            elif (
                step.step_id == "dk_search"
                and "dk_search_results" in step.output_data
            ):
                dk_results = step.output_data["dk_search_results"]
                self._display_dk_search_results(dk_results, step.step_id)

    def _display_dk_search_results(
        self, dk_results: List[Dict[str, Any]], step_id: str
    ):
        if not dk_results:
            self.add_pipeline_message(
                "Keine Klassifikationen (DK/RVK) gefunden", "info", step_id
            )
            return

        total_keywords = len(dk_results)
        total_classifications = sum(
            len(r.get("classifications", [])) for r in dk_results
        )
        cache_count = sum(1 for r in dk_results if r.get("source") == "cache")
        live_count = total_keywords - cache_count
        success_count = sum(1 for r in dk_results if r.get("classifications"))

        self.add_pipeline_message(
            f"🔍 Klassifikationssuche: {total_keywords} Keywords → "
            f"{success_count} erfolgreich → {total_classifications} Klassifikationen",
            "info",
            step_id,
        )
        if cache_count > 0 or live_count > 0:
            self.add_pipeline_message(
                f"   📦 Cache: {cache_count} | 🔍 Live: {live_count}",
                "debug",
                step_id,
            )

        for keyword_result in dk_results:
            keyword = keyword_result.get("keyword", "unknown")
            source = keyword_result.get("source", "unknown")
            search_time = keyword_result.get("search_time_ms", 0)
            classifications = keyword_result.get("classifications", [])
            if classifications:
                status_icon = "✅"
                msg_type = "info"
                status_text = f"{len(classifications)} Klassifikationen"
            else:
                status_icon = "⚠️"
                msg_type = "warning"
                status_text = "Keine Klassifikationen"
            source_icon = "📦" if source == "cache" else "🔍"
            timing_text = f"({search_time:.1f}ms)" if search_time > 0 else ""
            self.add_pipeline_message(
                f"{status_icon} {source_icon} {keyword} - {status_text} {timing_text}",
                msg_type,
                step_id,
            )
            if classifications:
                self.add_pipeline_message(
                    f"   ✓ {len(classifications)} Klassifikationen (DK/RVK) gefunden",
                    "debug",
                    step_id,
                )

    @pyqtSlot(object, str)
    def on_step_error(self, step: PipelineStep, error_message: str):
        self.add_pipeline_message(
            f"❌ Fehler in Schritt: {step.name}", "error", step.step_id
        )
        self.add_pipeline_message(
            f"Fehlermeldung: {error_message}", "error", step.step_id
        )

    @pyqtSlot(object)
    def on_pipeline_completed(self, analysis_state):
        total_duration = "unbekannt"
        if hasattr(self, "pipeline_start_time"):
            total_seconds = (
                datetime.now() - self.pipeline_start_time
            ).total_seconds()
            total_duration = f"{total_seconds:.1f}s"
        self.add_pipeline_message(
            f"\U0001f389 Pipeline vollständig abgeschlossen in {total_duration}!",
            "success",
        )

        if (
            analysis_state
            and hasattr(analysis_state, "final_llm_analysis")
            and analysis_state.final_llm_analysis
        ):
            kw_list = analysis_state.final_llm_analysis.extracted_gnd_keywords or []
            if kw_list:
                kw_display = ", ".join(kw_list)
                self.add_pipeline_message(
                    f"\U0001f4cc {len(kw_list)} GND-Schlagworte ausgewählt:\n{kw_display}",
                    "success",
                )
            response_text = (
                analysis_state.final_llm_analysis.response_full_text or ""
            )
            if (
                "Schlagwortketten" in response_text
                or "schlagwortketten" in response_text.lower()
            ):
                chain_lines = [
                    line
                    for line in response_text.split("\n")
                    if "→" in line or "->" in line
                ]
                if chain_lines:
                    self.add_pipeline_message(
                        "\U0001f517 Schlagwortketten:\n" + "\n".join(chain_lines[:10]),
                        "success",
                    )

        if analysis_state and getattr(analysis_state, "dk_classifications", None):
            dk_codes = analysis_state.dk_classifications
            dk_display_parts = []
            flat = getattr(analysis_state, "dk_search_results_flattened", None)
            if flat:
                for item in flat[:10]:
                    dk_code = item.get("dk", "")
                    title = (
                        ", ".join(item.get("titles", []))
                        if item.get("titles")
                        else ""
                    )
                    if dk_code:
                        dk_display_parts.append(
                            f"{dk_code} ({title})" if title else dk_code
                        )
            if not dk_display_parts:
                dk_display_parts = dk_codes[:10]
            self.add_pipeline_message(
                "\U0001f3f7 DK-Klassifikationen:\n" + ", ".join(dk_display_parts),
                "success",
            )

        if (
            analysis_state
            and hasattr(analysis_state, "rvk_provenance")
            and analysis_state.rvk_provenance
        ):
            rvk_count = len(analysis_state.rvk_provenance)
            if rvk_count:
                self.add_pipeline_message(
                    f"\U0001f4d6 {rvk_count} RVK-Klassifikationen zugeordnet",
                    "success",
                )

        # Auto-load chat context for the just-finished pipeline.
        try:
            self.load_context(analysis_state)
        except Exception:
            self.logger.exception("PipelineChatPanel: load_context after pipeline failed")

    @pyqtSlot(str)
    def on_llm_token_received(self, token: str):
        if self.current_step_id:
            self.add_streaming_token(token, self.current_step_id)

    def start_llm_streaming(self, step_id: str):
        self.start_streaming_line(step_id, "LLM Antwort: ")

    def end_llm_streaming(self):
        self.end_streaming_line()

    def clear_stream(self):
        self.stream_text.clear()
        self.add_pipeline_message("Stream geleert", "info")

    def save_stream_log(self):
        from PyQt6.QtWidgets import QFileDialog
        from pathlib import Path

        if self.current_working_title:
            default_filename = f"{self.current_working_title}_log.txt"
        else:
            default_filename = (
                f"pipeline_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            )

        docs_dir = Path.home() / "Documents"
        if not docs_dir.exists():
            docs_dir = Path.home()
        default_path = str(docs_dir / default_filename)

        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Pipeline-Log speichern",
            default_path,
            "Text Files (*.txt);;All Files (*)",
        )
        if filename:
            try:
                with open(filename, "w", encoding="utf-8") as f:
                    plain_text = self.stream_text.toPlainText()
                    f.write(f"ALIMA Pipeline Log - {datetime.now().isoformat()}\n")
                    f.write("=" * 50 + "\n\n")
                    f.write(plain_text)
                self.add_pipeline_message(
                    f"Log gespeichert: {filename}", "success"
                )
            except Exception as e:
                self.add_pipeline_message(f"Fehler beim Speichern: {e}", "error")

    def set_working_title(self, working_title: str):
        self.current_working_title = working_title
        self.logger.info(
            f"PipelineChatPanel: working_title set to '{working_title}'"
        )

    def refresh_styles(self):
        if hasattr(self, "stream_text"):
            self.stream_text.setFont(get_scaled_font(monospace=True))

    def reset_for_new_pipeline(self):
        self.current_step_id = None
        self.step_start_times.clear()
        self.is_streaming = False
        self.current_working_title = None
        self.clear_stream()
        self.hide_repetition_warning()
        if self.reset_toggle.isChecked():
            self.session.reset()
            self.current_context = ""
            self.working_title = ""
            self._assistant_block_open = False
            self._assistant_cell_cursor = None

    # ==================================================================
    # Chat-side rendering & lifecycle (ported from ChatWidget)
    # ==================================================================

    # -- Typing indicator ------------------------------------------------

    def _show_typing(self, model_label: str) -> None:
        self._typing_model = model_label or "…"
        self._typing_dots = 0
        self._tick_typing()
        self.typing_label.setVisible(True)
        self._typing_timer.start()

    def _tick_typing(self) -> None:
        self._typing_dots = (self._typing_dots % 3) + 1
        dots = "●" * self._typing_dots + "○" * (3 - self._typing_dots)
        self.typing_label.setText(f"🤖 {self._typing_model}  {dots}")

    def _hide_typing(self) -> None:
        self._typing_timer.stop()
        self.typing_label.setVisible(False)
        self.typing_label.setText("")

    # -- Model resolution & combo persistence ----------------------------

    def _refresh_model_status(self) -> None:
        try:
            provider, model = self._resolve_provider_model()
        except Exception:
            provider, model = "", ""
        if provider and model:
            self.model_status_label.setText(f"→ {provider} | {model}")
        else:
            self.model_status_label.setText("→ (kein Modell)")

    @pyqtSlot(int)
    def _on_model_combo_changed(self, _idx: int) -> None:
        self._refresh_model_status()
        if self.persist_combo_toggle.isChecked():
            self._persist_combo_to_chat_config()

    @pyqtSlot(int)
    def _on_autonomous_toggle_changed(self, _state: int) -> None:
        """Persist autonomous-mode toggle to ChatConfig + chat-side state."""
        new_value = self.autonomous_toggle.isChecked()
        try:
            from ..utils.config_manager import ConfigManager
            cm = ConfigManager()
            cfg = cm.get_unified_config()
            chat_cfg = getattr(cfg, "chat_config", None)
            if chat_cfg is None:
                return
            chat_cfg.autonomous_pipeline = new_value
            full = cm.load_config()
            cm.save_config(full, preserve_unified=True)
            msg = "aktiv" if new_value else "aus"
            self._append_system_message(f"🤖 Autonom-Modus: {msg}")
        except Exception:
            self.logger.exception(
                "PipelineChatPanel: persist autonomous_pipeline failed"
            )

    def _persist_combo_to_chat_config(self) -> None:
        data = self.model_combo.currentData()
        if not data:
            return
        try:
            provider, model = data.split("|", 1)
        except ValueError:
            return
        try:
            from ..utils.config_manager import ConfigManager

            cm = ConfigManager()
            cfg = cm.get_unified_config()
            chat_cfg = getattr(cfg, "chat_config", None)
            if chat_cfg is None:
                return
            chat_cfg.default_provider = provider
            chat_cfg.default_model = model
            full = cm.load_config()
            cm.save_config(full, preserve_unified=True)
            self._append_system_message(
                f"💾 Chat-Default gespeichert: {provider} | {model}"
            )
        except Exception:
            self.logger.exception(
                "PipelineChatPanel: persist default model failed"
            )

    def _populate_model_combo(self):
        try:
            self.model_combo.clear()
            self.model_combo.addItem("-- Auto --", None)
            from ..utils.config_manager import ConfigManager

            config_manager = ConfigManager()
            unified_config = config_manager.get_unified_config()
            for provider in unified_config.get_enabled_providers():
                models = getattr(provider, "available_models", []) or []
                if not models and getattr(provider, "preferred_model", None):
                    models = [provider.preferred_model]
                for model in models:
                    self.model_combo.addItem(
                        f"{provider.name} | {model}",
                        f"{provider.name}|{model}",
                    )
        except Exception as e:
            self.logger.error(f"Error populating model combo: {e}")

    def _resolve_provider_model(self) -> tuple[str, str]:
        override_data = self.model_combo.currentData()
        if override_data:
            provider, model = override_data.split("|", 1)
            if provider and model:
                return provider, model
        try:
            chat_cfg = self._get_chat_config()
            provider = getattr(chat_cfg, "default_provider", "") or ""
            model = getattr(chat_cfg, "default_model", "") or ""
            if provider and model:
                return provider, model
        except Exception:
            self.logger.exception(
                "PipelineChatPanel: ChatConfig provider lookup failed"
            )
        if self.pipeline_manager and hasattr(self.pipeline_manager, "config"):
            cfg = self.pipeline_manager.config
            provider = getattr(cfg, "global_provider_override", None)
            model = getattr(cfg, "global_model_override", None)
            if provider and model:
                return provider, model
        provider = getattr(self.llm_service, "current_provider", None) if self.llm_service else None
        model = getattr(self.llm_service, "current_model", None) if self.llm_service else None
        if provider and model:
            return provider, model
        try:
            clients = getattr(self.llm_service, "clients", {}) if self.llm_service else {}
            if clients:
                provider = list(clients.keys())[0]
                sp = getattr(self.llm_service, "supported_providers", {})
                pinfo = sp.get(provider, {})
                models = pinfo.get("models", [])
                if models:
                    return provider, models[0]
        except Exception:
            pass
        return "", ""

    def _get_chat_config(self):
        try:
            from ..utils.config_manager import ConfigManager

            cfg = ConfigManager().get_unified_config()
            chat_cfg = getattr(cfg, "chat_config", None)
            if chat_cfg is not None:
                return chat_cfg
        except Exception:
            self.logger.exception(
                "PipelineChatPanel: ChatConfig lookup failed"
            )
        from ..utils.config_models import ChatConfig

        return ChatConfig()

    # -- Context loading -------------------------------------------------

    def load_context(self, analysis_state) -> None:
        if self.reset_toggle.isChecked():
            # Don't wipe the pipeline log; only reset chat session state.
            self.session.reset()
            self.current_context = ""
            self.working_title = ""
            self._assistant_block_open = False
            self._assistant_cell_cursor = None

        self.session.reset()

        if analysis_state is None:
            self.current_context = ""
            self.session.last_shared_context = None
            self._append_system_message(
                "ℹ️ Kein Pipeline-Kontext geladen. Chat funktioniert trotzdem — "
                "stelle einfach eine Frage."
            )
            return

        parts = []
        if hasattr(analysis_state, "working_title") and analysis_state.working_title:
            self.working_title = analysis_state.working_title
            parts.append(f"Titel: {analysis_state.working_title}")
        if (
            hasattr(analysis_state, "original_abstract")
            and analysis_state.original_abstract
        ):
            abstract = analysis_state.original_abstract
            if len(abstract) > 500:
                abstract = abstract[:500] + "..."
            parts.append(f"Abstract: {abstract}")
        self.current_context = "\n".join(parts) or "(kein Titel/Abstract)"

        ctx = self._shared_context_from_analysis_state(analysis_state)
        self.session.last_shared_context = ctx

        kw_count = len(getattr(ctx, "extracted_keywords", []) or []) if ctx else 0
        if not kw_count and ctx:
            kw_count = len(getattr(ctx, "initial_keywords", []) or [])
        dk_count = len(getattr(ctx, "dk_classifications", []) or []) if ctx else 0
        self._append_system_message(
            f"✅ Kontext geladen: {self.working_title or 'Unbenannt'}"
            f" ({kw_count} Keywords, {dk_count} DK-Codes, Tools aktiv)"
        )

    @staticmethod
    def _shared_context_from_analysis_state(state) -> Optional[object]:
        # P-ζ: bridge promoted to SharedContext.from_keyword_analysis_state.
        # Wrapper retained one cycle for any external caller; remove afterwards.
        try:
            from src.core.agents.shared_context import SharedContext
            return SharedContext.from_keyword_analysis_state(state)
        except Exception:
            return None

    # -- Send / cancel / worker callbacks --------------------------------

    def send_message(self):
        text = self.input_field.toPlainText().strip()
        if not text:
            return
        if self.current_worker and self.current_worker.isRunning():
            return

        self._append_user_message(text)
        self.input_field.clear()

        user_prompt = self.USER_PROMPT_TEMPLATE.format(
            context=self.current_context,
            user_message=text,
        )

        provider, model = self._resolve_provider_model()
        if not provider or not model:
            self._append_system_message(
                "⚠️ Kein LLM-Provider konfiguriert. Bitte in Pipeline-Einstellungen "
                "ein Modell wählen."
            )
            return

        self._refresh_shared_context()
        chat_config = self._get_chat_config()
        kb_manager = None
        if self.pipeline_manager is not None:
            kb_manager = (
                getattr(self.pipeline_manager, "unified_knowledge_manager", None)
                or getattr(self.pipeline_manager, "knowledge_manager", None)
            )
        try:
            tool_registry = build_chat_toolset(
                session=self.session,
                chat_config=chat_config,
                mcp_registry=self.mcp_registry,
                pipeline_manager=self.pipeline_manager,
                kb_manager=kb_manager,
                proposal_gateway=self.proposal_gateway,
            )
        except Exception:
            self.logger.exception(
                "PipelineChatPanel: build_chat_toolset failed"
            )
            self._append_system_message(
                "❌ Tool-Setup fehlgeschlagen — siehe Log."
            )
            return

        self._set_ui_running(True)
        self.current_worker = ChatAgentWorker(
            llm_service=self.llm_service,
            tool_registry=tool_registry,
            system_prompt=self.system_prompt,
            user_prompt=user_prompt,
            provider=provider,
            model=model,
            temperature=getattr(chat_config, "temperature", 0.5),
            max_iterations=30,
            timeout_seconds=getattr(chat_config, "timeout_seconds", 600),
            history=list(self.session.messages[-6:]),
        )
        self.current_worker.token_received.connect(self._on_token)
        self.current_worker.status_message.connect(self._on_status_message)
        self.current_worker.tool_called.connect(self._on_tool_called)
        self.current_worker.tool_result.connect(self._on_tool_result)
        self.current_worker.generation_finished.connect(self._on_finished)
        self.current_worker.generation_error.connect(self._on_error)
        self._current_render_model = f"{provider} | {model}"
        self._refresh_model_status()
        self._show_typing(self._current_render_model)
        self.current_worker.start()

        self.message_sent.emit(text)

    def cancel_generation(self):
        if self.current_worker and self.current_worker.isRunning():
            self.current_worker.request_stop()
            self._append_system_message("⏹ Generation abgebrochen.")
            self._set_ui_running(False)

    @pyqtSlot(str)
    def _on_token(self, token: str):
        if not self._assistant_block_open:
            self._hide_typing()
            self._open_assistant_message(self._current_render_model)
            self._assistant_block_open = True
        self._append_assistant_token(token)

    @pyqtSlot(str)
    def _on_status_message(self, line: str):
        text = (line or "").strip()
        if not text:
            return
        if text.startswith(self._STATUS_SKIP_PREFIXES):
            return
        self._append_tool_marker(text)

    @pyqtSlot(str, dict)
    def _on_tool_called(self, name: str, args: dict):
        args_preview = self._format_tool_args(args)
        self._append_tool_marker(f"🔧 {name}({args_preview})")

    @pyqtSlot(str, str)
    def _on_tool_result(self, name: str, result_str: str):
        preview = (result_str or "").strip().replace("\n", " ")
        if len(preview) > 120:
            preview = preview[:120] + "…"
        self._append_tool_marker(f"↳ {preview}")

    @pyqtSlot(object)
    def _on_finished(self, result):
        try:
            # Merge full conversation (user + tool calls + tool results + assistant)
            for msg in getattr(result, "messages", []) or []:
                self.session.messages.append(dict(msg))
            final = getattr(result, "content", "") or ""
            if final and not self._assistant_block_open:
                self._hide_typing()
                self._open_assistant_message(self._current_render_model)
                self._assistant_block_open = True
                self._append_assistant_token(final)
        except Exception:
            self.logger.exception(
                "PipelineChatPanel: failed to append assistant turn"
            )
        self._hide_typing()
        self._finalize_assistant_message()
        self._set_ui_running(False)

    @pyqtSlot(str)
    def _on_error(self, error: str):
        self._hide_typing()
        self._append_system_message(f"❌ Fehler: {error}")
        self._assistant_block_open = False
        self._assistant_cell_cursor = None
        self._set_ui_running(False)
        self.logger.error(f"PipelineChatPanel: generation error: {error}")

    # -- Bus subscriptions for agentic-pipeline tool events --------------

    def _on_state_changed(self, _diff: dict) -> None:
        self._refresh_shared_context()

    def _on_bus_tool_called(self, payload: dict) -> None:
        try:
            name = payload.get("name", "") or "tool"
            args = payload.get("arguments", {}) or {}
            args_preview = self._format_tool_args(args)
            self._append_tool_marker(f"🔧 {name}({args_preview})")
        except Exception:
            self.logger.exception(
                "PipelineChatPanel: bus tool.called rendering failed"
            )

    def _on_bus_tool_result(self, payload: dict) -> None:
        try:
            result = payload.get("result", "") or ""
            preview = result.strip().replace("\n", " ")
            if len(preview) > 120:
                preview = preview[:120] + "…"
            self._append_tool_marker(f"↳ {preview}")
        except Exception:
            self.logger.exception(
                "PipelineChatPanel: bus tool.result rendering failed"
            )

    def _on_bus_pipeline_step(self, payload: dict) -> None:
        """Render step-progress markers for run_pipeline / rerun_step tools (P-ζ)."""
        try:
            status = payload.get("status", "")
            step_id = payload.get("step_id", "") or "?"
            name = payload.get("name", "") or step_id
            icon = "🔄" if status == "running" else ("✅" if status == "completed" else "•")
            self._append_tool_marker(f"{icon} Step {step_id}: {name}")
        except Exception:
            self.logger.exception(
                "PipelineChatPanel: bus state.pipeline_step rendering failed"
            )

    def _refresh_shared_context(self) -> None:
        try:
            if self.pipeline_manager is None:
                return
            ctx = getattr(self.pipeline_manager, "last_shared_context", None)
            if ctx is not None:
                self.session.last_shared_context = ctx
        except Exception:
            self.logger.exception(
                "PipelineChatPanel: shared-context refresh failed"
            )

    # -- UI helpers ------------------------------------------------------

    def _set_ui_running(self, running: bool):
        self.send_btn.setVisible(not running)
        self.cancel_btn.setVisible(running)
        self.input_field.setEnabled(not running)
        if running:
            self.input_field.setPlaceholderText("Antwort wird generiert...")
        else:
            self.input_field.setPlaceholderText(
                "Frage zu den Pipeline-Ergebnissen stellen..."
            )
            self.input_field.setFocus()

    # -- Chat bubble & marker rendering ---------------------------------

    def _append_user_message(self, text: str):
        self._insert_bubble(
            text,
            align=Qt.AlignmentFlag.AlignRight,
            width_percent=65,
            bg_color="#005c4b",
            fg_color="#e9edef",
        )

    def _insert_bubble(
        self,
        text: str,
        *,
        align: Qt.AlignmentFlag,
        width_percent: int,
        bg_color: str,
        fg_color: str,
    ) -> None:
        cursor = self.stream_text.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        if not self.stream_text.document().isEmpty():
            cursor.insertBlock(QTextBlockFormat())
        table_fmt = QTextTableFormat()
        table_fmt.setCellPadding(8)
        table_fmt.setCellSpacing(0)
        table_fmt.setBorder(0)
        table_fmt.setWidth(
            QTextLength(QTextLength.Type.PercentageLength, width_percent)
        )
        table_fmt.setAlignment(align)
        table = cursor.insertTable(1, 1, table_fmt)
        cell = table.cellAt(0, 0)
        cell_fmt = QTextTableCellFormat()
        cell_fmt.setBackground(QColor(bg_color))
        cell.setFormat(cell_fmt)
        body = self._escape_html(text).replace("\n", "<br>")
        cell.firstCursorPosition().insertHtml(
            f'<span style="color: {fg_color}; font-size: 10pt;">{body}</span>'
        )
        end_cursor = self.stream_text.textCursor()
        end_cursor.movePosition(QTextCursor.MoveOperation.End)
        end_cursor.insertBlock(QTextBlockFormat())
        if self.auto_scroll_checkbox.isChecked():
            self.auto_scroll_to_bottom()

    def _append_tool_marker(self, text: str):
        html = (
            f'<div style="margin: 2px 0 2px 8px; '
            f'font-family: monospace; font-size: 9pt; color: #888;">'
            f"{self._escape_html(text)}</div>"
        )
        self._append_html(html)

    def _append_system_message(self, text: str):
        html = (
            f'<div style="text-align: center; margin: 4px 0;">'
            f'<span style="color: #4caf50; font-size: 9pt; font-style: italic;">'
            f"{self._escape_html(text)}</span></div>"
        )
        self._append_html(html)

    def _open_assistant_message(self, model_label: str):
        self._current_assistant_text = ""
        cursor = self.stream_text.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        if not self.stream_text.document().isEmpty():
            cursor.insertBlock(QTextBlockFormat())
        cursor.insertHtml(
            f'<span style="color: #8be9fd; font-size: 9pt; font-style: italic;">'
            f'🤖 {self._escape_html(model_label or "Modell")}'
            f"</span>"
        )
        cursor.insertBlock(QTextBlockFormat())
        table_fmt = QTextTableFormat()
        table_fmt.setCellPadding(8)
        table_fmt.setCellSpacing(0)
        table_fmt.setBorder(0)
        table_fmt.setWidth(QTextLength(QTextLength.Type.PercentageLength, 75))
        table_fmt.setAlignment(Qt.AlignmentFlag.AlignLeft)
        table = cursor.insertTable(1, 1, table_fmt)
        cell = table.cellAt(0, 0)
        cell_fmt = QTextTableCellFormat()
        cell_fmt.setBackground(QColor("#202c33"))
        cell.setFormat(cell_fmt)
        self._assistant_cell_cursor = cell.firstCursorPosition()
        if self.auto_scroll_checkbox.isChecked():
            self.auto_scroll_to_bottom()

    def _append_assistant_token(self, token: str):
        if self._assistant_cell_cursor is None:
            return
        self._current_assistant_text += token
        html = self._escape_html(token).replace("\n", "<br>").replace(" ", "&nbsp;")
        self._assistant_cell_cursor.insertHtml(
            f'<span style="color: #e9edef; font-size: 10pt;">{html}</span>'
        )
        if self.auto_scroll_checkbox.isChecked():
            self.auto_scroll_to_bottom()

    def _finalize_assistant_message(self):
        if self._assistant_cell_cursor is not None and getattr(self, "_current_assistant_text", ""):
            try:
                import markdown
                md_html = markdown.markdown(
                    self._current_assistant_text,
                    extensions=["extra", "nl2br"],
                )
                cursor = self._assistant_cell_cursor
                cursor.movePosition(QTextCursor.MoveOperation.Start, QTextCursor.MoveMode.MoveAnchor)
                cursor.movePosition(QTextCursor.MoveOperation.End, QTextCursor.MoveMode.KeepAnchor)
                cursor.removeSelectedText()
                cursor.insertHtml(
                    f'<span style="color: #e9edef; font-size: 10pt;">{md_html}</span>'
                )
            except Exception:
                pass  # Keep raw text if markdown fails
        cursor = self.stream_text.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        cursor.insertBlock(QTextBlockFormat())
        self._assistant_block_open = False
        self._assistant_cell_cursor = None
        self._current_assistant_text = ""
        if self.auto_scroll_checkbox.isChecked():
            self.auto_scroll_to_bottom()

    def _append_html(self, html: str):
        cursor = self.stream_text.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        if not self.stream_text.document().isEmpty():
            cursor.insertBlock(QTextBlockFormat())
        cursor.insertHtml(html)
        if self.auto_scroll_checkbox.isChecked():
            self.auto_scroll_to_bottom()

    # -- P-ε: mutation-proposal inline bubble --------------------------

    @pyqtSlot(int, str, dict)
    def _render_proposal_bubble(
        self, audit_id: int, tool_name: str, payload: dict
    ) -> None:
        """Render a clickable confirmation bubble for a mutation proposal.

        Triggered by ``ProposalGateway.proposal_requested`` (auto-marshalled
        to the UI thread by Qt). The user clicks one of the embedded links;
        ``_on_anchor_clicked`` then routes the decision back to the gateway.
        """
        title_map = {
            "propose_keyword_replacement": "🔁 Vorschlag: Keyword ersetzen",
            "propose_dk_change": "🏷️ Vorschlag: DK-Klassifikation ändern",
        }
        title = title_map.get(tool_name, f"⚠️ Mutations-Vorschlag: {tool_name}")

        if tool_name == "propose_keyword_replacement":
            old = self._escape_html(str(payload.get("old", "")))
            new = self._escape_html(str(payload.get("new", "")))
            gnd = payload.get("gnd_id") or ""
            gnd_str = f" <span style='color: #888;'>(GND-ID: {self._escape_html(gnd)})</span>" if gnd else ""
            diff_html = f"<b>{old}</b> → <b>{new}</b>{gnd_str}"
        elif tool_name == "propose_dk_change":
            code = self._escape_html(str(payload.get("code", "")))
            action = str(payload.get("action", ""))
            verb = "hinzufügen" if action == "add" else "entfernen"
            diff_html = f"<b>{code}</b> ({verb})"
        else:
            diff_html = self._escape_html(str(payload))

        reason = self._escape_html(str(payload.get("reason", "") or "—"))
        accept_href = f"mutation://{audit_id}/accept"
        reject_href = f"mutation://{audit_id}/reject"

        html = (
            f'<div style="margin: 6px 12px; padding: 10px; '
            f'background-color: #2d3142; border-left: 3px solid #ffb86c; '
            f'border-radius: 4px;">'
            f'<div style="color: #ffb86c; font-weight: bold; font-size: 10pt;">{title}</div>'
            f'<div style="color: #f8f8f2; margin-top: 4px;">{diff_html}</div>'
            f'<div style="color: #888; font-size: 9pt; margin-top: 4px;">'
            f'Begründung: {reason}</div>'
            f'<div style="margin-top: 8px;">'
            f'<a href="{accept_href}" style="color: #50fa7b; '
            f'text-decoration: none; padding: 4px 10px; '
            f'border: 1px solid #50fa7b; border-radius: 3px; '
            f'margin-right: 8px;">✓ Akzeptieren</a>'
            f'<a href="{reject_href}" style="color: #ff5555; '
            f'text-decoration: none; padding: 4px 10px; '
            f'border: 1px solid #ff5555; border-radius: 3px;">✗ Ablehnen</a>'
            f'<span style="color: #555; font-size: 8pt; margin-left: 8px;">'
            f'#audit_{audit_id}</span>'
            f'</div></div>'
        )
        self._append_html(html)

    @pyqtSlot(QUrl)
    def _on_anchor_clicked(self, url: QUrl) -> None:
        """Route ``mutation://`` link clicks to ProposalGateway."""
        if url.scheme() != "mutation":
            return
        host_part = url.host()
        path_part = url.path().lstrip("/")
        try:
            audit_id = int(host_part)
        except (TypeError, ValueError):
            return
        action = path_part.lower()
        if action not in ("accept", "reject"):
            return
        accepted = action == "accept"
        try:
            self.proposal_gateway.resolve_decision(audit_id, accepted)
        except Exception:
            self.logger.exception(
                "PipelineChatPanel: resolve_decision failed"
            )
            return
        status = "✓ Akzeptiert" if accepted else "✗ Abgelehnt"
        color = "#50fa7b" if accepted else "#ff5555"
        self._append_html(
            f'<div style="margin: 2px 24px; color: {color}; font-size: 9pt;">'
            f'{status} (#audit_{audit_id})</div>'
        )

    @staticmethod
    def _escape_html(text: str) -> str:
        return (
            text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
        )

    @staticmethod
    def _format_tool_args(args: dict) -> str:
        if not args:
            return ""
        parts = []
        for k, v in args.items():
            sv = repr(v)
            if len(sv) > 40:
                sv = sv[:40] + "…"
            parts.append(f"{k}={sv}")
        joined = ", ".join(parts)
        if len(joined) > 80:
            joined = joined[:80] + "…"
        return joined

    # -- System-prompt dialog --------------------------------------------

    def show_system_prompt_dialog(self):
        dialog = SystemPromptDialog(self.system_prompt, parent=self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            new_prompt = dialog.get_prompt()
            if new_prompt:
                self.system_prompt = new_prompt
                self._append_system_message("✅ System-Prompt aktualisiert.")
