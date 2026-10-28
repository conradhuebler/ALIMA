"""
Unified UI styles for ALIMA application
"""

from PyQt6.QtGui import QFont

# Module-level dark mode state — Claude Generated
_dark_mode = False

# Module-level font size state — Claude Generated
_font_size: int = 10

# Font families used app-wide — Claude Generated
DEFAULT_FONT_FAMILY = "Segoe UI"
MONOSPACE_FONT_FAMILY = "Consolas"


def set_dark_mode(dark: bool):
    global _dark_mode
    _dark_mode = dark


def set_font_size(pt: int) -> None:
    """Set global base font size (clamped 8–16 pt). Call get_main_stylesheet() afterwards. — Claude Generated"""
    global _font_size
    _font_size = max(8, min(16, int(pt)))


def get_font_size() -> int:
    """Return current global base font size in pt. — Claude Generated"""
    return _font_size


def get_scaled_font(size_delta: int = 0, *, monospace: bool = False, bold: bool = False) -> QFont:
    """Return a QFont scaled from the global base font size.

    Replaces hardcoded QFont(family, N) calls so font-size changes cascade everywhere.
    — Claude Generated
    """
    family = MONOSPACE_FONT_FAMILY if monospace else DEFAULT_FONT_FAMILY
    pt = max(6, _font_size + size_delta)
    f = QFont(family, pt)
    if bold:
        f.setBold(True)
    return f


def get_colors() -> dict:
    return DARK_COLORS if _dark_mode else COLORS


def is_dark_mode() -> bool:
    """Current theme as set by ``set_dark_mode``. - Claude Generated"""
    return _dark_mode


def set_style_prop(widget, name: str, value) -> None:
    """Set a dynamic property that the main stylesheet styles, and re-polish.

    The colours for ``tone``/``box``/``variant``/``state``/``role`` live in
    ``get_main_stylesheet()`` (see ``_component_rules``), which the main window
    re-applies on every theme switch, so a widget marked this way follows the
    theme without its own refresh code. - Claude Generated
    """
    if widget is None:
        return
    widget.setProperty(name, value if value is not None else "")
    style = widget.style()
    if style is not None:
        style.unpolish(widget)
        style.polish(widget)
    widget.update()


def set_tone(widget, tone) -> None:
    """Text colour role: muted, light, heading, accent, info, violet, success,
    warning, error, divider ("" clears it). - Claude Generated"""
    set_style_prop(widget, "tone", tone or "")


def set_box(widget, kind) -> None:
    """Tinted message frame: success, warning, error, info ("" clears it). - Claude Generated"""
    set_style_prop(widget, "box", kind or "")


def set_status(label, kind) -> None:
    """Status line as in ``get_status_label_styles``: colour via ``tone``
    (follows a theme switch), bold for success/warning/error/info.
    ``kind`` "default" clears the colour. - Claude Generated"""
    set_tone(label, "" if kind in (None, "", "default") else kind)
    label.setStyleSheet(
        "font-weight: bold;" if kind in ("success", "warning", "error", "info") else ""
    )


def set_variant(button, variant) -> None:
    """Button variant from ``get_button_styles`` (primary, secondary, accent,
    success, warning, error) as a property, so it follows a theme switch.
    - Claude Generated"""
    set_style_prop(button, "variant", variant or "")


# Color palette — light theme. Derived from the ALIMA logo (alima.png):
# rose/coral/apricot from the book and branch lines, blue/violet from the
# trunk, warm neutral grounds. Token + contrast table: docs/design_refresh.md.
# ``primary`` is a FILL colour that external widgets pair with white text, so
# it stays deep enough for 4.5:1 in both themes. - Claude Generated
COLORS = {
    "primary": "#a8284c",  # Rose (book pages, deepened)
    "primary_hover": "#8e1f3f",
    "primary_pressed": "#74193a",
    "secondary": "#3f4fb0",  # Blue/violet counter-accent (trunk)
    "secondary_hover": "#34429a",
    "secondary_pressed": "#2b3683",
    "accent": "#f7a07f",  # Apricot (branch nodes)
    "accent_hover": "#f48d68",
    "accent_pressed": "#e97a55",
    "background": "#fffcfa",
    "background_light": "#f6f2ef",
    "background_dark": "#e8e0db",
    "text": "#231925",
    "text_light": "#5e4f5b",
    "text_muted": "#6f606b",
    "error": "#b42318",
    "error_hover": "#962015",
    "success": "#1a7044",
    "success_hover": "#155c38",
    "warning": "#8a5800",
    "warning_hover": "#724900",
    "border": "#8c7c88",  # control borders (>= 3:1)
    "border_light": "#e2d8d2",  # dividers, frames
    "border_dark": "#6f606b",
    "shadow": "rgba(60, 20, 40, 0.10)",
    # Text on solid fills
    "on_primary": "#ffffff",
    "on_secondary": "#ffffff",
    "on_accent": "#231925",
    "on_success": "#ffffff",
    "on_warning": "#ffffff",
    "on_error": "#ffffff",
    "on_gradient": "#ffffff",
    # Interaction
    "focus": "#a8284c",
    "indicator": "#a8284c",  # selected tab underline, splitter/scroll hover
    "surface_raised": "#f1ebe7",  # neutral button ground
    "hover_tint": "#f3e7e6",
    "secondary_tint": "#eceefa",
    "scroll_handle": "#bfb0a8",
    # Text roles and tinted message grounds (labels, status frames, table rows)
    "accent_text": "#9c2a4b",  # = --alima-accent (light)
    "violet": "#4a4fb0",  # = --alima-purple (light)
    "success_bg": "#e7f3ec",
    "warning_bg": "#f8eedb",
    "error_bg": "#f9e6e4",
    "info_bg": "#eceefa",
    # Logo gradient: buttons (with on_gradient text) and progress chunk
    # (with the theme's text colour drawn across it)
    "grad_start": "#b8483a",
    "grad_mid": "#a8284c",
    "grad_end": "#8c2e63",
    "progress_start": "#f6b493",
    "progress_mid": "#f08a7c",
    "progress_end": "#e85d70",
    # Confidence level colors (text, background) - Claude Generated
    "confidence_very_high_text": "#2d5016",
    "confidence_very_high_bg": "#d4edda",
    "confidence_high_text": "#0c5460",
    "confidence_high_bg": "#d1ecf1",
    "confidence_medium_text": "#664d03",
    "confidence_medium_bg": "#fff3cd",
    "confidence_low_text": "#721c24",
    "confidence_low_bg": "#f8d7da",
    # Comparison colors (shared=green, only_a=blue/teal, only_b=amber/orange) - Claude Generated
    "comparison_shared_bg": "#d4edda",
    "comparison_shared_text": "#2d5016",
    "comparison_only_a_bg": "#d1ecf1",
    "comparison_only_a_text": "#0c5460",
    "comparison_only_b_bg": "#fff3cd",
    "comparison_only_b_text": "#664d03",
}

# Dark mode color palette — the logo's own ground: aubergine night with light
# peach/coral lines. Same keys as COLORS (tests/test_design_tokens.py).
# - Claude Generated
DARK_COLORS = {
    "primary": "#b8325a",
    "primary_hover": "#c63d66",
    "primary_pressed": "#a02a4f",
    "secondary": "#9aa6f2",
    "secondary_hover": "#b0b9f5",
    "secondary_pressed": "#8592e8",
    "accent": "#f7a07f",
    "accent_hover": "#f9b497",
    "accent_pressed": "#f48d68",
    "background": "#120e1a",  # = --alima-bg, the GUI log ground
    "background_light": "#1b1625",
    "background_dark": "#0b0912",
    "text": "#f1ebf0",
    "text_light": "#c9bdcb",
    "text_muted": "#a497a8",
    "error": "#ff7b7b",
    "error_hover": "#ff9494",
    "success": "#7ad6a0",
    "success_hover": "#92e0b2",
    "warning": "#f3c56b",
    "warning_hover": "#f6d189",
    "border": "#7a6a88",
    "border_light": "#2e2539",
    "border_dark": "#9a8aa5",
    "shadow": "rgba(0, 0, 0, 0.45)",
    "on_primary": "#ffffff",
    "on_secondary": "#1b0c14",
    "on_accent": "#1b0c14",
    "on_success": "#1b0c14",
    "on_warning": "#1b0c14",
    "on_error": "#1b0c14",
    "on_gradient": "#1b0c14",
    "focus": "#f9cdb0",
    "indicator": "#f7a07f",
    "surface_raised": "#241d2f",
    "hover_tint": "#251d31",
    "secondary_tint": "#22233d",
    "scroll_handle": "#574869",
    "accent_text": "#f8b896",  # = --alima-accent (dark)
    "violet": "#aab0f6",  # = --alima-purple (dark)
    "success_bg": "#16261e",
    "warning_bg": "#2a2214",
    "error_bg": "#2c1619",
    "info_bg": "#22233d",
    "grad_start": "#f9cdb0",
    "grad_mid": "#f7907c",
    "grad_end": "#ea5b73",
    "progress_start": "#ad4338",
    "progress_mid": "#a8284c",
    "progress_end": "#8c2e63",
    # Confidence level colors — dark tints
    "confidence_very_high_text": "#a6e3a1",
    "confidence_very_high_bg": "#1e2e1e",
    "confidence_high_text": "#89dceb",
    "confidence_high_bg": "#1e2a2e",
    "confidence_medium_text": "#f9e2af",
    "confidence_medium_bg": "#2e2a1e",
    "confidence_low_text": "#f38ba8",
    "confidence_low_bg": "#2e1e22",
    # Comparison colors — dark tints - Claude Generated
    "comparison_shared_bg": "#1e2e1e",
    "comparison_shared_text": "#a6e3a1",
    "comparison_only_a_bg": "#1e2a2e",
    "comparison_only_a_text": "#89dceb",
    "comparison_only_b_bg": "#2e2a1e",
    "comparison_only_b_text": "#f9e2af",
}

# Standard layout constants (static values; for font size use get_font_size())
LAYOUT = {
    "margin": 15,
    "spacing": 12,
    "inner_spacing": 8,
    "border_radius": 8,
    "input_font_size": 11,  # kept for backward compat — use get_font_size()+1 for dynamic value
}


def _asset_url(name: str) -> str:
    """Path of a shared UI asset (src/webapp/static) for a QSS ``url()``.
    Qt style sheets load images from files, not from data: URIs. - Claude Generated"""
    from pathlib import Path

    return (Path(__file__).resolve().parent.parent / "webapp" / "static" / name).as_posix()


def _shade(hex_color: str, factor: int) -> str:
    """Lighten (factor > 100) or darken (factor < 100) a hex colour. - Claude Generated"""
    from PyQt6.QtGui import QColor

    c = QColor(hex_color)
    return (c.lighter(factor) if factor >= 100 else c.darker(int(10000 / factor))).name()


def _gradient(c: dict, shift: int = 100) -> str:
    """Logo gradient (left → right) for QSS. ``shift`` moves every stop away
    from the button's text colour on hover (darker in light mode, lighter in
    dark mode), so contrast never drops below the resting state. - Claude Generated"""
    stops = (c["grad_start"], c["grad_mid"], c["grad_end"])
    if shift != 100:
        factor = shift if _dark_mode else int(10000 / shift)
        stops = tuple(_shade(s, factor) for s in stops)
    return (
        "qlineargradient(x1:0, y1:0, x2:1, y2:0, "
        f"stop:0 {stops[0]}, stop:0.55 {stops[1]}, stop:1 {stops[2]})"
    )


def get_main_stylesheet():
    """Get the main application stylesheet. Uses _font_size for global scaling. — Claude Generated

    Neutral buttons by default; widgets mark their one main action with
    ``get_button_styles()["primary"]`` (the logo gradient). Rule order matters in QSS: for the same property the
    later rule wins, hence ``:hover:!selected`` wherever a hover rule follows a
    ``:selected`` rule (tests/test_provider_list_selection_contrast.py).
    """
    fs = _font_size          # base
    fs1 = _font_size + 1     # inputs / group box titles
    c = get_colors()
    combo_arrow = _asset_url("ui-chevron.svg")  # #8c7c88, >= 3:1 on both grounds
    check_mark = _asset_url("ui-check.svg")  # white, on the primary fill
    return f"""
    /* Main application styles */
    QWidget {{
        font-family: 'Segoe UI', Arial, sans-serif;
        font-size: {fs}pt;
        color: {c['text']};
    }}

    QToolTip {{
        background-color: {c['background']};
        color: {c['text']};
        border: 1px solid {c['border_light']};
        padding: 4px 6px;
    }}

    /* Group boxes */
    QGroupBox {{
        font-weight: bold;
        font-size: {fs1}pt;
        border: 1px solid {c['border_light']};
        border-radius: 8px;
        margin-top: 12px;
        padding-top: 10px;
        background-color: {c['background_light']};
    }}

    QGroupBox::title {{
        subcontrol-origin: margin;
        left: 10px;
        padding: 0 5px;
        background-color: {c['background_light']};
        color: {c['text']};
    }}

    /* Buttons: neutral by default */
    QPushButton {{
        border: 1px solid {c['border']};
        border-radius: 6px;
        padding: 1px 4px;  /* + 1px border = the former 2px 5px box */
        background-color: {c['surface_raised']};
        color: {c['text']};
        font-weight: bold;
        font-size: {fs}pt;
        min-height: 20px;
    }}

    QPushButton:hover {{
        background-color: {c['hover_tint']};
        border: 1px solid {c['indicator']};
    }}

    QPushButton:pressed {{
        background-color: {c['background_dark']};
    }}

    QPushButton:checked {{
        background-color: {c['secondary_tint']};
        border: 1px solid {c['secondary']};
    }}

    QPushButton:focus {{
        border: 1px solid {c['focus']};
    }}

    QPushButton:disabled {{
        background-color: {c['background_light']};
        color: {c['text_muted']};
        border: 1px solid {c['border_light']};
    }}

    /* Text inputs */
    QTextEdit {{
        border: 1px solid {c['border']};
        border-radius: 6px;
        padding: 8px;
        background-color: {c['background']};
        selection-background-color: {c['primary']};
        selection-color: {c['on_primary']};
        font-size: {fs1}pt;
        line-height: 1.4;
    }}

    QTextEdit:focus {{
        border: 2px solid {c['focus']};
    }}

    QLineEdit {{
        border: 1px solid {c['border']};
        border-radius: 6px;
        padding: 8px;
        background-color: {c['background']};
        selection-background-color: {c['primary']};
        selection-color: {c['on_primary']};
        font-size: {fs}pt;
    }}

    QLineEdit:focus {{
        border: 2px solid {c['focus']};
    }}

    /* Labels */
    QLabel {{
        color: {c['text']};
        font-size: {fs}pt;
    }}

    /* Combo boxes */
    QComboBox {{
        border: 1px solid {c['border']};
        border-radius: 6px;
        padding: 6px 8px;
        background-color: {c['background']};
        font-size: {fs}pt;
        min-height: 20px;
    }}

    QComboBox:hover {{
        border: 1px solid {c['indicator']};
    }}

    QComboBox:focus {{
        border: 2px solid {c['focus']};
    }}

    QComboBox::drop-down {{
        border: none;
        width: 20px;
    }}

    QComboBox::down-arrow {{
        image: url({combo_arrow});
        width: 10px;
        height: 6px;
    }}

    QComboBox QAbstractItemView {{
        background-color: {c['background']};
        color: {c['text']};
        border: 1px solid {c['border_light']};
        selection-background-color: {c['primary']};
        selection-color: {c['on_primary']};
    }}

    /* Checkboxes: a check mark, not colour alone, marks the checked state */
    QCheckBox {{
        color: {c['text']};
        spacing: 8px;
        font-size: {fs}pt;
    }}

    QCheckBox::indicator {{
        width: 16px;
        height: 16px;
        border: 1px solid {c['border']};
        border-radius: 4px;
        background-color: {c['background']};
    }}

    QCheckBox::indicator:hover {{
        border: 1px solid {c['indicator']};
    }}

    QCheckBox::indicator:checked {{
        background-color: {c['primary']};
        border: 1px solid {c['primary']};
        image: url({check_mark});
    }}

    /* Tables */
    QTableWidget {{
        border: 1px solid {c['border_light']};
        border-radius: 6px;
        background-color: {c['background']};
        gridline-color: {c['border_light']};
        selection-background-color: {c['primary']};
        selection-color: {c['on_primary']};
        font-size: {fs}pt;
    }}

    QTableWidget::item {{
        padding: 8px;
        border-bottom: 1px solid {c['border_light']};
    }}

    QTableWidget::item:selected {{
        background-color: {c['primary']};
        color: {c['on_primary']};
    }}

    QHeaderView::section {{
        background-color: {c['background_light']};
        color: {c['text_light']};
        padding: 8px;
        border: none;
        border-right: 1px solid {c['border_light']};
        border-bottom: 1px solid {c['border_light']};
        font-weight: bold;
    }}

    /* Progress bars: the logo gradient as the fill */
    QProgressBar {{
        border: 1px solid {c['border_light']};
        border-radius: 6px;
        background-color: {c['background_dark']};
        text-align: center;
        font-size: {fs}pt;
        font-weight: bold;
    }}

    QProgressBar::chunk {{
        background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0,
            stop:0 {c['progress_start']}, stop:0.55 {c['progress_mid']}, stop:1 {c['progress_end']});
        border-radius: 5px;
    }}

    /* Sliders */
    QSlider::groove:horizontal {{
        height: 6px;
        background: {c['background_dark']};
        border-radius: 3px;
    }}

    QSlider::handle:horizontal {{
        background: {c['primary']};
        border: 1px solid {c['primary']};
        width: 16px;
        margin: -5px 0;
        border-radius: 8px;
    }}

    QSlider::handle:horizontal:hover {{
        background: {c['primary_hover']};
    }}

    /* Spin boxes */
    QSpinBox {{
        border: 1px solid {c['border']};
        border-radius: 6px;
        padding: 6px 8px;
        background-color: {c['background']};
        font-size: {fs}pt;
    }}

    QSpinBox:focus {{
        border: 2px solid {c['focus']};
    }}

    /* Tab widgets */
    QTabWidget::pane {{
        border: 1px solid {c['border_light']};
        border-radius: 6px;
        background-color: {c['background']};
    }}

    QTabBar::tab {{
        background-color: {c['background_light']};
        color: {c['text_light']};
        border: 1px solid {c['border_light']};
        padding: 8px 16px;
        margin-right: 2px;
        border-bottom: none;
        border-top-left-radius: 6px;
        border-top-right-radius: 6px;
    }}

    QTabBar::tab:selected {{
        background-color: {c['background']};
        color: {c['text']};
        border-bottom: 2px solid {c['indicator']};
    }}

    QTabBar::tab:hover:!selected {{
        background-color: {c['hover_tint']};
        color: {c['text']};
    }}

    /* Splitters */
    QSplitter::handle {{
        background-color: {c['border_light']};
        border: none;
    }}

    QSplitter::handle:hover {{
        background-color: {c['indicator']};
    }}

    QSplitter::handle:horizontal {{
        width: 3px;
    }}

    QSplitter::handle:vertical {{
        height: 3px;
    }}

    /* Scrollbars */
    QScrollBar:vertical {{
        border: none;
        background: {c['background_light']};
        width: 10px;
        margin: 0px;
    }}

    QScrollBar::handle:vertical {{
        background: {c['scroll_handle']};
        min-height: 20px;
        border-radius: 5px;
    }}

    QScrollBar::handle:vertical:hover {{
        background: {c['indicator']};
    }}

    QScrollBar:horizontal {{
        border: none;
        background: {c['background_light']};
        height: 10px;
        margin: 0px;
    }}

    QScrollBar::handle:horizontal {{
        background: {c['scroll_handle']};
        min-width: 20px;
        border-radius: 5px;
    }}

    QScrollBar::handle:horizontal:hover {{
        background: {c['indicator']};
    }}

    /* Tree widgets */
    QTreeWidget {{
        border: 1px solid {c['border_light']};
        border-radius: 6px;
        background-color: {c['background']};
        font-size: {fs}pt;
    }}

    QTreeWidget::item {{
        padding: 4px;
        border-bottom: 1px solid {c['border_light']};
    }}

    QTreeWidget::item:selected {{
        background-color: {c['primary']};
        color: {c['on_primary']};
    }}

    QTreeWidget::item:hover:!selected {{
        background-color: {c['hover_tint']};
    }}
    """ + _component_rules(c, fs)


def _component_rules(c: dict, fs: int) -> str:
    """Rules for widgets marked with ``set_tone``/``set_box``/``set_variant``/
    ``set_style_prop``. Part of the main stylesheet, so they switch with the
    theme. Attribute selectors outrank the plain type rules above; ``:hover``
    and ``:disabled`` variants come after their base rule. - Claude Generated"""
    tones = {
        "muted": c["text_muted"],
        "light": c["text_light"],
        "heading": c["text_light"],
        "accent": c["accent_text"],
        "info": c["secondary"],
        "violet": c["violet"],
        "success": c["success"],
        "warning": c["warning"],
        "error": c["error"],
        "divider": c["border"],
    }
    rules = [f'*[tone="{k}"] {{ color: {v}; }}' for k, v in tones.items()]
    rules.append('*[tone="heading"] { font-weight: bold; }')
    for kind in ("success", "warning", "error", "info"):
        fg = c["secondary"] if kind == "info" else c[kind]
        rules.append(
            f'*[box="{kind}"] {{ background-color: {c[kind + "_bg"]}; color: {fg}; '
            f"border: 1px solid {fg}; border-radius: 4px; }}"
        )

    # Status-bar chips (provider, cache, pipeline)
    for kind in ("success", "warning", "error", "info"):
        fg = c["secondary"] if kind == "info" else c[kind]
        rules.append(
            f'QLabel[chip="{kind}"] {{ color: {fg}; background-color: {c[kind + "_bg"]}; '
            f"font-weight: bold; padding: 2px 8px; border-radius: 3px; }}"
        )

    def fill(variant, bg, bg_hover, fg, border="none", padding="10px 20px"):
        sel = f'QPushButton[variant="{variant}"]'
        return (
            f"{sel} {{ background-color: {bg}; color: {fg}; border: {border}; "
            f"border-radius: 6px; padding: {padding}; font-weight: bold; "
            f"font-size: {fs}pt; min-height: 20px; }}\n"
            f"{sel}:hover {{ background-color: {bg_hover}; }}\n"
            f"{sel}:pressed {{ background-color: {bg_hover}; }}\n"
            f"{sel}:disabled {{ background-color: {c['surface_raised']}; "
            f"color: {c['text_muted']}; border: none; }}"
        )

    rules += [
        fill("primary", _gradient(c), _gradient(c, 110), c["on_gradient"]),
        fill("secondary", c["secondary"], c["secondary_hover"], c["on_secondary"]),
        fill("accent", c["accent"], c["accent_hover"], c["on_accent"],
             border=f"1px solid {c['accent_pressed']}", padding="9px 19px"),
        fill("success", c["success"], c["success_hover"], c["on_success"]),
        fill("warning", c["warning"], c["warning_hover"], c["on_warning"]),
        fill("error", c["error"], c["error_hover"], c["on_error"]),
        # Small icon buttons and controls in panel headers
        f'QPushButton[role="icon"] {{ background: transparent; border: 1px solid {c["border"]}; '
        f'border-radius: 3px; color: {c["text_light"]}; font-size: 10px; padding: 2px 6px; }}',
        f'QPushButton[role="icon"]:hover {{ background: {c["hover_tint"]}; border-color: {c["indicator"]}; }}',
        f'QComboBox[role="compact"] {{ color: {c["text_light"]}; font-size: 10px; padding: 2px 6px; '
        f'border: 1px solid {c["border"]}; border-radius: 3px; }}',
        f'QCheckBox[role="compact"] {{ color: {c["text_light"]}; font-size: 10px; }}',
        # Pipeline step frame (state = pending | running | completed | error)
        f'QFrame#pipelineStep {{ border: 2px solid {c["border_light"]}; border-radius: 6px; '
        f'background-color: {c["background"]}; }}',
        f'QFrame#pipelineStep[state="running"] {{ border-color: {c["secondary"]}; background-color: {c["info_bg"]}; }}',
        f'QFrame#pipelineStep[state="completed"] {{ border-color: {c["success"]}; }}',
        f'QFrame#pipelineStep[state="error"] {{ border-color: {c["error"]}; background-color: {c["error_bg"]}; }}',
        # Drop / capture zones in the input step
        f'QFrame#dropZone {{ border: 2px dashed {c["border"]}; border-radius: 8px; '
        f'background-color: {c["background_light"]}; padding: 8px 10px; }}',
        f'QFrame#dropZone:hover {{ border-color: {c["indicator"]}; background-color: {c["hover_tint"]}; }}',
        f'QFrame#dropZone[state="drag"] {{ border: 2px solid {c["success"]}; background-color: {c["success_bg"]}; }}',
        # Same look as the drop zone: neither competes for attention, the hover
        # shows which one is about to act.
        f'QFrame#webcamZone {{ border: 2px dashed {c["border"]}; border-radius: 8px; '
        f'background-color: {c["background_light"]}; padding: 8px 10px; }}',
        f'QFrame#webcamZone:hover {{ border-color: {c["indicator"]}; background-color: {c["hover_tint"]}; }}',
        f'QFrame#optionsStrip {{ background-color: {c["background_dark"]}; border-radius: 6px; padding: 4px; }}',
        # Panel strips (toolbars, chat header/input, status strips)
        f'QFrame#panelStrip {{ background-color: {c["background_light"]}; border: none; '
        f'border-bottom: 1px solid {c["border_light"]}; }}',
        f'QFrame#panelStripTop {{ background-color: {c["background_light"]}; border: none; '
        f'border-top: 1px solid {c["border_light"]}; }}',
        f'QLabel#statusStrip {{ background-color: {c["background_light"]}; color: {c["text_light"]}; '
        f'font-size: 8pt; padding: 2px 8px; border-bottom: 1px solid {c["border_light"]}; }}',
        f'QLabel#typingIndicator {{ color: {c["accent_text"]}; font-size: 9pt; font-style: italic; '
        f'padding: 2px 12px; background-color: {c["background"]}; border-top: 1px solid {c["border_light"]}; }}',
        f'QSplitter#chatSplitter::handle {{ background-color: {c["border_light"]}; }}',
        f'QSplitter#chatSplitter::handle:hover {{ background-color: {c["indicator"]}; }}',
        # Chat input with the floating send button
        f'QTextEdit#chatInput {{ background-color: {c["background"]}; color: {c["text"]}; '
        f'border: 1px solid {c["border"]}; border-radius: 10px; padding: 8px 52px 8px 12px; font-size: 11pt; }}',
        f'QTextEdit#chatInput:focus {{ border: 1px solid {c["focus"]}; }}',
        f'QPushButton#chatSend {{ background-color: {_gradient(c)}; color: {c["on_gradient"]}; border: none; '
        f'border-radius: 18px; font-size: 15pt; font-weight: bold; padding: 0px 0px 2px 0px; min-height: 0px; }}',
        f'QPushButton#chatSend:hover {{ background-color: {_gradient(c, 110)}; }}',
        f'QPushButton#chatSend:disabled {{ background-color: {c["background_dark"]}; color: {c["text_muted"]}; }}',
    ]
    return "\n    ".join(rules) + "\n"


def _filled_button(c: dict, fs: int, bg: str, bg_hover: str, fg: str, border: str = "none") -> str:
    """QSS for a solid-fill button; a border is taken out of the padding so
    every variant keeps the same outer size. - Claude Generated"""
    padding = "10px 20px" if border == "none" else "9px 19px"
    return f"""
            QPushButton {{
                background-color: {bg};
                color: {fg};
                border: {border};
                border-radius: 6px;
                padding: {padding};
                font-weight: bold;
                font-size: {fs}pt;
                min-height: 20px;
            }}
            QPushButton:hover {{
                background-color: {bg_hover};
            }}
            QPushButton:pressed {{
                background-color: {bg_hover};
            }}
            QPushButton:disabled {{
                background-color: {c['surface_raised']};
                color: {c['text_muted']};
                border: none;
            }}
        """


def get_button_styles():
    """Get button style variations. Uses global _font_size. — Claude Generated

    primary   — the logo gradient (the one main action of a view)
    secondary — the cool blue/violet counter-accent (logo trunk)
    accent    — apricot fill
    success / warning / error — semantic fills with their own text colour

    Every variant is a self-contained fill + text pair and never relies on
    the ground around it: tabs apply these strings once at construction and
    a later theme switch only refreshes the main stylesheet, so an outline
    style computed for one theme would end up on the other theme's ground.
    """
    fs = _font_size
    c = get_colors()
    return {
        "primary": f"""
            QPushButton {{
                background-color: {_gradient(c)};
                color: {c['on_gradient']};
                border: none;
                border-radius: 6px;
                padding: 10px 20px;
                font-weight: bold;
                font-size: {fs}pt;
                min-height: 20px;
            }}
            QPushButton:hover {{
                background-color: {_gradient(c, 110)};
            }}
            QPushButton:pressed {{
                background-color: {_gradient(c, 118)};
            }}
            QPushButton:disabled {{
                background-color: {c['surface_raised']};
                color: {c['text_muted']};
            }}
        """,
        "secondary": _filled_button(
            c, fs, c["secondary"], c["secondary_hover"], c["on_secondary"]
        ),
        "accent": _filled_button(
            c, fs, c["accent"], c["accent_hover"], c["on_accent"],
            border=f"1px solid {c['accent_pressed']}",
        ),
        "success": _filled_button(c, fs, c["success"], c["success_hover"], c["on_success"]),
        "warning": _filled_button(c, fs, c["warning"], c["warning_hover"], c["on_warning"]),
        "error": _filled_button(c, fs, c["error"], c["error_hover"], c["on_error"]),
    }


def get_status_label_styles():
    """Get status label styles"""
    return {
        "default": f"color: {get_colors()['text']};",
        "success": f"color: {get_colors()['success']}; font-weight: bold;",
        "warning": f"color: {get_colors()['warning']}; font-weight: bold;",
        "error": f"color: {get_colors()['error']}; font-weight: bold;",
        # Cool accent: readable as text on both grounds (primary is a fill colour)
        "info": f"color: {get_colors()['secondary']}; font-weight: bold;",
        "muted": f"color: {get_colors()['text_muted']};",
    }


def get_confidence_style(count: int) -> tuple:
    """Get color styling based on confidence/frequency count - Claude Generated

    Args:
        count: Number of catalog hits or unique titles

    Returns:
        Tuple of (text_color, bg_color, label, bar)
        where bar is emoji confidence indicator
    """
    if count > 50:
        return (
            get_colors()["confidence_very_high_text"],
            get_colors()["confidence_very_high_bg"],
            "Very High",
            "\U0001f7e9" * 5,
        )
    elif count > 20:
        return (
            get_colors()["confidence_high_text"],
            get_colors()["confidence_high_bg"],
            "High",
            "\U0001f7e9" * 3,
        )
    elif count > 5:
        return (
            get_colors()["confidence_medium_text"],
            get_colors()["confidence_medium_bg"],
            "Medium",
            "\U0001f7e9" * 2,
        )
    else:
        return (
            get_colors()["confidence_low_text"],
            get_colors()["confidence_low_bg"],
            "Low",
            "\U0001f7e9",
        )


def get_image_preview_style():
    """Get image preview container style"""
    return f"""
        QLabel {{
            border: 2px dashed {get_colors()['border']};
            border-radius: 8px;
            background-color: {get_colors()['background_light']};
            color: {get_colors()['text_muted']};
            font-size: {_font_size + 2}pt;
            text-align: center;
            padding: 20px;
        }}
    """
