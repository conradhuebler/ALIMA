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
    """Token table of the active colour scheme and brightness. - Claude Generated"""
    return scheme_colors(_color_scheme, is_dark_mode())


def is_dark_mode() -> bool:
    """Effective brightness: the user's choice (``set_dark_mode``), except in
    the Desktop scheme, where the system decides. - Claude Generated"""
    if _color_scheme == "desktop":
        return _desktop_dark
    return _dark_mode


def dark_mode_preference() -> bool:
    """The user's own light/dark choice, independent of the scheme. - Claude Generated"""
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

# ---------------------------------------------------------------------------
# Colour schemes. "alima" (above, the default), "neutral" (below) and
# "desktop" (derived from the operating system's palette at start-up). Every
# table has exactly the keys of COLORS (tests/test_design_tokens.py), so any
# widget code works under every scheme. - Claude Generated
# ---------------------------------------------------------------------------

COLOR_SCHEMES = ("alima", "neutral", "desktop")
DEFAULT_COLOR_SCHEME = "alima"

#: Menu / settings labels and tooltips (German UI chrome).
COLOR_SCHEME_LABELS = {
    "alima": "ALIMA",
    "neutral": "Neutral",
    "desktop": "Desktop",
}
COLOR_SCHEME_TOOLTIPS = {
    "alima": "Farben aus dem ALIMA-Logo, hell oder dunkel.",
    "neutral": "Zurückhaltende Grautöne mit einem ruhigen blauen Akzent, ohne Verlauf.",
    "desktop": "Farben und Hell/Dunkel aus den Systemeinstellungen des Rechners.",
}

# Neutral: blue-grey greys, one quiet blue accent, flat fills (the gradient
# stops are all the same colour). Semantic colours keep distinct hues plus
# their symbols. - Claude Generated
NEUTRAL_COLORS = dict(COLORS)
NEUTRAL_COLORS.update({
    "primary": "#3a5f8f", "primary_hover": "#2f4f78", "primary_pressed": "#26405f",
    "secondary": "#4f5d6e", "secondary_hover": "#424f5e", "secondary_pressed": "#36414e",
    "accent": "#dde3ea", "accent_hover": "#cfd7e1", "accent_pressed": "#b8c2ce",
    "background": "#fbfcfd", "background_light": "#f3f5f7", "background_dark": "#e3e7ec",
    "text": "#1c2330", "text_light": "#465161", "text_muted": "#596473",
    "error": "#a82a2a", "error_hover": "#8f2222",
    "success": "#256b45", "success_hover": "#1d5838",
    "warning": "#7a5600", "warning_hover": "#654700",
    "border": "#7f8995", "border_light": "#dce1e7", "border_dark": "#596473",
    "shadow": "rgba(20, 30, 45, 0.10)",
    "on_primary": "#ffffff", "on_secondary": "#ffffff", "on_accent": "#1c2330",
    "on_success": "#ffffff", "on_warning": "#ffffff", "on_error": "#ffffff",
    "on_gradient": "#ffffff",
    "focus": "#3a5f8f", "indicator": "#3a5f8f",
    "surface_raised": "#edf0f3", "hover_tint": "#e6ebf1",
    "secondary_tint": "#e4eaf2", "scroll_handle": "#b6bec8",
    "accent_text": "#34557f", "violet": "#4a5487",
    "success_bg": "#e7f1eb", "warning_bg": "#f5efe1", "error_bg": "#f7e7e7", "info_bg": "#e8edf4",
    "grad_start": "#3a5f8f", "grad_mid": "#3a5f8f", "grad_end": "#3a5f8f",
    "progress_start": "#b9cbe2", "progress_mid": "#b9cbe2", "progress_end": "#b9cbe2",
})

NEUTRAL_DARK_COLORS = dict(DARK_COLORS)
NEUTRAL_DARK_COLORS.update({
    "primary": "#3c6aa6", "primary_hover": "#4776b3", "primary_pressed": "#345d92",
    "secondary": "#93abc8", "secondary_hover": "#a8bdd6", "secondary_pressed": "#7f99ba",
    "accent": "#c8d1dc", "accent_hover": "#d7dee7", "accent_pressed": "#b6c1ce",
    "background": "#14171c", "background_light": "#1b1f26", "background_dark": "#0e1014",
    "text": "#e7eaef", "text_light": "#bcc4ce", "text_muted": "#9ca5b0",
    "error": "#f28b8b", "error_hover": "#f5a3a3",
    "success": "#7cc89d", "success_hover": "#95d4b0",
    "warning": "#e3bd68", "warning_hover": "#ead08f",
    "border": "#6c7582", "border_light": "#2b313a", "border_dark": "#8b949f",
    "shadow": "rgba(0, 0, 0, 0.45)",
    "on_primary": "#ffffff", "on_secondary": "#10141a", "on_accent": "#10141a",
    "on_success": "#10141a", "on_warning": "#10141a", "on_error": "#10141a",
    "on_gradient": "#10141a",
    "focus": "#93b9e8", "indicator": "#93b9e8",
    "surface_raised": "#232830", "hover_tint": "#262c35",
    "secondary_tint": "#1f2836", "scroll_handle": "#4a525e",
    "accent_text": "#a3c2e8", "violet": "#aab3de",
    "success_bg": "#16251d", "warning_bg": "#2a2416", "error_bg": "#2c1a1b", "info_bg": "#1d2533",
    "grad_start": "#93b9e8", "grad_mid": "#93b9e8", "grad_end": "#93b9e8",
    "progress_start": "#33557f", "progress_mid": "#33557f", "progress_end": "#33557f",
})

_color_scheme = DEFAULT_COLOR_SCHEME
_desktop_tables: dict = {}   # {"light": {...}, "dark": {...}} once a snapshot exists
_desktop_dark = False        # the system's own light/dark choice


def normalize_color_scheme(name) -> str:
    """Known scheme name or the default. - Claude Generated"""
    name = str(name or "").strip().lower()
    return name if name in COLOR_SCHEMES else DEFAULT_COLOR_SCHEME


def set_color_scheme(name) -> str:
    """Select the colour scheme; unknown names fall back to "alima".
    Returns the scheme now active. Call get_main_stylesheet() afterwards.
    - Claude Generated"""
    global _color_scheme
    _color_scheme = normalize_color_scheme(name)
    return _color_scheme


def get_color_scheme() -> str:
    """Active colour scheme ("alima" | "neutral" | "desktop"). - Claude Generated"""
    return _color_scheme


def scheme_colors(scheme: str, dark: bool) -> dict:
    """Token table of one scheme and brightness. Desktop without a palette
    snapshot (tests, headless tools) falls back to Neutral. - Claude Generated"""
    scheme = normalize_color_scheme(scheme)
    if scheme == "neutral":
        return NEUTRAL_DARK_COLORS if dark else NEUTRAL_COLORS
    if scheme == "desktop":
        table = _desktop_tables.get("dark" if dark else "light")
        if table:
            return table
        return NEUTRAL_DARK_COLORS if dark else NEUTRAL_COLORS
    return DARK_COLORS if dark else COLORS


def all_scheme_values(key: str) -> set:
    """Every value a token can take across schemes and brightness (used to
    recognise a colour set earlier under another theme). - Claude Generated"""
    tables = [COLORS, DARK_COLORS, NEUTRAL_COLORS, NEUTRAL_DARK_COLORS, *_desktop_tables.values()]
    return {t[key].lower() for t in tables if key in t}


# --- colour arithmetic (pure Python; usable without a QApplication) --------

def _rgb(hex_color: str) -> tuple:
    h = hex_color.strip().lstrip("#")
    if len(h) == 3:
        h = "".join(ch * 2 for ch in h)
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _hex(rgb) -> str:
    return "#%02x%02x%02x" % tuple(max(0, min(255, round(v))) for v in rgb)


def _mix(a: str, b: str, t: float) -> str:
    """``t`` of the way from colour ``a`` to colour ``b``."""
    ra, rb = _rgb(a), _rgb(b)
    return _hex(tuple(ra[i] + (rb[i] - ra[i]) * t for i in range(3)))


def luminance(hex_color: str) -> float:
    """WCAG relative luminance. - Claude Generated"""
    def lin(v):
        v /= 255
        return v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4

    r, g, b = _rgb(hex_color)
    return 0.2126 * lin(r) + 0.7152 * lin(g) + 0.0722 * lin(b)


def contrast_ratio(a: str, b: str) -> float:
    """WCAG contrast ratio of two opaque colours. - Claude Generated"""
    la, lb = sorted((luminance(a), luminance(b)), reverse=True)
    return (la + 0.05) / (lb + 0.05)


def best_text_on(bg: str) -> str:
    """Black or white, whichever reads better on ``bg`` (always >= 4.58:1)."""
    return "#000000" if contrast_ratio("#000000", bg) >= contrast_ratio("#ffffff", bg) else "#ffffff"


def _push_apart(color: str, against, minimum: float) -> str:
    """Move ``color`` towards black or white, whichever separates it from all
    colours in ``against``, until every pair reaches ``minimum``. Keeps the
    hue as long as possible; ends at pure black/white if nothing else works.
    - Claude Generated"""
    against = [against] if isinstance(against, str) else list(against)

    def worst(c):
        return min(contrast_ratio(c, x) for x in against)

    if worst(color) >= minimum:
        return color
    best = None
    for pole in ("#000000", "#ffffff"):
        for step in range(1, 21):
            cand = _mix(color, pole, step / 20)
            if worst(cand) >= minimum:
                if best is None or step / 20 < best[0]:
                    best = (step / 20, cand)
                break
    if best:
        return best[1]
    return max(("#000000", "#ffffff"), key=worst)


def _tinted(fg: str, ground: str, alpha: float) -> str:
    """``fg`` laid over ``ground`` at ``alpha`` (badge chips in the log)."""
    return _mix(ground, fg, alpha)


def _fit_badge(color: str, card: str, page: str, alpha: float) -> str:
    """Raise the contrast of a badge colour to its grounds step by step until
    the chip (the colour on a tint of itself over the card) reads at 4.5:1.
    - Claude Generated"""
    if contrast_ratio(color, _tinted(color, card, alpha)) >= 4.5:
        return color
    for target in (5.0, 5.5, 6.0, 6.5, 7.0, 8.0, 9.0, 10.0, 12.0, 15.0, 21.0):
        cand = _push_apart(color, [card, page], target)
        if contrast_ratio(cand, _tinted(cand, card, alpha)) >= 4.5:
            return cand
    return best_text_on(card)


#: Pairs every Desktop table must satisfy, in repair order:
#: (foreground key, background key, minimum ratio, side to adjust).
#: Covers the Qt pairs measured in tests/test_design_tokens.py and the log
#: pairs of ``log_css_vars``; the badge pairs leave room for the tinted chip
#: ground. - Claude Generated
DESKTOP_CONTRAST_RULES = (
    ("text", "background", 4.5, "fg"),
    ("text", "background_light", 4.5, "fg"),
    ("text", "surface_raised", 4.5, "bg"),
    ("text", "hover_tint", 4.5, "bg"),
    ("text", "secondary_tint", 4.5, "bg"),
    ("text", "background_dark", 4.5, "bg"),
    ("text", "success_bg", 4.5, "bg"),
    ("text", "warning_bg", 4.5, "bg"),
    ("text", "error_bg", 4.5, "bg"),
    ("text", "info_bg", 4.5, "bg"),
    ("text", "progress_start", 4.5, "bg"),
    ("text", "progress_mid", 4.5, "bg"),
    ("text", "progress_end", 4.5, "bg"),
    ("text_light", "background_light", 4.5, "fg"),
    ("text_light", "background", 4.5, "fg"),
    ("text_muted", "background_light", 4.5, "fg"),
    ("text_muted", "background", 4.5, "fg"),
    ("text_muted", "warning_bg", 4.5, "fg"),
    ("text_muted", "info_bg", 4.5, "fg"),
    ("secondary", "background", 4.5, "fg"),
    ("secondary", "background_light", 4.5, "fg"),
    ("secondary", "info_bg", 4.5, "fg"),
    ("accent_text", "background", 4.5, "fg"),
    ("accent_text", "background_light", 4.5, "fg"),
    ("accent_text", "background_dark", 4.5, "fg"),
    ("violet", "background", 4.5, "fg"),
    ("violet", "background_light", 4.5, "fg"),
    ("success", "background", 4.5, "fg"),
    ("success", "background_light", 4.5, "fg"),
    ("success", "success_bg", 4.5, "fg"),
    ("warning", "background", 4.5, "fg"),
    ("warning", "background_light", 4.5, "fg"),
    ("warning", "warning_bg", 4.5, "fg"),
    ("error", "background", 4.5, "fg"),
    ("error", "background_light", 4.5, "fg"),
    ("error", "error_bg", 4.5, "fg"),
    ("border", "background", 3.0, "fg"),
    ("border", "background_light", 3.0, "fg"),
    ("focus", "background", 3.0, "fg"),
    ("focus", "background_light", 3.0, "fg"),
    ("indicator", "background", 3.0, "fg"),
    ("primary", "background", 3.0, "fg"),
)

#: Badge chips in the log: colour on a tint of itself over the card ground.
DESKTOP_BADGE_RULES = (
    ("accent_text", 0.16), ("warning", 0.16), ("violet", 0.16),
    ("success", 0.14), ("error", 0.14),
)

#: Text on solid fills: the "on" colour is chosen per fill, black or white
#: when the system's own text colour does not reach 4.5:1.
_FILL_PAIRS = (
    ("on_primary", ("primary", "primary_hover", "primary_pressed")),
    ("on_secondary", ("secondary", "secondary_hover")),
    ("on_accent", ("accent", "accent_hover")),
    ("on_success", ("success", "success_hover")),
    ("on_warning", ("warning",)),
    ("on_error", ("error", "error_hover")),
    ("on_gradient", ("grad_start", "grad_mid", "grad_end")),
)


def _away(color: str, from_color: str, t: float) -> str:
    """Shade ``color`` by ``t`` away from ``from_color`` (hover states keep or
    raise the contrast to their text)."""
    pole = "#000000" if luminance(from_color) > luminance(color) else "#ffffff"
    if luminance(from_color) == luminance(color):
        pole = "#000000"
    return _mix(color, pole, t)


def derive_desktop_colors(roles: dict, dark: bool) -> dict:
    """Token table from an operating-system palette.

    ``roles`` maps lower-case QPalette role names (window, window_text, base,
    alternate_base, text, button, button_text, highlight, highlighted_text,
    link, mid, placeholder_text) to hex colours. Grounds, text and the accent
    come from the system; success/warning/error keep the Neutral hues. Every
    pair in ``DESKTOP_CONTRAST_RULES``, every fill/"on" pair and every badge
    is then checked, and a colour that falls short is moved towards black or
    white until it passes: a readable pair wins over the system's exact
    shade. - Claude Generated
    """
    ref = NEUTRAL_DARK_COLORS if dark else NEUTRAL_COLORS
    g = {k: str(v).lower() for k, v in roles.items() if v}
    window = g.get("window", ref["background_light"])
    base = g.get("base", window)
    wtext = g.get("window_text", g.get("text", ref["text"]))
    text = g.get("text", wtext)
    button = g.get("button", window)
    highlight = g.get("highlight", ref["primary"])
    hl_text = g.get("highlighted_text", best_text_on(highlight))
    link = g.get("link", highlight)
    mid = g.get("mid", _mix(window, wtext, 0.45))
    placeholder = g.get("placeholder_text", _mix(wtext, window, 0.4))

    c = dict(ref)
    c.update({
        "background": base,
        "background_light": window,
        "background_dark": _mix(window, wtext, 0.08),
        "surface_raised": button,
        "hover_tint": _mix(button, highlight, 0.15),
        "text": text,
        "text_light": _mix(wtext, window, 0.25),
        "text_muted": placeholder,
        "border": mid,
        "border_light": _mix(window, wtext, 0.15),
        "primary": highlight,
        "secondary": link,
        "accent": _mix(highlight, base, 0.75),
        "focus": highlight,
        "indicator": highlight,
        "secondary_tint": _mix(base, link, 0.12),
        "scroll_handle": mid,
        "accent_text": highlight,
        "violet": link,
        "success_bg": _mix(base, ref["success"], 0.12),
        "warning_bg": _mix(base, ref["warning"], 0.12),
        "error_bg": _mix(base, ref["error"], 0.12),
        "info_bg": _mix(base, link, 0.12),
        "progress_start": _mix(highlight, base, 0.5),
        "shadow": "rgba(0, 0, 0, 0.45)" if dark else "rgba(0, 0, 0, 0.12)",
    })
    c["progress_mid"] = c["progress_end"] = c["progress_start"]

    for _ in range(4):
        for fg, bg, minimum, side in DESKTOP_CONTRAST_RULES:
            if contrast_ratio(c[fg], c[bg]) >= minimum:
                continue
            if side == "fg":
                c[fg] = _push_apart(c[fg], c[bg], minimum)
            else:
                c[bg] = _push_apart(c[bg], c[fg], minimum)
        for key, alpha in DESKTOP_BADGE_RULES:
            c[key] = _fit_badge(c[key], c["background_light"], c["background"], alpha)
            if contrast_ratio(c[key], _tinted(c[key], c["background_light"], alpha)) < 4.5:
                # No colour reads on this card ground (mid grey): move the
                # ground itself away from the badge colour.
                pole = "#ffffff" if luminance(c[key]) < 0.18 else "#000000"
                for step in range(1, 21):
                    cand = _mix(c["background_light"], pole, step / 20)
                    if contrast_ratio(c[key], _tinted(c[key], cand, alpha)) >= 4.5:
                        c["background_light"] = cand
                        break
        c["progress_mid"] = c["progress_end"] = c["progress_start"]

    # Hover / pressed shades of the fills, then the text on each fill.
    c["grad_start"] = c["grad_mid"] = c["grad_end"] = c["primary"]
    system_on = {"on_primary": hl_text, "on_gradient": hl_text}
    for on_key, fills in _FILL_PAIRS:
        base_fill = c[fills[0]]
        on = system_on.get(on_key) or best_text_on(base_fill)
        if contrast_ratio(on, base_fill) < 4.5:
            on = best_text_on(base_fill)
        c[on_key] = on
    c["primary_hover"] = _away(c["primary"], c["on_primary"], 0.12)
    c["primary_pressed"] = _away(c["primary"], c["on_primary"], 0.22)
    c["secondary_hover"] = _away(c["secondary"], c["on_secondary"], 0.12)
    c["secondary_pressed"] = _away(c["secondary"], c["on_secondary"], 0.22)
    c["accent_hover"] = _away(c["accent"], c["on_accent"], 0.08)
    c["accent_pressed"] = _away(c["accent"], c["on_accent"], 0.16)
    for k in ("success", "warning", "error"):
        c[k + "_hover"] = _away(c[k], c["on_" + k], 0.12)
    # The "on" colour must hold for the shaded fills as well.
    for on_key, fills in _FILL_PAIRS:
        for f in fills:
            if contrast_ratio(c[on_key], c[f]) < 4.5:
                c[f] = _push_apart(c[f], c[on_key], 4.5)
    c["border_dark"] = c["text_muted"]
    return c


def set_desktop_palette(roles: dict, dark: bool) -> None:
    """Store the system palette (taken before ALIMA changes anything) and
    derive the Desktop tables from it. ``dark`` is the system's own choice;
    the other brightness is derived from the same roles for completeness.
    - Claude Generated"""
    global _desktop_dark
    _desktop_dark = bool(dark)
    table = derive_desktop_colors(roles, dark)
    _desktop_tables.clear()
    _desktop_tables["dark" if dark else "light"] = table
    _desktop_tables["light" if dark else "dark"] = table


def desktop_is_dark() -> bool:
    """The system's light/dark choice from the snapshot. - Claude Generated"""
    return _desktop_dark


def has_desktop_palette() -> bool:
    """True once ``set_desktop_palette`` ran. - Claude Generated"""
    return bool(_desktop_tables)


def palette_roles(palette) -> dict:
    """Hex colours of the QPalette roles ``derive_desktop_colors`` reads.
    - Claude Generated"""
    from PyQt6.QtGui import QPalette

    r = QPalette.ColorRole
    names = {
        "window": r.Window, "window_text": r.WindowText, "base": r.Base,
        "alternate_base": r.AlternateBase, "text": r.Text, "button": r.Button,
        "button_text": r.ButtonText, "highlight": r.Highlight,
        "highlighted_text": r.HighlightedText, "link": r.Link, "mid": r.Mid,
        "placeholder_text": r.PlaceholderText,
    }
    return {k: palette.color(QPalette.ColorGroup.Active, role).name() for k, role in names.items()}


def log_css_vars(c: dict) -> dict:
    """The shared log's ``--alima-*`` variables from a Qt token table (the
    Desktop scheme sets them at run time; ALIMA and Neutral have them in
    alima_render.css). - Claude Generated"""
    flat = f"linear-gradient(180deg, {c['primary']} 0%, {c['primary']} 100%)"
    return {
        "--alima-bg": c["background"],
        "--alima-fg": c["text"],
        "--alima-body": c["text"],
        "--alima-muted": c["text_muted"],
        "--alima-muted-strong": c["text_muted"],
        "--alima-meta": c["text_muted"],
        "--alima-ts": c["violet"],
        "--alima-accent": c["accent_text"],
        "--alima-purple": c["violet"],
        "--alima-green": c["success"],
        "--alima-yellow": c["warning"],
        "--alima-orange": c["warning"],
        "--alima-error": c["error"],
        "--alima-error-soft": c["error"],
        "--alima-link": c["secondary"],
        "--alima-link-ext": c["warning"],
        "--alima-surface": c["background_light"],
        "--alima-surface-deep": c["background_dark"],
        "--alima-surface-alt": c["background_light"],
        "--alima-border": c["border_light"],
        "--alima-border-soft": c["border_light"],
        "--alima-user-bubble": c["info_bg"],
        "--alima-system": c["success"],
        "--alima-stream-live": c["text_light"],
        "--alima-info": c["secondary"],
        "--alima-info-bg": c["info_bg"],
        "--alima-info-border": c["secondary"],
        "--alima-grad": flat,
        "--alima-mark-opacity": "0.4",
        "--alima-mark-filter": "grayscale(1)",
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
        # Light text on the fill: darken on hover; dark text: lighten.
        light_text = luminance(c["on_gradient"]) > 0.5
        factor = int(10000 / shift) if light_text else shift
        stops = tuple(_shade(s, factor) for s in stops)
    if stops[0] == stops[1] == stops[2]:
        return stops[0]  # flat fill (Neutral, Desktop)
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
    if _color_scheme == "alima":
        combo_arrow = _asset_url("ui-chevron.svg")  # #8c7c88, >= 3:1 on both grounds
    else:
        # black or white, whichever separates from the field ground
        dark_arrow = contrast_ratio("#000000", c["background"]) >= contrast_ratio("#ffffff", c["background"])
        combo_arrow = _asset_url("ui-chevron-black.svg" if dark_arrow else "ui-chevron-white.svg")
    # check mark on the primary fill: white unless black reads better
    check_mark = _asset_url(
        "ui-check-dark.svg" if best_text_on(c["primary"]) == "#000000" else "ui-check.svg"
    )
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
