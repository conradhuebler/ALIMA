"""Design tokens: both themes define the same tokens, and the token pairs the
UI actually draws keep their WCAG contrast. - Claude Generated

The webapp (styles.css), the shared log chrome (alima_render.css, also inlined
into the Qt log) and the Qt stylesheet (src/ui/styles.py) carry the same
logo-derived palette. Values are parsed from the real files, so a token edit
that breaks a pair fails here instead of in front of a reader.

Thresholds: text 4.5:1, control borders / focus indicators 3:1 (WCAG 2.x
1.4.3 and 1.4.11). Translucent grounds are composited over the surface they
sit on before measuring. Token table and measured values:
docs/design_refresh.md.
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STYLES_CSS = ROOT / "src/webapp/static/styles.css"
RENDER_CSS = ROOT / "src/webapp/static/alima_render.css"


# ── colour helpers ───────────────────────────────────────────────────────────

def _rgba(value: str):
    """'#rrggbb' | 'rgba(r, g, b, a)' → (r, g, b, a) with r,g,b in 0..255."""
    value = value.strip()
    m = re.fullmatch(r"#([0-9a-fA-F]{6})", value)
    if m:
        h = m.group(1)
        return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16), 1.0
    m = re.fullmatch(r"rgba\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*([\d.]+)\s*\)", value)
    if m:
        return int(m.group(1)), int(m.group(2)), int(m.group(3)), float(m.group(4))
    raise ValueError(f"not a plain colour: {value!r}")


def _over(fg, bg):
    """Composite fg (r,g,b,a) over an opaque bg (r,g,b,1)."""
    a = fg[3]
    return tuple(round(fg[i] * a + bg[i] * (1 - a)) for i in range(3)) + (1.0,)


def _luminance(rgb) -> float:
    def lin(c):
        c /= 255
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

    r, g, b = rgb[:3]
    return 0.2126 * lin(r) + 0.7152 * lin(g) + 0.0722 * lin(b)


def contrast(fg: str, bg: str, base: str | None = None, alpha: float | None = None) -> float:
    """WCAG contrast of fg on bg. ``bg`` may be translucent (then composited
    over ``base``); ``alpha`` tints ``fg`` itself over ``base`` to form the
    ground (badge chips: color-mix(fg 16%, transparent) on a surface)."""
    f = _rgba(fg)
    if alpha is not None:
        b = _over(f[:3] + (alpha,), _rgba(base))
    else:
        b = _rgba(bg)
        if b[3] < 1:
            b = _over(b, _rgba(base))
    if f[3] < 1:
        f = _over(f, b)
    la, lb = sorted((_luminance(f), _luminance(b)), reverse=True)
    return (la + 0.05) / (lb + 0.05)


# ── token parsing ────────────────────────────────────────────────────────────

def _block(css: str, selector: str) -> str:
    """Body of the first rule whose selector text is exactly ``selector``."""
    css = re.sub(r"/\*.*?\*/", "", css, flags=re.S)
    pat = r"(?:^|[}\s])" + re.escape(selector) + r"\s*\{"
    m = re.search(pat, css)
    if not m:
        raise AssertionError(f"selector not found: {selector}")
    depth, i = 1, m.end()
    while depth:
        depth += {"{": 1, "}": -1}.get(css[i], 0)
        i += 1
    return css[m.end(): i - 1]


def _tokens(body: str) -> dict:
    body = re.sub(r"/\*.*?\*/", "", body, flags=re.S)
    out = {}
    for name, value in re.findall(r"(--[\w-]+)\s*:\s*([^;]+);", body):
        out[name] = " ".join(value.split())
    return out


def webapp_tokens() -> dict:
    css = STYLES_CSS.read_text(encoding="utf-8")
    render = RENDER_CSS.read_text(encoding="utf-8")
    light = _tokens(_block(css, ":root"))
    dark_media = _tokens(_block(css, ':root:not([data-theme="light"])'))
    dark_manual = _tokens(_block(css, '[data-theme="dark"]'))
    # The log palette lives in alima_render.css (shared with the GUI log).
    log_light_media = _tokens(_block(render, ':root:not([data-theme="dark"]) #log'))
    log_light_manual = _tokens(_block(render, '[data-theme="light"] #log'))
    log_dark = _tokens(_block(render, "#log"))
    return {
        "light": light,
        "dark": dark_manual,
        "dark_media": dark_media,
        "log_dark": log_dark,
        "log_light": log_light_manual,
        "log_light_media": log_light_media,
    }


def _is_colour_token(value: str) -> bool:
    return bool(re.match(r"(#|rgba\(|linear-gradient\(|url\()", value))


# ── the pairs the UI draws ───────────────────────────────────────────────────
# (label, fg token, bg token, threshold[, base token for translucent grounds])
WEB_PAIRS = [
    ("Text / Seite", "--clr-text", "--clr-bg", 4.5),
    ("Text / Fläche", "--clr-text", "--clr-surface", 4.5),
    ("Text / Fläche 2", "--clr-text", "--clr-surface-2", 4.5),
    ("Text / Fläche 3 (Hover)", "--clr-text", "--clr-surface-3", 4.5),
    ("Gedämpft / Fläche", "--clr-muted", "--clr-surface", 4.5),
    ("Gedämpft / Fläche 2", "--clr-muted", "--clr-surface-2", 4.5),
    ("Gedämpft / Seite", "--clr-muted", "--clr-bg", 4.5),
    ("Hinweis+Platzhalter / Fläche", "--clr-subtle", "--clr-surface", 4.5),
    ("Hinweis+Platzhalter / Fläche 2", "--clr-subtle", "--clr-surface-2", 4.5),
    ("Platzhalter / Eingabegrund", "--clr-subtle", "--clr-bg", 4.5),
    ("Primär (Text) / Fläche", "--clr-primary", "--clr-surface", 4.5),
    ("Primär (Text) / Fläche 2", "--clr-primary", "--clr-surface-2", 4.5),
    ("Kühler Akzent / Fläche 2", "--clr-cool", "--clr-surface-2", 4.5),
    ("DK-Fortschritt / Leiste", "--clr-stream-text", "--clr-surface-2", 4.5),
    ("Log-Untertitel / Log-Karte", "--clr-stream-muted", "--clr-stream-bg", 4.5),
    ("Erfolg / Fläche", "--clr-success", "--clr-surface", 4.5),
    ("Warnung / Fläche 2", "--clr-warning", "--clr-surface-2", 4.5),
    ("Fehler / Fläche", "--clr-error", "--clr-surface", 4.5),
    ("Info / Fläche", "--clr-info", "--clr-surface", 4.5),
    ("Knopftext / Fehler-Füllung", "--clr-on-solid", "--clr-error", 4.5),
    ("Knopftext / Erfolg-Füllung", "--clr-on-solid", "--clr-success", 4.5),
    ("Knopftext / Warnung-Füllung", "--clr-on-solid", "--clr-warning", 4.5),
    ("Knopftext / Info-Füllung", "--clr-on-solid", "--clr-info", 4.5),
    ("Banner-Text / Banner", "--clr-reconnect-text", "--clr-reconnect-bg-to", 4.5),
    ("Bedienrand / Fläche", "--clr-border-2", "--clr-surface", 3.0),
    ("Bedienrand / Fläche 2", "--clr-border-2", "--clr-surface-2", 3.0),
    ("Bedienrand / Seite", "--clr-border-2", "--clr-bg", 3.0),
    ("Fokusring / Fläche", "--clr-focus", "--clr-surface", 3.0),
    ("Fokusring / Seite", "--clr-focus", "--clr-bg", 3.0),
]

LOG_PAIRS = [
    ("Log-Text / Log", "--alima-fg", "--alima-bg", 4.5),
    ("Fließtext / Blase", "--alima-body", "--alima-surface", 4.5),
    ("Fließtext / Nutzerblase", "--alima-body", "--alima-user-bubble", 4.5),
    ("Fließtext / Tabellenkopf", "--alima-body", "--alima-surface-deep", 4.5),
    ("Gedämpft / Log", "--alima-muted", "--alima-bg", 4.5),
    ("Gedämpft / Blase", "--alima-muted", "--alima-surface", 4.5),
    ("Tool-Zeile / Log", "--alima-muted-strong", "--alima-bg", 4.5),
    ("Meta / Karte", "--alima-meta", "--alima-surface", 4.5),
    ("Zeitstempel / Log", "--alima-ts", "--alima-bg", 4.5),
    ("Akzent / Log", "--alima-accent", "--alima-bg", 4.5),
    ("Akzent / Tabellenkopf", "--alima-accent", "--alima-surface-deep", 4.5),
    ("Violett / Log", "--alima-purple", "--alima-bg", 4.5),
    ("Violett / Karte", "--alima-purple", "--alima-surface", 4.5),
    ("Grün / Log", "--alima-green", "--alima-bg", 4.5),
    ("Gelb / Log", "--alima-yellow", "--alima-bg", 4.5),
    ("Orange / Karte", "--alima-orange", "--alima-surface", 4.5),
    ("Fehler / Log", "--alima-error", "--alima-bg", 4.5),
    ("Fehler-Text / Log", "--alima-error-soft", "--alima-bg", 4.5),
    ("Link / Log", "--alima-link", "--alima-bg", 4.5),
    ("Link / Blase", "--alima-link", "--alima-surface", 4.5),
    ("Fremd-Link / Log", "--alima-link-ext", "--alima-bg", 4.5),
    ("Systemzeile / Log", "--alima-system", "--alima-bg", 4.5),
    ("Live-Stream / Log", "--alima-stream-live", "--alima-bg", 4.5),
    ("Info-Zeile / Info-Grund", "--alima-info", "--alima-info-bg", 4.5, "--alima-bg"),
    ("Vorschlag-ID / Karte", "--alima-muted-strong", "--alima-surface", 4.5),
]

# Badge chips: text colour on a 16 % (14 %) tint of itself over the card.
LOG_BADGES = [
    ("Badge DK", "--alima-accent", 0.16),
    ("Badge DDC", "--alima-orange", 0.16),
    ("Badge RVK", "--alima-purple", 0.16),
    ("Badge standard", "--alima-green", 0.14),
    ("Badge nicht standard", "--alima-yellow", 0.14),
    ("Badge Fehler", "--alima-error", 0.14),
]

GRAD_STOP = re.compile(r"#[0-9a-fA-F]{6}")

# Qt: (label, fg key, bg key, threshold)
QT_PAIRS = [
    ("Text / Eingabe", "text", "background", 4.5),
    ("Text / Gruppe", "text", "background_light", 4.5),
    ("Text / neutraler Knopf", "text", "surface_raised", 4.5),
    ("Text / Hover", "text", "hover_tint", 4.5),
    ("Nebentext / Gruppe", "text_light", "background_light", 4.5),
    ("Gedämpft / Gruppe", "text_muted", "background_light", 4.5),
    ("Gedämpft / Eingabe", "text_muted", "background", 4.5),
    ("Weiß / Primär", "on_primary", "primary", 4.5),
    ("Weiß / Primär Hover", "on_primary", "primary_hover", 4.5),
    ("Weiß / Primär gedrückt", "on_primary", "primary_pressed", 4.5),
    ("Statustext Info / Eingabe", "secondary", "background", 4.5),
    ("Statustext Info / Gruppe", "secondary", "background_light", 4.5),
    ("Knopftext / Sekundär", "on_secondary", "secondary", 4.5),
    ("Knopftext / Sekundär Hover", "on_secondary", "secondary_hover", 4.5),
    ("Text / gedrückter Umschalter", "text", "secondary_tint", 4.5),
    ("Knopftext / Akzent", "on_accent", "accent", 4.5),
    ("Knopftext / Akzent Hover", "on_accent", "accent_hover", 4.5),
    ("Knopftext / Erfolg", "on_success", "success", 4.5),
    ("Knopftext / Erfolg Hover", "on_success", "success_hover", 4.5),
    ("Knopftext / Warnung", "on_warning", "warning", 4.5),
    ("Knopftext / Fehler", "on_error", "error", 4.5),
    ("Knopftext / Fehler Hover", "on_error", "error_hover", 4.5),
    ("Verlaufstext / Verlauf Anfang", "on_gradient", "grad_start", 4.5),
    ("Verlaufstext / Verlauf Mitte", "on_gradient", "grad_mid", 4.5),
    ("Verlaufstext / Verlauf Ende", "on_gradient", "grad_end", 4.5),
    ("Fortschrittstext / Balken Anfang", "text", "progress_start", 4.5),
    ("Fortschrittstext / Balken Mitte", "text", "progress_mid", 4.5),
    ("Fortschrittstext / Balken Ende", "text", "progress_end", 4.5),
    ("Fortschrittstext / Balkengrund", "text", "background_dark", 4.5),
    ("Statustext Erfolg / Gruppe", "success", "background_light", 4.5),
    ("Statustext Warnung / Gruppe", "warning", "background_light", 4.5),
    ("Statustext Fehler / Gruppe", "error", "background_light", 4.5),
    ("Bedienrand / Eingabe", "border", "background", 3.0),
    ("Bedienrand / Gruppe", "border", "background_light", 3.0),
    ("Fokusrand / Eingabe", "focus", "background", 3.0),
    ("Fokusrand / Gruppe", "focus", "background_light", 3.0),
    ("Tab-Markierung / Eingabe", "indicator", "background", 3.0),
    ("Checkbox gefüllt / Eingabe", "primary", "background", 3.0),
    # Widget roles (set_tone / set_box / chips / bars), see styles._component_rules
    ("Akzenttext / Eingabe", "accent_text", "background", 4.5),
    ("Akzenttext / Gruppe", "accent_text", "background_light", 4.5),
    ("Violett / Eingabe", "violet", "background", 4.5),
    ("Violett / Gruppe", "violet", "background_light", 4.5),
    ("Erfolg / Erfolgsgrund", "success", "success_bg", 4.5),
    ("Warnung / Warngrund", "warning", "warning_bg", 4.5),
    ("Fehler / Fehlergrund", "error", "error_bg", 4.5),
    ("Info / Infogrund", "secondary", "info_bg", 4.5),
    ("Text / Erfolgsgrund (Tabellenzeile)", "text", "success_bg", 4.5),
    ("Text / Warngrund (Leiste)", "text", "warning_bg", 4.5),
    ("Text / Fehlergrund (Schritt)", "text", "error_bg", 4.5),
    ("Text / Infogrund (Schritt, Zeile)", "text", "info_bg", 4.5),
    ("Gedämpft / Warngrund (Vorschlag)", "text_muted", "warning_bg", 4.5),
    ("Gedämpft / Infogrund (Kamera)", "text_muted", "info_bg", 4.5),
    ("Erfolg / Eingabe (Tabellenzeile)", "success", "background", 4.5),
    ("Fehler / Eingabe", "error", "background", 4.5),
    ("Warnung / Eingabe", "warning", "background", 4.5),
    ("Trennlinie / Eingabe", "border", "background", 3.0),
]


def measure_all() -> list:
    """Every pair → (scope, theme, label, fg, bg, ratio, threshold)."""
    from src.ui.styles import COLORS, DARK_COLORS

    t = webapp_tokens()
    rows = []
    for theme in ("light", "dark"):
        tok = t[theme]
        for label, fg, bg, need, *base in WEB_PAIRS:
            r = contrast(tok[fg], tok[bg], tok[base[0]] if base else None)
            rows.append(("Webapp", theme, label, tok[fg], tok[bg], r, need))
        for stop in GRAD_STOP.findall(tok["--clr-grad"]):
            r = contrast(tok["--clr-on-solid"], stop)
            rows.append(("Webapp", theme, "Knopftext / Verlauf-Stopp", tok["--clr-on-solid"], stop, r, 4.5))
        log = t["log_light" if theme == "light" else "log_dark"]
        for label, fg, bg, need, *base in LOG_PAIRS:
            r = contrast(log[fg], log[bg], log[base[0]] if base else None)
            rows.append(("Log", theme, label, log[fg], log[bg], r, need))
        for label, fg, alpha in LOG_BADGES:
            r = contrast(log[fg], "", base=log["--alima-surface"], alpha=alpha)
            rows.append(("Log", theme, label, log[fg], f"{int(alpha * 100)}% über Karte", r, 4.5))
        colours = COLORS if theme == "light" else DARK_COLORS
        for label, fg, bg, need in QT_PAIRS:
            r = contrast(colours[fg], colours[bg])
            rows.append(("Qt", theme, label, colours[fg], colours[bg], r, need))
    return rows


class TokenParityTest(unittest.TestCase):
    def setUp(self):
        self.t = webapp_tokens()

    def test_both_dark_blocks_are_identical(self):
        self.assertEqual(self.t["dark"], self.t["dark_media"])

    def test_both_light_log_blocks_are_identical(self):
        self.assertEqual(self.t["log_light"], self.t["log_light_media"])

    def test_every_light_colour_token_has_a_dark_value(self):
        colour = {k for k, v in self.t["light"].items() if _is_colour_token(v)}
        self.assertEqual(colour - set(self.t["dark"]), set())
        self.assertEqual(set(self.t["dark"]) - set(self.t["light"]), set())

    def test_log_light_override_covers_every_log_token(self):
        self.assertEqual(set(self.t["log_dark"]), set(self.t["log_light"]))

    def test_qt_palettes_share_keys(self):
        from src.ui.styles import COLORS, DARK_COLORS

        self.assertEqual(set(COLORS), set(DARK_COLORS))


class ContrastTest(unittest.TestCase):
    def test_measured_pairs_meet_thresholds(self):
        failures = [
            f"{scope}/{theme}: {label} {fg} on {bg} = {ratio:.2f} < {need}"
            for scope, theme, label, fg, bg, ratio, need in measure_all()
            if ratio < need
        ]
        self.assertEqual(failures, [])

    def test_contrast_helper_matches_known_values(self):
        self.assertAlmostEqual(contrast("#ffffff", "#000000"), 21.0, places=2)
        self.assertAlmostEqual(contrast("#777777", "#ffffff"), 4.48, places=2)


class OfflineAssetsTest(unittest.TestCase):
    """The page loads no external fonts or scripts; the mark stays small."""

    def test_template_and_css_reference_no_external_hosts(self):
        html = (ROOT / "src/webapp/templates/webapp.html").read_text(encoding="utf-8")
        for text in (html, STYLES_CSS.read_text(encoding="utf-8"), RENDER_CSS.read_text(encoding="utf-8")):
            self.assertNotRegex(text, r"(?:src|href)=[\"']https?://")
            self.assertNotRegex(text, r"@import\s+url\(\s*['\"]?https?://")

    def test_mark_svgs_are_small(self):
        for name in ("alima-mark.svg", "alima-icon.svg"):
            size = (ROOT / "src/webapp/static" / name).stat().st_size
            self.assertLess(size, 4096, name)


if __name__ == "__main__":
    unittest.main()
