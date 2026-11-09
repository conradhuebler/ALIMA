"""Category-grouped plugin settings tab - Claude Generated.

Replaces the checkbox-only ``SearchProviderSelectorWidget`` ("nur ticks"). Each
plugin *instance* gets its own panel: what the type does ("Wirkung"), its state
(An / Nur Cache / Aus, stored as ``enabled`` + ``settings["cache_only"]``), label, an agent
``usage_hint``, and a form auto-generated from the plugin type's declared
``config_fields`` (so finc-URL, Libero-token, DOI contact-email … live *with* the
plugin, not scattered in the config). Instances can be added, duplicated and
removed, and several instances of one type may coexist (e.g. two finc endpoints).

The tab reads/writes ``AlimaConfig.plugins`` directly and is authoritative on
save. The DOI ``SystemConfig`` fields are still a derived mirror, kept exact by
``plugin_migration.derive_input_mirrors`` on save; the ``CatalogConfig`` /
``SearchProviderConfig`` mirrors are gone (WP P7). See ``docs/plugin_system.md``.

The instance list (Nov 2026): a three-state checkbox per row that is the same
state as the form's "Zustand" (stored only in the panel's instance list),
order by drag and drop or Alt+Up/Down (the list order is what ``collect`` and
``apply_to`` write), an origin tag per row derived on load (not stored), and
for search sources the DK position from ``factory.describe_dk_source`` over
the unsaved dialog state. - Claude Generated
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PyQt6.QtCore import QEvent, QModelIndex, Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListView,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QStyle,
    QStyledItemDelegate,
    QStyleOptionViewItem,
    QTabWidget,
    QTextEdit,
    QToolTip,
    QVBoxLayout,
    QWidget,
)

from src.core.plugins import CHOICE, INT, BOOL, SECRET, get_category, list_categories
from src.utils.config_models import PluginInstanceConfig
from .styles import set_tone

logger = logging.getLogger(__name__)

# Human labels for the known categories (fallback = the raw name).
# Capability -> what it means for the operator. Keys are the SearchCapability
# enum values; a type declaring none (input sources, lookups) shows nothing.
# - Claude Generated
_CAPABILITY_LABELS = {
    "gnd_keywords": "GND-Schlagwortsuche",
    "title_records": "Titeldaten",
    "subject_facets": "Sachfacetten",
    "classification": "DK/RVK-Klassifikation",
}

_CATEGORY_LABELS = {
    "search_provider": "🔎 Suchquellen",
    "input_source": "📥 Input-Quellen",
    "lookup": "🔖 Lookups (API)",
}

# Per-type actions rendered as buttons at the bottom of an instance form.
# Key: (category, provider_id) → list of (button_label, callback(panel, inst, parent)).
# Lets a plugin type expose an interactive action (e.g. the webindex crawl)
# without the generic form hardcoding it. - Claude Generated
_TYPE_ACTIONS: Dict[tuple, list] = {}


def register_type_action(category: str, provider_id: str, label: str,
                         callback) -> None:
    """Register a per-instance action button for a plugin type - Claude Generated."""
    _TYPE_ACTIONS.setdefault((category, provider_id), []).append((label, callback))


# Per-type read-only info lines in the instance form: (category, provider_id)
# → list of (row label, fn(inst) -> str); an empty string shows nothing.
# - Claude Generated
_TYPE_INFO: Dict[tuple, list] = {}


def register_type_info(category: str, provider_id: str, label: str, fn) -> None:
    """Register a read-only info line for the form of a plugin type - Claude Generated."""
    _TYPE_INFO.setdefault((category, provider_id), []).append((label, fn))


def geo_sachdb_info(inst: PluginInstanceConfig) -> str:
    """State of the database of an available ``geo_sachdb`` instance, from its
    ``meta`` table (source, date, checksum); empty when not available.
    - Claude Generated"""
    from src.utils.lookups.geo_sachdb.store import GeoSachdbStore

    path = str((inst.settings or {}).get("db_path") or "").strip()
    if not path:
        return ""
    status = GeoSachdbStore(path).status()
    if not status.get("available"):
        return ""
    prov = status.get("provenance") or {}
    parts = []
    for key, name in (("source", "Quelle"), ("source_date", "Stand"),
                      ("source_sha256", "Prüfsumme")):
        if prov.get(key):
            parts.append(f"{name} {prov[key]}")
    return ", ".join(parts)


register_type_info("lookup", "geo_sachdb", "Datenbank", geo_sachdb_info)


# Instance state ("Zustand") shown instead of the bare enable checkbox. Stored as
# before: ``enabled`` plus, for search sources with a cache path, the
# ``cache_only`` setting. The constants and the three helpers below live in the
# Qt-free ``src/core/search/cache_only.py`` (shared with the institution
# presets); imported here under their old names. - Claude Generated
from src.core.search.cache_only import (  # noqa: E402
    OFFER_BLOCKED,
    OFFER_CACHE_ONLY,
    OFFER_NONE,
    STATE_CACHE_ONLY,
    STATE_OFF,
    STATE_ON,
    apply_state,
    cache_only_offer,
    instance_state,
)

_STATE_LABELS = {STATE_ON: "An", STATE_CACHE_ONLY: "Nur Cache", STATE_OFF: "Aus"}
_LIST_SUFFIX = {STATE_ON: "", STATE_CACHE_ONLY: "  (nur Cache)", STATE_OFF: "  (aus)"}

# A setting the instance does not store stays absent as long as its field shows
# what it showed when the form was built (``_CategoryPanel._shown_absent``), so
# looking at a row and saving leaves config.json unchanged. This was a list of
# two keys (timeout_s, dk_enabled); every other field wrote its default on the
# first save of the selected row. ``dk_enabled``: on catalog an absent key has
# its own meaning (DK source only while no other instance has the switch on);
# the field shows the effective state there. - Claude Generated


# The list checkbox shows the state: checked = An, partially checked = Nur
# Cache, unchecked = Aus. A click moves on An → Nur Cache → Aus → An and skips
# "Nur Cache" where ``cache_only_offer`` does not offer it. - Claude Generated
_STATE_CYCLE = (STATE_ON, STATE_CACHE_ONLY, STATE_OFF)
_CHECK_OF_STATE = {
    STATE_ON: Qt.CheckState.Checked,
    STATE_CACHE_ONLY: Qt.CheckState.PartiallyChecked,
    STATE_OFF: Qt.CheckState.Unchecked,
}
_STATE_MEANING = {
    STATE_ON: "An: die Instanz wird benutzt.",
    STATE_CACHE_ONLY: ("Nur Cache: die Instanz wird benutzt, stellt aber keine Anfragen ins "
                       "Netz; Ergebnisse kommen nur aus dem lokalen Cache."),
    STATE_OFF: "Aus: die Instanz wird nicht benutzt.",
}
_STATE_STATUS = {
    STATE_ON: "an",
    STATE_CACHE_ONLY: "nur Cache (keine Abfragen im Netz)",
    STATE_OFF: "aus",
}


def next_state(state: str, offer: str) -> str:
    """The state one click on the list checkbox moves to - Claude Generated"""
    cycle = [s for s in _STATE_CYCLE if s != STATE_CACHE_ONLY or offer == OFFER_CACHE_ONLY]
    pos = cycle.index(state) if state in cycle else -1
    return cycle[(pos + 1) % len(cycle)]


class _InstancesView:
    """The panel's instances as the ``enabled_instances_for`` view that
    ``factory.describe_dk_source`` reads - Claude Generated"""

    def __init__(self, instances: List[PluginInstanceConfig]):
        self._instances = instances

    def enabled_instances_for(self, category: str) -> List[PluginInstanceConfig]:
        return [i for i in self._instances if i.category == category and i.enabled]


def dk_source_info(
    instances: List[PluginInstanceConfig],
) -> Tuple[str, Dict[str, Tuple[int, int]]]:
    """The "DK-Quelle(n) im Moment" line for a list of search instances and the
    position of each instance id in that chain, ``{instance_id: (n, total)}``.
    One ``factory.describe_dk_source`` call for both, so the line and the
    "DK n" tags in the list cannot disagree. - Claude Generated"""
    from src.core.search.factory import describe_dk_source

    info = describe_dk_source(_InstancesView(instances))
    chain = info.get("chain") or []
    positions: Dict[str, Tuple[int, int]] = {}
    for n, entry in enumerate(chain, 1):
        positions.setdefault(str(entry.get("instance_id")), (n, len(chain)))
    if not chain:
        text = str(info.get("text") or "").replace("DK-Quelle:", "DK-Quelle im Moment:", 1)
        return text, positions

    def _one(entry) -> str:
        text = f"{entry['label']} [{entry['instance_id']}]"
        return text + (", nur Cache" if entry.get("cache_only") else "")

    if len(chain) == 1:
        return f"DK-Quelle im Moment: {_one(chain[0])}", positions
    return ("DK-Quellen im Moment: " + "; dann ".join(_one(e) for e in chain)
            + " (Rückfall je Schlagwort)"), positions


def dk_source_text(instances: List[PluginInstanceConfig]) -> str:
    """The "DK-Quelle(n) im Moment" line for a list of search instances
    (``factory.describe_dk_source``) - Claude Generated"""
    return dk_source_info(instances)[0]


def _dk_tag_tip(n: int, total: int) -> str:
    """Tooltip sentence of a "DK n" tag - Claude Generated"""
    if total == 1:
        return "DK 1: die einzige DK-Quelle im Moment; der DK-Schritt fragt nur sie."
    return (f"DK {n}: Stelle {n} von {total} der DK-Kette; der DK-Schritt fragt die Quellen "
            "in dieser Reihenfolge, ein Schlagwort, das eine Quelle nicht beantwortet, geht "
            "an die nächste.")


ORDER_RULE_TEXT = ("Reihenfolge der Liste gilt: erste aktive Instanz je Typ; DK-Schritt und "
                   "search_titles fragen ihre Quellen in dieser Reihenfolge. Zum Ändern "
                   "ziehen oder Alt+↑ / Alt+↓.")
ORDER_RULE_TEXT_OTHER = ("Reihenfolge der Liste gilt: erste aktive Instanz je Typ. Zum Ändern "
                         "ziehen oder Alt+↑ / Alt+↓.")


def _ensure_categories() -> None:
    import src.core.search  # noqa: F401
    import src.utils.input_sources  # noqa: F401
    import src.utils.lookups  # noqa: F401


# ---------------------------------------------------------------------------
# Origin of an instance ("Herkunft"). Not stored in config.json: derived when
# the tab loads, from the module of the registered type class (built-in types
# live under ``src.``, the loader mounts a code plugin as the package
# ``alima_plugin_<id>``), the plugin.toml manifests in the plugin directory
# (read structurally, no plugin code runs), the approvals in the config and the
# result of a scan in this dialog. - Claude Generated
# ---------------------------------------------------------------------------
ORIGIN_BUILTIN = "builtin"
ORIGIN_DIRECTORY = "directory"
ORIGIN_CODE = "code"
ORIGIN_UNKNOWN = "unknown"

_BUILTIN_MODULE_PREFIX = "src."

_TIP_BUILTIN = "Eingebaut: der Typ ist Teil von ALIMA (Quellbaum src/)."
_TIP_DIR_UNREAD = (" Das Plugin-Verzeichnis wurde nicht gelesen; ob die Instanz aus einer "
                   "plugin.toml stammt, ist nicht bekannt.")
_TIP_CODE_LOADED = ("Code-Plugin: Python-Code aus dem Plugin-Verzeichnis, freigegeben und in "
                    "diesem Programmlauf geladen; er läuft mit vollen Rechten im Programm.")
_TIP_UNKNOWN_REGISTERED = ("Herkunft unbekannt: der Typ ist registriert, stammt aber weder aus "
                           "dem Quellbaum von ALIMA noch aus dem Plugin-Verzeichnis.")


def _code_package_prefix() -> str:
    """Package prefix the loader gives every code plugin - Claude Generated"""
    from src.core.plugins.loader import _plugin_package_name

    return _plugin_package_name("")


@dataclass(frozen=True)
class Origin:
    """Where one instance comes from, as the list shows it - Claude Generated"""

    kind: str
    tag: str
    tooltip: str


def read_plugin_manifests(plugins_dir: Path) -> Dict[str, Any]:
    """``{manifest id: PluginManifest}`` of the plugin directory; ``{}`` when it
    does not exist. Reads ``plugin.toml`` only (no plugin code runs); an
    unreadable manifest is left out. - Claude Generated"""
    from src.core.plugins.manifest import ManifestError, load_manifest_file

    root = Path(plugins_dir)
    out: Dict[str, Any] = {}
    if not root.is_dir():
        return out
    for sub in sorted(p for p in root.iterdir() if p.is_dir()):
        path = sub / "plugin.toml"
        if not path.is_file():
            continue
        try:
            manifest = load_manifest_file(path)
        except ManifestError:
            continue
        out.setdefault(manifest.id, manifest)
    return out


class PluginOrigins:
    """What the tab knows about where instances come from - Claude Generated.

    ``plugins_dir`` ``None``: the directory is not known, so an instance from a
    declarative plugin cannot be told apart from one of a built-in type.
    ``scan``: ``{manifest id: LoadedPlugin}`` of a scan run in this dialog.
    """

    def __init__(self, plugins_dir: Optional[Path] = None, *, approved=None,
                 code_enabled: bool = False, scan=None):
        self.plugins_dir = Path(plugins_dir) if plugins_dir is not None else None
        self.manifests: Dict[str, Any] = {}
        self.dir_read = False
        if self.plugins_dir is not None:
            try:
                self.manifests = read_plugin_manifests(self.plugins_dir)
                self.dir_read = True
            except OSError as e:
                from src.utils.error_visibility import log_caught

                log_caught(logger, e, "Plugin-Verzeichnis lesen")
        self.approved = dict(approved or {})
        self.code_enabled = bool(code_enabled)
        self.scan = dict(scan or {})
        self._digests: Dict[str, Optional[str]] = {}

    def _digest(self, manifest) -> Optional[str]:
        from src.core.plugins.security import hash_dir

        if manifest.id not in self._digests:
            try:
                self._digests[manifest.id] = hash_dir(manifest.source_dir)
            except OSError:
                self._digests[manifest.id] = None
        return self._digests[manifest.id]

    def _blocked_reason(self, manifest) -> str:
        """Why an unloaded code plugin may not load; ``""`` when nothing stops it."""
        if not self.code_enabled:
            return ("Code-Plugins sind nicht erlaubt (Schalter „Code-Plugins (Tier 2) "
                    "erlauben“).")
        approved = self.approved.get(manifest.id)
        if not approved:
            return "das Plugin ist nicht freigegeben (Plugin-Verzeichnis scannen und freigeben)."
        if approved != self._digest(manifest):
            return ("die Dateien haben sich seit der Freigabe geändert; es braucht eine neue "
                    "Freigabe (Plugin-Verzeichnis scannen).")
        return ""

    def origin(self, inst: PluginInstanceConfig, *, registered: bool,
               type_cls: Any = None) -> Origin:
        pid = inst.provider_id
        if registered:
            module = str(getattr(type_cls, "__module__", "") or "")
            if module.startswith(_code_package_prefix()):
                return Origin(ORIGIN_CODE, "Code-Plugin, freigegeben", _TIP_CODE_LOADED)
            if module.startswith(_BUILTIN_MODULE_PREFIX):
                manifest = self.manifests.get(inst.instance_id)
                if (manifest is not None and manifest.is_declarative
                        and manifest.kind == pid and manifest.category == inst.category):
                    where = Path(manifest.source_dir or "") / "plugin.toml"
                    return Origin(ORIGIN_DIRECTORY, "Verzeichnis", (
                        f"Verzeichnis: die Instanz stammt aus {where}, einem deklarativen "
                        f"Plugin ohne Code, und nutzt den eingebauten Typ {pid}."))
                tip = _TIP_BUILTIN if self.dir_read else _TIP_BUILTIN + _TIP_DIR_UNREAD
                return Origin(ORIGIN_BUILTIN, "eingebaut", tip)
            return Origin(ORIGIN_UNKNOWN, "unbekannt", _TIP_UNKNOWN_REGISTERED)

        manifest = self.manifests.get(pid)
        if manifest is not None and manifest.is_code and manifest.category == inst.category:
            reason = self._blocked_reason(manifest)
            if reason:
                return Origin(ORIGIN_CODE, "Code-Plugin, gesperrt",
                              f"Code-Plugin aus dem Plugin-Verzeichnis, nicht geladen: {reason}")
            tip = ("Code-Plugin aus dem Plugin-Verzeichnis, freigegeben, aber in diesem "
                   "Programmlauf nicht geladen (Plugin-Verzeichnis scannen).")
            last = self.scan.get(manifest.id)
            if last is not None and getattr(last, "status", "") == "error":
                tip += f" Letzter Scan: {getattr(last, 'detail', '')}"
            return Origin(ORIGIN_CODE, "Code-Plugin, nicht geladen", tip)
        tip = (f"Herkunft unbekannt: der Typ '{pid}' ist nicht geladen, und im "
               "Plugin-Verzeichnis liegt kein Code-Plugin mit dieser ID.")
        if not self.dir_read:
            tip += " Das Plugin-Verzeichnis wurde nicht gelesen."
        return Origin(ORIGIN_UNKNOWN, "unbekannt, nicht geladen", tip)


def _existing_plugins_dir() -> Optional[Path]:
    """The plugin directory of the ConfigManager that is already running.

    Never creates one: a ConfigManager made here (in a test, say) would bind
    the singleton to whatever HOME is set at that moment. - Claude Generated
    """
    try:
        from src.utils.config_manager import ConfigManager
    except ImportError:
        return None
    cm = getattr(ConfigManager, "_instance", None)
    if cm is None or not getattr(ConfigManager, "_initialized", False):
        return None
    try:
        return Path(cm.plugins_dir)
    except (AttributeError, TypeError):
        return None


# Item data of a list row - Claude Generated
_KEY_ROLE = int(Qt.ItemDataRole.UserRole)          # index into the panel's instance list
_CHECK_TIP_ROLE = int(Qt.ItemDataRole.UserRole) + 1  # tooltip of the checkbox
_LOCKED_ROLE = int(Qt.ItemDataRole.UserRole) + 2     # type not loaded: checkbox locked
_ROW_FLAGS = (Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled
              | Qt.ItemFlag.ItemIsDragEnabled)
_CHECKABLE_FLAGS = (_ROW_FLAGS | Qt.ItemFlag.ItemIsUserCheckable
                    | Qt.ItemFlag.ItemIsUserTristate)


def indicator_colors(colors: Dict[str, str], state: str, *, selected: bool = False,
                     locked: bool = False) -> Dict[str, Optional[str]]:
    """Colours of the list checkbox from the design tokens - Claude Generated.

    The tokens of ``QCheckBox::indicator`` in the main stylesheet: frame
    ``border`` (the token pair border/background and border/background_light
    is held at 3:1, the list ground is one of the two), fill ``background``
    (Aus), ``primary`` with a mark in ``on_primary`` (An), an inner ``primary``
    square (Nur Cache). On a selected row (highlight = ``primary``) the frame is
    ``on_primary``; a locked row uses ``text_muted`` where the others use
    ``primary``. Fusion drew the frame of an item checkbox from the palette
    role Dark, which has almost no contrast against the list in the dark
    themes. Returns ``{"border", "fill", "mark"}``; ``mark`` is ``None`` for Aus.
    """
    accent = colors["text_muted"] if locked else colors["primary"]
    border = colors["border"]
    if state == STATE_ON:
        fill, mark = accent, (colors["background"] if locked else colors["on_primary"])
    else:
        fill = colors["background"]
        mark = accent if state == STATE_CACHE_ONLY else None
    if selected:
        border = colors["on_primary"]
    return {"border": border, "fill": fill, "mark": mark}


_STATE_OF_CHECK = {int(v.value): k for k, v in _CHECK_OF_STATE.items()}


class _InstanceDelegate(QStyledItemDelegate):
    """Draws a locked row disabled and the checkbox from the design tokens;
    the checkbox has a tooltip of its own, the rest of the row explains the
    tags - Claude Generated"""

    def paint(self, painter, option, index):
        super().paint(painter, option, index)
        check = index.data(int(Qt.ItemDataRole.CheckStateRole))
        if check is None:
            return
        opt = QStyleOptionViewItem(option)
        self.initStyleOption(opt, index)
        view = self.parent() if isinstance(self.parent(), QAbstractItemView) else None
        style = view.style() if view is not None else QApplication.style()
        rect = style.subElementRect(QStyle.SubElement.SE_ItemViewItemCheckIndicator, opt, view)
        if not rect.isValid():
            return
        from .styles import get_colors

        state = _STATE_OF_CHECK.get(int(getattr(check, "value", check)), STATE_OFF)
        colors = indicator_colors(
            get_colors(), state, selected=bool(opt.state & QStyle.StateFlag.State_Selected),
            locked=bool(index.data(_LOCKED_ROLE)))
        _paint_indicator(painter, rect, state, colors)


    def initStyleOption(self, option, index):
        super().initStyleOption(option, index)
        if index.data(_LOCKED_ROLE):
            option.state = option.state & ~QStyle.StateFlag.State_Enabled
            # The app stylesheet sets one text colour for every state, so the
            # disabled state alone does not show; the muted token is read at
            # paint time and follows a theme switch. - Claude Generated
            from PyQt6.QtGui import QColor, QPalette

            from .styles import get_colors

            muted = QColor(get_colors()["text_muted"])
            for group in (QPalette.ColorGroup.Active, QPalette.ColorGroup.Inactive,
                          QPalette.ColorGroup.Disabled):
                option.palette.setColor(group, QPalette.ColorRole.Text, muted)

    def sizeHint(self, option, index):
        # No wider than the list: a long row is shortened at its end (the name
        # comes first) instead of scrolling sideways. - Claude Generated
        size = super().sizeHint(option, index)
        view = self.parent()
        if isinstance(view, QAbstractItemView) and view.viewport().width() > 0:
            size.setWidth(min(size.width(), view.viewport().width()))
        return size

    def tooltip_at(self, view, option, index, pos) -> str:
        opt = QStyleOptionViewItem(option)
        self.initStyleOption(opt, index)
        style = view.style() if view is not None else QApplication.style()
        indicator = style.subElementRect(
            QStyle.SubElement.SE_ItemViewItemCheckIndicator, opt, view)
        role = _CHECK_TIP_ROLE if indicator.contains(pos) else int(Qt.ItemDataRole.ToolTipRole)
        return str(index.data(role) or "")

    def helpEvent(self, event, view, option, index):
        if event is not None and event.type() == QEvent.Type.ToolTip and index.isValid():
            text = self.tooltip_at(view, option, index, event.pos())
            if text:
                QToolTip.showText(event.globalPos(), text, view)
            else:
                QToolTip.hideText()
            return True
        return super().helpEvent(event, view, option, index)


def _paint_indicator(painter, rect, state: str, colors: Dict[str, Optional[str]]) -> None:
    """Paint the checkbox over the one the style drew - Claude Generated"""
    from PyQt6.QtCore import QPointF
    from PyQt6.QtGui import QColor, QPainter, QPainterPath, QPen

    painter.save()
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
    painter.fillRect(rect, QColor(colors["fill"]))
    painter.setPen(QPen(QColor(colors["border"]), 1))
    painter.setBrush(Qt.BrushStyle.NoBrush)
    painter.drawRect(rect.adjusted(0, 0, -1, -1))
    mark = colors.get("mark")
    if mark and state == STATE_CACHE_ONLY:
        inset = max(3, rect.width() // 4)
        painter.fillRect(rect.adjusted(inset, inset, -inset, -inset), QColor(mark))
    elif mark and state == STATE_ON:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        w, h, x, y = rect.width(), rect.height(), rect.x(), rect.y()
        path = QPainterPath(QPointF(x + 0.22 * w, y + 0.52 * h))
        path.lineTo(QPointF(x + 0.42 * w, y + 0.72 * h))
        path.lineTo(QPointF(x + 0.78 * w, y + 0.30 * h))
        painter.setPen(QPen(QColor(mark), max(1.5, w / 7.0)))
        painter.drawPath(path)
    painter.restore()


class _InstanceList(QListWidget):
    """The instance list: internal move by drag and drop, Alt+Up/Down moves
    the selected row - Claude Generated"""

    move_requested = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        self.setDefaultDropAction(Qt.DropAction.MoveAction)
        self.setDropIndicatorShown(True)
        self.setResizeMode(QListView.ResizeMode.Adjust)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setTextElideMode(Qt.TextElideMode.ElideRight)
        self._delegate = _InstanceDelegate(self)
        self.setItemDelegate(self._delegate)

    def keyPressEvent(self, event):
        mods = event.modifiers() & ~Qt.KeyboardModifier.KeypadModifier
        if mods == Qt.KeyboardModifier.AltModifier and event.key() in (
                Qt.Key.Key_Up, Qt.Key.Key_Down):
            self.move_requested.emit(-1 if event.key() == Qt.Key.Key_Up else 1)
            event.accept()
            return
        super().keyPressEvent(event)


class _CategoryPanel(QWidget):
    """Instance list + per-instance form for one plugin category."""

    def __init__(self, category: str, parent=None):
        super().__init__(parent)
        self.category = category
        self._adapter = get_category(category)
        self._instances: List[PluginInstanceConfig] = []
        self._field_widgets: Dict[str, QWidget] = {}
        self._current_row = -1
        # Value the form showed for a key the instance does not store;
        # unchanged it stays unstored. - Claude Generated
        self._shown_absent: Dict[str, Any] = {}
        self._meta_cache: Dict[str, Any] = {}
        self._origins = PluginOrigins(None)
        self._syncing = False  # our own item updates, not the operator's
        self._moving = False   # between rowsAboutToBeMoved and rowsMoved

        root = QHBoxLayout(self)

        # Left: instance list + actions
        left = QVBoxLayout()
        # Search sources: which source the DK step uses right now, above the
        # list (computed with factory.describe_dk_source, the chain of
        # resolve_dk_extractor). Every list: the order rule. - Claude Generated
        self._dk_label: Optional[QLabel] = None
        if category == "search_provider":
            self._dk_label = QLabel("")
            self._dk_label.setWordWrap(True)
            left.addWidget(self._dk_label)
        order = QLabel(ORDER_RULE_TEXT if category == "search_provider" else ORDER_RULE_TEXT_OTHER)
        order.setWordWrap(True)
        set_tone(order, "muted")
        left.addWidget(order)
        self.list = _InstanceList()
        self.list.currentRowChanged.connect(self._on_row_changed)
        self.list.itemChanged.connect(self._on_item_changed)
        self.list.move_requested.connect(self.move_current)
        model = self.list.model()
        model.rowsAboutToBeMoved.connect(self._on_rows_about_to_move)
        model.rowsMoved.connect(self._on_rows_moved)
        left.addWidget(self.list)
        # Feedback after a state change; emptied when another row is selected.
        # - Claude Generated
        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        set_tone(self.status_label, "info")
        left.addWidget(self.status_label)

        self.type_combo = QComboBox()
        for t in self._adapter.list_types():
            self.type_combo.addItem(self._adapter.type_meta(t).label, t)
        left.addWidget(self.type_combo)

        btns = QHBoxLayout()
        add = QPushButton("Hinzufügen")
        dup = QPushButton("Duplizieren")
        rem = QPushButton("Entfernen")
        add.clicked.connect(self._add_instance)
        dup.clicked.connect(self._duplicate_instance)
        rem.clicked.connect(self._remove_instance)
        for b in (add, dup, rem):
            btns.addWidget(b)
        left.addLayout(btns)
        # 2:3 (was 1:2): a row with its tags fits at 1100 px. - Claude Generated
        root.addLayout(left, 2)

        # Right: form for the selected instance
        self.form_box = QGroupBox("Instanz")
        self.form_layout = QFormLayout(self.form_box)
        root.addWidget(self.form_box, 3)

    # -- data -----------------------------------------------------------------
    def load(self, instances: List[PluginInstanceConfig]) -> None:
        # The form of a previous load belongs to another instance list; without
        # clearing it the next row change wrote it into the new instance at the
        # old row. - Claude Generated
        self._instances = [copy.deepcopy(i) for i in instances]
        self._meta_cache = {}
        self._current_row = -1
        self._clear_form()
        self._set_status("")
        self._rebuild_list()
        if self._instances:
            self.list.setCurrentRow(0)
        else:
            self._build_form(None)

    def collect(self) -> List[PluginInstanceConfig]:
        """Flush the visible form and return the edited instances in list order."""
        self._flush_form()
        return [copy.deepcopy(i) for i in self._instances]

    def set_origins(self, origins: PluginOrigins) -> None:
        """Use ``origins`` for the origin tags and refresh the rows - Claude Generated"""
        self._origins = origins
        self._refresh_view()

    # -- type information ----------------------------------------------------
    def _meta(self, provider_id: str):
        """``type_meta`` of a type, ``None`` when it is not registered (cached
        per load: types are registered on load and scan only) - Claude Generated"""
        if provider_id not in self._meta_cache:
            try:
                self._meta_cache[provider_id] = self._adapter.type_meta(provider_id)
            except KeyError:
                self._meta_cache[provider_id] = None
        return self._meta_cache[provider_id]

    def _offer(self, provider_id: str) -> str:
        """``cache_only_offer`` of a type; ``OFFER_NONE`` for an unloaded one - Claude Generated"""
        meta = self._meta(provider_id)
        return cache_only_offer(meta) if meta is not None else OFFER_NONE

    def _type_class(self, provider_id: str) -> Any:
        getter = getattr(self._adapter, "type_class", None)
        if getter is None:
            return None
        try:
            return getter(provider_id)
        except KeyError:
            return None

    def _origin(self, inst: PluginInstanceConfig) -> Origin:
        return self._origins.origin(
            inst, registered=self._meta(inst.provider_id) is not None,
            type_cls=self._type_class(inst.provider_id))

    # -- list -----------------------------------------------------------------
    def _rebuild_list(self) -> None:
        """One row per instance, keyed by its index; keeps the selection - Claude Generated"""
        self._syncing = True
        self.list.blockSignals(True)
        try:
            self.list.clear()
            for index in range(len(self._instances)):
                item = QListWidgetItem(self.list)
                item.setData(_KEY_ROLE, index)
            if 0 <= self._current_row < self.list.count():
                self.list.setCurrentRow(self._current_row)
        finally:
            self.list.blockSignals(False)
            self._syncing = False
        self._refresh_view()

    def _dialog_state(self) -> List[PluginInstanceConfig]:
        """The instances as the dialog shows them: the stored list plus the
        unsaved form of the selected row (copies) - Claude Generated"""
        state = [copy.deepcopy(i) for i in self._instances]
        if 0 <= self._current_row < len(state) and hasattr(self, "_state_combo"):
            self._write_form_into(state[self._current_row])
        return state

    def _refresh_view(self) -> None:
        """Rows, DK line, DK switch and "Wirkung" from the dialog state - Claude Generated"""
        if self._moving:
            return
        state = self._dialog_state()
        positions = self._update_dk_line(state)
        self._syncing = True
        try:
            for row in range(min(self.list.count(), len(state))):
                self._render_item(self.list.item(row), state[row], positions)
        finally:
            self._syncing = False
        self._sync_dk_switch(positions)
        self._refresh_effect_line()

    def _update_dk_line(self, state: List[PluginInstanceConfig]) -> Dict[str, Tuple[int, int]]:
        """Recompute the "DK-Quelle im Moment" line; returns the DK positions - Claude Generated"""
        if self._dk_label is None:
            return {}
        try:
            text, positions = dk_source_info(state)
        except Exception as e:  # a broken plugin must not break the tab
            from src.utils.error_visibility import log_caught

            log_caught(logger, e, "DK-Quelle im Plugin-Tab")
            text, positions = "DK-Quelle im Moment: nicht bestimmbar", {}
        self._dk_label.setText(text)
        return positions

    def _render_item(self, item: QListWidgetItem, inst: PluginInstanceConfig,
                     positions: Dict[str, Tuple[int, int]]) -> None:
        """Text, checkbox, tags and tooltips of one row - Claude Generated"""
        meta = self._meta(inst.provider_id)
        offer = cache_only_offer(meta) if meta is not None else OFFER_NONE
        state = instance_state(inst, offer)
        origin = self._origin(inst)
        tags: List[str] = []
        tips: List[str] = []
        dk = positions.get(inst.instance_id)
        if dk is not None:
            tags.append(f"DK {dk[0]}")
            tips.append(_dk_tag_tip(*dk))
        tags.append(origin.tag)
        tips.append(origin.tooltip)
        text = f"{inst.display_label()} [{inst.provider_id}]{_LIST_SUFFIX[state]}"
        item.setText(text + "".join(f"  ·  {t}" for t in tags))
        item.setToolTip("\n".join(tips))
        item.setCheckState(_CHECK_OF_STATE[state])
        locked = meta is None
        item.setData(_LOCKED_ROLE, locked)
        if locked:
            item.setFlags(_ROW_FLAGS)
            item.setData(_CHECK_TIP_ROLE, f"Nicht schaltbar. {origin.tooltip}")
        else:
            item.setFlags(_CHECKABLE_FLAGS)
            nxt = _STATE_LABELS[next_state(state, offer)]
            item.setData(_CHECK_TIP_ROLE,
                         f"{_STATE_MEANING[state]} Ein Klick schaltet auf „{nxt}“.")

    def _set_status(self, text: str) -> None:
        self.status_label.setText(text)

    def _row_name(self, row: int) -> str:
        """Display name of a row, with the unsaved name field of the selected row."""
        inst = self._instances[row]
        if row == self._current_row and hasattr(self, "_label_edit"):
            return self._label_edit.text() or inst.instance_id or inst.provider_id
        return inst.display_label()

    def _on_item_changed(self, item: QListWidgetItem) -> None:
        """A click on a row's checkbox: move that row's state on - Claude Generated"""
        if self._syncing or self._moving:
            return
        row = self.list.row(item)
        if not (0 <= row < len(self._instances)):
            return
        inst = self._instances[row]
        meta = self._meta(inst.provider_id)
        offer = cache_only_offer(meta) if meta is not None else OFFER_NONE
        current = instance_state(inst, offer)
        if item.checkState() == _CHECK_OF_STATE[current]:
            return  # another role of the item changed
        if meta is None:
            self._refresh_view()  # locked: back to the stored state
            return
        self._set_state(row, next_state(current, offer))

    def _set_state(self, row: int, state: str) -> None:
        """Write ``state`` into the row's instance (the only place it is kept),
        show it in the checkbox and, for the selected row, in "Zustand".
        - Claude Generated"""
        inst = self._instances[row]
        apply_state(inst, state, self._offer(inst.provider_id))
        combo = getattr(self, "_state_combo", None)
        if row == self._current_row and combo is not None:
            index = combo.findData(state)
            if index >= 0 and index != combo.currentIndex():
                combo.blockSignals(True)
                combo.setCurrentIndex(index)
                combo.blockSignals(False)
        self._set_status(f"{self._row_name(row)}: {_STATE_STATUS[state]}")
        self._refresh_view()

    def _on_state_combo_changed(self, *_args) -> None:
        if 0 <= self._current_row < len(self._instances):
            self._set_state(self._current_row, self._state_combo.currentData())

    def _on_form_edited(self, *_args) -> None:
        """Keep rows, DK line and DK switch in step with the unsaved form - Claude Generated"""
        self._refresh_view()

    def _sync_dk_switch(self, positions: Dict[str, Tuple[int, int]]) -> None:
        """An unstored DK switch shows the effective state (the instance is in
        the DK chain or not), as long as the operator has not changed it.
        - Claude Generated"""
        from src.core.search.factory import DK_SWITCH_KEY

        if self._dk_label is None or not (0 <= self._current_row < len(self._instances)):
            return
        w = self._field_widgets.get(DK_SWITCH_KEY)
        inst = self._instances[self._current_row]
        if (not isinstance(w, QCheckBox) or DK_SWITCH_KEY in (inst.settings or {})
                or DK_SWITCH_KEY not in self._shown_absent):
            return
        shown = bool(self._shown_absent[DK_SWITCH_KEY])
        if w.isChecked() != shown:
            return  # changed by the operator: saved literally
        effective = inst.instance_id in positions
        if effective != shown:
            w.blockSignals(True)
            w.setChecked(effective)
            w.blockSignals(False)
            self._shown_absent[DK_SWITCH_KEY] = effective

    def _effect_text(self, inst: PluginInstanceConfig, meta) -> str:
        effect = getattr(meta, "effect", "") or ""
        siblings = sum(1 for i in self._instances if i.provider_id == inst.provider_id)
        if siblings > 1 and getattr(meta, "effect_multi", ""):
            effect = f"{effect} {meta.effect_multi}".strip()
        return effect

    def _refresh_effect_line(self) -> None:
        label = getattr(self, "_effect_label", None)
        if label is None or not (0 <= self._current_row < len(self._instances)):
            return
        inst = self._instances[self._current_row]
        meta = self._meta(inst.provider_id)
        if meta is not None:
            label.setText(self._effect_text(inst, meta))

    # -- order ----------------------------------------------------------------
    def _on_rows_about_to_move(self, *_args) -> None:
        self._flush_form()
        self._moving = True

    def _on_rows_moved(self, *_args) -> None:
        """Take the order of the rows (drag and drop or ``move_current``) into the
        instance list; the selected instance stays selected - Claude Generated"""
        self._moving = False
        keys = [self.list.item(r).data(_KEY_ROLE) for r in range(self.list.count())]
        if sorted(k for k in keys if isinstance(k, int)) != list(range(len(self._instances))):
            logger.error("Plugin-Liste passt nicht mehr zu den Instanzen (%s); neu aufgebaut",
                         keys)
            self._rebuild_list()
            return
        moved = self._current_row
        self._instances = [self._instances[k] for k in keys]
        self._syncing = True
        try:
            for row in range(self.list.count()):
                self.list.item(row).setData(_KEY_ROLE, row)
        finally:
            self._syncing = False
        if 0 <= moved < len(keys):
            self._current_row = keys.index(moved)
            if self.list.currentRow() != self._current_row:
                self.list.blockSignals(True)
                self.list.setCurrentRow(self._current_row)
                self.list.blockSignals(False)
        self._refresh_view()

    def move_current(self, delta: int) -> None:
        """Move the selected row up (-1) or down (+1); same path as a drop - Claude Generated"""
        row = self._current_row
        target = row + delta
        if not (0 <= row < len(self._instances)) or not (0 <= target < len(self._instances)):
            return
        destination = target + 1 if delta > 0 else target
        if not self.list.model().moveRow(QModelIndex(), row, QModelIndex(), destination):
            self._flush_form()
            self._instances.insert(target, self._instances.pop(row))
            self._current_row = target
            self._rebuild_list()

    # -- list ops -------------------------------------------------------------
    def _unique_id(self, base: str) -> str:
        existing = {i.instance_id for i in self._instances}
        if base not in existing:
            return base
        n = 2
        while f"{base}-{n}" in existing:
            n += 1
        return f"{base}-{n}"

    def _add_instance(self) -> None:
        type_id = self.type_combo.currentData()
        if not type_id:
            return
        meta = self._adapter.type_meta(type_id)
        settings = {f.key: f.default for f in meta.config_fields}
        inst = PluginInstanceConfig(
            instance_id=self._unique_id(type_id),
            category=self.category,
            provider_id=type_id,
            label=meta.label,
            enabled=True,
            settings=settings,
        )
        self._flush_form()
        self._instances.append(inst)
        self._rebuild_list()
        self.list.setCurrentRow(len(self._instances) - 1)

    def _duplicate_instance(self) -> None:
        if not (0 <= self._current_row < len(self._instances)):
            return
        self._flush_form()
        src = self._instances[self._current_row]
        clone = copy.deepcopy(src)
        clone.instance_id = self._unique_id(src.instance_id)
        clone.label = f"{src.display_label()} (Kopie)"
        self._instances.append(clone)
        self._rebuild_list()
        self.list.setCurrentRow(len(self._instances) - 1)

    def _remove_instance(self) -> None:
        if not (0 <= self._current_row < len(self._instances)):
            return
        del self._instances[self._current_row]
        self._current_row = -1
        self._clear_form()
        self._set_status("")
        self._rebuild_list()
        if self._instances:
            self.list.setCurrentRow(0)
        else:
            self._build_form(None)

    def _on_row_changed(self, row: int) -> None:
        if self._syncing or self._moving:
            return
        self._flush_form()
        self._current_row = row
        self._build_form(self._instances[row] if 0 <= row < len(self._instances) else None)
        self._set_status("")
        self._refresh_view()

    # -- form -----------------------------------------------------------------
    def _clear_form(self) -> None:
        while self.form_layout.rowCount():
            self.form_layout.removeRow(0)
        self._field_widgets = {}
        # Die Widgets sind mit dem Formular gelöscht; ohne das Entfernen schreibt
        # _flush_form in tote Qt-Objekte. - Claude Generated
        for name in ("_state_combo", "_label_edit", "_hint_edit", "_effect_label"):
            self.__dict__.pop(name, None)

    def _build_form(self, inst: Optional[PluginInstanceConfig]) -> None:
        self._clear_form()
        self._shown_absent = {}
        if inst is None:
            self.form_layout.addRow(QLabel("Keine Instanz ausgewählt."))
            return

        # Eine gespeicherte Instanz kann zu einem Typ gehören, der nicht geladen
        # ist (Code-Plugin gesperrt, nicht freigegeben oder entfernt). Die
        # Exception in diesem Slot beendete sonst das ganze Programm (PyQt
        # bricht bei unbehandelten Fehlern in Slots ab). - Claude Generated
        try:
            meta = self._adapter.type_meta(inst.provider_id)
        except KeyError:
            note = QLabel(
                f"Das Plugin '{inst.provider_id}' ist nicht geladen. Code-Plugins "
                "müssen erlaubt und das Plugin freigegeben sein (Plugin-Verzeichnis "
                "neu einlesen). Die Einstellungen der Instanz bleiben unverändert "
                "gespeichert."
            )
            note.setWordWrap(True)
            self.form_layout.addRow(note)
            origin = QLabel(self._origin(inst).tooltip)
            origin.setWordWrap(True)
            set_tone(origin, "muted")
            self.form_layout.addRow(origin)
            return

        # Natural-language self-description (what it does + input/output).
        doc = getattr(meta, "doc", None)
        if doc is not None and (doc.description or doc.input or doc.output):
            doc_lbl = QLabel(doc.as_text())
            doc_lbl.setWordWrap(True)
            # Palette roles resolved by Qt at paint time, so the box follows a
            # theme switch (a hex snapshot of the palette would not).
            # - Claude Generated
            doc_lbl.setStyleSheet(
                "background: palette(alternate-base); color: palette(window-text);"
                " padding:6px; border-radius:4px;"
            )
            self.form_layout.addRow(doc_lbl)

        # What this type can do. Capabilities are what the runtime actually
        # selects on (factory.enabled_gnd_provider_ids, resolve_dk_extractor);
        # type_meta has computed them all along and the form never showed them,
        # so the operator saw prose and settings but not the selection rule.
        # - Claude Generated
        caps = list(getattr(meta, "capabilities", []) or [])
        if caps:
            cap_lbl = QLabel("Kann: " + ", ".join(_CAPABILITY_LABELS.get(c, str(c)) for c in caps))
            cap_lbl.setWordWrap(True)
            set_tone(cap_lbl, "muted")
            cap_lbl.setStyleSheet("font-size: 10px;")
            self.form_layout.addRow(cap_lbl)

        # What an instance of this type does to a run ("Wirkung"), read-only;
        # the multi-instance rule only when the list holds several of the type.
        # - Claude Generated
        effect = self._effect_text(inst, meta)
        if effect:
            effect_lbl = QLabel(effect)
            effect_lbl.setWordWrap(True)
            effect_lbl.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            self._effect_label = effect_lbl
            self.form_layout.addRow("Wirkung", effect_lbl)

        # Read-only info lines of the type (e.g. the state of the geo_sachdb
        # database). - Claude Generated
        for info_label, info_fn in _TYPE_INFO.get((self.category, inst.provider_id), []):
            try:
                text = info_fn(inst)
            except Exception as exc:  # an unreadable file must not break the form
                from src.utils.error_visibility import log_caught

                log_caught(logger, exc, f"Info-Zeile {info_label} für {inst.instance_id}")
                text = ""
            if text:
                info = QLabel(text)
                info.setWordWrap(True)
                info.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
                set_tone(info, "muted")
                self.form_layout.addRow(info_label, info)

        # State instead of an enable checkbox: An / Nur Cache / Aus. "Nur Cache"
        # only where the type has a cache path; shown disabled with the reason
        # for a network source without one. - Claude Generated
        offer = cache_only_offer(meta)
        self._state_offer = offer
        self._state_combo = QComboBox()
        for state in (STATE_ON, STATE_CACHE_ONLY, STATE_OFF):
            if state == STATE_CACHE_ONLY and offer == OFFER_NONE:
                continue
            self._state_combo.addItem(_STATE_LABELS[state], state)
            if state == STATE_CACHE_ONLY and offer == OFFER_BLOCKED:
                idx = self._state_combo.count() - 1
                item = self._state_combo.model().item(idx)
                if item is not None:
                    item.setEnabled(False)
                self._state_combo.setItemData(
                    idx, meta.cache_only_note, Qt.ItemDataRole.ToolTipRole
                )
        current = self._state_combo.findData(instance_state(inst, offer))
        self._state_combo.setCurrentIndex(current if current >= 0 else 0)
        self.form_layout.addRow("Zustand", self._state_combo)
        if offer == OFFER_CACHE_ONLY:
            from src.core.search.cache_only import cache_only_field

            state_hint = QLabel(f"Nur Cache: {cache_only_field().help}")
        elif offer == OFFER_BLOCKED:
            state_hint = QLabel(meta.cache_only_note)
        else:
            state_hint = None
        if state_hint is not None:
            state_hint.setWordWrap(True)
            set_tone(state_hint, "muted")
            state_hint.setStyleSheet("font-size: 10px;")
            self.form_layout.addRow("", state_hint)
        self._state_combo.currentIndexChanged.connect(self._on_state_combo_changed)

        self._label_edit = QLineEdit(inst.label)
        self._label_edit.textChanged.connect(self._on_form_edited)
        self.form_layout.addRow("Name", self._label_edit)

        # Collapsed unless it has content. It is a real feature (the text is
        # appended to the generated MCP tool description, see
        # tool_registry._describe_with_hint), but it only matters with two
        # instances of one type — a case no install has had — while occupying a
        # 64px box on all 16 panels. - Claude Generated
        self._hint_edit = QTextEdit(inst.usage_hint)
        self._hint_edit.setPlaceholderText(
            "Wann soll der Agent gerade diese Instanz nehmen? Nur nötig, wenn es "
            "mehrere Instanzen desselben Typs gibt."
        )
        self._hint_edit.setMaximumHeight(64)
        self._hint_row = QWidget()
        _hint_box = QVBoxLayout(self._hint_row)
        _hint_box.setContentsMargins(0, 0, 0, 0)
        _hint_box.addWidget(self._hint_edit)
        self._hint_toggle = QCheckBox("Agent-Hinweis")
        self._hint_toggle.setChecked(bool(inst.usage_hint))
        self._hint_row.setVisible(bool(inst.usage_hint))
        self._hint_toggle.toggled.connect(self._hint_row.setVisible)
        self.form_layout.addRow(self._hint_toggle, self._hint_row)

        meta = self._adapter.type_meta(inst.provider_id)
        for fld in meta.config_fields:
            if self._state_covers(fld.key):
                continue  # shown as the "Nur Cache" state above - Claude Generated
            w = self._make_field_widget(
                fld, inst.settings.get(fld.key, fld.default), instance_id=inst.instance_id
            )
            if fld.key not in inst.settings:
                # What the field shows for an absent key (the default, as the
                # widget holds it); unchanged it stays absent. - Claude Generated
                self._shown_absent[fld.key] = self._read_field_widget(fld, w)
                if fld.key == "dk_enabled":
                    # Synced to the effective state by _sync_dk_switch. - Claude Generated
                    w.setToolTip(
                        "Nicht gespeichert: das Kästchen zeigt, ob die Instanz im Moment "
                        "DK-Quelle ist (Zeile über der Liste). Erst eine Änderung wird "
                        "gespeichert."
                    )
            self._field_widgets[fld.key] = w
            self._connect_edited(w)
            label = f"{fld.label} *" if fld.required else fld.label
            self.form_layout.addRow(label, w)
            if fld.help:
                hint = QLabel(fld.help)
                hint.setWordWrap(True)
                set_tone(hint, "muted")
                hint.setStyleSheet("font-size: 10px;")
                self.form_layout.addRow("", hint)

        # Per-type action buttons (e.g. webindex "Seite indizieren"). The form is
        # flushed first so the action sees the just-edited settings. - Claude Generated
        actions = _TYPE_ACTIONS.get((self.category, inst.provider_id), [])
        for btn_label, cb in actions:
            btn = QPushButton(btn_label)
            btn.clicked.connect(lambda _checked, c=cb: self._run_type_action(c))
            self.form_layout.addRow(btn)

    def _run_type_action(self, callback) -> None:
        """Flush the form, then run a registered per-type action on the current
        instance (the callback receives the flushed instance + this panel)."""
        self._flush_form()
        if not (0 <= self._current_row < len(self._instances)):
            return
        callback(self._instances[self._current_row], parent=self.window())

    def _make_field_widget(self, fld, value, instance_id: str = "") -> QWidget:
        if fld.kind == BOOL:
            w = QCheckBox()
            w.setChecked(bool(value))
            return w
        if fld.kind == INT:
            w = QSpinBox()
            # Declared bounds (e.g. a timeout's 1..120 s) - Claude Generated
            low = fld.minimum if getattr(fld, "minimum", None) is not None else 0
            high = fld.maximum if getattr(fld, "maximum", None) is not None else 10_000_000
            w.setRange(low, high)
            try:
                w.setValue(int(value or 0))
            except (TypeError, ValueError):
                w.setValue(0)
            return w
        if fld.kind == CHOICE:
            w = QComboBox()
            for c in (fld.choices or []):
                w.addItem(str(c), c)
            # A stored value that is not among the choices (a legacy bool, a preset
            # dropped from the list) used to leave the combo on index 0, and
            # _flush_form then wrote that back — a silent value change on the next
            # save. ConfigField.coerce has the right fallback: the declared
            # default. - Claude Generated
            idx = w.findData(fld.coerce(value))
            w.setCurrentIndex(idx if idx >= 0 else 0)
            return w
        w = QLineEdit("" if value is None else str(value))
        if fld.kind == SECRET:
            w.setEchoMode(QLineEdit.EchoMode.Password)
            # Surface an active env-var override (runtime wins over the stored
            # value; the stored value stays editable). - Claude Generated
            import os

            from src.core.plugins.schema import env_var_name

            var = env_var_name(instance_id, fld.key)
            if os.environ.get(var):
                w.setPlaceholderText(f"überschrieben durch Umgebungsvariable {var}")
                w.setToolTip(
                    f"Zur Laufzeit gewinnt die Umgebungsvariable {var}; der hier "
                    "gespeicherte Wert wird dann ignoriert."
                )
        return w

    def _read_field_widget(self, fld, w):
        if fld.kind == BOOL:
            return w.isChecked()
        if fld.kind == INT:
            return w.value()
        if fld.kind == CHOICE:
            return w.currentData()
        return w.text()

    def _state_covers(self, key: str) -> bool:
        """Whether the "Zustand" combo stands for this settings key - Claude Generated"""
        from src.core.search.cache_only import CACHE_ONLY_KEY

        return key == CACHE_ONLY_KEY and getattr(self, "_state_offer", OFFER_NONE) == OFFER_CACHE_ONLY

    def _connect_edited(self, w: QWidget) -> None:
        """Re-evaluate the DK line when a field changes (search sources only) - Claude Generated"""
        if self._dk_label is None:
            return
        for signal in ("toggled", "textChanged", "valueChanged", "currentIndexChanged"):
            sig = getattr(w, signal, None)
            if sig is not None:
                sig.connect(self._on_form_edited)
                return

    def _write_form_into(self, inst: PluginInstanceConfig) -> None:
        """Write the visible form's values into ``inst`` (the shown instance or
        a preview copy of it). - Claude Generated"""
        apply_state(inst, self._state_combo.currentData(), self._state_offer)
        inst.label = self._label_edit.text()
        inst.usage_hint = self._hint_edit.toPlainText()
        meta = self._adapter.type_meta(inst.provider_id)
        for fld in meta.config_fields:
            w = self._field_widgets.get(fld.key)
            if w is None:
                continue
            value = self._read_field_widget(fld, w)
            # A key the instance does not store stays unstored while the field
            # shows what it showed on building the form (for dk_enabled: the
            # effective state, see _sync_dk_switch). - Claude Generated
            if (fld.key not in inst.settings and fld.key in self._shown_absent
                    and value == self._shown_absent[fld.key]):
                continue
            inst.settings[fld.key] = value

    def _flush_form(self) -> None:
        """Write the visible form back into the current instance (data only;
        the rows follow through ``_refresh_view``)."""
        if not (0 <= self._current_row < len(self._instances)):
            return
        if not hasattr(self, "_state_combo"):
            return
        self._write_form_into(self._instances[self._current_row])


def _local_time(stamp) -> str:
    """An ISO time stamp of the ledger as local date and time - Claude Generated"""
    from datetime import datetime

    try:
        return datetime.fromisoformat(str(stamp)).astimezone().strftime("%d.%m.%Y %H:%M")
    except (TypeError, ValueError):
        return str(stamp or "?")


class _PresetWorker(QThread):
    """Runs one preset job (plan, apply, revert, export) off the UI thread:
    hashing and copying a large database must not freeze the dialog.
    - Claude Generated"""

    succeeded = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, job, parent=None):
        super().__init__(parent)
        self._job = job

    def run(self) -> None:
        try:
            self.succeeded.emit(self._job())
        except Exception as exc:  # PresetError, OSError, a defect: shown to the operator
            from src.utils.error_visibility import log_caught

            log_caught(logger, exc, "Preset-Aktion im Plugin-Reiter")
            self.failed.emit(str(exc) or type(exc).__name__)


class _PresetPlanDialog(QDialog):
    """Preview of a preset plan with "Anwenden" / "Abbrechen" - Claude Generated.

    "Anwenden" is offered only for a plan without errors that changes
    something.
    """

    def __init__(self, plan, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Preset anwenden: Vorschau")
        self.resize(760, 520)
        v = QVBoxLayout(self)
        self.text = QTextEdit()
        self.text.setReadOnly(True)
        self.text.setPlainText("\n".join(plan.lines()))
        v.addWidget(self.text, 1)
        if plan.errors:
            note = ("Das Preset hat Fehler und wird nicht angewendet. Dateien mit falscher "
                    "Prüfsumme oder falschem Schema werden nie installiert.")
            tone = "error"
        elif plan.is_empty:
            note = "Nichts zu ändern."
            tone = "muted"
        else:
            note = ("Die Änderungen werden sofort gespeichert, unabhängig von OK oder Abbrechen "
                    "im Einstellungsdialog. „Rückgängig“ in der Gruppe nimmt sie zurück.")
            tone = "muted"
        self.note = QLabel(note)
        self.note.setWordWrap(True)
        set_tone(self.note, tone)
        v.addWidget(self.note)
        buttons = QDialogButtonBox()
        self.apply_button = buttons.addButton("Anwenden", QDialogButtonBox.ButtonRole.AcceptRole)
        self.apply_button.setEnabled(plan.ok and not plan.is_empty)
        buttons.addButton("Abbrechen", QDialogButtonBox.ButtonRole.RejectRole)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        v.addWidget(buttons)


class _ExportPresetDialog(QDialog):
    """Preset id/version/name and a checklist of the instances to export.

    - Claude Generated
    """

    def __init__(self, instances, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Preset exportieren")
        v = QVBoxLayout(self)
        form = QFormLayout()
        self.id_edit = QLineEdit()
        self.id_edit.setPlaceholderText("z. B. ub-musterstadt")
        self.version_edit = QLineEdit("1")
        self.label_edit = QLineEdit()
        form.addRow("Preset-ID:", self.id_edit)
        form.addRow("Version:", self.version_edit)
        form.addRow("Name:", self.label_edit)
        v.addLayout(form)
        hint = QLabel("Instanzen im Paket (Zustand, Reihenfolge und Einstellungen; ohne "
                      "Schlüssel und Tokens; Datenbank- und Vokabulardateien werden mit "
                      "Prüfsumme mitkopiert):")
        hint.setWordWrap(True)
        v.addWidget(hint)
        self.list = QListWidget()
        for category, iid, label, enabled in instances:
            text = f"{_CATEGORY_LABELS.get(category, category)}: {iid}"
            if label and label != iid:
                text += f"  ·  {label}"
            if not enabled:
                text += "  (aus)"
            item = QListWidgetItem(text)
            item.setData(Qt.ItemDataRole.UserRole, iid)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked)
            self.list.addItem(item)
        v.addWidget(self.list, 1)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        v.addWidget(buttons)

    def selected_instance_ids(self) -> List[str]:
        out = []
        for i in range(self.list.count()):
            item = self.list.item(i)
            if item.checkState() == Qt.CheckState.Checked:
                out.append(item.data(Qt.ItemDataRole.UserRole))
        return out


class PluginSettingsTab(QWidget):
    """Top-level plugins tab: one sub-tab per registered plugin category."""

    #: A preset was applied or reverted and saved: (replay steps, saved config).
    #: The settings dialog writes the steps into its own copy. - Claude Generated
    preset_applied = pyqtSignal(object, object)

    def __init__(self, parent=None):
        super().__init__(parent)
        _ensure_categories()
        # Origin context of the list tags, set on load - Claude Generated
        self._plugins_dir: Optional[Path] = None
        self._approved: Dict[str, str] = {}
        self._scan_results: Dict[str, Any] = {}
        self._origins = PluginOrigins(None)
        layout = QVBoxLayout(self)
        info = QLabel(
            "Plugins bündeln ihre eigene Konfiguration (Endpunkte, Tokens …). Jede "
            "Instanz hat einen Zustand: An, Nur Cache (bei Suchquellen mit Cache) oder "
            "Aus. Das Kästchen vor der Instanz schaltet der Reihe nach weiter (Haken: "
            "An, halb: Nur Cache, leer: Aus), das Feld „Zustand“ zeigt denselben Wert. "
            "Eine Instanz kann einen Agent-Hinweis bekommen. Mehrere Instanzen eines "
            "Typs sind möglich; die erste aktive vertritt den Typ dort, wo nur eine "
            "gemeint sein kann."
        )
        info.setWordWrap(True)
        layout.addWidget(info)

        self._tabs = QTabWidget()
        self._panels: Dict[str, _CategoryPanel] = {}
        for cat in list_categories():
            panel = _CategoryPanel(cat)
            self._panels[cat] = panel
            self._tabs.addTab(panel, _CATEGORY_LABELS.get(cat, cat))
        layout.addWidget(self._tabs, 1)  # free height to the lists, not the preset group

        # Directory-plugin rescan with the Tier-2 approval dialog (config load
        # runs headless = deny; this button is the interactive path). The
        # enable_code_plugins master gate lives here too (was config.json-only).
        # - Claude Generated
        scan_row = QHBoxLayout()
        self._code_plugins_cb = QCheckBox("Code-Plugins (Tier 2) erlauben")
        self._code_plugins_cb.setToolTip(
            "Master-Schalter für Code-Plugins aus ~/.config/alima/plugins/.\n"
            "Zusätzlich erfordert jedes Code-Plugin eine einzelne Freigabe\n"
            "(Security-Scan + Hash). Freigegebene Plugins laufen in-process\n"
            "mit vollen Rechten — kein Sandbox. Deklarative Plugins (nur\n"
            "plugin.toml) laden unabhängig von diesem Schalter."
        )
        self._code_plugins_cb.toggled.connect(self._on_code_plugins_toggled)
        scan_row.addWidget(self._code_plugins_cb)
        self._scan_btn = QPushButton("📂 Plugin-Verzeichnis scannen…")
        self._scan_btn.setToolTip(
            "Scannt ~/.config/alima/plugins/ erneut; nicht freigegebene "
            "Code-Plugins werden mit Scan-Findings zur Freigabe angezeigt."
        )
        self._scan_btn.clicked.connect(self._rescan_plugins)
        scan_row.addWidget(self._scan_btn)
        scan_row.addStretch(1)
        layout.addLayout(scan_row)

        layout.addWidget(self._build_preset_group())

    def load(self, config, plugins_dir: Optional[Path] = None) -> None:
        """Populate every category panel from ``config.plugins``.

        ``plugins_dir``: the directory scanned for directory plugins (origin
        tags); default the last one given, else that of the running
        ConfigManager. - Claude Generated
        """
        plugins = list(getattr(config, "plugins", []) or [])
        if plugins_dir is not None:
            self._plugins_dir = Path(plugins_dir)
        elif self._plugins_dir is None:
            self._plugins_dir = _existing_plugins_dir()
        self._approved = dict(getattr(config, "approved_plugins", None) or {})
        self._code_plugins_cb.blockSignals(True)
        self._code_plugins_cb.setChecked(
            bool(getattr(getattr(config, "system_config", None), "enable_code_plugins", False))
        )
        self._code_plugins_cb.blockSignals(False)
        self._origins = PluginOrigins(
            self._plugins_dir, approved=self._approved,
            code_enabled=self._code_plugins_cb.isChecked(), scan=self._scan_results)
        for cat, panel in self._panels.items():
            panel._origins = self._origins
            panel.load([p for p in plugins if p.category == cat])

    def _on_code_plugins_toggled(self, checked: bool) -> None:
        """The origin tags name the switch as a reason; follow it - Claude Generated"""
        self._origins.code_enabled = bool(checked)
        for panel in self._panels.values():
            panel.set_origins(self._origins)

    def apply_to(self, config) -> None:
        """Write the edited instances back into ``config.plugins`` (authoritative)."""
        collected: List[PluginInstanceConfig] = []
        managed = set(self._panels)
        for panel in self._panels.values():
            collected.extend(panel.collect())
        # Preserve instances of any category this tab doesn't manage.
        other = [p for p in (getattr(config, "plugins", []) or []) if p.category not in managed]
        config.plugins = collected + other
        if getattr(config, "system_config", None) is not None:
            config.system_config.enable_code_plugins = self._code_plugins_cb.isChecked()
        self._warn_on_operator_urls(collected)

    # ---- Institution preset (alima_presets.json) ---------------------------
    def _build_preset_group(self) -> QGroupBox:
        """Found preset, last applied preset, apply / revert / export.

        The jobs run in a worker thread (``_PresetWorker``); plan and result
        come from the Qt-free ``src/utils/preset_apply.py`` and
        ``preset_export.py``, the same functions as ``alima preset``.
        - Claude Generated
        """
        box = QGroupBox("Einrichtung (Preset)")
        v = QVBoxLayout(box)
        self._preset_found_label = QLabel()
        self._preset_found_label.setWordWrap(True)
        self._preset_found_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        v.addWidget(self._preset_found_label)
        self._preset_last_label = QLabel()
        self._preset_last_label.setWordWrap(True)
        set_tone(self._preset_last_label, "muted")
        v.addWidget(self._preset_last_label)

        row = QHBoxLayout()
        self._preset_apply_btn = QPushButton("Preset anwenden…")
        self._preset_apply_btn.setToolTip(
            "Plugin-Zustände und -Einstellungen, Einrichtung (ISIL/ELN), Standard-Workflow "
            "und mitgelieferte Dateien aus einem Preset übernehmen; vorher zeigt eine "
            "Vorschau jede Änderung."
        )
        menu = QMenu(self._preset_apply_btn)
        self._preset_found_action = menu.addAction("Gefundenes Preset")
        self._preset_found_action.triggered.connect(self._apply_found_preset)
        menu.addAction("Preset-Datei wählen…").triggered.connect(self._choose_preset_file)
        menu.addAction("Preset-Ordner wählen…").triggered.connect(self._choose_preset_folder)
        self._preset_apply_btn.setMenu(menu)
        row.addWidget(self._preset_apply_btn)
        self._preset_revert_btn = QPushButton("Rückgängig")
        self._preset_revert_btn.setToolTip(
            "Setzt die Werte und Dateien des zuletzt angewendeten Presets zurück; seitdem "
            "geänderte Werte bleiben."
        )
        self._preset_revert_btn.clicked.connect(self._revert_preset_clicked)
        row.addWidget(self._preset_revert_btn)
        self._preset_export_btn = QPushButton("Preset exportieren…")
        self._preset_export_btn.setToolTip(
            "Schreibt die gespeicherte Einrichtung als Paket (Ordner mit alima_presets.json "
            "und den Dateien); ohne Schlüssel und Tokens."
        )
        self._preset_export_btn.clicked.connect(self.start_preset_export)
        row.addWidget(self._preset_export_btn)
        row.addStretch(1)
        v.addLayout(row)
        self._preset_status = QLabel()
        self._preset_status.setWordWrap(True)
        v.addWidget(self._preset_status)
        self._preset_worker: Optional[_PresetWorker] = None
        self._preset_on_done = None
        self._found_preset: Tuple[Optional[Path], str] = (None, "")
        self.refresh_preset_info()
        return box

    def refresh_preset_info(self, config=None) -> None:
        """Show the found preset and the last applied one - Claude Generated"""
        from src.utils import preset_apply as pa

        try:
            path, source = pa.find_preset()
        except Exception as exc:  # never let a broken lookup break the tab
            from src.utils.error_visibility import log_caught

            log_caught(logger, exc, "Preset suchen")
            path, source = None, ""
        self._found_preset = (path, source)
        if path is None:
            self._preset_found_label.setText(
                "Kein Preset gefunden (gesucht: Umgebungsvariable ALIMA_PRESET, "
                f"{pa.PRESET_FILENAME} im Konfigurationsverzeichnis, Projektverzeichnis)."
            )
        else:
            try:
                preset = pa.load_preset(path, source)
                name = f"{preset.label} {preset.version}".strip()
            except pa.PresetError as exc:
                name = f"nicht lesbar ({exc})"
            self._preset_found_label.setText(
                f"Gefunden: {name}  ·  Quelle: {pa.SOURCE_LABELS.get(source, source)}  ·  {path}"
            )
        self._preset_found_action.setEnabled(path is not None)
        if config is None:
            try:
                from src.utils.config_manager import ConfigManager

                config = ConfigManager().load_config()
            except Exception as exc:  # config unreadable: show the tab anyway
                from src.utils.error_visibility import log_caught

                log_caught(logger, exc, "Konfiguration für den Preset-Stand")
                config = None
        entry = pa.last_applied(config) if config is not None else None
        if entry is None:
            self._preset_last_label.setText("Zuletzt angewendet: keines")
        else:
            when = _local_time(entry.get("applied_at"))
            self._preset_last_label.setText(
                f"Zuletzt angewendet: {entry.get('label') or entry.get('id')} "
                f"{entry.get('version') or ''} am {when} "
                f"({len(entry.get('changes') or [])} Änderungen, "
                f"{len(entry.get('files') or [])} Dateien)"
            )
        self._preset_revert_btn.setEnabled(entry is not None)

    def _set_preset_status(self, text: str, tone: str = "muted") -> None:
        self._preset_status.setText(text)
        set_tone(self._preset_status, tone)

    def _preset_busy(self, text: Optional[str]) -> None:
        """Lock the buttons while a job runs; afterwards "Rückgängig" follows
        the ledger again (``refresh_preset_info``) - Claude Generated"""
        busy = text is not None
        for btn in (self._preset_apply_btn, self._preset_revert_btn, self._preset_export_btn):
            btn.setEnabled(not busy)
        if busy:
            self._set_preset_status(text)
        else:
            self.refresh_preset_info()

    def _run_preset_job(self, job, on_done, busy_text: str) -> None:
        """Run ``job`` in a worker; ``on_done(result)`` in the UI thread, an
        error as a message box and in the status line - Claude Generated"""
        self._preset_busy(busy_text)
        worker = _PresetWorker(job, self)
        self._preset_worker = worker
        self._preset_on_done = on_done
        # Bound methods of this widget: delivered in the UI thread.
        worker.succeeded.connect(self._on_preset_job_done)
        worker.failed.connect(self._on_preset_job_failed)
        worker.start()

    def _on_preset_job_done(self, result) -> None:
        self._preset_busy(None)
        on_done, self._preset_on_done = self._preset_on_done, None
        if on_done is not None:
            on_done(result)

    def _on_preset_job_failed(self, message: str) -> None:
        self._preset_busy(None)
        self._preset_on_done = None
        self._set_preset_status(f"Fehlgeschlagen: {message}", "error")
        QMessageBox.critical(self, "Preset", message)

    def _apply_found_preset(self) -> None:
        path, source = self._found_preset
        if path is not None:
            self.preview_preset(path, source)

    def _choose_preset_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Preset-Datei wählen", "", "Preset (alima_presets.json *.json);;Alle Dateien (*)"
        )
        if path:
            self.preview_preset(path)

    def _choose_preset_folder(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Preset-Ordner wählen")
        if path:
            self.preview_preset(path)

    def preview_preset(self, path, source: str = "") -> None:
        """Build the plan for ``path`` in the worker, then show the preview
        and apply on confirmation - Claude Generated"""
        from src.utils import preset_apply as pa

        def job():
            return pa.build_plan(pa.load_preset(path, source or pa.SOURCE_GIVEN))

        self._run_preset_job(job, self._show_plan, "Vorschau wird erstellt …")

    def _show_plan(self, plan) -> None:
        from src.utils import preset_apply as pa

        dialog = _PresetPlanDialog(plan, self)
        self._preset_dialog = dialog
        if plan.errors:
            self._set_preset_status(f"Preset nicht anwendbar: {len(plan.errors)} Fehler "
                                    "(siehe Vorschau)", "error")
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        steps = plan.steps()

        def done(entry):
            files = len((entry or {}).get("files") or [])
            self._after_preset(steps, f"Preset {plan.label} angewendet: {len(plan.changes)} "
                                      f"Änderungen, {files} Dateien.")

        self._run_preset_job(lambda: pa.apply_plan(plan), done,
                             "Preset wird angewendet (Dateien werden kopiert) …")

    def _revert_preset_clicked(self) -> None:
        from src.utils import preset_apply as pa
        from src.utils.config_manager import ConfigManager

        entry = pa.last_applied(ConfigManager().load_config())
        if entry is None:
            return
        if QMessageBox.question(
            self, "Preset zurücksetzen?",
            f"{entry.get('label') or entry.get('id')} {entry.get('version') or ''} "
            "zurücksetzen?\nJeder Wert, den das Preset gesetzt hat, bekommt seinen vorherigen "
            "Wert, solange er seitdem nicht geändert wurde; neue Dateien werden gelöscht, "
            "ersetzte kommen aus der .bak-Datei zurück. Wird sofort gespeichert.",
        ) != QMessageBox.StandardButton.Yes:
            return

        def done(result):
            self._after_preset(result.steps, "Preset zurückgesetzt.")
            QMessageBox.information(self, "Preset zurückgesetzt", "\n".join(result.lines()))

        self._run_preset_job(pa.revert_last, done, "Preset wird zurückgesetzt …")

    def _after_preset(self, steps, message: str) -> None:
        """Show the saved result without dropping unsaved edits of this tab:
        the steps are written into the dialog state, then ``preset_applied``
        lets the settings dialog follow - Claude Generated"""
        from src.utils.config_manager import ConfigManager
        from src.utils.preset_apply import replay_steps

        saved = ConfigManager().load_config()
        state = copy.deepcopy(saved)
        dialog_state = [i for panel in self._panels.values() for i in panel.collect()]
        state.plugins = dialog_state + [p for p in saved.plugins if p.category not in self._panels]
        state.system_config.enable_code_plugins = self._code_plugins_cb.isChecked()
        replay_steps(state, steps)
        self.load(state)
        self.refresh_preset_info(saved)
        self._set_preset_status(message, "success")
        self.preset_applied.emit(list(steps), saved)

    def start_preset_export(self) -> None:
        """Checklist of instances, folder choice, export in the worker.

        Exports the saved configuration. - Claude Generated
        """
        from src.utils.config_manager import ConfigManager
        from src.utils.preset_export import exportable_instances, export_preset

        config = ConfigManager().load_config()
        dialog = _ExportPresetDialog(exportable_instances(config), self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        selected = dialog.selected_instance_ids()
        pid = dialog.id_edit.text().strip()
        version = dialog.version_edit.text().strip()
        label = dialog.label_edit.text().strip()
        parent_dir = QFileDialog.getExistingDirectory(self, "Ordner für das Paket wählen")
        if not parent_dir:
            return
        dest = Path(parent_dir) / (pid or "alima_preset")

        def done(result):
            self._set_preset_status(f"Exportiert: {result.path}", "success")
            QMessageBox.information(
                self, "Preset exportiert",
                "\n".join(result.lines()) + "\n\nExportiert wurde der gespeicherte Stand "
                "(Einstellungen vorher speichern). Keine Schlüssel und Tokens; Endpunkte und "
                "Dateien vor dem Verteilen prüfen.",
            )

        self._run_preset_job(
            lambda: export_preset(None, dest, selected, preset_id=pid, version=version,
                                  label=label),
            done, "Preset wird exportiert …")

    def _rescan_plugins(self) -> None:
        """Re-run directory discovery with the interactive approval gate - Claude Generated."""
        from src.utils.config_manager import ConfigManager
        from src.utils.plugin_discovery import discover_plugins

        cm = ConfigManager()
        config = cm.load_config()
        # The checkbox governs the scan live (persisted on settings save).
        enable_code = self._code_plugins_cb.isChecked()
        result = discover_plugins(
            config,
            plugins_dir=cm.plugins_dir,
            approve_cb=self._approval_dialog,
            enable_code_plugins=enable_code,
            persist=lambda: cm.save_config(config),
        )
        if result is None:
            QMessageBox.information(
                self, "Plugin-Verzeichnis",
                f"Kein Plugin-Verzeichnis gefunden:\n{cm.plugins_dir}",
            )
            return
        # Der Scan arbeitet auf einer frisch gelesenen Konfiguration. load() mit
        # ihr hätte den Haken "Code-Plugins erlauben" auf den gespeicherten Wert
        # (aus) zurückgesetzt und ungespeicherte Instanzänderungen verworfen; mit
        # OK wäre dann "aus" gespeichert worden. Darum: Dialogzustand behalten,
        # nur neu gefundene Instanzen ergänzen. - Claude Generated
        dialog_state = [i for panel in self._panels.values() for i in panel.collect()]
        have = {(i.category, i.instance_id) for i in dialog_state}
        found = [i for i in result.instances if (i.category, i.instance_id) not in have]
        other = [p for p in config.plugins if p.category not in self._panels]
        config.plugins = dialog_state + found + other
        config.system_config.enable_code_plugins = enable_code
        self._scan_results = {p.manifest.id: p for p in result.plugins}
        self.load(config, plugins_dir=cm.plugins_dir)  # reflect freshly merged instances/types
        lines = []
        for p in result.plugins:
            mark = {"loaded": "✅", "blocked": "⛔", "denied": "🚫", "error": "❌"}.get(p.status, "•")
            detail = f" — {p.detail}" if p.detail else ""
            lines.append(f"{mark} {p.manifest.id} [{p.status}]{detail}")
        if not lines:
            lines = ["(keine Plugins im Verzeichnis)"]
        if not enable_code and any(p.status == "blocked" for p in result.plugins):
            lines.append("")
            lines.append("Hinweis: Code-Plugins sind deaktiviert (enable_code_plugins).")
        QMessageBox.information(self, "Plugin-Scan", "\n".join(lines))

    def _approval_dialog(self, manifest, findings, digest) -> bool:
        """Tier-2 consent dialog: manifest + severity-sorted findings + hash - Claude Generated.

        Default is decline; approval pins the SHA-256 (any file change re-prompts).
        """
        by_sev = {"high": 0, "medium": 0, "low": 0}
        for f in findings:
            by_sev[f.severity] = by_sev.get(f.severity, 0) + 1
        summary = ", ".join(f"{n}× {sev}" for sev, n in by_sev.items() if n) or "keine Findings"

        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Warning)
        box.setWindowTitle("Code-Plugin freigeben?")
        box.setText(
            f"<b>{manifest.label}</b> (id: <code>{manifest.id}</code>, "
            f"Kategorie: {manifest.category})<br>"
            f"Quelle: <code>{manifest.source_dir}</code><br><br>"
            f"Security-Scan: <b>{summary}</b><br><br>"
            "⚠️ Nach Freigabe läuft dieses Plugin <b>in-process mit vollen "
            "Rechten</b> — der Scan ist eine Abschreckung, <b>kein Sandbox</b>. "
            "Nur freigeben, wenn Sie der Quelle vertrauen.<br>"
            "Jede Dateiänderung erfordert eine erneute Freigabe (Hash-Pinning)."
        )
        detail_lines = [str(f) for f in findings] or ["(keine Findings)"]
        detail_lines += ["", f"SHA-256: {digest}"]
        box.setDetailedText("\n".join(detail_lines))
        box.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        box.setDefaultButton(QMessageBox.StandardButton.No)
        box.button(QMessageBox.StandardButton.Yes).setText("Freigeben")
        box.button(QMessageBox.StandardButton.No).setText("Ablehnen")
        return box.exec() == QMessageBox.StandardButton.Yes

    def _warn_on_operator_urls(self, instances: List[PluginInstanceConfig]) -> None:
        """Sanity warnings for configured base URLs (never blocks the save) - Claude Generated.

        Uses ``net_guard.check_operator_url`` posture (a): scheme/parseability
        only — intranet endpoints are legitimate and are NOT flagged.
        """
        from src.core.plugins.schema import URL
        from src.utils.net_guard import check_operator_url

        warnings: List[str] = []
        for inst in instances:
            if not getattr(inst, "enabled", True):
                continue
            try:
                fields = get_category(inst.category).config_fields(inst.provider_id)
            except Exception:
                continue
            for fld in fields:
                if fld.kind != URL:
                    continue
                value = (inst.settings or {}).get(fld.key)
                for msg in check_operator_url(str(value or "")):
                    warnings.append(f"[{inst.label or inst.instance_id}] {msg}")
        if warnings:
            QMessageBox.warning(
                self,
                "Plugin-URLs prüfen",
                "Gespeichert. Hinweise zu konfigurierten Endpunkten:\n\n"
                + "\n".join(f"• {w}" for w in warnings),
            )


# Register the webindex crawl action onto the per-type action registry. Done at
# module load (idempotent — register_type_action appends) so the "Seite indizieren"
# button appears on the webindex lookup instance form. - Claude Generated
try:
    from . import webindex_crawl
    webindex_crawl.register(register_type_action)
except Exception:  # noqa: BLE001 — optional UI action must not break the tab
    import logging as _logging
    _logging.getLogger(__name__).warning(
        "webindex crawl action could not be registered", exc_info=True
    )
