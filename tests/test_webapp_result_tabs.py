"""Result tabs of the webapp: which render event goes where - Claude Generated.

The webapp shows the log and the result cards in tabs (``webapp.html``,
``app.js``). The cards come from the shared render layer unchanged; the server
adds the card's kind to its block event as ``card``
(``render_bridge.WebSocketRenderTransport``), and ``app.js`` sorts by it
(``RESULT_TAB_BY_CARD``). Held here:

* every card kind the server emits has an entry in that table, and every tab
  of the table has its tab, panel and region in the template (inside ``#log``,
  so the ``#log``-scoped chrome of ``alima_render.css`` reaches it);
* the tag rides on the card's block only, ``kind`` stays ``html_block``;
* the GUI side is unchanged: its transport ignores the extra field, and
  ``render_pipeline_result`` still opens with the completion line unless told
  otherwise (only a loaded file is).
"""
from __future__ import annotations

import os
import re
import unittest
from html.parser import HTMLParser
from pathlib import Path
from unittest.mock import MagicMock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parent.parent
APP_JS = ROOT / "src/webapp/static/app.js"
TEMPLATE = ROOT / "src/webapp/templates/webapp.html"
CARD_PRODUCERS = (ROOT / "src/utils/pipeline_formatters.py", ROOT / "src/webapp/routers/analysis.py")


def _js_object(name: str) -> dict:
    """``const NAME = { key: 'value', … };`` from app.js as a dict."""
    text = APP_JS.read_text(encoding="utf-8")
    body = re.search(r"const %s = \{(.*?)\};" % name, text, re.S).group(1)
    return dict(re.findall(r"(\w+):\s*'([\w-]+)'", body))


def _js_list(name: str) -> list:
    text = APP_JS.read_text(encoding="utf-8")
    body = re.search(r"const %s = \[(.*?)\];" % name, text, re.S).group(1)
    return re.findall(r"'([\w-]+)'", body)


def _emitted_card_kinds() -> set:
    """``kind="…"`` of every render_html_block / render_markdown_block call."""
    kinds = set()
    for path in CARD_PRODUCERS:
        text = path.read_text(encoding="utf-8")
        for call in re.finditer(r"render_(?:html|markdown)_block\((.*?)\)\n", text, re.S):
            kinds.update(re.findall(r'kind="(\w+)"', call.group(1)))
    return kinds


class _Elements(HTMLParser):
    """id → (tag, attrs, ancestor ids) of every element with an id."""

    VOID = {"br", "hr", "img", "input", "meta", "link"}

    def __init__(self):
        super().__init__()
        self.stack = []
        self.by_id = {}

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if attrs.get("id"):
            self.by_id[attrs["id"]] = (tag, attrs, [a.get("id") for _t, a in self.stack])
        if tag not in self.VOID:
            self.stack.append((tag, attrs))

    def handle_endtag(self, tag):
        for i in range(len(self.stack) - 1, -1, -1):
            if self.stack[i][0] == tag:
                del self.stack[i:]
                break


class TabMapping(unittest.TestCase):
    def test_every_card_kind_the_server_emits_has_a_tab(self):
        emitted = _emitted_card_kinds()
        self.assertTrue({"final_result", "dk_classifications", "dk_statistics",
                         "workflow_report", "dk_search"} <= emitted, emitted)
        mapping = _js_object("RESULT_TAB_BY_CARD")
        self.assertEqual(sorted(emitted - set(mapping)), [],
                         "a card kind without a tab lands in the log")

    def test_the_mapping_points_at_existing_tabs(self):
        tabs = _js_list("RESULT_TABS")
        self.assertEqual(tabs[0], "log")
        self.assertTrue(set(_js_object("RESULT_TAB_BY_CARD").values()) <= set(tabs))
        self.assertEqual(set(_js_object("RESULT_REGION_ID")), set(tabs))

    def test_the_template_has_tab_panel_and_region_for_each(self):
        parser = _Elements()
        parser.feed(TEMPLATE.read_text(encoding="utf-8"))
        regions = _js_object("RESULT_REGION_ID")
        tablist = parser.by_id["result-tablist"]
        self.assertEqual(tablist[1].get("role"), "tablist")
        for tab in _js_list("RESULT_TABS"):
            button = parser.by_id[f"result-tab-{tab}"]
            panel = parser.by_id[f"result-panel-{tab}"]
            region = parser.by_id[regions[tab]]
            self.assertEqual(button[1].get("role"), "tab")
            self.assertIn("result-tablist", button[2])
            self.assertEqual(button[1].get("aria-controls"), f"result-panel-{tab}")
            self.assertEqual(button[1].get("aria-selected"), "true" if tab == "log" else "false")
            self.assertEqual(panel[1].get("role"), "tabpanel")
            self.assertEqual(panel[1].get("aria-labelledby"), f"result-tab-{tab}")
            self.assertIn("log", panel[2], "panels sit inside #log for the shared chrome")
            self.assertIn(f"result-panel-{tab}", region[2])
            self.assertEqual("hidden" in panel[1], tab != "log")

    def test_the_chat_input_is_outside_the_panels(self):
        parser = _Elements()
        parser.feed(TEMPLATE.read_text(encoding="utf-8"))
        ancestors = parser.by_id["chat-input"][2]
        self.assertIn("chat-input-area", ancestors)
        self.assertNotIn("log", ancestors)

    def test_the_default_render_target_is_the_log_region(self):
        text = APP_JS.read_text(encoding="utf-8")
        self.assertIn("window.__alimaLogId = RESULT_REGION_ID.log;", text)


class CardTagging(unittest.TestCase):
    def _session_renderer(self):
        from src.webapp.render_bridge import _build_session_renderer
        from src.webapp.session_state import Session

        session = Session("tabs")
        return session, _build_session_renderer(session)

    def test_result_cards_carry_their_kind(self):
        from tests.test_webapp_result_load import fixture_state
        from src.utils.pipeline_formatters import render_pipeline_result

        state = fixture_state()
        state.report_markdown = "| a | b |\n|---|---|\n| 1 | 2 |"
        session, renderer = self._session_renderer()
        render_pipeline_result(renderer, state)
        blocks = [e for e in session.render_buffer if e["type"] == "block"]
        self.assertEqual([e.get("card") for e in blocks if e.get("card")],
                         ["final_result", "dk_classifications", "workflow_report"])
        for event in blocks:
            if event.get("card"):
                self.assertEqual(event["kind"], "html_block")
            else:
                self.assertEqual(event["kind"], "pipeline_log")

    def test_the_tag_ends_with_the_card(self):
        session, renderer = self._session_renderer()
        renderer.render_html_block("<div>a</div>", kind="dk_search")
        renderer.render_pipeline_log("danach", "info")
        tid = renderer.render_tool_call("t", {})
        renderer.render_tool_result(tid, "ok")
        renderer.render_html_block("<div>b</div>")
        cards = [e.get("card") for e in session.render_buffer]
        self.assertEqual(cards[0], "dk_search")
        self.assertEqual(set(cards[1:]), {None})

    def test_nested_tags_restore_the_outer_one(self):
        from src.webapp.render_bridge import WebSocketRenderTransport
        from src.webapp.session_state import Session

        session = Session("nest")
        transport = WebSocketRenderTransport(session)
        with transport.card("final_result"):
            with transport.card("dk_search"):
                transport.send({"type": "block", "html": "inner"})
            transport.send({"type": "block", "html": "outer"})
            transport.send({"type": "collapsible", "id": "c", "summary": "", "body": ""})
        transport.send({"type": "block", "html": "after"})
        self.assertEqual([e.get("card") for e in session.render_buffer],
                         ["dk_search", "final_result", None, None])

    def test_the_gui_transport_ignores_the_tag(self):
        from src.ui.render_transport import WebLogViewTransport

        view = MagicMock()
        WebLogViewTransport(view).send(
            {"type": "block", "html": "<div>x</div>", "kind": "html_block", "card": "final_result"})
        view.append_block.assert_called_once_with("<div>x</div>")


class CompletionLine(unittest.TestCase):
    def _calls(self, **kwargs):
        from tests.test_webapp_result_load import fixture_state
        from src.utils.pipeline_formatters import render_pipeline_result

        renderer = MagicMock()
        render_pipeline_result(renderer, fixture_state(), **kwargs)
        return renderer.mock_calls

    def test_default_opens_with_the_completion_line(self):
        first = self._calls()[0]
        self.assertEqual(first[0], "render_pipeline_log")
        self.assertIn("Pipeline vollständig abgeschlossen", first[1][0])

    def test_a_loaded_result_has_the_same_cards_without_it(self):
        default = self._calls()
        loaded = self._calls(announce_completion=False)
        self.assertEqual(loaded, default[1:])


if __name__ == "__main__":
    unittest.main()


def test_page_versions_static_assets():
    """Claude Generated: the page must reference app.js/styles with ?v=, so a cached old script is never combined with the new HTML."""
    import re
    from fastapi.testclient import TestClient
    from src.webapp.app import app
    html = TestClient(app).get("/webapp").text
    for name in ("app.js", "styles.css", "alima_render.js", "alima_render.css"):
        assert re.search(rf'/static/{re.escape(name)}\?v=\d+', html), name
