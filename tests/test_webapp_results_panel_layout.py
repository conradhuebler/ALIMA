"""The results summary must survive a collapsed input zone - Claude Generated.

The webapp collapses ``.input-zone-body`` when a run starts (``max-height: 0``).
While ``#results-panel`` lived inside that body, showing the summary at the end
of a run required re-expanding the whole input, which pushed the result out of
view: the operator had to collapse the input again to read what came out.

The panel therefore sits OUTSIDE the collapsing body. This test holds that
structure, because moving it back inside brings the old behaviour with it and
nothing else would notice.
"""

import unittest
from html.parser import HTMLParser
from pathlib import Path

_TEMPLATE = Path(__file__).resolve().parent.parent / "src/webapp/templates/webapp.html"


class _AncestorFinder(HTMLParser):
    """Record the open-tag stack at the point where ``target_id`` appears."""

    def __init__(self, target_id: str):
        super().__init__()
        self.target_id = target_id
        self._stack: list = []
        self.ancestors: list = []
        self.attrs_at_target: dict = {}
        self.found = False

    def handle_starttag(self, tag, attrs):
        attrs_d = dict(attrs)
        if not self.found and attrs_d.get("id") == self.target_id:
            self.found = True
            self.ancestors = list(self._stack)
            self.attrs_at_target = attrs_d
        # void elements never open a scope
        if tag not in ("br", "hr", "img", "input", "meta", "link"):
            self._stack.append((tag, attrs_d))

    def handle_endtag(self, tag):
        for i in range(len(self._stack) - 1, -1, -1):
            if self._stack[i][0] == tag:
                del self._stack[i:]
                break


class TestResultsPanelOutsideCollapsingBody(unittest.TestCase):
    def setUp(self):
        self.parser = _AncestorFinder("results-panel")
        self.parser.feed(_TEMPLATE.read_text(encoding="utf-8"))

    def test_panel_exists(self):
        self.assertTrue(self.parser.found, "#results-panel missing from the template")

    def test_panel_is_not_inside_the_collapsing_input_body(self):
        classes = [a.get("class", "") for _tag, a in self.parser.ancestors]
        self.assertNotIn(
            "input-zone-body",
            " ".join(classes).split(),
            "#results-panel is inside .input-zone-body again — a finished run "
            "would have to re-open the input to show the summary",
        )

    def test_panel_still_lives_in_the_input_zone(self):
        ids = [a.get("id") for _tag, a in self.parser.ancestors]
        self.assertIn("input-zone", ids)

    def test_panel_carries_its_own_styling_hook(self):
        """Outside the body it no longer inherits `.input-zone-body .card`."""
        self.assertIn("input-zone-results", self.parser.attrs_at_target.get("class", ""))


if __name__ == "__main__":
    unittest.main()
