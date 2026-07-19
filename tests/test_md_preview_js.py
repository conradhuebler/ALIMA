"""JS markdown-preview contract (alima_render.js) - Claude Generated.

Runs the shared stream formatter under Node (skipped when node is absent):
closed-form constructs render, unclosed ones degrade to literals, HTML is
escaped. Guards the incremental-preview path added in the Chat-UX WP
(appendToken → throttled alimaFormatStreamBlockHtml re-render).
"""

import shutil
import subprocess
import unittest
from pathlib import Path

_JS = Path(__file__).resolve().parents[1] / "src" / "webapp" / "static" / "alima_render.js"

_NODE_SNIPPET = r"""
global.window = {};
const src = require('fs').readFileSync(process.argv[1], 'utf8');
eval(src);
const f = window.alimaFormatStreamBlockHtml;
const cases = [
  ['**unclosed bold', 'unclosed'],
  ['**closed** rest', '<strong>closed</strong>'],
  ['| a | b |', '| a | b |'],
  ['| a | b |\n|---|---|\n| 1 | 2 |', 'stream-md-table'],
  ['<script>x</script>', '&lt;script&gt;'],
];
let fails = 0;
for (const [input, expect] of cases) {
  const out = f(input);
  if (!out.includes(expect)) { console.error('FAIL', JSON.stringify(input)); fails++; }
}
if (typeof appendToken !== 'function' || typeof _renderMdPreview !== 'function') {
  console.error('FAIL preview functions missing'); fails++;
}
// GUI same-window flag: _ensureLinksNewTab must be a no-op (a stamped
// target="_blank" makes Chromium request a popup the GUI never creates,
// so the link click dies silently — July 19 regression).
window.__alimaSameWindowLinks = true;
let touched = false;
_ensureLinksNewTab({ querySelectorAll: () => { touched = true; return []; } });
if (touched) { console.error('FAIL _ensureLinksNewTab ignored same-window flag'); fails++; }
delete window.__alimaSameWindowLinks;
_ensureLinksNewTab({ querySelectorAll: () => { touched = true; return []; } });
if (!touched) { console.error('FAIL _ensureLinksNewTab inactive without flag'); fails++; }
process.exit(fails ? 1 : 0);
"""


@unittest.skipIf(shutil.which("node") is None, "node not available")
class MdPreviewJsTest(unittest.TestCase):
    def test_formatter_contract(self):
        proc = subprocess.run(
            ["node", "-e", _NODE_SNIPPET, str(_JS)],
            capture_output=True, text=True, timeout=30,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr or proc.stdout)


if __name__ == "__main__":
    unittest.main()
