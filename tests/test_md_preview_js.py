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
// Copy helpers (result card). The attribute names what to copy; folded
// <details> travel with it, the buttons themselves do not.
// The fake behaves like the DOM does: text inside a CLOSED <details> is not
// part of innerText, and a removed button contributes nothing. So the
// assertions below actually depend on the clone being manipulated.
function fakeNode(opts) {
  return {
    cloneNode: function () {
      return {
        buttons: opts.buttons || 0,
        detailsOpen: false,
        querySelectorAll: function (sel) {
          const holder = this;
          if (sel.indexOf('details') >= 0) {
            return opts.folded
              ? [{ setAttribute: () => { holder.detailsOpen = true; } }]
              : [];
          }
          return Array.from({ length: holder.buttons }, () => ({
            remove: () => { holder.buttons--; },
          }));
        },
        get innerText() {
          let out = '  ' + (opts.text || '') + '  ';
          if (opts.folded && this.detailsOpen) { out += '\n\n\n\n' + opts.folded; }
          if (this.buttons > 0) { out += '\nKopieren'; }
          return out;
        },
      };
    },
  };
}
const copied = _alimaCopyableText(fakeNode({ text: 'a', buttons: 2, folded: 'b' }));
if (copied !== 'a\n\nb') { console.error('FAIL copyable text', JSON.stringify(copied)); fails++; }
const noFold = _alimaCopyableText(fakeNode({ text: 'a', buttons: 1 }));
if (noFold !== 'a') { console.error('FAIL button stripping', JSON.stringify(noFold)); fails++; }

const scope = {
  querySelector: (sel) => ({ sel: sel }),
  _isScope: true,
};
const btnWhole = { closest: (s) => (s === '.alima-copy-scope' ? scope : null), getAttribute: () => null };
if (_alimaCopyTarget(btnWhole) !== scope) { console.error('FAIL whole-scope copy'); fails++; }
const btnPart = { closest: (s) => (s === '.alima-copy-scope' ? scope : null), getAttribute: () => '.x' };
if (_alimaCopyTarget(btnPart).sel !== '.x') { console.error('FAIL selector copy'); fails++; }

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
