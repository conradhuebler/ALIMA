/* alima_render.js — shared render-layer DOM dispatcher (WP12).
 *
 * Single source of truth for the log/chat/stream/collapsible render functions
 * used by BOTH the GUI (WebLogView, QWebEngineView — inlined at construction
 * and driven via page().runJavaScript) and the webapp (served as a static
 * asset, driven via WebSocket render-events). The functions map 1:1 to the
 * render-event protocol in src/core/render_events.py.
 *
 * Both frontends must provide a container element. Its id defaults to "log";
 * a frontend may override it by setting window.__alimaLogId before this script
 * runs (the webapp uses a dedicated region to avoid id collisions). All
 * functions are append-only and idempotent per element id, so a webapp client
 * can replay the event buffer on reconnect safely. - Claude Generated
 */
window.__autoscroll = true;
var curStream = null;       // live assistant bubble target
var curStreamBlock = null;  // live <details> stream-block body target

function maybeScroll() {
  if (window.__autoscroll) { window.scrollTo(0, document.body.scrollHeight); }
}
function _log() { return document.getElementById(window.__alimaLogId || 'log'); }

function appendBlock(html) {
  var d = document.createElement('div');
  d.className = 'block';
  d.innerHTML = html;
  _log().appendChild(d);
  maybeScroll();
}
function appendCollapsible(id, summary, body, open) {
  // Idempotent per id: a replay/duplicate updates the existing block.
  if (document.getElementById(id)) { updateCollapsible(id, summary, body); return; }
  var det = document.createElement('details');
  det.id = id;
  if (open) det.open = true;
  var s = document.createElement('summary');
  s.innerHTML = summary;
  var b = document.createElement('div');
  b.className = 'tc-body';
  b.innerHTML = body || '';
  det.appendChild(s);
  det.appendChild(b);
  _log().appendChild(det);
  maybeScroll();
}
function updateCollapsible(id, summary, body) {
  var det = document.getElementById(id);
  if (!det) return;
  var s = det.querySelector('summary');
  if (s) s.innerHTML = summary;
  var b = det.querySelector('.tc-body');
  if (b) b.innerHTML = body || '';
  maybeScroll();
}
function openAssistant(header) {
  var wrap = document.createElement('div');
  wrap.className = 'assistant';
  var h = document.createElement('div');
  h.className = 'ahdr';
  h.innerHTML = header;
  var s = document.createElement('div');
  s.className = 'stream';
  wrap.appendChild(h);
  wrap.appendChild(s);
  _log().appendChild(wrap);
  curStream = s;
  maybeScroll();
}
function appendToken(text) {
  if (!curStream) return;
  curStream.appendChild(document.createTextNode(text));
  maybeScroll();
}
function finalizeAssistant(html) {
  if (!curStream) return;
  curStream.className = 'rendered';
  curStream.innerHTML = html;
  curStream = null;
  maybeScroll();
}
function openStreamBlock(id, summary) {
  // A <details>, OPEN while the LLM streams, that collapses on close.
  if (document.getElementById(id)) { curStreamBlock = document.getElementById(id).querySelector('.tc-body'); return; }
  var det = document.createElement('details');
  det.id = id;
  det.open = true;
  var s = document.createElement('summary');
  s.innerHTML = summary;
  var b = document.createElement('div');
  b.className = 'tc-body sl-body';
  det.appendChild(s);
  det.appendChild(b);
  _log().appendChild(det);
  curStreamBlock = b;
  maybeScroll();
}
function appendStreamBlock(text) {
  if (!curStreamBlock) return;
  curStreamBlock.appendChild(document.createTextNode(text));
  maybeScroll();
}
function closeStreamBlock(id, summary, collapse) {
  var det = document.getElementById(id);
  if (det) {
    if (summary) { var s = det.querySelector('summary'); if (s) s.innerHTML = summary; }
    det.open = !collapse;
  }
  curStreamBlock = null;
  maybeScroll();
}
function clearLog() {
  var el = _log();
  if (el) el.innerHTML = '';
  curStream = null;
  curStreamBlock = null;
}
