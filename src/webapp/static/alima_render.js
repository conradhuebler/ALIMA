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
  _ensureLinksNewTab(d);
  _log().appendChild(d);
  maybeScroll();
}

function _ensureLinksNewTab(root) {
  root.querySelectorAll('a:not([target])').forEach(function (a) {
    a.setAttribute('target', '_blank');
    a.setAttribute('rel', 'noopener noreferrer');
  });
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
  _ensureLinksNewTab(b);
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
  if (b) {
    b.innerHTML = body || '';
    _ensureLinksNewTab(b);
  }
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
  _ensureLinksNewTab(curStream);
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
    // WP12: re-render the accumulated raw stream text through the shared
    // formatter so pipe/markdown tables (e.g. RVK-Zweitranking) render
    // identically to the webapp — single source of truth. - Claude Generated
    var body = det.querySelector('.tc-body');
    if (body && !body.dataset.alimaFormatted) {
      var raw = body.textContent || '';
      if (raw.trim()) {
        body.innerHTML = alimaFormatStreamBlockHtml(raw);
        body.dataset.alimaFormatted = '1';
      }
    }
    det.open = !collapse;
    _ensureLinksNewTab(det);
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
function alimaT(key, fallback) {
  // UI-chrome i18n: the embedding page injects window.__alimaI18n (the js.*
  // subset of locales/<lang>.json); fall back to the literal when absent.
  var cat = window.__alimaI18n || {};
  return (cat[key] !== undefined) ? cat[key] : fallback;
}
var typingEl = null;
function showTyping(model) {
  if (!typingEl) {
    typingEl = document.createElement('div');
    typingEl.id = 'alima-typing-indicator';
    typingEl.className = 'typing-indicator';
    _log().appendChild(typingEl);
  }
  typingEl.style.display = 'block';
  typingEl.textContent = '🤖 ' + (model || alimaT('js.typing.model_fallback', 'Modell'))
    + ' ' + alimaT('js.typing.suffix', 'schreibt …');
  maybeScroll();
}
function hideTyping() {
  if (typingEl) typingEl.style.display = 'none';
}

/* ----------------------------------------------------------------------------
 * Shared streamed-content formatter — single source of truth (WP12).
 *
 * Ported verbatim from the webapp's former App.streamBufferToHtml (Jens
 * Mittelbach) so the GUI #log stream-blocks AND the webapp render streamed
 * pipeline text — including the RVK-Zweitranking pipe/markdown tables —
 * identically. app.js delegates to window.alimaFormatStreamBlockHtml. Emitted
 * class names match the webapp's styles.css; alima_render.css styles them for
 * the #log surface. - Claude Generated
 * ------------------------------------------------------------------------- */
function _alimaEscapeHtml(value) {
  return String(value == null ? '' : value)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

function _alimaRenderInlineMarkdown(text) {
  var html = _alimaEscapeHtml(text);
  html = html.replace(/`([^`]+)`/g, '<code>$1</code>');
  html = html.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
  html = html.replace(/\*([^*]+)\*/g, '<em>$1</em>');
  return html;
}

function _alimaNormalizeSpecialLogLine(line) {
  var normalized = String(line || '');
  normalized = normalized.replace(/<\|begin_of_thought\|>/g, '## Analyse');
  normalized = normalized.replace(/<\|end_of_thought\|>/g, '').replace(/\s+$/, '');
  return normalized;
}

function _alimaIsStructuredSectionLabel(line) {
  var trimmed = String(line || '').trim();
  return [
    'DK-Profil für RVK-Zweitranking:',
    'RVK-Kandidaten für DK-basiertes Zweitranking:',
    'RVK-Bewertung aus DK-basiertem Zweitranking:',
    'RVK-Auswahl nach DK-basiertem Zweitranking:',
  ].indexOf(trimmed) !== -1;
}

function _alimaParseMarkdownTableRow(line) {
  var trimmed = String(line || '').trim().replace(/^\|/, '').replace(/\|$/, '');
  return trimmed.split('|').map(function (cell) { return cell.trim(); });
}

function _alimaIsMarkdownTableSeparator(line) {
  var trimmed = String(line || '').trim();
  if (trimmed.indexOf('-') === -1) return false;
  var normalized = trimmed.replace(/^\|/, '').replace(/\|$/, '').trim();
  var cells = normalized.split('|').map(function (c) { return c.trim(); }).filter(Boolean);
  return cells.length >= 2 && cells.every(function (c) { return /^:?-{3,}:?$/.test(c); });
}

function _alimaIsPotentialMarkdownTableRow(line) {
  var trimmed = String(line || '').trim();
  if (!trimmed || trimmed.indexOf('|') === -1) return false;
  var pipeCount = (trimmed.match(/\|/g) || []).length;
  if (pipeCount < 2) return false;
  if (trimmed.indexOf('🔍 ') === 0 || trimmed.indexOf('✅ ') === 0 || trimmed.indexOf('⚠️ ') === 0 || trimmed.indexOf('❌ ') === 0) {
    return false;
  }
  return true;
}

function _alimaIsMarkdownTableStart(lines, index) {
  if (index + 1 >= lines.length) return false;
  return _alimaIsPotentialMarkdownTableRow(lines[index]) && _alimaIsMarkdownTableSeparator(lines[index + 1]);
}

function _alimaIsPipeRecordLine(line) {
  var trimmed = String(line || '').trim();
  if (!trimmed || _alimaIsMarkdownTableSeparator(trimmed)) return false;
  if (trimmed.indexOf('═══') === 0 || trimmed.indexOf('[') === 0 || trimmed.indexOf('ℹ️ ') === 0 || trimmed.indexOf('✅ ') === 0 || trimmed.indexOf('⚠️ ') === 0 || trimmed.indexOf('❌ ') === 0 || trimmed.indexOf('🔎 ') === 0) {
    return false;
  }
  var pipeCount = (trimmed.match(/\|/g) || []).length;
  return pipeCount >= 2;
}

function _alimaIsPipeRecordBlockStart(lines, index) {
  if (index + 1 >= lines.length) return false;
  return _alimaIsPipeRecordLine(lines[index]) && _alimaIsPipeRecordLine(lines[index + 1]);
}

function _alimaSplitPipeRecordLine(line) {
  return String(line || '').split('|').map(function (c) { return c.trim(); }).filter(Boolean);
}

function _alimaRenderPipeRecordTable(lines, startIndex) {
  var rows = [];
  var index = startIndex;
  var keyValueRows = [];
  var renderAsKeyValue = true;

  while (index < lines.length && _alimaIsPipeRecordLine(lines[index])) {
    var row = _alimaSplitPipeRecordLine(lines[index]);
    if (row.length >= 2) {
      rows.push(row);
      var head = row[0] || '';
      var detailCells = row.slice(1);
      var details = [];
      for (var i = 0; i < detailCells.length; i++) {
        var match = detailCells[i].match(/^([^:]+):\s*(.+)$/);
        if (match) {
          details.push({ label: match[1].trim(), value: match[2].trim() });
        } else {
          renderAsKeyValue = false;
        }
      }
      if (details.length > 0) {
        keyValueRows.push({ head: head, details: details });
      } else {
        renderAsKeyValue = false;
      }
    }
    index += 1;
  }

  if (renderAsKeyValue && keyValueRows.length > 0) {
    var bodyHtml = keyValueRows.map(function (r) {
      var detailHtml = r.details.map(function (d) {
        return '<div class="stream-record-detail"><span class="stream-record-detail__label">' +
          _alimaRenderInlineMarkdown(d.label) + '</span><span class="stream-record-detail__value">' +
          _alimaRenderInlineMarkdown(d.value) + '</span></div>';
      }).join('');
      return '<tr><th class="stream-record-head">' + _alimaRenderInlineMarkdown(r.head) + '</th><td>' + detailHtml + '</td></tr>';
    }).join('');
    return {
      html: '<table class="stream-md-table stream-md-table--records stream-md-table--keyvalue"><tbody>' + bodyHtml + '</tbody></table>',
      nextIndex: index,
    };
  }

  var maxColumns = 0;
  for (var j = 0; j < rows.length; j++) { maxColumns = Math.max(maxColumns, rows[j].length); }
  var bodyHtml2 = rows.map(function (r) {
    var padded = [];
    for (var k = 0; k < maxColumns; k++) { padded.push(r[k] || ''); }
    return '<tr>' + padded.map(function (c) { return '<td>' + _alimaRenderInlineMarkdown(c) + '</td>'; }).join('') + '</tr>';
  }).join('');
  return {
    html: '<table class="stream-md-table stream-md-table--records"><tbody>' + bodyHtml2 + '</tbody></table>',
    nextIndex: index,
  };
}

function _alimaRenderMarkdownTable(lines, startIndex) {
  var headers = _alimaParseMarkdownTableRow(lines[startIndex]);
  var rows = [];
  var index = startIndex + 2;
  while (index < lines.length && _alimaIsPotentialMarkdownTableRow(lines[index])) {
    rows.push(_alimaParseMarkdownTableRow(lines[index]));
    index += 1;
  }
  var headerHtml = headers.map(function (c) { return '<th>' + _alimaRenderInlineMarkdown(c) + '</th>'; }).join('');
  var bodyHtml = rows.map(function (r) {
    var cells = headers.map(function (_, ci) { return r[ci] || ''; });
    return '<tr>' + cells.map(function (c) { return '<td>' + _alimaRenderInlineMarkdown(c) + '</td>'; }).join('') + '</tr>';
  }).join('');
  return {
    html: '<table class="stream-md-table"><thead><tr>' + headerHtml + '</tr></thead><tbody>' + bodyHtml + '</tbody></table>',
    nextIndex: index,
  };
}

function alimaFormatStreamBlockHtml(text) {
  var lines = String(text || '').replace(/\r\n/g, '\n').split('\n');
  var html = [];
  var index = 0;

  while (index < lines.length) {
    var line = _alimaNormalizeSpecialLogLine(lines[index]);
    var trimmed = line.trim();

    if (!trimmed) {
      html.push('<div class="stream-log-blank"></div>');
      index += 1;
      continue;
    }
    if (trimmed === '## Analyse') {
      html.push('<h2 class="stream-md-heading stream-md-heading--2">Analyse</h2>');
      index += 1;
      continue;
    }
    if (_alimaIsStructuredSectionLabel(trimmed)) {
      html.push('<h3 class="stream-md-heading stream-md-heading--3">' + _alimaRenderInlineMarkdown(trimmed.replace(/:$/, '')) + '</h3>');
      index += 1;
      continue;
    }
    if (trimmed.indexOf('ℹ️ RVK-Zweitranking') === 0) {
      html.push('<div class="stream-log-line stream-log-line--info">' + _alimaRenderInlineMarkdown(trimmed) + '</div>');
      index += 1;
      continue;
    }
    if (_alimaIsMarkdownTableStart(lines, index)) {
      var t1 = _alimaRenderMarkdownTable(lines, index);
      html.push(t1.html);
      index = t1.nextIndex;
      continue;
    }
    if (_alimaIsPipeRecordBlockStart(lines, index)) {
      var t2 = _alimaRenderPipeRecordTable(lines, index);
      html.push(t2.html);
      index = t2.nextIndex;
      continue;
    }
    if (/^#{1,3}\s+/.test(trimmed)) {
      var level = Math.min(3, trimmed.match(/^#+/)[0].length);
      var content = trimmed.replace(/^#{1,3}\s+/, '');
      html.push('<h' + level + ' class="stream-md-heading stream-md-heading--' + level + '">' + _alimaRenderInlineMarkdown(content) + '</h' + level + '>');
      index += 1;
      continue;
    }
    if (/^\s*[-*]\s+/.test(line)) {
      var items = [];
      while (index < lines.length && /^\s*[-*]\s+/.test(lines[index])) {
        items.push(lines[index].replace(/^\s*[-*]\s+/, ''));
        index += 1;
      }
      var itemsHtml = items.map(function (it) { return '<li>' + _alimaRenderInlineMarkdown(it) + '</li>'; }).join('');
      html.push('<ul class="stream-md-list">' + itemsHtml + '</ul>');
      continue;
    }
    html.push('<div class="stream-log-line">' + _alimaRenderInlineMarkdown(line) + '</div>');
    index += 1;
  }
  return html.join('');
}
window.alimaFormatStreamBlockHtml = alimaFormatStreamBlockHtml;
