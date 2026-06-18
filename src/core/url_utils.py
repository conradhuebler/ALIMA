"""Canonical URL builders for GND / SWB identifiers + JSON URL extraction.

Claude Generated. Qt-free so CLI/HTTP/GUI/Webapp frontends can all import it.

The chat agent pre-formats clickable URLs at the data/tool layer (analogous to
the existing ``web_url`` field on catalog records) instead of letting the LLM
construct URLs from bare identifiers. This module is the single source of truth
for those URL formats so they cannot drift or be malformed.
"""
from __future__ import annotations

import re
from typing import Any, Set

# d-nb.info is the canonical GND authority; the bare-id form (no /about/lds) is
# the human-facing landing page, e.g. https://d-nb.info/gnd/4047979-1
GND_URL_BASE = "https://d-nb.info/gnd/"

# SWB (BSZ) union-catalog PPN landing page. DB=2.104 is the GND/SWD database,
# INDEXSET=21 selects the default presentation. The session-specific ``sid=`` is
# intentionally omitted — the OPAC resolves a PPN without it.
SWB_PPN_URL_TMPL = (
    "https://swb.bsz-bw.de/DB=2.104/PPNSET?PPN={ppn}&INDEXSET=21"
)

# GND ids are either a plain numeric block (person records, e.g. 118540238) or a
# numeric block + single check character after a hyphen (subject records, e.g.
# 4047979-1; the check char may be the letter X).
_GND_ID_RE = re.compile(r"^[0-9]+(?:-[0-9X])?$")

# PPNs are a numeric block with an optional trailing check character X.
_PPN_RE = re.compile(r"^[0-9]+X?$")


def gnd_url(gnd_id: str) -> str:
    """Return the canonical d-nb.info landing URL for a GND id, or "" if malformed.

    Claude Generated. Returning "" for invalid input lets callers add the field
    unconditionally without fabricating broken links.
    """
    if not gnd_id or not isinstance(gnd_id, str):
        return ""
    gid = gnd_id.strip()
    if not _GND_ID_RE.match(gid):
        return ""
    return f"{GND_URL_BASE}{gid}"


def swb_ppn_url(ppn: str) -> str:
    """Return the SWB/BSZ PPN landing URL for a PPN, or "" if missing/malformed.

    Claude Generated. Only an additional link to a GND record (when a PPN is
    stored); the canonical GND link is always :func:`gnd_url`.
    """
    if not ppn or not isinstance(ppn, str):
        return ""
    p = ppn.strip()
    if not _PPN_RE.match(p):
        return ""
    return SWB_PPN_URL_TMPL.format(ppn=p)


def extract_urls_from_json(data: Any) -> Set[str]:
    """Recursively collect all http(s) URL strings from a parsed JSON value.

    Claude Generated. Moved here from ``PipelineChatPanel`` so both the GUI and
    the webapp can register tool-result URLs as trusted (renderer skips the
    external-link warning for them).
    """
    urls: Set[str] = set()
    if isinstance(data, str):
        if data.startswith(("http://", "https://")):
            urls.add(data)
    elif isinstance(data, dict):
        for v in data.values():
            urls |= extract_urls_from_json(v)
    elif isinstance(data, (list, tuple)):
        for item in data:
            urls |= extract_urls_from_json(item)
    return urls
