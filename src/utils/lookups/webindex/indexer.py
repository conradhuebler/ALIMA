"""Website crawler + indexer for the webindex lookup plugin - Claude Generated.

Fills the :class:`WebIndexStore` by crawling every URL that is a child of a
configured ``base_url`` (BFS, same host + path prefix). Each page is fetched via
the shared guarded fetch (``fetch_guarded_response`` — SSRF guard, redirect-per-hop,
size cap), its main content extracted with BeautifulSoup (reusing the
``scrape_url`` heuristic), and its keywords derived from two sources:

* **Meta/heading keywords** (deterministic, free): ``meta[name=keywords]``,
  ``meta[name=description]`` tokens, ``<title>`` and ``h1``–``h3``.
* **LLM keywords** (optional): one ``generate_response`` call per page that returns a
  JSON list of subject keywords — skipped silently if no ``llm_service``/provider/model
  is configured or the call fails.

The crawler is an **operator** action (driven by the ``alima webindex crawl`` CLI),
never an agent tool — the agent only reads the index. ``fetch_func`` is injectable so
tests run without network.
"""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.parse import urldefrag, urljoin, urlparse

from .store import WebIndexStore, _normalize_keyword

_DEFAULT_UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"


# --- text + link extraction ----------------------------------------------- #
def _extract_main_text(soup) -> Tuple[str, str]:
    """Return (title, cleaned_main_text) mirroring ``scrape_url``'s heuristic.

    Strips ``script/style/nav/header/footer/aside``, prefers
    ``main``/``article``/``div.content``, collapses whitespace. - Claude Generated
    """
    title_tag = soup.find("title")
    title = title_tag.get_text(strip=True) if title_tag else ""
    for tag in soup(["script", "style", "nav", "header", "footer", "aside"]):
        tag.decompose()
    main_content = (
        soup.find("main") or soup.find("article") or soup.find("div", class_="content")
    )
    if main_content:
        text = main_content.get_text(separator="\n", strip=True)
    else:
        body = soup.find("body")
        text = (
            body.get_text(separator="\n", strip=True)
            if body
            else soup.get_text(separator="\n", strip=True)
        )
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    text = re.sub(r" +", " ", text)
    # Fall back to the first heading if no <title>.
    if not title:
        h1 = soup.find("h1")
        title = h1.get_text(strip=True) if h1 else ""
    return title, text


def _discover_child_links(soup, base_url: str) -> List[str]:
    """Absolute, de-fragmented child URLs under ``base_url`` (same host + path prefix).

    Filters out non-http, different hosts, and paths that do not start with the
    base path. Deduplicates while preserving discovery order. - Claude Generated
    """
    base = urlparse(base_url)
    base_prefix = (base.scheme, base.netloc, (base.path or "/").rstrip("/"))
    out: List[str] = []
    seen = set()
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if not href or href.startswith(("#", "mailto:", "tel:", "javascript:", "data:")):
            continue
        absolute = urldefrag(urljoin(base_url, href))[0]
        parsed = urlparse(absolute)
        if parsed.scheme not in ("http", "https"):
            continue
        if parsed.netloc != base.netloc:
            continue
        child_path = (parsed.path or "/").rstrip("/")
        # Child must live under the base path prefix.
        if not child_path.startswith(base_prefix[2]):
            continue
        if absolute in seen:
            continue
        seen.add(absolute)
        out.append(absolute)
    return out


# --- keyword extraction --------------------------------------------------- #
def _extract_meta_keywords(soup) -> List[Tuple[str, str]]:
    """Deterministic keywords from HTML meta tags + headings. Returns
    ``(display, source)`` tuples. - Claude Generated"""
    out: List[Tuple[str, str]] = []

    def add(raw: str, source: str) -> None:
        for piece in re.split(r"[,;]", str(raw or "")):
            kw = piece.strip()
            if kw and _normalize_keyword(kw):
                out.append((kw, source))

    kw_meta = soup.find("meta", attrs={"name": "keywords"})
    if kw_meta and kw_meta.get("content"):
        add(kw_meta["content"], "meta")
    desc_meta = soup.find("meta", attrs={"name": "description"})
    if desc_meta and desc_meta.get("content"):
        # Description is a sentence, not a keyword list — split on common
        # delimiters but only keep short fragments (<=6 words) as candidate terms.
        for piece in re.split(r"[,;]", str(desc_meta["content"])):
            frag = piece.strip()
            if frag and len(frag.split()) <= 6 and _normalize_keyword(frag):
                out.append((frag, "meta"))
    title = soup.find("title")
    if title:
        t = title.get_text(strip=True)
        if t and _normalize_keyword(t):
            out.append((t, "heading"))
    for level in ("h1", "h2", "h3"):
        for h in soup.find_all(level):
            txt = h.get_text(strip=True)
            if txt and _normalize_keyword(txt):
                out.append((txt, "heading"))
    return out


# --- person extraction ----------------------------------------------------- #
# Matches an obfuscated "spamspan"-style email (a well-known CMS/Drupal anti-
# scraping pattern: local-part and domain each in their own span, with literal
# "[dot]" markers instead of ".") - Claude Generated.
_EMAIL_OBFUSCATION_CLASS = "spamspan"
_PHONE_RE = re.compile(r"(\+?\d[\d\s\-/]{6,}\d)")


def _deobfuscate_spamspan(span) -> str:
    """Reconstruct a plain email address from a spamspan-obfuscated span.

    Returns "" if the expected inner structure (``span.u`` local-part,
    ``span.d`` domain) is missing — never guesses. - Claude Generated
    """
    user = span.find("span", class_="u")
    domain = span.find("span", class_="d")
    if not user or not domain:
        return ""
    user_text = re.sub(r"\s*\[dot\]\s*", ".", user.get_text())
    domain_text = re.sub(r"\s*\[dot\]\s*", ".", domain.get_text())
    user_text = user_text.strip()
    domain_text = domain_text.strip()
    if not user_text or not domain_text:
        return ""
    return f"{user_text}@{domain_text}"


def _extract_people(soup) -> List[Dict[str, str]]:
    """Deterministic person-record extraction from staff/team pages.

    Finds every obfuscated email (spamspan) or plain ``mailto:`` link, then
    reads the name from the nearest ``<strong>`` in the same table row/paragraph
    and the role/phone from the surrounding text. Purely structural — no LLM,
    no invented fields; a record with no discoverable name is dropped rather
    than guessed. This exists because the chat agent previously answered
    "is X UB staff?" / "what's their email?" by pattern-matching prose, which
    both missed real staff and fabricated plausible-looking emails. - Claude
    Generated
    """
    people: List[Dict[str, str]] = []
    seen_containers = set()

    email_spans = soup.find_all("span", class_=_EMAIL_OBFUSCATION_CLASS)
    for span in email_spans:
        email = _deobfuscate_spamspan(span)
        if not email:
            continue
        # The row (table) or paragraph (accordion-style) holding this contact.
        container = span.find_parent("tr") or span.find_parent("p")
        if container is None:
            continue
        if id(container) in seen_containers:
            continue
        seen_containers.add(id(container))

        name_cell = container.find("td") if container.name == "tr" else container
        strong = name_cell.find("strong") if name_cell else None
        if strong is None:
            continue  # no reliable name anchor — skip rather than guess

        # The full name often continues as plain text after </strong> on the
        # same line (e.g. "<strong>Nagel</strong>, Stefanie Dr." in a table
        # cell, vs. "<strong>Dr. Meyer, Julia</strong>" alone in the accordion
        # layout) — take everything in name_cell up to the first <br>, not just
        # the <strong> text, or the surname-only match would miss "Stefanie"
        # entirely and break find_person("Stefanie Nagel"). - Claude Generated
        first_br = name_cell.find("br")
        if first_br is not None:
            name_parts = [str(s) for s in first_br.find_previous_siblings(string=True)][::-1]
            name = (strong.get_text(strip=True) + " " + " ".join(
                p.strip() for p in name_parts if p.strip()
            )).strip()
        else:
            name = name_cell.get_text(strip=True)
        if not name:
            name = strong.get_text(strip=True)
        if not name:
            continue

        full_text = container.get_text(separator="|", strip=True)
        phone_m = _PHONE_RE.search(full_text)
        phone = phone_m.group(1).strip() if phone_m else ""

        # Role = the text segment right after the name, before phone/email
        # fragments (e.g. "Direktorin" between "Dr. Meyer, Julia" and the
        # phone number). Best-effort: empty string if nothing distinct found.
        segments = [s.strip() for s in full_text.split("|") if s.strip()]
        role = ""
        try:
            name_idx = segments.index(name)
            for seg in segments[name_idx + 1:]:
                if _PHONE_RE.search(seg) or "@" in seg or "[at]" in seg.lower():
                    break
                role = seg
                break
        except ValueError:
            pass

        people.append({"name": name, "role": role, "email": email, "phone": phone})

    return people


def _extract_llm_keywords(
    text: str,
    keyword_extractor: Optional[Callable[[str, int], List[str]]],
    max_keywords: int,
) -> List[str]:
    """LLM-extracted keywords via the injected extractor (the webindex_keywords
    workflow). Returns [] when no extractor is configured or the call fails — the
    crawl then keeps the deterministic meta/heading keywords. - Claude Generated"""
    if keyword_extractor is None:
        return []
    try:
        return list(keyword_extractor(text, int(max_keywords)) or [])
    except Exception as e:  # noqa: BLE001 — LLM call must never break the crawl
        logging.getLogger(__name__).warning(f"LLM keyword extraction failed: {e}")
        return []


def _parse_keyword_list(raw: str, max_keywords: int) -> List[str]:
    """Defensively parse an LLM JSON-list response into clean keywords - Claude Generated."""
    if not raw:
        return []
    candidates: List[str] = []
    # Try strict JSON first.
    try:
        data = json.loads(raw.strip())
        if isinstance(data, list):
            candidates = [str(x) for x in data if str(x).strip()]
    except json.JSONDecodeError:
        # Fallback: grab the first JSON-looking array in the text.
        m = re.search(r"\[.*?\]", raw, re.DOTALL)
        if m:
            try:
                data = json.loads(m.group(0))
                if isinstance(data, list):
                    candidates = [str(x) for x in data if str(x).strip()]
            except json.JSONDecodeError:
                pass
    if not candidates:
        # Last resort: split on commas/newlines if the model ignored the JSON rule.
        candidates = [p.strip().strip('"') for p in re.split(r"[\n,]", raw) if p.strip()]
    out: List[str] = []
    seen = set()
    for c in candidates:
        kw = c.strip().strip('"').strip()
        n = _normalize_keyword(kw)
        if n and n not in seen:
            seen.add(n)
            out.append(kw)
        if len(out) >= max_keywords:
            break
    return out


# --- crawl ---------------------------------------------------------------- #
def crawl_site(
    store: WebIndexStore,
    *,
    base_url: str,
    max_depth: int = 2,
    max_pages: int = 50,
    include_re: Optional[str] = None,
    exclude_re: Optional[str] = None,
    keyword_extractor: Optional[Callable[[str, int], List[str]]] = None,
    fetch_timeout: int = 20,
    user_agent: str = _DEFAULT_UA,
    min_chars: int = 50,
    max_keywords: int = 15,
    fetch_func: Optional[Callable[..., Any]] = None,
    progress_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    should_stop: Optional[Callable[[], bool]] = None,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Crawl ``base_url`` + its children and populate ``store``. - Claude Generated.

    ``fetch_func`` defaults to :func:`fetch_guarded_response` and is injectable for
    tests. ``keyword_extractor`` (built from the ``webindex_keywords`` workflow via
    :mod:`.keywords`) supplies the LLM keywords; ``None`` ⇒ meta/heading only.
    ``should_stop`` lets the GUI worker cancel the crawl between pages. Returns a
    stats dict (pages_indexed, pages_skipped, errors, urls).
    """
    from bs4 import BeautifulSoup

    logger = logging.getLogger(__name__)
    fetch = fetch_func or _default_fetch

    base_url = base_url.strip()
    if not base_url.startswith(("http://", "https://")):
        raise ValueError(f"base_url must be http(s): {base_url!r}")

    include_re_c = re.compile(include_re) if include_re else None
    exclude_re_c = re.compile(exclude_re) if exclude_re else None

    visited: set = set()
    queue: List[Tuple[str, int]] = [(base_url, 0)]
    pages_indexed = 0
    pages_skipped = 0
    errors: List[str] = []
    indexed_urls: List[str] = []

    while queue and len(visited) < max_pages:
        if should_stop and should_stop():
            logger.info("Crawl cancelled by caller (should_stop).")
            break
        url, depth = queue.pop(0)
        if url in visited:
            continue
        visited.add(url)
        # include/exclude scope discovered children; the seed (depth 0) is always
        # fetched so the crawl has an entry point even under a restrictive filter.
        if depth > 0:
            if include_re_c and not include_re_c.search(url):
                continue
            if exclude_re_c and exclude_re_c.search(url):
                continue

        try:
            resp = fetch(url, user_agent=user_agent, timeout=fetch_timeout)
        except Exception as e:  # noqa: BLE001 — one bad URL must not kill the crawl
            errors.append(f"{url}: {e}")
            logger.warning(f"Fetch failed for {url}: {e}")
            continue

        content_type = ""
        status = None
        content: bytes = b""
        try:
            content_type = (resp.headers.get("Content-Type") or "").lower()
            status = int(getattr(resp, "status_code", 0) or 0)
            content = resp.content or b""
        except Exception:
            content = b""

        looks_pdf = "application/pdf" in content_type or url.lower().split("?")[0].endswith(".pdf")

        meta_pairs: List[Tuple[str, str]] = []
        people: List[Dict[str, str]] = []
        try:
            if looks_pdf:
                title, text, child_urls = _index_pdf(url, content, fetch_timeout)
            else:
                soup = BeautifulSoup(content, "html.parser")
                title, text = _extract_main_text(soup)
                meta_pairs = _extract_meta_keywords(soup)
                people = _extract_people(soup)
                child_urls = (
                    _discover_child_links(soup, base_url) if depth < max_depth else []
                )
        except Exception as e:  # noqa: BLE001
            errors.append(f"{url}: parse {e}")
            logger.warning(f"Parse failed for {url}: {e}")
            continue

        truncated = len(text) > 200_000
        if truncated:
            text = text[:200_000]

        if not text or len(text.strip()) < min_chars:
            pages_skipped += 1
        elif dry_run:
            # Discover-only: record the URL without writing to the store.
            indexed_urls.append(url)
            logger.info(f"[dry-run] would index {url} ({len(text)} chars)")
        else:
            store.upsert_page(
                url=url,
                base_url=base_url,
                title=title,
                text=text,
                http_status=status,
                content_type=content_type,
                text_truncated=truncated,
            )
            links = _build_keyword_links(
                text=text,
                meta_pairs=meta_pairs,
                keyword_extractor=keyword_extractor,
                max_keywords=max_keywords,
            )
            store.set_page_keywords(url, links)
            store.set_page_people(url, people)
            pages_indexed += 1
            indexed_urls.append(url)
            logger.info(
                f"Indexed {url} ({len(text)} chars, {len(links)} keywords, "
                f"{len(people)} people)"
            )

        if progress_callback:
            progress_callback(url, {"depth": depth, "chars": len(text), "status": status})

        for child in child_urls:
            if child not in visited:
                queue.append((child, depth + 1))

    return {
        "base_url": base_url,
        "pages_indexed": pages_indexed,
        "pages_skipped": pages_skipped,
        "errors": errors,
        "indexed_urls": indexed_urls,
        "visited": len(visited),
    }


def _default_fetch(url: str, *, user_agent: str, timeout: int):
    """Production fetch entry point — the shared SSRF-guarded fetch - Claude Generated."""
    from src.utils.input_sources.url_fetch import fetch_guarded_response

    return fetch_guarded_response(url, user_agent=user_agent, timeout=timeout)


def _index_pdf(url: str, content: bytes, timeout: int) -> Tuple[str, str, List[str]]:
    """Extract text from a PDF via ``pdf_extractor``; no child links (PDFs have none).
    - Claude Generated"""
    from src.utils.pdf_extractor import extract_text as _pdf_extract

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name
    try:
        result = _pdf_extract(tmp_path, max_chars=None)
        return os.path.basename(url.split("?")[0]), result.get("text", "") or "", []
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def _build_keyword_links(
    *,
    text: str,
    meta_pairs: Optional[List[Tuple[str, str]]] = None,
    keyword_extractor: Optional[Callable[[str, int], List[str]]] = None,
    max_keywords: int = 15,
) -> List[Tuple[str, float, str]]:
    """Combine meta/heading + LLM keywords into ``(display, weight, source)`` tuples.

    Meta/heading keywords get weight 1.0; LLM keywords get weight 1.5 (the
    operator-chosen "LLM-Ergänzung" boost). Duplicates across sources keep the
    higher weight/source. - Claude Generated
    """
    # norm -> (display, weight, source)
    links: Dict[str, Tuple[str, float, str]] = {}
    if meta_pairs is None:
        meta_pairs = []
    for display, source in meta_pairs:
        norm = _normalize_keyword(display)
        if not norm:
            continue
        if norm not in links or links[norm][1] < 1.0:
            links[norm] = (display, 1.0, source)
    llm_kws = _extract_llm_keywords(text, keyword_extractor, max_keywords)
    for display in llm_kws:
        norm = _normalize_keyword(display)
        if not norm:
            continue
        if norm not in links or links[norm][1] < 1.5:
            links[norm] = (display, 1.5, "llm")
    return [(display, weight, source) for (display, weight, source) in links.values()]