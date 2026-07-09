"""CLI handler for the ``alima webindex`` command group - Claude Generated.

Operator surface for the website-RAG index: crawl a base URL into the index DB,
inspect the keyword catalogue / stats, and run a retrieval query from the shell.
The agent-facing tools (``search_webindex`` / ``fetch_page`` /
``list_webindex_keywords``) are auto-generated from the lookup plugin's
``mcp_tool_specs`` and need no CLI counterpart beyond the ``search`` convenience
command here.
"""

from __future__ import annotations

import logging
import sys
from typing import Any, Dict, Optional

from src.utils.config_manager import ConfigManager


def _instance_settings(config, instance_id: Optional[str], logger) -> Dict[str, Any]:
    """Resolve the webindex lookup instance's settings from config.

    Falls back to an empty dict (defaults / flags) when no instance is configured,
    so the CLI works before the operator has created a plugin instance. - Claude
    Generated
    """
    instances = [p for p in config.enabled_instances_for("lookup") if p.provider_id == "webindex"]
    if instance_id:
        match = [p for p in instances if p.instance_id == instance_id]
        if not match:
            # Also search disabled instances so the operator can inspect a disabled one.
            match = [p for p in config.plugins
                     if p.category == "lookup" and p.provider_id == "webindex"
                     and p.instance_id == instance_id]
            if not match:
                logger.error(f"No webindex instance with id '{instance_id}'.")
                return {}
            logger.info(f"Note: instance '{instance_id}' is disabled.")
        return dict(match[0].settings or {})
    if instances:
        return dict(instances[0].settings or {})
    return {}


def _build_store(settings: Dict[str, Any]):
    from src.utils.lookups.webindex.store import WebIndexStore

    return WebIndexStore(settings, f"webindex_cli_{id(settings)}")


def _print_kv(rows, *, keys) -> None:
    widths = {k: max(len(str(k)), *(len(str(r.get(k, ""))) for r in rows)) for k in keys}
    print("  ".join(str(k).ljust(widths[k]) for k in keys))
    for r in rows:
        print("  ".join(str(r.get(k, "")).ljust(widths[k]) for k in keys))


def handle_webindex(args, logger: logging.Logger) -> int:
    action = getattr(args, "webindex_action", None)
    if action is None:
        logger.error("No webindex subcommand. See `alima webindex --help`.")
        return 2

    config = ConfigManager().load_config()

    if action == "crawl":
        return _handle_crawl(args, config, logger)
    if action == "stats":
        settings = _instance_settings(config, getattr(args, "instance", None), logger)
        if getattr(args, "db_path", None):
            settings["db_path"] = args.db_path
        store = _build_store(settings)
        stats = store.stats()
        print(f"DB-Pfad:        {stats['db_path']}")
        print(f"Seiten:         {stats['pages']}")
        print(f"Keywords:       {stats['keywords']}")
        print(f"Verknüpfungen:  {stats['page_keywords']}")
        print(f"Letzter Fetch:  {stats['last_fetched'] or '—'}")
        return 0
    if action == "list-keywords":
        settings = _instance_settings(config, getattr(args, "instance", None), logger)
        if getattr(args, "db_path", None):
            settings["db_path"] = args.db_path
        store = _build_store(settings)
        rows = store.list_keywords(limit=int(getattr(args, "limit", 200) or 200),
                                   contains=getattr(args, "contains", "") or "")
        if not rows:
            print("(keine Keywords im Index — zuerst `alima webindex crawl` laufen lassen)")
            return 0
        _print_kv(rows, keys=["page_count", "keyword", "display"])
        print(f"\n{len(rows)} Keyword(s).")
        return 0
    if action == "search":
        return _handle_search(args, config, logger)

    logger.error(f"Unknown webindex action: {action}")
    return 2


def _handle_crawl(args, config, logger) -> int:
    from src.utils.lookups.webindex.indexer import crawl_site
    from src.utils.lookups.webindex.keywords import (
        build_keyword_extractor, resolve_crawl_model,
    )

    settings = _instance_settings(config, getattr(args, "instance", None), logger)
    # CLI flags override instance settings; fall back to instance values.
    base_url = (getattr(args, "base_url", None) or settings.get("base_url") or "").strip()
    if not base_url:
        logger.error("No base_url. Pass --base-url or set it on the webindex instance.")
        return 2
    if getattr(args, "db_path", None):
        settings["db_path"] = args.db_path

    store = _build_store(settings)

    # Model: CLI flags → instance llm_provider/llm_model → global agentic default.
    provider, model = resolve_crawl_model(
        config, settings,
        cli_provider=getattr(args, "provider", None),
        cli_model=getattr(args, "model", None),
    )
    keyword_extractor = None
    if provider:
        from src.llm.llm_service import LlmService

        llm_service = LlmService(providers=None, config_manager=ConfigManager())
        keyword_extractor = build_keyword_extractor(llm_service, provider, model, logger=logger)
        if keyword_extractor is None:
            logger.warning("Keyword-Workflow nicht ladbar; crawle mit Meta-Keywords nur.")
        else:
            logger.info(f"LLM-Keyword-Extraktion via {provider}/{model or 'auto'}")
    else:
        logger.info("Kein LLM-Provider konfiguriert (CLI/Instanz/Default) — "
                    "crawle mit Meta-/Überschriften-Keywords nur.")

    def _flag(name, default, cast=int):
        v = getattr(args, name, None)
        if v in (None, "", False):
            return cast(settings.get(name, default) or default) if cast is int else default
        return cast(v) if cast is int else v

    result = crawl_site(
        store,
        base_url=base_url,
        max_depth=_flag("max_depth", 2),
        max_pages=_flag("max_pages", 50),
        include_re=getattr(args, "include", None) or (settings.get("include_re") or None),
        exclude_re=getattr(args, "exclude", None) or (settings.get("exclude_re") or None),
        keyword_extractor=keyword_extractor,
        fetch_timeout=int(getattr(args, "timeout", 20) or settings.get("fetch_timeout", 20) or 20),
        user_agent=getattr(args, "user_agent", None) or "ALIMA-webindex",
        min_chars=_flag("min_chars", 50),
        max_keywords=_flag("max_keywords", 15),
        dry_run=bool(getattr(args, "dry_run", False)),
    )
    label = "would index" if args.dry_run else "indexed"
    print(f"Base-URL:       {result['base_url']}")
    print(f"Besucht:        {result['visited']}")
    print(f"{label.capitalize()}: {result['pages_indexed']}")
    print(f"Übersprungen:   {result['pages_skipped']}")
    if result["errors"]:
        print(f"Fehler:         {len(result['errors'])}")
        for e in result["errors"][:10]:
            print(f"  - {e}")
    if result["indexed_urls"]:
        print(f"\nURLs ({label}):")
        for u in result["indexed_urls"]:
            print(f"  - {u}")
    return 0 if not result["errors"] else 1


def _handle_search(args, config, logger) -> int:
    settings = _instance_settings(config, getattr(args, "instance", None), logger)
    if getattr(args, "db_path", None):
        settings["db_path"] = args.db_path
    store = _build_store(settings)
    from src.utils.lookups.webindex.provider import WebIndexLookup

    # Build a transient lookup to reuse its search_keyword ranking + snippets.
    lookup = WebIndexLookup.__new__(WebIndexLookup)
    lookup._config = settings
    lookup._store = store
    query = getattr(args, "query", "")
    max_results = int(getattr(args, "max_results", 0) or 0) or int(settings.get("max_results", 10) or 10)
    result = lookup.search_keyword(query, max_results=max_results)
    print(f"Query: {result['query']}  (terms: {', '.join(result['terms']) or '—'})")
    print(f"Treffer: {result['count']}")
    for i, h in enumerate(result["hits"], 1):
        print(f"\n[{i}] {h['title'] or h['url']}")
        print(f"    URL:     {h['url']}")
        print(f"    Match:   {', '.join(h['matched_keywords'])}  "
              f"(score={h['score']}, {h['matched_count']} kw)")
        print(f"    Snippet: {h['snippet']}")
    return 0 if result["count"] else 1


def main(argv=None) -> int:  # pragma: no cover — convenience entry
    """Module-level entry for ``python -m`` style invocation."""
    import argparse

    parser = argparse.ArgumentParser(prog="alima webindex")
    parser.add_argument("webindex_action",
                        choices=["crawl", "stats", "list-keywords", "search"])
    parser.add_argument("--base-url")
    parser.add_argument("--instance")
    parser.add_argument("--db-path")
    parser.add_argument("--max-depth", type=int, default=2)
    parser.add_argument("--max-pages", type=int, default=50)
    parser.add_argument("--include")
    parser.add_argument("--exclude")
    parser.add_argument("--provider")
    parser.add_argument("--model")
    parser.add_argument("--timeout", type=int, default=20)
    parser.add_argument("--user-agent", default="ALIMA-webindex")
    parser.add_argument("--min-chars", type=int, default=50)
    parser.add_argument("--max-keywords", type=int, default=15)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--contains", default="")
    parser.add_argument("--max-results", type=int, default=0)
    parser.add_argument("query", nargs="*", default=[])
    args = parser.parse_args(argv)
    args.query = " ".join(args.query) if isinstance(args.query, list) else args.query
    logging.basicConfig(level=logging.INFO)
    return handle_webindex(args, logging.getLogger("alima.webindex"))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())