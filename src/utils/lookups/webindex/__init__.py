"""Webindex lookup plugin — a website/URL keyword index as a retrieval source for
the RAG chatbot. Importing this package registers the lookup plugin via its
``provider`` module. - Claude Generated"""

from . import provider  # noqa: F401 — registers WebIndexLookup via @register_lookup

__all__ = ["provider"]