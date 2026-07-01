"""Registry + contract for input sources - Claude Generated.

An *input source* turns a raw input reference (text, a file path, a URL, a DOI)
into extracted text. This mirrors the search-provider registry
(``src/core/search/registry.py``): a module-level ``id -> class`` dict populated
by the ``@register_input_source`` decorator so sources self-register on import.

Replacing the former if/elif in ``execute_input_extraction`` (Debt D-11), every
source declares its ``config_fields`` (so it is a real, per-instance-configurable
plugin — the reason the DOI resolver is split into three separately configurable
sources), an ``id``/``label`` for the registry + UI, ``can_handle`` for
auto-detection, and ``extract`` for the work.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple, runtime_checkable

from src.core.plugins.schema import ConfigField

INPUT_SOURCE_REGISTRY: Dict[str, type] = {}


@dataclass
class InputToolSpec:
    """Declares that an input source is callable as an MCP tool by the agent.

    Kept minimal: an input source takes one input value (a DOI, a URL). ``param``
    names that value; the handler builds the source from its instance settings and
    calls ``mcp_execute`` (rich JSON) or falls back to ``extract``. A source that
    returns ``None`` from :meth:`mcp_tool_spec` is not exposed as a tool (e.g. the
    file-path sources text/file/pdf/image). - Claude Generated
    """

    name: str
    description: str
    param: str = "source"
    param_description: str = ""


@runtime_checkable
class InputSource(Protocol):
    """Structural contract every input source implements."""

    id: str
    label: str

    @classmethod
    def config_fields(cls) -> List[ConfigField]: ...

    def can_handle(self, source: str, input_type: str) -> bool: ...

    def extract(
        self,
        source: str,
        *,
        llm_service: Any = None,
        stream_callback: Optional[Callable[[str], None]] = None,
        logger: Any = None,
        **opts: Any,
    ) -> Tuple[str, str, str]: ...


def register_input_source(cls: type) -> type:
    """Class decorator: register an input source by its class-level ``id``."""
    sid = getattr(cls, "id", None)
    if not sid or not isinstance(sid, str):
        raise ValueError(f"Input source {cls.__name__} must define a non-empty string `id`")
    if sid in INPUT_SOURCE_REGISTRY and INPUT_SOURCE_REGISTRY[sid] is not cls:
        raise ValueError(
            f"Input source id '{sid}' already registered to {INPUT_SOURCE_REGISTRY[sid].__name__}"
        )
    INPUT_SOURCE_REGISTRY[sid] = cls
    return cls


def get_input_source(sid: str) -> type:
    if sid not in INPUT_SOURCE_REGISTRY:
        raise KeyError(
            f"Unknown input source '{sid}'. Registered: {sorted(INPUT_SOURCE_REGISTRY)}"
        )
    return INPUT_SOURCE_REGISTRY[sid]


def list_input_sources() -> List[str]:
    return sorted(INPUT_SOURCE_REGISTRY)


def _reset_for_tests() -> None:
    INPUT_SOURCE_REGISTRY.clear()
