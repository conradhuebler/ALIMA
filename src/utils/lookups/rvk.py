"""RVK classification lookup plugin - Claude Generated.

Wraps the bare RVK-API client (`src/utils/clients/rvk_api_client.py`) as a lookup
plugin exposing two agent tools: keyword → ranked RVK notations, and notation
validation (label + ancestor path). This is the *raw* RVK API as a plugin/tool;
the composed ``rvk_lookup`` core tool (pipeline anchor machinery + catalog RVK
search + this API for validation) stays as-is.
"""

from __future__ import annotations

from typing import Any, List

from src.core.plugins.schema import INT, ConfigField, PluginDoc

from .registry import LookupToolSpec, register_lookup


@register_lookup
class RvkLookup:
    id = "rvk_api"
    label = "RVK-API (Regensburg)"

    def __init__(self, **config: Any):
        self._config = config or {}

    @classmethod
    def config_fields(cls) -> List[ConfigField]:
        return [ConfigField(key="timeout", label="Timeout (s)", kind=INT, default=8)]

    @classmethod
    def doc(cls) -> PluginDoc:
        return PluginDoc(
            description="RVK-Klassifikation über die offizielle RVK-API der UB "
            "Regensburg: Schlagwort→Notationen-Suche und Notationsvalidierung.",
            input="Ein Schlagwort (Suche) bzw. eine RVK-Notation (Validierung).",
            output="Gerankte RVK-Notationen mit Label/Zweig bzw. Validierung + "
            "Ahnenpfad einer Notation.",
        )

    @classmethod
    def mcp_tool_specs(cls) -> List[LookupToolSpec]:
        return [
            LookupToolSpec(
                name="rvk_search",
                description="Search the RVK classification by subject keyword. "
                "Returns ranked RVK notations (code, label, branch family) validated "
                "against the official RVK API.",
                parameters={
                    "type": "object",
                    "properties": {
                        "keyword": {"type": "string", "description": "Subject keyword"},
                        "max_results": {"type": "integer", "default": 8, "description": "Max notations"},
                    },
                    "required": ["keyword"],
                },
                method="search_keyword",
                cache_key_param="keyword",
            ),
            LookupToolSpec(
                name="rvk_validate",
                description="Validate an RVK notation against the official RVK API. "
                "Returns whether it exists, its label, and its ancestor path.",
                parameters={
                    "type": "object",
                    "properties": {
                        "notation": {"type": "string", "description": "RVK notation, e.g. 'QK 300'"},
                    },
                    "required": ["notation"],
                },
                method="validate_notation",
                cache_key_param="notation",
            ),
        ]

    def _client(self):
        from src.utils.clients.rvk_api_client import RvkApiClient

        return RvkApiClient(timeout=int(self._config.get("timeout", 8) or 8))

    # --- tool methods (called by the generated handler) ------------------- #
    def search_keyword(self, keyword: str, max_results: int = 8) -> dict:
        results = self._client().search_keyword(str(keyword), max_results=int(max_results or 8))
        return {"keyword": keyword, "count": len(results), "results": results}

    def validate_notation(self, notation: str) -> dict:
        return {"notation": notation, "result": self._client().validate_notation(str(notation))}
