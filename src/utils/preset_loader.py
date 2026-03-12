#!/usr/bin/env python3
"""
ALIMA Institution Presets Loader - Claude Generated

Loads institution-specific configuration defaults from alima_presets.json
placed in the project root directory. Enables institutions to pre-fill
wizard fields without hardcoding values in source code.
"""

import json
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class InstitutionPresets:
    """Institution-specific default configuration values - Claude Generated"""
    institution_name: str = ""
    # LLM provider
    llm_provider_type: str = ""     # 'ollama', 'openai_compatible', 'gemini', 'anthropic'
    llm_provider_name: str = ""
    llm_base_url: str = ""
    llm_api_key: str = ""
    # LLM model defaults - Claude Generated
    llm_default_model: str = ""
    llm_task_models: dict = field(default_factory=dict)  # {task_type_value: model_name}
    # Catalog SOAP
    catalog_soap_search_url: str = ""
    catalog_soap_details_url: str = ""
    catalog_token: str = ""
    # Catalog web fallback (BiblioClient scraping)
    catalog_web_search_url: str = ""
    catalog_web_record_url: str = ""

    def has_llm(self) -> bool:
        return bool(self.llm_provider_type or self.llm_base_url)

    def has_catalog(self) -> bool:
        return bool(self.catalog_soap_search_url or self.catalog_soap_details_url)

    def has_models(self) -> bool:
        """True if preset contains model configuration - Claude Generated"""
        return bool(self.llm_default_model or self.llm_task_models)


class PresetLoader:
    """Loads institution presets from alima_presets.json - Claude Generated"""

    PRESET_FILENAME = "alima_presets.json"

    @classmethod
    def load(cls) -> InstitutionPresets:
        """Load presets from the project directory.

        Returns empty InstitutionPresets if file is missing or unreadable.
        Never raises an exception.
        """
        try:
            from .path_utils import get_project_root
            path = get_project_root() / cls.PRESET_FILENAME
            if not path.exists():
                return InstitutionPresets()
            data = json.loads(path.read_text(encoding='utf-8'))
            if not isinstance(data, dict):
                logger.warning("alima_presets.json: unexpected format (not a JSON object)")
                return InstitutionPresets()
            presets = cls._parse(data)
            if presets.institution_name:
                logger.info(f"Institution presets loaded: {presets.institution_name}")
            return presets
        except Exception as e:
            logger.warning(f"Could not load alima_presets.json: {e}")
            return InstitutionPresets()

    @classmethod
    def _parse(cls, data: dict) -> InstitutionPresets:
        llm = data.get('llm') or {}
        catalog = data.get('catalog') or {}
        return InstitutionPresets(
            institution_name=data.get('institution_name', ''),
            llm_provider_type=llm.get('provider_type', ''),
            llm_provider_name=llm.get('provider_name', ''),
            llm_base_url=llm.get('base_url', ''),
            llm_api_key=llm.get('api_key', ''),
            llm_default_model=llm.get('default_model', ''),
            llm_task_models=llm.get('task_models', {}),
            catalog_soap_search_url=catalog.get('soap_search_url', ''),
            catalog_soap_details_url=catalog.get('soap_details_url', ''),
            catalog_token=catalog.get('token', ''),
            catalog_web_search_url=catalog.get('web_search_url', ''),
            catalog_web_record_url=catalog.get('web_record_url', ''),
        )
