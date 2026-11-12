#!/usr/bin/env python3
"""
ALIMA Institution Presets Loader - Claude Generated

Reads the wizard pre-fill values (LLM address and models, database, Libero
addresses) from the institution preset ``alima_presets.json``. The file is
found by ``preset_apply.find_preset`` (environment variable ``ALIMA_PRESET``,
then the configuration directory, then the project directory). API keys and
tokens in the file are not used (``preset_apply.load_preset`` drops them; the
key a ``providers`` entry may carry is not handed to the wizard either).
The LLM pre-fill comes from ``providers`` and ``default_models``; a block
``llm`` of an older preset is not read. The other parts of a preset
(institution, workflow, plugins, files, providers) are applied by
``preset_apply.build_plan``/``apply_plan``.
"""

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class InstitutionPresets:
    """Institution-specific default configuration values - Claude Generated"""
    institution_name: str = ""
    # Where the preset came from (empty without one) - Claude Generated
    path: str = ""
    source: str = ""
    label: str = ""
    # LLM provider
    llm_provider_type: str = ""     # 'ollama', 'openai_compatible', 'gemini', 'anthropic'
    llm_provider_name: str = ""
    llm_base_url: str = ""
    # LLM model defaults - Claude Generated
    llm_default_model: str = ""
    llm_task_models: dict = field(default_factory=dict)  # {task_type_value: model_name}
    # Catalog SOAP
    catalog_soap_search_url: str = ""
    catalog_soap_details_url: str = ""
    # Catalog web fallback (BiblioClient scraping)
    catalog_web_search_url: str = ""
    catalog_web_record_url: str = ""
    # Database configuration - Claude Generated
    db_type: str = ""              # 'sqlite', 'mysql', 'mariadb'
    db_sqlite_path: str = ""       # Path for SQLite
    db_host: str = ""              # Host for MySQL/MariaDB
    db_port: int = 3306            # Port for MySQL/MariaDB
    db_database: str = ""           # Database name
    db_username: str = ""           # Username
    db_password: str = ""           # Password
    db_charset: str = "utf8mb4"    # Character set
    # GND data status - Claude Generated
    gnd_data_preloaded: bool = False  # True if DB already contains GND data

    def has_llm(self) -> bool:
        return bool(self.llm_provider_type or self.llm_base_url)

    def has_catalog(self) -> bool:
        return bool(self.catalog_soap_search_url or self.catalog_soap_details_url)

    def has_models(self) -> bool:
        """True if preset contains model configuration - Claude Generated"""
        return bool(self.llm_default_model or self.llm_task_models)

    def pipeline_model(self) -> str:
        """The model for every non-vision pipeline step - Claude Generated

        ``llm_task_models`` keeps its per-task dict shape for backward
        compatibility with hand-written preset files, but the wizard only
        asks about two scopes now (pipeline, vision). Every real preset in
        use has the same model under every non-vision key, so the first one
        found is as good as any.
        """
        for key, model in self.llm_task_models.items():
            if key != "vision" and model:
                return model
        return self.llm_default_model

    def vision_model(self) -> str:
        """The model for image/OCR analysis - Claude Generated"""
        return self.llm_task_models.get("vision") or self.llm_default_model

    def has_database_config(self) -> bool:
        """True if preset contains database configuration - Claude Generated"""
        return bool(self.db_type)

    def display_name(self) -> str:
        """Name for the wizard banner - Claude Generated"""
        return self.institution_name or self.label


class PresetLoader:
    """Loads institution presets from alima_presets.json - Claude Generated"""

    PRESET_FILENAME = "alima_presets.json"

    @classmethod
    def load(cls) -> InstitutionPresets:
        """Load the preset found by ``preset_apply.find_preset``.

        Without a preset in the environment variable or the configuration
        directory the project directory is used, as before. Returns empty
        InstitutionPresets if no file is found or it is unreadable. Never
        raises an exception.
        """
        try:
            from .preset_apply import find_preset, load_preset

            path, source = find_preset()
            if path is None:
                return InstitutionPresets()
            preset = load_preset(path, source)
            presets = cls._parse(preset.data)
            presets.path = str(preset.path)
            presets.source = source
            presets.label = preset.label
            if presets.display_name():
                logger.info(f"Institution presets loaded: {presets.display_name()}")
            return presets
        except Exception as e:
            logger.warning(f"Could not load alima_presets.json: {e}")
            return InstitutionPresets()

    @staticmethod
    def _catalog_from_plugins(data: dict) -> dict:
        """Libero addresses from the ``catalog`` entry under ``plugins`` when
        the preset has no ``catalog`` block, so the wizard's catalog page shows
        what the preset sets afterwards. - Claude Generated"""
        for entry in data.get('plugins') or []:
            if isinstance(entry, dict) and entry.get('instance') == 'catalog':
                s = entry.get('settings') if isinstance(entry.get('settings'), dict) else {}
                return {
                    'soap_search_url': s.get('catalog_search_url', ''),
                    'soap_details_url': s.get('catalog_details', ''),
                    'web_search_url': s.get('catalog_web_search_url', ''),
                    'web_record_url': s.get('catalog_web_record_url', ''),
                }
        return {}

    @staticmethod
    def _llm_prefill(data: dict) -> dict:
        """The wizard's one provider from ``providers``/``default_models`` - Claude Generated.

        The provider is the one ``default_models.general`` names, else the first
        enabled entry. Returned: ``provider_type``, ``provider_name``,
        ``base_url`` (for Ollama built from host/port/SSL when the entry has no
        ``base_url``), ``default_model`` and ``task_models`` (``pipeline``,
        ``vision``; only models of a scope that names this provider). Never the
        key. Empty without a usable entry.
        """
        entries = [p for p in data.get('providers') or []
                   if isinstance(p, dict) and isinstance(p.get('name'), str) and p['name'].strip()
                   and isinstance(p.get('provider_type'), str) and p['provider_type']
                   and p.get('enabled', True) is not False]
        if not entries:
            return {}
        defaults = data.get('default_models') if isinstance(data.get('default_models'), dict) else {}

        def scope(name: str) -> dict:
            item = defaults.get(name)
            return item if isinstance(item, dict) else {}

        def same(a, b) -> bool:
            return str(a or '').strip().lower() == str(b or '').strip().lower()

        general = scope('general')
        chosen = next((p for p in entries if same(p['name'], general.get('provider'))), entries[0])
        base_url = str(chosen.get('base_url') or '')
        if not base_url and chosen['provider_type'] == 'ollama' and chosen.get('host'):
            try:
                from .ollama_endpoint import base_url as ollama_base_url

                base_url = ollama_base_url(str(chosen['host']), int(chosen.get('port') or 11434),
                                           bool(chosen.get('use_ssl')))
            except (TypeError, ValueError) as exc:  # a malformed port in the file
                logger.warning(f"Preset: Ollama-Adresse von {chosen['name']} nicht lesbar ({exc})")
        task_models = {}
        for key in ('pipeline', 'vision'):
            item = scope(key)
            if item.get('model') and same(item.get('provider'), chosen['name']):
                task_models[key] = str(item['model'])
        default_model = (str(general.get('model') or '') if same(general.get('provider'), chosen['name'])
                         else '') or str(chosen.get('preferred_model') or '')
        return {
            'provider_type': chosen['provider_type'],
            'provider_name': chosen['name'],
            'base_url': base_url,
            'default_model': default_model,
            'task_models': task_models,
        }

    @classmethod
    def _parse(cls, data: dict) -> InstitutionPresets:
        llm = cls._llm_prefill(data)
        catalog = data.get('catalog') or cls._catalog_from_plugins(data)
        database = data.get('database') or {}  # Claude Generated
        return InstitutionPresets(
            institution_name=data.get('institution_name', ''),
            llm_provider_type=llm.get('provider_type', ''),
            llm_provider_name=llm.get('provider_name', ''),
            llm_base_url=llm.get('base_url', ''),
            llm_default_model=llm.get('default_model', ''),
            llm_task_models=llm.get('task_models', {}),
            catalog_soap_search_url=catalog.get('soap_search_url', ''),
            catalog_soap_details_url=catalog.get('soap_details_url', ''),
            catalog_web_search_url=catalog.get('web_search_url', ''),
            catalog_web_record_url=catalog.get('web_record_url', ''),
            # Database configuration - Claude Generated
            db_type=database.get('type', ''),
            db_sqlite_path=database.get('sqlite_path', ''),
            db_host=database.get('host', ''),
            db_port=database.get('port', 3306),
            db_database=database.get('database', ''),
            db_username=database.get('username', ''),
            db_password=database.get('password', ''),
            db_charset=database.get('charset', 'utf8mb4'),
            gnd_data_preloaded=database.get('gnd_preloaded', False),
        )
