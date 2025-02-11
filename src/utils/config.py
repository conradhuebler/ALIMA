from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import json
import os
from pathlib import Path
import logging
from enum import Enum
import yaml

class AIProvider(Enum):
    """Verfügbare KI-Provider"""
    GEMINI = "gemini"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    LOCAL = "local"

@dataclass
class AIProviderConfig:
    """Konfiguration für einen AI-Provider"""
    name: str
    models: List[str]
    api_url: str
    default_model: str

@dataclass
class AIConfig:
    """KI-bezogene Konfiguration"""
    provider: AIProvider = AIProvider.GEMINI
    api_key: str = ""
    temperature: float = 0.7
    max_tokens: int = 1000
    providers: Dict[str, AIProviderConfig] = field(default_factory=lambda: {
        "gemini": AIProviderConfig(
            name="Google Gemini",
            api_url="https://generativelanguage.googleapis.com/v1beta/models/modelname:generateContent",
            default_model="gemini-1.5-flash",
            models = ["gemini-1.5-pro", "gemini-1.5-flash", "gemini-1.5-flash-8b", "gemini-2.0-flash-exp"]
        ),
        "openai": AIProviderConfig(
            name="OpenAI",
            models=["gpt-3.5-turbo", "gpt-4"],
            api_url="https://api.openai.com/v1/chat/completions",
            default_model="gpt-3.5-turbo"
        ),
        "anthropic": AIProviderConfig(
            name="Anthropic",
            models=["claude-2", "claude-instant"],
            api_url="https://api.anthropic.com/v1/complete",
            default_model="claude-2"
        ),
        "local": AIProviderConfig(
            name="Local LLM",
            models=["llama2", "mistral"],
            api_url="http://localhost:8000/v1/chat/completions",
            default_model="llama2"
        )
    })

class ConfigSection(Enum):
    """Verfügbare Konfigurationsbereiche"""
    GENERAL = "general"
    AI = "ai"
    SEARCH = "search"
    CACHE = "cache"
    UI = "ui"
    EXPORT = "export"
    PROMPTS = "prompts"

@dataclass
class GeneralConfig:
    """Allgemeine Konfiguration"""
    language: str = "de"
    debug: bool = False
    log_level: str = "INFO"

@dataclass
class PromptTemplate:
    """Template für einen KI-Prompt"""
    name: str
    description: str
    template: str
    required_variables: List[str]
    model: str = "gemini-1.5-flash"

@dataclass
class PromptConfig:
    """Konfiguration für Prompts"""
    templates: Dict[str, PromptTemplate] = field(default_factory=lambda: {
        "abstract_analysis": PromptTemplate(
            name="abstract_analysis",
            description="Analysiert einen Abstract und extrahiert relevante Schlagworte",
            template="""Basierend auf folgendem Abstract und Keywords, schlage passende deutsche Schlagworte vor. Bei chemischen Verbindungen als Summenformeln oder Elementen gibt zusätzlich jeden Elementnamen als weiteres Schlagwort mit an.
            
            Abstract:
            {abstract}
            
            Vorhandene Keywords:
            {keywords}
            
            Bitte gibt die Schlagworte kommagetrennt aus, nutze nur eine Zeile und gib keinen weiteren Text aus. Verwende auch das Wort Schlagwort nicht.            Die Schlagworte sollten möglichst präzise und spezifisch sein.""",
            required_variables=["abstract", "keywords"]
        ),
        "results_verification": PromptTemplate(
            name="results_verification",
            description="Überprüft die Qualität der gefundenen GND-Schlagworte",
            template="""
             Wähle aus der Liste der OGND-Schlagworte diejenigen heraus, die zur inhaltlichen Beschreibung des Abstraktes verwendet werden können. Nutze nur Schlagworte, die in der OGND-Liste korrekt auftauchen und keine Synonyme. Führe auch keine weitere Erschließung durch, außer in der letzten Diskussion.
                 Abstract:
                {abstract}

                Gefundene GND-Schlagworte:
                {keywords}

                Bitte gib deine Antwort in folgendem Format:

                                ANALYSE:
                [Deine qualitative Analyse der Verschlagwortung]

                Schlagworte:
                [Liste der passende Schlagwort aus dem Prompt - bitte kommagetrennt. ***Nutze keine Synonyme oder alternative Schreibweisen/Formulierungen***] 

                Schlagworte OGND Eintrage:
                [Liste der passende Konzepte mit der zugeörigen OGND-ID aus dem Prompt  - bitte kommagetrennt]

                Schlagwortketten:
                [Nutze Kombinationen von OGND-Schlagworten um bestimmte Themenbereiche konkret zu beschreiben oder um Konzepte, die durch ein Schlagwort nicht korrekt abgedeckt sind. Trenne die Schlagworte (mit GND-ID) in den Ketten mit Komma. Nimm für jede Schlagwortkette eine neue Zeile - Kommentiere zu jeder Schlagwortkette kurz, wieso diese passend ist]	

                FEHLENDE KONZEPTE:
                [Liste von Konzepten, die noch nicht durch GND abgedeckt sind]""",

            required_variables=["abstract", "keywords"]
        ),
        "concept_extraction": PromptTemplate(
            name="concept_extraction",
            description="Extrahiert Konzepte aus einem Text",
            template="""Extrahiere die wichtigsten Konzepte aus folgendem Text:

            Text:
            {text}
            
            Bitte liste die Konzepte in der Reihenfolge ihrer Wichtigkeit auf.
            Berücksichtige dabei:
            - Fachbegriffe
            - Forschungsmethoden
            - Theoretische Konzepte
            - Untersuchungsgegenstände""",
            required_variables=["text"]
        ),
        "ub_search": PromptTemplate(
            name="ub_search",
            description="Verknüpft Schlagworte und Klassifizierung aus dem UB-Katalog",
         #   template="""Wähle aus der folgendenen Liste von Schlagworten - {keywords} - exakt diejenige, die auf die Titelliste passen?, 
         #   Folgende Titel sind verfügbar:
         #   {abstract}
         #   
         #   Gibt deine Anwort in folgendem Format:
         #   Am besten passendes Schlagwort (nur eines):
         #   [1. Schlagwort]
         #   Zweit bestes passendes Schlagwort (nur eines):
         #   [2. Schlagwort]
         #   ANALYSE:
         #   [Deine qualitative Analyse der Zuordnung]""",
         template="""### Aufgabe: Schlagwort-Klassifizierung
        Bewerte jedes Schlagwort in der Liste mit einem Score zwischen 0 (am wenigsten passend) und 10 (am besten passend). Gibt den Score in Klammern hinter das Schlagwort und sortiere die Ausgabe mit absteigendem Score! Nutze dabei ***ausschließlich*** Schlagworte aus der Liste und ***keine*** Synonyme oder alternative Bezeichnungen und schlage ***keine*** anderen Schlagworte vor.

        **Schlagwortliste:**
        {keywords}

        **Text:**
        {abstract}""",
        required_variables=["keywords", "abstract"])
    })

@dataclass
class SearchConfig:
    """Konfiguration für die Suche"""
    default_threshold: float = 1.0
    max_results: int = 1000
    timeout: int = 30
    retry_attempts: int = 3
    sort_options: List[str] = field(default_factory=lambda: [
        "Häufigkeit", "Alphabetisch", "Relevanz"
    ])
    api_url: str = "https://lobid.org/resources/search"
    batch_size: int = 100

@dataclass
class CacheConfig:
    """Konfiguration für das Caching"""
    enabled: bool = True
    max_age_hours: int = 24
    db_path: str = "search_cache.db"
    cleanup_interval: int = 24  # Stunden
    max_entries: int = 10000
    compression: bool = True

@dataclass
class UIConfig:
    """Konfiguration für die Benutzeroberfläche"""
    theme: str = "system"
    font_size: int = 12
    window_size: Dict[str, int] = field(default_factory=lambda: {"width": 1200, "height": 800})
    show_tooltips: bool = True
    save_window_position: bool = True
    autosave_interval: int = 5  # Minuten
    max_recent_searches: int = 10

@dataclass
class ExportConfig:
    """Konfiguration für Export-Funktionen"""
    default_format: str = "csv"
    export_directory: str = "exports"
    include_metadata: bool = True
    date_format: str = "%Y-%m-%d_%H-%M-%S"
    formats: Dict[str, Dict] = field(default_factory=lambda: {
        "csv": {"delimiter": ",", "encoding": "utf-8"},
        "json": {"indent": 2, "ensure_ascii": False},
        "xlsx": {"include_headers": True}
    })

class Config:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    @staticmethod
    def get_instance():
        if Config._instance is None:
            Config()
        return Config._instance
    
    def __init__(self):
        if self._initialized:
            return
        self.logger = logging.getLogger(__name__)
        self.config_dir = os.path.expanduser("~/.config/gnd-fetcher")
        self.config_file = os.path.join(self.config_dir, "config.yaml")
        
        # Initialisiere sections
        self.sections = {}
        
        # Erstelle Standardkonfiguration mit Default-Prompts
        self._create_default_config()
        #self.logger.info(f"Verfügbare Konfigurationsbereiche: {self.sections}")
        # Lade bestehende Konfiguration, falls vorhanden
        self.load_config()
        
        # Debug-Ausgabe
        self.logger.debug(f"Verfügbare Prompts nach Initialisierung: {self.sections[ConfigSection.PROMPTS].templates.keys()}")
        self._initialized = True

    def get_section(self, section: ConfigSection) -> Any:
        """Gibt eine Konfigurationssektion zurück"""
        if section not in self.sections:
            self.logger.warning(f"Sektion {section} nicht gefunden")
            return None
        return self.sections.get(section)

    def _create_default_config(self):
        """Erstellt die Standardkonfiguration mit Default-Prompts"""
        # Setze die Standardkonfiguration
        self.sections = {
            ConfigSection.GENERAL: GeneralConfig(),
            ConfigSection.AI: AIConfig(),
            ConfigSection.SEARCH: SearchConfig(),
            ConfigSection.CACHE: CacheConfig(),
            ConfigSection.UI: UIConfig(),
            ConfigSection.EXPORT: ExportConfig(),
            ConfigSection.PROMPTS: PromptConfig()
        }
        #self.logger.info("Konfig {self.sections}")
    

            
    def save_config(self):
        """Speichert die Konfiguration in der Datei"""
        self.logger.info("Starte Speichervorgang der Konfiguration...")
        try:
            if not os.path.exists(self.config_dir):
                os.makedirs(self.config_dir)
            
            # Konvertiere Konfiguration in dict
            config_dict = {}
            #self.logger.info(f"self.sections: {self.sections}")
            for section, config in self.sections.items():
                #self.logger.info(f"Verarbeite Section {section}")
                section_dict = {}
                for key, value in vars(config).items():
                    self.logger.info(f"Verarbeite Key {key} and Value {value}")
                    if not key.startswith('_'):
                        if isinstance(value, AIProvider):
                            value = value.value
                        elif key == 'providers' and isinstance(value, dict):
                            providers_dict = {}
                            for provider_name, provider_config in value.items():
                                providers_dict[provider_name] = {
                                    'name': provider_config.name,
                                    'models': provider_config.models,
                                    'api_url': provider_config.api_url,
                                    'default_model': provider_config.default_model
                                }
                            value = providers_dict
                            self.logger.debug(f"Verarbeite AI-Provider-Dict: {value}")
                        
                        elif isinstance(value, AIProviderConfig):
                            value = {
                                'name': value.name,
                                'models': value.models,
                                'api_url': value.api_url,
                                'default_model': value.default_model
                            }
                            self.logger.info(f"Verarbeite AI-ProviderKonfiguration: {value}")
                        elif isinstance(value, AIConfig):
                            providers_dict = {}
                            for provider_name, provider_config in value.providers.items():
                                providers_dict[provider_name] = {
                                    'name': provider_config.name,
                                    'models': provider_config.models,
                                    'api_url': provider_config.api_url,
                                    'default_model': provider_config.default_model
                                }
                                self.logger.info(f"Verarbeite AI-Provider-Konfiguration: {providers_dict}")
                            value = {
                                'provider': value.provider,
                                'api_key': value.api_key,
                                'temperature': value.temperature,
                                'max_tokens': value.max_tokens,
                                'providers': providers_dict
                            }
                            self.logger.info(f"Verarbeite AI-Konfiguration: {value}")
                        elif key == 'templates' and isinstance(value, dict):
                            #self.logger.info(f"Verarbeite Templates: {value}")
                            templates_dict = {}
                            for template_name, template in value.items():
                                self.logger.info(f"Verarbeite Template {template_name}")
                                templates_dict[template_name] = {
                                    'name': template.name,
                                    'description': template.description,
                                    'template': template.template,
                                    'required_variables': template.required_variables,
                                    'model': template.model
                                }
                            value = templates_dict
                        section_dict[key] = value
                config_dict[section.value] = section_dict

            self.logger.info(f"Finales config_dict zum Speichern: {config_dict}")
            
            with open(self.config_file, 'w', encoding='utf-8') as f:
                yaml.safe_dump(config_dict, f, default_flow_style=False, allow_unicode=True)
            
            self.logger.info(f"Konfiguration erfolgreich in {self.config_file} gespeichert")
        except Exception as e:
            self.logger.error(f"Fehler beim Speichern der Konfiguration: {e}")
            raise

    def load_config(self):
        """Lädt die Konfiguration aus der YAML-Datei"""
        self.logger.debug("Starte Laden der Konfiguration...")
        try:
            if not os.path.exists(self.config_file):
                self.logger.warning(f"Keine Konfigurationsdatei gefunden unter {self.config_file}")
                self._create_default_config()
                self.logger.info(f"Geladene Rohdaten: {self.sections}")
                return

            with open(self.config_file, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)

            if not config_data:
                self.logger.warning("Leere Konfigurationsdatei gefunden")
                self._create_default_config()
                self.logger.info(f"Geladene Rohdaten: {self.sections}")
                return

            self.logger.info(f"Geladene Rohdaten von der Config: {config_data}")
            
            # Verarbeite jede Sektion
            for section_name, section_data in config_data.items():
                section = ConfigSection(section_name)
                
                if section == ConfigSection.AI:
                    provider = section_data.get('provider', 'openai')
                    providers = section_data.get('providers', {})
                    providers_dict = {
                        provider_name: AIProviderConfig(**provider_config)
                        for provider_name, provider_config in providers.items()
                    }
                    self.sections[section] = AIConfig(
                        provider=provider,
                        api_key=section_data.get('api_key', ''),
                        temperature=section_data.get('temperature', 0.7),
                        max_tokens=section_data.get('max_tokens', 1000),
                        providers=providers_dict
                    )

                elif section == ConfigSection.PROMPTS:
                    # Verarbeite Prompt-Templates
                    templates = {}
                    if 'templates' in section_data:
                        for name, template_data in section_data['templates'].items():
                            templates[name] = PromptTemplate(
                                name=template_data.get('name', ''),
                                description=template_data.get('description', ''),
                                template=template_data.get('template', ''),
                                required_variables=template_data.get('required_variables', []),
                                model=template_data.get('model', '')
                            )
                    self.sections[section] = PromptConfig(templates=templates)

                elif section == ConfigSection.CACHE:
                    # Verarbeite Cache-Konfiguration
                    self.sections[section] = CacheConfig(
                    #    enabled=section_data.get('enabled', True),
                    #    max_age=section_data.get('max_age', 7),
                    #    max_size=section_data.get('max_size', 1000)
                    )

                elif section == ConfigSection.SEARCH:
                    # Verarbeite Such-Konfiguration
                    self.sections[section] = SearchConfig(
                    #    max_results=section_data.get('max_results', 10),
                    #    min_score=section_data.get('min_score', 0.5),
                    #    timeout=section_data.get('timeout', 30)
                    )

                else:
                    self.logger.warning(f"Unbekannte Konfigurationssektion: {section_name}")

            self.logger.info("Konfiguration erfolgreich geladen")
            self.logger.info(f"Geladene Sektionen: {self.sections}")

        except Exception as e:
            self.logger.error(f"Fehler beim Laden der Konfiguration: {e}")
            self._create_default_config()
            raise

    def update_section(self, section: ConfigSection, updates: Dict) -> None:
        """Aktualisiert eine bestimmte Konfigurationssektion"""
        config_obj = self.get_section(section)
        if config_obj:
            self._update_dataclass(config_obj, updates)
            self.save_config()

    def reset_section(self, section: ConfigSection) -> None:
        """Setzt eine Konfigurationssektion auf Standardwerte zurück"""
        default_map = {
            ConfigSection.AI: AIConfig(),
            ConfigSection.SEARCH: SearchConfig(),
            ConfigSection.CACHE: CacheConfig(),
            ConfigSection.UI: UIConfig(),
            ConfigSection.EXPORT: ExportConfig()
        }
        
        if section in default_map:
            setattr(self, section.value, default_map[section])
            self.save_config()

    def get_ai_provider_config(self, provider: str) -> Dict:
        """Gibt die Konfiguration für einen bestimmten AI-Provider zurück"""
        ai_config = self.get_section(ConfigSection.AI)
        if not ai_config:
            self.logger.error("Keine AI-Konfiguration gefunden")
            return {}
        
        self.logger.debug(f"AI-Konfiguration: {ai_config}")
        self.logger.debug(f"Verfügbare Provider: {ai_config.providers.keys()}")
        
        providers = ai_config.providers.get(provider)
        if not providers:
            self.logger.error(f"Keine Konfiguration für Provider {provider} gefunden")
            return {}
        
        return {
            "name": providers.name,
            "models": providers.models,
            "api_url": providers.api_url,
            "default_model": providers.default_model
        }

    @staticmethod
    def _update_dataclass(obj: Any, data: Dict) -> None:
        """Aktualisiert ein Dataclass-Objekt mit neuen Werten"""
        for key, value in data.items():
            if hasattr(obj, key):
                setattr(obj, key, value)

    @staticmethod
    def _dataclass_to_dict(obj: Any) -> Dict:
        """Konvertiert ein Dataclass-Objekt in ein Dictionary"""
        return {k: v for k, v in obj.__dict__.items() if not k.startswith('_')}

    def validate_config(self) -> List[str]:
        """Überprüft die Konfiguration auf Fehler"""
        errors = []

        # Überprüfe AI-Konfiguration
        if not self.ai.api_key and self.ai.provider != "local":
            errors.append("Kein API-Key für den AI-Provider konfiguriert")

        # Überprüfe Cache-Konfiguration
        if self.cache.enabled and not self.cache.db_path:
            errors.append("Kein Datenbankpfad für den Cache konfiguriert")

        # Überprüfe Export-Konfiguration
        if not os.path.exists(self.export.export_directory):
            try:
                os.makedirs(self.export.export_directory)
            except Exception as e:
                errors.append(f"Export-Verzeichnis konnte nicht erstellt werden: {e}")

        return errors

    def get_environment_overrides(self) -> None:
        """Liest Überschreibungen aus Umgebungsvariablen"""
        # AI-Konfiguration
        if api_key := os.getenv('GND_SEARCH_AI_KEY'):
            self.ai.api_key = api_key
        if provider := os.getenv('GND_SEARCH_AI_PROVIDER'):
            self.ai.provider = provider

        # Cache-Konfiguration
        if cache_path := os.getenv('GND_SEARCH_CACHE_PATH'):
            self.cache.db_path = cache_path
        if cache_enabled := os.getenv('GND_SEARCH_CACHE_ENABLED'):
            self.cache.enabled = cache_enabled.lower() == 'true'

        # Export-Konfiguration
        if export_dir := os.getenv('GND_SEARCH_EXPORT_DIR'):
            self.export.export_directory = export_dir

    def list_available_templates(self) -> Dict[str, str]:
        """
        Listet alle verfügbaren Templates auf.
        
        Returns:
            Dict mit Template-Namen und deren Beschreibungen
        """
        prompt_config = self.get_section(ConfigSection.PROMPTS)
        if not prompt_config or not prompt_config.templates:
            return {}
            
        return {
            name: template.description 
            for name, template in prompt_config.templates.items()
        }