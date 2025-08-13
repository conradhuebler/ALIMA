#!/usr/bin/env python3
"""
ALIMA Configuration Manager
Handles all application configuration with system-wide and fallback support.
Includes LLM providers, database settings, catalog tokens, and all other configs.
Claude Generated
"""

import json
import os
import sys
import platform
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass, asdict, field
import logging


@dataclass
class DatabaseConfig:
    """Database configuration - Claude Generated"""
    db_type: str = 'sqlite'  # 'sqlite' or 'mysql'/'mariadb' 
    
    # SQLite specific
    sqlite_path: str = 'alima_knowledge.db'
    
    # MySQL/MariaDB specific  
    host: str = 'localhost'
    port: int = 3306
    database: str = 'alima_knowledge'
    username: str = 'alima'
    password: str = ''
    
    # Connection settings
    connection_timeout: int = 30
    auto_create_tables: bool = True
    charset: str = 'utf8mb4'
    ssl_disabled: bool = False


@dataclass
class OpenAICompatibleProvider:
    """Configuration for OpenAI-compatible API providers - Claude Generated"""
    name: str = ''                    # Provider name (e.g. "ChatAI", "DeepSeek") 
    base_url: str = ''               # API base URL
    api_key: str = ''                # API key for authentication
    enabled: bool = True             # Whether provider is active
    models: List[str] = field(default_factory=list)  # Available models (optional)
    description: str = ''            # Description for UI display
    
    def __post_init__(self):
        """Validation after initialization - Claude Generated"""
        if not self.name:
            raise ValueError("Provider name cannot be empty")
        if not self.base_url:
            raise ValueError("Base URL cannot be empty")


@dataclass
class OllamaProvider:
    """Flexible Ollama provider configuration similar to OpenAI providers - Claude Generated"""
    name: str  # Alias name (e.g., "local_home", "work_server", "cloud_instance")
    host: str  # Server host (e.g., "localhost", "192.168.1.100", "ollama.example.com")
    port: int = 11434  # Server port
    api_key: str = ''  # Optional API key for authenticated access
    enabled: bool = True  # Provider enabled/disabled state
    description: str = ''  # Human-readable description
    use_ssl: bool = False  # Use HTTPS instead of HTTP
    connection_type: str = 'native_client'  # 'native_client' (native ollama library) or 'openai_compatible' (OpenAI API format)
    
    def __post_init__(self):
        """Validation after initialization - Claude Generated"""
        if not self.name:
            raise ValueError("Ollama provider name cannot be empty")
        if not self.host:
            raise ValueError("Ollama host cannot be empty")
        if self.port <= 0 or self.port > 65535:
            raise ValueError("Port must be between 1 and 65535")
            
    @property
    def base_url(self) -> str:
        """Get the complete base URL for this Ollama provider - Claude Generated"""
        # Handle case where host already contains protocol
        if self.host.startswith(('http://', 'https://')):
            host_without_protocol = self.host.split('://', 1)[1]
            protocol = 'https' if self.use_ssl else 'http'
        else:
            host_without_protocol = self.host
            protocol = 'https' if self.use_ssl else 'http'
        
        # Only add port for local/IP addresses, not for domain names with standard ports - Claude Generated
        if self._needs_explicit_port():
            port_part = f":{self.port}"
        else:
            port_part = ""
            
        if self.connection_type == 'openai_compatible':
            return f"{protocol}://{host_without_protocol}{port_part}/v1"
        else:
            return f"{protocol}://{host_without_protocol}{port_part}"
    
    def _needs_explicit_port(self) -> bool:
        """Check if explicit port is needed - Claude Generated"""
        # Extract hostname without protocol
        if '://' in self.host:
            host_part = self.host.split('://', 1)[1]
        else:
            host_part = self.host
            
        # Remove port if already in host
        host_part = host_part.split(':')[0]
        
        # Standard HTTPS/HTTP ports don't need explicit port
        if self.use_ssl and self.port == 443:
            return False
        elif not self.use_ssl and self.port == 80:
            return False
            
        # localhost and IP addresses typically need explicit ports
        if host_part in ['localhost', '127.0.0.1'] or host_part.count('.') == 3:
            return True
            
        # Domain names with non-standard ports need explicit port
        return True
    
    @property
    def display_name(self) -> str:
        """Get display name with connection info - Claude Generated"""
        status = "🔐" if self.api_key else "🔓"
        ssl_indicator = "🔒" if self.use_ssl else ""
        return f"{status}{ssl_indicator} {self.name} ({self.host}:{self.port})"

@dataclass 
class LLMConfig:
    """LLM provider configuration - Claude Generated"""
    # Individual provider API Keys (non-OpenAI compatible)
    gemini: str = ''
    anthropic: str = ''
    
    # OpenAI-compatible providers (flexible list)
    openai_compatible_providers: List[OpenAICompatibleProvider] = field(default_factory=list)
    
    # Ollama providers (flexible multi-instance list)
    ollama_providers: List[OllamaProvider] = field(default_factory=list)
    
    # Legacy ollama settings (for backward compatibility)
    ollama_host: str = 'localhost'
    ollama_port: int = 11434
    
    def get_provider_by_name(self, name: str) -> Optional[OpenAICompatibleProvider]:
        """Get OpenAI-compatible provider by name - Claude Generated"""
        for provider in self.openai_compatible_providers:
            if provider.name.lower() == name.lower():
                return provider
        return None
    
    def add_provider(self, provider: OpenAICompatibleProvider) -> bool:
        """Add new OpenAI-compatible provider - Claude Generated"""
        if self.get_provider_by_name(provider.name):
            return False  # Provider already exists
        self.openai_compatible_providers.append(provider)
        return True
    
    def remove_provider(self, name: str) -> bool:
        """Remove OpenAI-compatible provider by name - Claude Generated"""
        for i, provider in enumerate(self.openai_compatible_providers):
            if provider.name.lower() == name.lower():
                del self.openai_compatible_providers[i]
                return True
        return False
    
    def get_enabled_providers(self) -> List[OpenAICompatibleProvider]:
        """Get list of enabled OpenAI-compatible providers - Claude Generated"""
        return [p for p in self.openai_compatible_providers if p.enabled]
    
    # Ollama provider management methods - Claude Generated
    def get_ollama_provider_by_name(self, name: str) -> Optional[OllamaProvider]:
        """Get Ollama provider by name - Claude Generated"""
        for provider in self.ollama_providers:
            if provider.name.lower() == name.lower():
                return provider
        return None
    
    def add_ollama_provider(self, provider: OllamaProvider) -> bool:
        """Add new Ollama provider - Claude Generated"""
        if self.get_ollama_provider_by_name(provider.name):
            return False  # Provider already exists
        self.ollama_providers.append(provider)
        return True
    
    def remove_ollama_provider(self, name: str) -> bool:
        """Remove Ollama provider by name - Claude Generated"""
        for i, provider in enumerate(self.ollama_providers):
            if provider.name.lower() == name.lower():
                del self.ollama_providers[i]
                return True
        return False
    
    def get_enabled_ollama_providers(self) -> List[OllamaProvider]:
        """Get list of enabled Ollama providers - Claude Generated"""
        return [p for p in self.ollama_providers if p.enabled]
    
    def get_enabled_openai_providers(self) -> List[OpenAICompatibleProvider]:
        """Get list of enabled OpenAI-compatible providers - Claude Generated"""
        return [p for p in self.openai_compatible_providers if p.enabled]
    
    def get_primary_ollama_provider(self) -> Optional[OllamaProvider]:
        """Get first enabled Ollama provider (primary) - Claude Generated"""
        enabled = self.get_enabled_ollama_providers()
        return enabled[0] if enabled else None
    
    def resolve_provider_type(self, config_name: str) -> tuple[str, dict]:
        """
        Resolve configuration name to provider type and config - Claude Generated
        
        Args:
            config_name: Configuration name (e.g., "LLMachine", "ChatAI", "Gemini")
            
        Returns:
            Tuple of (provider_type, provider_config)
            
        Examples:
            "LLMachine" → ("ollama", {"host": "139.20.140.163", "port": 11434})
            "ChatAI" → ("openai", {"base_url": "...", "api_key": "..."})
            "Gemini" → ("gemini", {"api_key": "..."})
        """
        # Check Ollama providers first
        for ollama_provider in self.get_enabled_ollama_providers():
            if ollama_provider.name == config_name:
                config = {
                    "host": ollama_provider.host,
                    "port": ollama_provider.port,
                    "api_key": ollama_provider.api_key,
                    "use_ssl": ollama_provider.use_ssl,
                    "connection_type": ollama_provider.connection_type
                }
                return ("ollama", config)
                
        # Check OpenAI-compatible providers
        for openai_provider in self.get_enabled_openai_providers():
            if openai_provider.name == config_name:
                config = {
                    "base_url": openai_provider.base_url,
                    "api_key": openai_provider.api_key
                }
                # Return the specific provider name, not generic "openai"
                return (openai_provider.name.lower(), config)
                
        # Check static providers
        if config_name.lower() == "gemini":
            config = {"api_key": self.gemini}
            return ("gemini", config)
        elif config_name.lower() == "anthropic":
            config = {"api_key": self.anthropic}
            return ("anthropic", config)
            
        # Unknown provider - return as-is for backward compatibility
        return (config_name, {})
    
    @classmethod
    def create_default(cls) -> 'LLMConfig':
        """Create LLMConfig with default OpenAI-compatible and Ollama providers - Claude Generated"""
        default_openai_providers = [
            OpenAICompatibleProvider(
                name="ChatAI",
                base_url="http://chat-ai.academiccloud.de/v1",
                api_key="",
                enabled=True,
                description="GWDG Academic Cloud ChatAI"
            ),
            OpenAICompatibleProvider(
                name="OpenAI",
                base_url="https://api.openai.com/v1", 
                api_key="",
                enabled=False,
                description="Official OpenAI API"
            ),
            OpenAICompatibleProvider(
                name="Comet",
                base_url="https://api.cometapi.com/v1",
                api_key="",
                enabled=False,
                description="Comet API"
            )
        ]
        
        default_ollama_providers = [
            OllamaProvider(
                name="localhost",
                host="localhost",
                port=11434,
                api_key="",
                enabled=True,
                description="Local Ollama instance",
                use_ssl=False,
                connection_type="native_client"
            )
        ]
        
        return cls(
            gemini="",
            anthropic="",
            openai_compatible_providers=default_openai_providers,
            ollama_providers=default_ollama_providers,
            ollama_host="localhost",
            ollama_port=11434
        )


@dataclass
class CatalogConfig:
    """Library catalog configuration - Claude Generated"""
    catalog_token: str = ''
    catalog_search_url: str = ''
    catalog_details_url: str = ''


@dataclass
class SystemConfig:
    """System-wide configuration - Claude Generated"""
    debug: bool = False
    log_level: str = 'INFO'
    cache_dir: str = 'cache'
    data_dir: str = 'data'
    temp_dir: str = '/tmp'


@dataclass
class AlimaConfig:
    """Complete ALIMA configuration - Claude Generated"""
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    llm: LLMConfig = field(default_factory=lambda: LLMConfig.create_default())
    catalog: CatalogConfig = field(default_factory=CatalogConfig)
    system: SystemConfig = field(default_factory=SystemConfig)
    
    # Version and metadata
    config_version: str = '1.0'
    last_updated: str = ''


class ConfigManager:
    """Manages all ALIMA configuration with OS-specific paths and priority fallback - Claude Generated"""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        
        # Get OS-specific configuration paths
        self._setup_config_paths()
        
        self._config: Optional[AlimaConfig] = None
    
    def _setup_config_paths(self):
        """Setup OS-specific configuration file paths - Claude Generated"""
        system_name = platform.system().lower()
        
        # Project config is always in current directory
        self.project_config_path = Path("alima_config.json")
        
        if system_name == "windows":
            # Windows paths
            appdata = os.environ.get("APPDATA", str(Path.home() / "AppData" / "Roaming"))
            local_appdata = os.environ.get("LOCALAPPDATA", str(Path.home() / "AppData" / "Local"))
            programdata = os.environ.get("PROGRAMDATA", "C:\\ProgramData")
            
            self.user_config_path = Path(appdata) / "ALIMA" / "config.json"
            self.system_config_path = Path(programdata) / "ALIMA" / "config.json"
            self.legacy_config_path = Path.home() / ".alima_config.json"
            
        elif system_name == "darwin":  # macOS
            # macOS paths following Apple guidelines
            self.user_config_path = Path.home() / "Library" / "Application Support" / "ALIMA" / "config.json"
            self.system_config_path = Path("/Library") / "Application Support" / "ALIMA" / "config.json"
            self.legacy_config_path = Path.home() / ".alima_config.json"
            
        else:  # Linux and other Unix-like systems
            # Follow XDG Base Directory Specification
            xdg_config_home = os.environ.get("XDG_CONFIG_HOME", str(Path.home() / ".config"))
            xdg_config_dirs = os.environ.get("XDG_CONFIG_DIRS", "/etc/xdg").split(":")
            
            self.user_config_path = Path(xdg_config_home) / "alima" / "config.json"
            # Use first directory from XDG_CONFIG_DIRS, fallback to /etc
            system_config_dir = Path(xdg_config_dirs[0]) if xdg_config_dirs else Path("/etc")
            self.system_config_path = system_config_dir / "alima" / "config.json"
            self.legacy_config_path = Path.home() / ".alima_config.json"
        
        self.logger.debug(f"Config paths for {system_name}:")
        self.logger.debug(f"  Project: {self.project_config_path}")
        self.logger.debug(f"  User: {self.user_config_path}")
        self.logger.debug(f"  System: {self.system_config_path}")
        self.logger.debug(f"  Legacy: {self.legacy_config_path}")
    
    def get_config_info(self) -> Dict[str, str]:
        """Get information about configuration paths - Claude Generated"""
        system_name = platform.system()
        return {
            "os": system_name,
            "project_config": str(self.project_config_path),
            "user_config": str(self.user_config_path),
            "system_config": str(self.system_config_path),
            "legacy_config": str(self.legacy_config_path)
        }
        
    def load_config(self) -> AlimaConfig:
        """Load configuration with priority fallback - Claude Generated"""
        if self._config is not None:
            return self._config
            
        # Try loading from different sources (priority order)
        config_sources = [
            ("project", self.project_config_path),
            ("user", self.user_config_path), 
            ("system", self.system_config_path),
            ("legacy", self.legacy_config_path)
        ]
        
        for source_name, config_path in config_sources:
            if config_path.exists():
                try:
                    config_data = self._load_config_file(config_path)
                    if config_data:
                        self._config = self._parse_config(config_data, source_name)
                        self.logger.info(f"Loaded config from {source_name}: {config_path}")
                        return self._config
                except Exception as e:
                    self.logger.warning(f"Failed to load config from {source_name} ({config_path}): {e}")
                    continue
        
        # Use default configuration if nothing found
        self.logger.info("Using default configuration")
        self._config = AlimaConfig()
        return self._config
    
    def _load_config_file(self, config_path: Path) -> Optional[Dict[str, Any]]:
        """Load JSON config file - Claude Generated"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            self.logger.error(f"Invalid JSON in config ({config_path}): {e}")
            return None
        except Exception as e:
            self.logger.error(f"Error reading config ({config_path}): {e}")
            return None
    
    def _parse_config(self, config_data: Dict[str, Any], source_name: str) -> AlimaConfig:
        """Parse configuration data into AlimaConfig - Claude Generated"""
        
        # Handle legacy config format
        if source_name == "legacy":
            return self._parse_legacy_config(config_data)
        
        # Parse new format
        config = AlimaConfig()
        
        # Database section
        if "database" in config_data:
            db_data = config_data["database"]
            config.database = DatabaseConfig(
                db_type=db_data.get("db_type", "sqlite"),
                sqlite_path=db_data.get("sqlite_path", "alima_knowledge.db"),
                host=db_data.get("host", "localhost"),
                port=db_data.get("port", 3306),
                database=db_data.get("database", "alima_knowledge"),
                username=db_data.get("username", "alima"),
                password=db_data.get("password", ""),
                connection_timeout=db_data.get("connection_timeout", 30),
                auto_create_tables=db_data.get("auto_create_tables", True),
                charset=db_data.get("charset", "utf8mb4"),
                ssl_disabled=db_data.get("ssl_disabled", False)
            )
        
        # LLM section
        if "llm" in config_data:
            llm_data = config_data["llm"]
            
            # Parse OpenAI-compatible providers
            openai_providers = []
            if "openai_compatible_providers" in llm_data:
                # New format: list of provider objects
                for provider_data in llm_data["openai_compatible_providers"]:
                    try:
                        provider = OpenAICompatibleProvider(
                            name=provider_data.get("name", ""),
                            base_url=provider_data.get("base_url", ""),
                            api_key=provider_data.get("api_key", ""),
                            enabled=provider_data.get("enabled", True),
                            models=provider_data.get("models", []),
                            description=provider_data.get("description", "")
                        )
                        openai_providers.append(provider)
                    except ValueError as e:
                        self.logger.warning(f"Skipping invalid provider: {e}")
            else:
                # Legacy format: convert individual fields to providers
                if llm_data.get("openai"):
                    openai_providers.append(OpenAICompatibleProvider(
                        name="OpenAI",
                        base_url="https://api.openai.com/v1",
                        api_key=llm_data["openai"],
                        enabled=True,
                        description="Official OpenAI API"
                    ))
                if llm_data.get("chatai"):
                    openai_providers.append(OpenAICompatibleProvider(
                        name="ChatAI",
                        base_url="http://chat-ai.academiccloud.de/v1",
                        api_key=llm_data["chatai"],
                        enabled=True,
                        description="GWDG Academic Cloud ChatAI"
                    ))
                if llm_data.get("comet"):
                    openai_providers.append(OpenAICompatibleProvider(
                        name="Comet",
                        base_url="https://api.cometapi.com/v1",
                        api_key=llm_data["comet"],
                        enabled=True,
                        description="Comet API"
                    ))
            
            # Parse Ollama providers (new multi-instance format)
            ollama_providers = []
            if "ollama_providers" in llm_data:
                # New format: list of Ollama provider objects
                for provider_data in llm_data["ollama_providers"]:
                    try:
                        provider = OllamaProvider(
                            name=provider_data.get("name", ""),
                            host=provider_data.get("host", "localhost"),
                            port=provider_data.get("port", 11434),
                            api_key=provider_data.get("api_key", ""),
                            enabled=provider_data.get("enabled", True),
                            description=provider_data.get("description", ""),
                            use_ssl=provider_data.get("use_ssl", False),
                            connection_type=provider_data.get("connection_type", "native_client")
                        )
                        ollama_providers.append(provider)
                    except ValueError as e:
                        self.logger.warning(f"Skipping invalid Ollama provider: {e}")
            elif "ollama" in llm_data:
                # Migration from old OllamaConfig format
                ollama_data = llm_data["ollama"]
                if ollama_data.get("local_enabled", True):
                    ollama_providers.append(OllamaProvider(
                        name="localhost_migrated",
                        host=ollama_data.get("local_host", "localhost"),
                        port=ollama_data.get("local_port", 11434),
                        enabled=True,
                        description="Migrated from legacy config",
                        connection_type="native_client"
                    ))
            else:
                # Legacy fallback: create default localhost provider
                ollama_providers.append(OllamaProvider(
                    name="localhost",
                    host=llm_data.get("ollama_host", "localhost"),
                    port=llm_data.get("ollama_port", 11434),
                    enabled=True,
                    description="Legacy configuration",
                    connection_type="native_client"
                ))
            
            config.llm = LLMConfig(
                gemini=llm_data.get("gemini", ""),
                anthropic=llm_data.get("anthropic", ""),
                openai_compatible_providers=openai_providers,
                ollama_providers=ollama_providers,
                ollama_host=llm_data.get("ollama_host", "localhost"),
                ollama_port=llm_data.get("ollama_port", 11434)
            )
        
        # Catalog section
        if "catalog" in config_data:
            cat_data = config_data["catalog"]
            config.catalog = CatalogConfig(
                catalog_token=cat_data.get("catalog_token", ""),
                catalog_search_url=cat_data.get("catalog_search_url", ""),
                catalog_details_url=cat_data.get("catalog_details_url", "")
            )
        
        # System section  
        if "system" in config_data:
            sys_data = config_data["system"]
            config.system = SystemConfig(
                debug=sys_data.get("debug", False),
                log_level=sys_data.get("log_level", "INFO"),
                cache_dir=sys_data.get("cache_dir", "cache"),
                data_dir=sys_data.get("data_dir", "data"),
                temp_dir=sys_data.get("temp_dir", "/tmp")
            )
        
        # Metadata
        config.config_version = config_data.get("config_version", "1.0")
        config.last_updated = config_data.get("last_updated", "")
        
        return config
    
    def _parse_legacy_config(self, config_data: Dict[str, Any]) -> AlimaConfig:
        """Parse legacy .alima_config.json format - Claude Generated"""
        config = AlimaConfig()
        
        # Map legacy fields to new structure
        config.llm.gemini = config_data.get("gemini", "")
        config.llm.anthropic = config_data.get("anthropic", "")
        config.llm.openai = config_data.get("openai", "")
        config.llm.comet = config_data.get("comet", "")
        config.llm.chatai = config_data.get("chatai", "")
        config.llm.ollama_host = config_data.get("ollama_host", "localhost")
        config.llm.ollama_port = int(config_data.get("ollama_port", 11434))
        
        config.catalog.catalog_token = config_data.get("catalog_token", "")
        config.catalog.catalog_search_url = config_data.get("catalog_search_url", "")
        config.catalog.catalog_details_url = config_data.get("catalog_details", "")
        
        # Database defaults to SQLite for legacy
        config.database.db_type = "sqlite"
        config.database.sqlite_path = "alima_knowledge.db"
        
        self.logger.info("Converted legacy config to new format")
        return config
    
    def save_config(self, config: AlimaConfig, scope: str = "user") -> bool:
        """Save configuration - Claude Generated"""
        try:
            if scope == "system":
                config_path = self.system_config_path
            elif scope == "project":
                config_path = self.project_config_path
            else:  # user (default)
                config_path = self.user_config_path
            
            # Create directory if needed for all scopes
            config_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Update metadata
            from datetime import datetime
            config.last_updated = datetime.now().isoformat()
            
            # Create configuration structure
            config_dict = asdict(config)
            config_data = {
                **config_dict,
                "_metadata": {
                    "version": config.config_version,
                    "created_by": "ALIMA Configuration Manager",
                    "scope": scope
                }
            }
            
            # Write configuration file
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Saved config to {scope}: {config_path}")
            
            # Update cached config
            self._config = config
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save {scope} config: {e}")
            return False
    
    def get_database_connection_string(self) -> str:
        """Get database connection string - Claude Generated"""
        config = self.load_config()
        db = config.database
        
        if db.db_type == 'sqlite':
            return f"sqlite:///{db.sqlite_path}"
        elif db.db_type in ['mysql', 'mariadb']:
            ssl_part = "?ssl_disabled=true" if db.ssl_disabled else ""
            return f"mysql://{db.username}:{db.password}@{db.host}:{db.port}/{db.database}{ssl_part}"
        else:
            raise ValueError(f"Unsupported database type: {db.db_type}")
    
    def get_database_connection_info(self) -> Dict[str, Any]:
        """Get database connection info for direct access - Claude Generated"""
        config = self.load_config()
        db = config.database
        
        if db.db_type == 'sqlite':
            return {
                'type': 'sqlite',
                'path': db.sqlite_path,
                'connection_timeout': db.connection_timeout
            }
        elif db.db_type in ['mysql', 'mariadb']:
            return {
                'type': db.db_type,
                'host': db.host,
                'port': db.port,
                'database': db.database,
                'username': db.username,
                'password': db.password,
                'charset': db.charset,
                'connection_timeout': db.connection_timeout,
                'ssl_disabled': db.ssl_disabled
            }
        else:
            raise ValueError(f"Unsupported database type: {db.db_type}")
    
    def test_database_connection(self) -> tuple[bool, str]:
        """Test database connection - Claude Generated"""
        try:
            config = self.load_config()
            db = config.database
            
            if db.db_type == 'sqlite':
                import sqlite3
                conn = sqlite3.connect(db.sqlite_path, timeout=db.connection_timeout)
                cursor = conn.cursor()
                cursor.execute("SELECT 1")
                conn.close()
                return True, f"SQLite connection successful: {db.sqlite_path}"
                
            elif db.db_type in ['mysql', 'mariadb']:
                try:
                    import pymysql
                except ImportError:
                    return False, "PyMySQL not installed. Install with: pip install pymysql"
                
                connection = pymysql.connect(
                    host=db.host,
                    port=db.port,
                    user=db.username,
                    password=db.password,
                    database=db.database,
                    charset=db.charset,
                    connect_timeout=db.connection_timeout,
                    ssl_disabled=db.ssl_disabled
                )
                with connection.cursor() as cursor:
                    cursor.execute("SELECT 1")
                connection.close()
                return True, f"MySQL connection successful: {db.username}@{db.host}:{db.port}/{db.database}"
                
            else:
                return False, f"Unsupported database type: {db.db_type}"
                
        except Exception as e:
            return False, f"Connection failed: {str(e)}"
    
    def get_llm_config(self) -> Dict[str, Any]:
        """Get LLM configuration in legacy format for compatibility - Claude Generated"""
        config = self.load_config()
        llm = config.llm
        
        # Create legacy format with individual provider keys for backwards compatibility
        result = {
            "gemini": llm.gemini,
            "anthropic": llm.anthropic,
            "ollama_host": llm.ollama_host,
            "ollama_port": str(llm.ollama_port)
        }
        
        # Add individual provider keys for legacy compatibility
        for provider in llm.openai_compatible_providers:
            if provider.name.lower() == "openai":
                result["openai"] = provider.api_key
            elif provider.name.lower() == "chatai":
                result["chatai"] = provider.api_key  
            elif provider.name.lower() == "comet":
                result["comet"] = provider.api_key
        
        return result
    
    def get_catalog_config(self) -> Dict[str, Any]:
        """Get catalog configuration in unified format - Claude Generated"""
        config = self.load_config()
        cat = config.catalog
        
        return {
            "catalog_token": cat.catalog_token,
            "catalog_search_url": cat.catalog_search_url,
            "catalog_details_url": cat.catalog_details_url  # Unified key name - Claude Generated
        }
    
    def create_sample_configs(self) -> Dict[str, str]:
        """Create sample configuration files - Claude Generated"""
        samples = {}
        
        # Complete sample config
        sample_config = AlimaConfig(
            database=DatabaseConfig(
                db_type='sqlite',
                sqlite_path='alima_knowledge.db'
            ),
            llm=LLMConfig(
                gemini='your_gemini_api_key_here',
                anthropic='your_anthropic_api_key_here',
                openai='your_openai_api_key_here',
                ollama_host='localhost',
                ollama_port=11434
            ),
            catalog=CatalogConfig(
                catalog_token='your_catalog_token_here',
                catalog_search_url='https://your-catalog-server.com/search',
                catalog_details_url='https://your-catalog-server.com/details'
            ),
            system=SystemConfig(
                debug=False,
                log_level='INFO',
                cache_dir='cache',
                data_dir='data'
            )
        )
        
        samples['complete'] = json.dumps(asdict(sample_config), indent=2)
        
        # MySQL variant
        mysql_config = AlimaConfig(
            database=DatabaseConfig(
                db_type='mysql',
                host='localhost',
                port=3306,
                database='alima_knowledge',
                username='alima',
                password='your_password_here',
                charset='utf8mb4'
            )
        )
        samples['mysql'] = json.dumps(asdict(mysql_config), indent=2)
        
        return samples


# Global instance for easy access
_config_manager = None

def get_config_manager() -> ConfigManager:
    """Get global configuration manager - Claude Generated"""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager


# Convenience functions for backward compatibility
def get_config() -> AlimaConfig:
    """Get current configuration - Claude Generated"""
    return get_config_manager().load_config()

def get_llm_config() -> Dict[str, Any]:
    """Get LLM configuration - Claude Generated"""
    return get_config_manager().get_llm_config()

def get_catalog_config() -> Dict[str, Any]: 
    """Get catalog configuration - Claude Generated"""
    return get_config_manager().get_catalog_config()


if __name__ == "__main__":
    # Demo/test functionality
    import argparse
    
    parser = argparse.ArgumentParser(description="ALIMA Configuration Manager")
    parser.add_argument("--show-config", action="store_true", help="Show current configuration")
    parser.add_argument("--show-paths", action="store_true", help="Show OS-specific configuration paths")
    parser.add_argument("--test-db", action="store_true", help="Test database connection")
    parser.add_argument("--create-samples", action="store_true", help="Show sample configurations")
    parser.add_argument("--convert-legacy", action="store_true", help="Convert legacy config to new format")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    manager = ConfigManager()
    
    if args.show_config:
        config = manager.load_config()
        print("Current Configuration:")
        print(json.dumps(asdict(config), indent=2))
    
    if args.show_paths:
        config_info = manager.get_config_info()
        print(f"🖥️  Configuration Paths for {config_info['os']}:")
        print(f"   Project:  {config_info['project_config']}")
        print(f"   User:     {config_info['user_config']}")
        print(f"   System:   {config_info['system_config']}")
        print(f"   Legacy:   {config_info['legacy_config']}")
        print()
        
        # Show which files exist
        from pathlib import Path
        paths = [
            ("Project", config_info['project_config']),
            ("User", config_info['user_config']),
            ("System", config_info['system_config']),
            ("Legacy", config_info['legacy_config'])
        ]
        
        print("📁 File Status:")
        for name, path in paths:
            exists = Path(path).exists()
            status = "✅ EXISTS" if exists else "❌ NOT FOUND"
            print(f"   {name:8}: {status}")
    
    if args.test_db:
        success, message = manager.test_database_connection()
        print(f"Database Test: {'✅ SUCCESS' if success else '❌ FAILED'}")
        print(f"Message: {message}")
    
    if args.create_samples:
        samples = manager.create_sample_configs()
        print("=== Complete Configuration Sample ===")
        print(samples['complete'])
        print("\n=== MySQL Configuration Sample ===")
        print(samples['mysql'])
    
    if args.convert_legacy:
        config = manager.load_config()
        success = manager.save_config(config, "user")
        if success:
            print("✅ Legacy config converted and saved to ~/.alima_config_v2.json")
        else:
            print("❌ Failed to convert legacy config")