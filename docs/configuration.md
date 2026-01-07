# ALIMA Configuration System

ALIMA uses an OS-specific configuration system that follows platform conventions for storing application settings.

## Configuration File Locations

The configuration system uses a priority-based fallback system that searches for configuration files in the following order:

### Linux (XDG Base Directory Specification)
1. **Project**: `./alima_config.json` (current directory)
2. **User**: `~/.config/alima/config.json` (XDG_CONFIG_HOME)
3. **System**: `/etc/xdg/alima/config.json` (XDG_CONFIG_DIRS)
4. **Legacy**: `~/.alima_config.json` (fallback)

### macOS (Apple Guidelines)
1. **Project**: `./alima_config.json`
2. **User**: `~/Library/Application Support/ALIMA/config.json`
3. **System**: `/Library/Application Support/ALIMA/config.json`
4. **Legacy**: `~/.alima_config.json`

### Windows
1. **Project**: `./alima_config.json`
2. **User**: `%APPDATA%\ALIMA\config.json`
3. **System**: `%PROGRAMDATA%\ALIMA\config.json`
4. **Legacy**: `~/.alima_config.json`

## Configuration Structure

The configuration file uses JSON format with the following structure:

```json
{
  "database": {
    "db_type": "sqlite",
    "sqlite_path": "alima_knowledge.db",
    "host": "localhost",
    "port": 3306,
    "database": "alima_knowledge",
    "username": "alima",
    "password": "",
    "connection_timeout": 30,
    "auto_create_tables": true,
    "charset": "utf8mb4",
    "ssl_disabled": false
  },
  "llm": {
    "gemini": "your_api_key_here",
    "anthropic": "your_api_key_here",
    "openai": "your_api_key_here",
    "comet": "",
    "chatai": "",
    "ollama_host": "localhost",
    "ollama_port": 11434
  },
  "catalog": {
    "catalog_token": "your_token_here",
    "catalog_search_url": "https://your-catalog.com/search",
    "catalog_details_url": "https://your-catalog.com/details"
  },
  "system": {
    "debug": false,
    "log_level": "INFO",
    "cache_dir": "cache",
    "data_dir": "data",
    "temp_dir": "/tmp"
  },
  "config_version": "1.0",
  "last_updated": "2025-08-07T09:14:52.784000"
}
```

## CLI Configuration Management

### View Configuration Paths
```bash
python alima_cli.py db-config paths
```

### Show Current Database Configuration
```bash
python alima_cli.py db-config show
```

### Test Database Connection
```bash
python alima_cli.py db-config test
```

### Configure SQLite Database
```bash
# User scope (default)
python alima_cli.py db-config set-sqlite --path /path/to/database.db

# Project scope
python alima_cli.py db-config set-sqlite --path ./data/alima.db --scope project

# System scope (requires privileges)
python alima_cli.py db-config set-sqlite --path /var/lib/alima/db.sqlite --scope system
```

### Configure MySQL/MariaDB Database
```bash
# Basic configuration
python alima_cli.py db-config set-mysql --host db.server.com --database alima --username alima

# Full configuration with options
python alima_cli.py db-config set-mysql \
  --host db.server.com \
  --port 3306 \
  --database alima_prod \
  --username alima_user \
  --password 'secure_password' \
  --charset utf8mb4 \
  --scope user
```

## Database Support

### SQLite (Local Database)
- **Default**: Uses local SQLite database file
- **Path**: Configurable database file location
- **Advantages**: No server setup required, portable
- **Use Case**: Development, single-user environments

### MySQL/MariaDB (Remote Database)
- **Requirements**: PyMySQL package (`pip install pymysql`)
- **Features**: Full remote database support
- **SSL**: Configurable SSL connection
- **Use Case**: Production, multi-user environments

## Legacy Configuration Migration

The system automatically detects and converts legacy `~/.alima_config.json` files to the new format. You can also manually convert:

```bash
python src/utils/config_manager.py --convert-legacy
```

This will:
1. Load the legacy configuration
2. Convert to new structured format
3. Save to OS-appropriate user configuration path
4. Preserve all existing settings

## Configuration Scopes

### Project Scope
- Stored in current working directory
- Takes highest priority
- Useful for project-specific settings
- Not recommended for sensitive data (API keys)

### User Scope
- Stored in user's configuration directory
- Personal settings and API keys
- Most common for development

### System Scope
- Stored in system-wide configuration directory
- Shared across all users
- Requires administrative privileges
- Useful for enterprise deployments

## Environment Variables

The configuration system respects standard environment variables:

### Linux
- `XDG_CONFIG_HOME`: Override user config directory
- `XDG_CONFIG_DIRS`: Override system config directories

### Windows
- `APPDATA`: User application data directory
- `LOCALAPPDATA`: Local application data directory
- `PROGRAMDATA`: System-wide application data

### macOS
Standard macOS paths are used without environment variable overrides.

## Security Considerations

### API Keys and Passwords
- Store sensitive data in user or system scope
- Avoid committing API keys in project scope
- Use environment variables for CI/CD pipelines

### Database Passwords
- The CLI will prompt for passwords if not provided
- Passwords are stored in configuration files (consider encryption for production)
- Use secure database authentication methods

## Examples

### Development Setup
```bash
# Configure local SQLite database
python alima_cli.py db-config set-sqlite --path ./dev_database.db --scope project

# Set API keys in user config
# (Edit ~/.config/alima/config.json manually or use legacy conversion)
```

### Production Setup
```bash
# Configure remote MySQL database
python alima_cli.py db-config set-mysql \
  --host prod-db.company.com \
  --database alima_production \
  --username alima_prod \
  --scope system

# Test the connection
python alima_cli.py db-config test
```

### Multi-Environment Setup
```bash
# Development (project scope)
python alima_cli.py db-config set-sqlite --path ./dev.db --scope project

# Staging (user scope)  
python alima_cli.py db-config set-mysql --host staging-db --database alima_staging --username staging_user --scope user

# Production uses system scope configuration
```