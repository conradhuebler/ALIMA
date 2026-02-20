#!/usr/bin/env python3
"""
ALIMA CLI First-Start Setup Wizard
Interactive terminal-based setup for new users
Claude Generated
"""

import sys
import logging
from typing import Optional, List

from .config_manager import ConfigManager
from .config_models import TaskType
from .setup_utils import (
    OllamaConnectionValidator, APIKeyValidator, GNDDatabaseDownloader,
    ConfigurationBuilder, PromptValidator, SetupResult
)


logger = logging.getLogger(__name__)


class CLISetupWizard:
    """Interactive CLI setup wizard - Claude Generated"""

    def __init__(self):
        self.config_manager = ConfigManager()
        self.provider_type = None
        self.provider_name = None
        self.base_url = None
        self.api_key = None
        self.available_models = []
        self.task_model_selections = {}  # {task_type.value: model_name}
        # Catalog SOAP settings (optional) - Claude Generated
        self.catalog_search_url = ""
        self.catalog_details_url = ""
        self.catalog_token = ""
        # Load institution presets - Claude Generated
        from .preset_loader import PresetLoader
        self._presets = PresetLoader.load()
        if self._presets.institution_name:
            print(f"💼 Institutions-Presets geladen: {self._presets.institution_name}")

    def run(self) -> bool:
        """Run the setup wizard - Claude Generated

        Returns:
            bool: True if setup completed successfully
        """
        try:
            self._print_welcome()
            self._setup_llm_provider()
            self._collect_task_model_selections()  # Collect task-specific model preferences
            self._setup_gnd_database()
            self._setup_catalog()  # Optional SOAP catalog step - Claude Generated
            self._print_summary()

            # Create and save configuration
            config = ConfigurationBuilder.create_initial_config(
                provider_type=self.provider_type,
                provider_name=self.provider_name,
                base_url=self.base_url,
                api_key=self.api_key,
                models=self.available_models,
                task_model_selections=self.task_model_selections  # Pass task selections
            )
            config.system_config.first_run_completed = True

            # Apply catalog SOAP settings if configured - Claude Generated
            if self.catalog_search_url or self.catalog_details_url or self.catalog_token:
                from .config_models import CatalogConfig
                config.catalog_config = CatalogConfig(
                    catalog_search_url=self.catalog_search_url,
                    catalog_details_url=self.catalog_details_url,
                    catalog_token=self.catalog_token,
                )

            self.config_manager.save_config(config)
            logger.info("CLI setup wizard completed successfully")
            print("\n✅ Konfiguration gespeichert! Sie können ALIMA jetzt starten.\n")
            return True

        except KeyboardInterrupt:
            print("\n\n❌ Setup vom Benutzer abgebrochen")
            return False
        except Exception as e:
            logger.error(f"Setup error: {str(e)}")
            print(f"\n❌ Setup-Fehler: {str(e)}")
            return False

    def _print_welcome(self):
        """Print welcome message - Claude Generated"""
        print("\n" + "=" * 60)
        print("🚀 Willkommen beim ALIMA Setup-Assistenten")
        print("=" * 60)
        print("\nALIMA (Automatische Bibliotheksindexierung und Metadatenanalyse)")
        print("hilft Ihnen bei der Analyse von Bibliotheksmaterialien und der")
        print("Extraktion von Metadaten mit KI.")
        print("\nDieser Assistent führt Sie durch:")
        print("  1. Einrichtung eines LLM-Anbieters (lokal oder Cloud)")
        print("  2. Optionaler Download der GND-Normdaten")
        print("  3. Überprüfung Ihrer Konfiguration")
        print("\n💡 Sie können diese Einstellungen jederzeit im Einstellungsmenü ändern.")
        print("=" * 60 + "\n")

    def _setup_llm_provider(self):
        """Interactive LLM provider setup - Claude Generated"""
        print("\n📌 Schritt 1: LLM-Anbieter Konfiguration\n")
        print("Wählen Sie Ihren LLM-Anbieter:\n")
        print("  1) Ollama (Lokaler Server) - Empfohlen")
        print("  2) OpenAI-Compatible API")
        print("  3) Google Gemini API")
        print("  4) Anthropic Claude API")

        while True:
            choice = input("\nAnbieter auswählen (1-4): ").strip()
            if choice in ['1', '2', '3', '4']:
                break
            print("❌ Ungültige Auswahl. Bitte wählen Sie 1-4.")

        if choice == '1':
            self._setup_ollama()
        elif choice == '2':
            self._setup_openai_compatible()
        elif choice == '3':
            self._setup_gemini()
        elif choice == '4':
            self._setup_anthropic()

    def _setup_ollama(self):
        """Setup Ollama provider - Claude Generated"""
        print("\n🔹 Ollama Konfiguration\n")

        # Use preset URL as default if available - Claude Generated
        preset_host = "localhost"
        preset_port = 11434
        if self._presets.llm_provider_type in ('ollama', '') and self._presets.llm_base_url:
            try:
                from urllib.parse import urlparse
                parsed = urlparse(self._presets.llm_base_url)
                preset_host = parsed.hostname or preset_host
                preset_port = parsed.port or preset_port
            except Exception:
                pass

        host = input(f"Ollama Host (Standard: {preset_host}): ").strip() or preset_host
        port_str = input(f"Ollama Port (Standard: {preset_port}): ").strip() or str(preset_port)

        try:
            port = int(port_str)
        except ValueError:
            port = 11434
            print(f"Ungültiger Port, verwende Standard: {port}")

        print("\n🧪 Teste Verbindung...\n")
        result = OllamaConnectionValidator.test_native(host, port)

        if result.success:
            print(f"✅ {result.message}\n")
            print(f"Verfügbare Modelle: {', '.join(result.data[:5])}")
            if len(result.data) > 5:
                print(f"   ... und {len(result.data) - 5} weitere")

            self.provider_type = "ollama"
            self.provider_name = self.provider_type  # BUGFIX: Use provider_type for consistency
            self.base_url = f"http://{host}:{port}"
            self.available_models = result.data
        else:
            print(f"❌ {result.message}\n")
            retry = input("Andere Einstellungen versuchen? (j/n): ").lower()
            if retry == 'j':
                self._setup_ollama()
            else:
                raise Exception("Ollama-Setup fehlgeschlagen")

    def _setup_openai_compatible(self):
        """Setup OpenAI-compatible provider - Claude Generated"""
        print("\n🔷 OpenAI-Compatible API Konfiguration\n")

        # Use preset URL/key as defaults if provider type matches - Claude Generated
        preset_url = ""
        preset_key = ""
        if self._presets.llm_provider_type == 'openai_compatible':
            preset_url = self._presets.llm_base_url
            preset_key = self._presets.llm_api_key

        prompt_url = f"API Basis-URL (Standard: {preset_url}): " if preset_url else "API Basis-URL (z.B. http://localhost:8000/v1): "
        base_url = input(prompt_url).strip() or preset_url
        if not base_url:
            print("❌ Basis-URL erforderlich")
            return self._setup_openai_compatible()

        prompt_key = f"API-Schlüssel (Standard: {'(aus Preset)' if preset_key else 'leer'}): " if preset_key else "API-Schlüssel (oder leer lassen falls nicht erforderlich): "
        api_key = input(prompt_key).strip() or preset_key

        print("\n🧪 Teste Verbindung...\n")
        result = OllamaConnectionValidator.test_openai_compatible(base_url, api_key)

        if result.success:
            print(f"✅ {result.message}\n")
            print(f"Verfügbare Modelle: {', '.join(result.data[:5])}")
            if len(result.data) > 5:
                print(f"   ... und {len(result.data) - 5} weitere")

            self.provider_type = "openai_compatible"
            self.provider_name = f"OpenAI-Compatible ({base_url})"
            self.base_url = base_url
            self.api_key = api_key
            self.available_models = result.data
        else:
            print(f"❌ {result.message}\n")
            retry = input("Andere Einstellungen versuchen? (j/n): ").lower()
            if retry == 'j':
                self._setup_openai_compatible()
            else:
                raise Exception("OpenAI-compatible Setup fehlgeschlagen")

    def _setup_gemini(self):
        """Setup Gemini provider - Claude Generated"""
        print("\n🟡 Google Gemini API Konfiguration\n")
        print("API-Schlüssel erhalten Sie unter: https://aistudio.google.com/app/apikey\n")

        # Use preset key as default if provider type matches - Claude Generated
        preset_key = self._presets.llm_api_key if self._presets.llm_provider_type == 'gemini' else ""
        prompt = f"Gemini API-Schlüssel (Standard: aus Preset): " if preset_key else "Geben Sie Ihren Gemini API-Schlüssel ein: "
        api_key = input(prompt).strip() or preset_key
        if not api_key:
            print("❌ API-Schlüssel erforderlich")
            return self._setup_gemini()

        print("\n🧪 Teste API-Schlüssel...\n")
        result = APIKeyValidator.validate_gemini(api_key)

        if result.success:
            print(f"✅ {result.message}\n")
            self.provider_type = "gemini"
            self.provider_name = "Google Gemini"
            self.api_key = api_key
            self.available_models = ["gemini-1.5-flash", "gemini-1.5-pro"]
        else:
            print(f"❌ {result.message}\n")
            retry = input("Anderen API-Schlüssel versuchen? (j/n): ").lower()
            if retry == 'j':
                self._setup_gemini()
            else:
                raise Exception("Gemini-Setup fehlgeschlagen")

    def _setup_anthropic(self):
        """Setup Anthropic provider - Claude Generated"""
        print("\n🔴 Anthropic Claude API Konfiguration\n")
        print("API-Schlüssel erhalten Sie unter: https://console.anthropic.com/\n")

        # Use preset key as default if provider type matches - Claude Generated
        preset_key = self._presets.llm_api_key if self._presets.llm_provider_type == 'anthropic' else ""
        prompt = f"Anthropic API-Schlüssel (Standard: aus Preset): " if preset_key else "Geben Sie Ihren Anthropic API-Schlüssel ein: "
        api_key = input(prompt).strip() or preset_key
        if not api_key:
            print("❌ API-Schlüssel erforderlich")
            return self._setup_anthropic()

        print("\n🧪 Teste API-Schlüssel...\n")
        result = APIKeyValidator.validate_anthropic(api_key)

        if result.success:
            print(f"✅ {result.message}\n")
            self.provider_type = "anthropic"
            self.provider_name = "Anthropic Claude"
            self.api_key = api_key
            self.available_models = ["claude-3-5-haiku-20241022", "claude-3-5-sonnet-20241022"]
        else:
            print(f"❌ {result.message}\n")
            retry = input("Anderen API-Schlüssel versuchen? (j/n): ").lower()
            if retry == 'j':
                self._setup_anthropic()
            else:
                raise Exception("Anthropic-Setup fehlgeschlagen")

    def _collect_task_model_selections(self):
        """Collect task-specific model preferences from user - Claude Generated"""
        if not self.available_models:
            print("⚠️  Keine Modelle verfügbar, überspringe Task-Auswahl")
            return

        print("\n\n📌 Schritt 2: Modell-Auswahl für Pipeline-Schritte\n")
        print("Verschiedene Schritte haben unterschiedliche Anforderungen:")
        print("  • Initialisation & Keywords: Benötigen Reasoning-Fähigkeiten")
        print("  • Classification: Benötigt strukturiertes Denken")
        print("  • Vision: Benötigt Bildverständnis\n")

        # Use shared LLM task configuration from config_models - Claude Generated
        # Single source of truth for consistent task lists across all wizards/dialogs
        from .config_models import LLM_TASK_DISPLAY_INFO

        # Convert shared constant to CLI format (remove icons from labels)
        llm_tasks = []
        for task_type, icon_label, description in LLM_TASK_DISPLAY_INFO:
            # Extract label without emoji icon (remove first ~2-3 chars: emoji + space)
            # Example: "🔤 Initialisation" -> "Initialisation"
            label = icon_label.split(' ', 1)[1] if ' ' in icon_label else icon_label
            llm_tasks.append((task_type, label, description))

        print("Verfügbare Modelle:\n")
        for idx, model in enumerate(self.available_models[:10]):
            print(f"  {idx + 1}) {model}")
        if len(self.available_models) > 10:
            print(f"  ... und {len(self.available_models) - 10} weitere")

        print("\n" + "-" * 60)
        print("Drücken Sie ENTER um das Standard-Modell für alle Tasks zu verwenden")
        print("Oder geben Sie die Modellnummer (1-X) ein\n")

        default_model = self.available_models[0]

        for task_type, task_label, task_desc in llm_tasks:
            print(f"\n{task_label} ({task_desc})")
            print(f"  Standard: {default_model}")

            choice = input(f"Modell für {task_label} (1-{len(self.available_models)}/Enter für Standard): ").strip()

            if choice:
                try:
                    idx = int(choice) - 1
                    if 0 <= idx < len(self.available_models):
                        selected_model = self.available_models[idx]
                        self.task_model_selections[task_type.name] = selected_model
                        print(f"  ✓ {task_label}: {selected_model}")
                    else:
                        print(f"  ⚠️  Ungültige Auswahl, verwende Standard")
                        self.task_model_selections[task_type.name] = default_model
                except ValueError:
                    print(f"  ⚠️  Ungültige Eingabe, verwende Standard")
                    self.task_model_selections[task_type.name] = default_model
            else:
                self.task_model_selections[task_type.name] = default_model
                print(f"  ✓ {task_label}: {default_model}")

        print("\n✅ Task-Modell-Auswahl gespeichert\n")

    def _setup_gnd_database(self):
        """Interactive GND database setup - Claude Generated"""
        print("\n\n📌 Schritt 3: GND-Normdatenbank (Optional)\n")
        print("Die GND (Gemeinsame Normdatei) Datenbank enthält deutsche Schlagwörter.")
        print("Der Download verbessert Schlagwort-Vorschläge und Suchgenauigkeit.\n")
        print("Optionen:")
        print("  1) Von DNB herunterladen (~300 MB, ~5-10 Min. Import)")
        print("  2) Datenbankdatei importieren (schnell, wenn .db-Datei vorhanden)")
        print("  3) Überspringen (Lobid-API verwenden)\n")

        while True:
            choice = input("Option auswählen (1-3): ").strip()
            if choice in ['1', '2', '3']:
                break
            print("❌ Ungültige Auswahl. Bitte wählen Sie 1-3.")

        if choice == '1':
            self._download_gnd_database()
        elif choice == '2':
            self._import_gnd_database_file()
        else:
            print("⏭️  GND-Datenbank übersprungen (verwende Lobid-API)")

    def _download_gnd_database(self):
        """Download GND database - Claude Generated"""
        print("\n🌐 Lade GND-Datenbank von der DNB herunter...")
        print("   Dies kann einige Minuten dauern...\n")

        def progress_callback(percent):
            # Print progress bar
            bar_length = 40
            filled = int(bar_length * percent // 100)
            bar = '█' * filled + '░' * (bar_length - filled)
            print(f"\r📦 [{bar}] {percent}%", end='', flush=True)

        result = GNDDatabaseDownloader.download(progress_callback)

        if result.success:
            print(f"\n✅ GND-Datenbank erfolgreich heruntergeladen")
            print(f"   Speicherort: {result.data}\n")
        else:
            print(f"\n❌ Download fehlgeschlagen: {result.message}\n")
            retry = input("Download erneut versuchen? (j/n): ").lower()
            if retry == 'j':
                self._download_gnd_database()

    def _import_gnd_database_file(self):
        """Import an existing SQLite database file by copying it - Claude Generated"""
        import shutil
        import sqlite3
        from pathlib import Path

        print("\n🗄️  Datenbankdatei importieren\n")

        while True:
            source_path = input("Pfad zur .db-Datei eingeben: ").strip()
            if not source_path:
                print("❌ Kein Pfad angegeben.")
                retry = input("Erneut versuchen? (j/n): ").lower()
                if retry != 'j':
                    return
                continue

            source = Path(source_path)
            if not source.exists():
                print(f"❌ Datei nicht gefunden: {source_path}")
                retry = input("Anderen Pfad versuchen? (j/n): ").lower()
                if retry != 'j':
                    return
                continue

            # Basic SQLite validity check
            try:
                conn = sqlite3.connect(str(source))
                conn.execute("SELECT 1")
                conn.close()
            except sqlite3.DatabaseError:
                print(f"❌ Keine gültige SQLite-Datenbank: {source_path}")
                retry = input("Anderen Pfad versuchen? (j/n): ").lower()
                if retry != 'j':
                    return
                continue

            break

        config = self.config_manager.load_config()
        target = Path(config.database_config.sqlite_path)
        target.parent.mkdir(parents=True, exist_ok=True)

        try:
            shutil.copy2(str(source), str(target))
            print(f"\n✅ Datenbank erfolgreich importiert")
            print(f"   Quelle: {source_path}")
            print(f"   Ziel:   {target}\n")
        except Exception as e:
            print(f"\n❌ Import fehlgeschlagen: {e}\n")

    def _setup_catalog(self):
        """Optional Libero SOAP catalog configuration step - Claude Generated"""
        print("\n\n📌 Schritt 4: Katalog-Konfiguration (Optional)\n")
        print("Für die DK-Analyse und UB-Suche kann ein Libero-SOAP-Katalog eingebunden werden.")
        print("Dieser Schritt ist optional.\n")

        # Show preset values if available - Claude Generated
        preset_search = self._presets.catalog_soap_search_url
        preset_details = self._presets.catalog_soap_details_url
        preset_token = self._presets.catalog_token
        has_presets = bool(preset_search or preset_details)

        if has_presets:
            print("💼 Institutions-Presets verfügbar:")
            if preset_search:
                print(f"   Search URL: {preset_search}")
            if preset_details:
                print(f"   Details URL: {preset_details}")
            if preset_token:
                print(f"   Token: (gesetzt)")
            print("   Eingabe leer lassen = Preset übernehmen\n")
            default_choice = 'j'
        else:
            default_choice = 'n'

        choice = input(f"Katalog konfigurieren? (j/n, Standard: {default_choice}): ").lower().strip() or default_choice
        if choice != 'j':
            # If presets exist and user skips, still apply presets silently
            if has_presets:
                self.catalog_search_url = preset_search
                self.catalog_details_url = preset_details
                self.catalog_token = preset_token
                print("✅ Katalog-Presets übernommen")
            else:
                print("⏭️  Katalog übersprungen")
            return

        search_prompt = f"SOAP Search URL (Standard: {preset_search or 'leer'}): " if preset_search else (
            "SOAP Search URL\n"
            "  (z.B. https://katalog.ub.example.de/libero/LiberoWebServices.CatalogueSearcher.cls): "
        )
        self.catalog_search_url = input(search_prompt).strip() or preset_search

        details_prompt = f"SOAP Details URL (Standard: {preset_details or 'leer'}): " if preset_details else (
            "SOAP Details URL\n"
            "  (z.B. https://katalog.ub.example.de/libero/LiberoWebServices.LibraryAPI.cls): "
        )
        self.catalog_details_url = input(details_prompt).strip() or preset_details

        token_prompt = "Auth-Token (Standard: aus Preset): " if preset_token else "Auth-Token (leer lassen falls nicht erforderlich): "
        self.catalog_token = input(token_prompt).strip() or preset_token

        if self.catalog_search_url or self.catalog_token:
            print("✅ Katalog-Konfiguration gespeichert")
        else:
            print("⚠️  Keine Daten eingegeben, Katalog-Konfiguration übersprungen")
            self.catalog_search_url = ""
            self.catalog_details_url = ""
            self.catalog_token = ""

    def _print_summary(self):
        """Print configuration summary - Claude Generated"""
        print("\n" + "=" * 60)
        print("✅ Konfigurations-Zusammenfassung")
        print("=" * 60)
        print(f"\nLLM-Anbieter Konfiguration:")
        print(f"  Typ: {self.provider_type}")
        print(f"  Name: {self.provider_name}")
        if self.base_url:
            print(f"  Basis-URL: {self.base_url}")
        if self.api_key:
            print(f"  API-Schlüssel: {'*' * len(self.api_key[:-4]) + self.api_key[-4:]}")
        print(f"  Verfügbare Modelle: {len(self.available_models)}")

        print(f"\nDatenbank:")
        print(f"  Typ: SQLite")
        config = self.config_manager.load_config()
        print(f"  Pfad: {config.database_config.sqlite_path}")

        # Catalog summary - Claude Generated
        print(f"\nKatalog-SOAP:")
        if self.catalog_search_url:
            print(f"  Search URL: {self.catalog_search_url}")
            print(f"  Details URL: {self.catalog_details_url or '(nicht gesetzt)'}")
            print(f"  Token: {'(gesetzt)' if self.catalog_token else '(nicht gesetzt)'}")
        else:
            print(f"  Nicht konfiguriert (übersprungen)")

        print("\n" + "=" * 60)
        print("✅ ALIMA ist bereit zur Verwendung!")
        print("=" * 60 + "\n")


def run_cli_setup_wizard() -> bool:
    """Run the CLI setup wizard - Claude Generated

    Returns:
        bool: True if setup completed successfully
    """
    wizard = CLISetupWizard()
    return wizard.run()
