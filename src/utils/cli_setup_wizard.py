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
        self.catalog_search_url = ''  # Claude Generated: Catalog URLs
        self.catalog_details_url = ''

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
            self._setup_catalog_urls()  # Claude Generated: Setup catalog configuration
            self._print_summary()

            # Create and save configuration
            config = ConfigurationBuilder.create_initial_config(
                provider_type=self.provider_type,
                provider_name=self.provider_name,
                base_url=self.base_url,
                api_key=self.api_key,
                models=self.available_models,
                task_model_selections=self.task_model_selections,  # Pass task selections
                catalog_search_url=self.catalog_search_url,  # Claude Generated: Pass catalog URLs
                catalog_details_url=self.catalog_details_url
            )
            config.system_config.first_run_completed = True

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

        host = input("Ollama Host (Standard: localhost): ").strip() or "localhost"
        port_str = input("Ollama Port (Standard: 11434): ").strip() or "11434"

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

        base_url = input("API Basis-URL (z.B. http://localhost:8000/v1): ").strip()
        if not base_url:
            print("❌ Basis-URL erforderlich")
            return self._setup_openai_compatible()

        api_key = input("API-Schlüssel (oder leer lassen falls nicht erforderlich): ").strip()

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

        api_key = input("Geben Sie Ihren Gemini API-Schlüssel ein: ").strip()
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

        api_key = input("Geben Sie Ihren Anthropic API-Schlüssel ein: ").strip()
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
        print("Der Download verbessert Schlagwort-Vorschläge und Suchgenauigkeit.")
        print("Dies ist optional - Sie können stattdessen die Lobid-API verwenden.\n")

        choice = input("GND-Datenbank herunterladen? (j/n): ").lower().strip()

        if choice == 'j':
            self._download_gnd_database()
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

    def _setup_catalog_urls(self):
        """Setup optional catalog URLs - Claude Generated"""
        print("\n\n📌 Schritt 4: Katalog-Konfiguration (Optional)\n")
        print("Sie können optionale Katalog-URLs eingeben, um die DK-Klassifikation zu aktivieren.")
        print("Falls Sie diese leer lassen, wird der DK-Schritt übersprungen.\n")

        search_url = input("Katalog-Such-URL (leer = überspringen): ").strip()
        if search_url:
            details_url = input("Katalog-Details-URL (erforderlich für Such-URL): ").strip()
            if details_url:
                self.catalog_search_url = search_url
                self.catalog_details_url = details_url
                print("✅ Katalog-URLs konfiguriert")
            else:
                print("⚠️  Details-URL erforderlich, Katalog-Konfiguration übersprungen")
        else:
            print("⏭️  Katalog-Konfiguration übersprungen (DK-Schritt wird nicht ausgeführt)")

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

        # Claude Generated: Show catalog configuration
        if self.catalog_search_url and self.catalog_details_url:
            print(f"\nKatalog-Konfiguration:")
            print(f"  Such-URL: {self.catalog_search_url}")
            print(f"  Details-URL: {self.catalog_details_url}")
            print(f"  DK-Klassifikation: ✅ Aktiviert")
        else:
            print(f"\nKatalog-Konfiguration:")
            print(f"  DK-Klassifikation: ⏭️ Übersprungen")

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
