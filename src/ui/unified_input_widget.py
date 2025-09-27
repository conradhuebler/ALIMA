"""
Unified Input Widget - Konvergierende UX für verschiedene Input-Typen
Claude Generated - Drag-n-Drop, Copy-Paste, und verschiedene Input-Quellen
"""

from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QTextEdit,
    QLabel,
    QPushButton,
    QTabWidget,
    QGroupBox,
    QFileDialog,
    QMessageBox,
    QProgressBar,
    QFrame,
    QSplitter,
    QScrollArea,
    QApplication,
    QLineEdit,
)
from PyQt6.QtCore import Qt, pyqtSignal, QThread, QMimeData, QUrl, pyqtSlot
from PyQt6.QtGui import QDragEnterEvent, QDropEvent, QFont, QPalette
from typing import Optional, Dict, Any, List, Tuple
import logging
import os
import PyPDF2
import requests
from datetime import datetime

from ..llm.llm_service import LlmService
from .crossref_tab import CrossrefTab
from .image_analysis_tab import ImageAnalysisTab
from ..core.alima_manager import AlimaManager
from ..utils.doi_resolver import resolve_input_to_text


class TextExtractionWorker(QThread):
    """Worker für Textextraktion aus verschiedenen Quellen - Claude Generated"""

    text_extracted = pyqtSignal(str, str)  # extracted_text, source_info
    error_occurred = pyqtSignal(str)
    progress_updated = pyqtSignal(str)

    def __init__(
        self,
        source_type: str,
        source_data: Any,
        llm_service: Optional[LlmService] = None,
        alima_manager: Optional[AlimaManager] = None,
    ):
        super().__init__()
        self.source_type = source_type  # pdf, image, doi, url
        self.source_data = source_data
        self.llm_service = llm_service
        self.alima_manager = alima_manager
        self.logger = logging.getLogger(__name__)

    def run(self):
        """Extract text based on source type - Claude Generated"""
        try:
            if self.source_type == "pdf":
                self._extract_from_pdf()
            elif self.source_type == "image":
                self._extract_from_image()
            elif self.source_type == "doi":
                self._extract_from_doi()
            elif self.source_type == "url":
                self._extract_from_url()
            else:
                self.error_occurred.emit(f"Unbekannter Quelltyp: {self.source_type}")

        except Exception as e:
            self.logger.error(f"Error extracting text from {self.source_type}: {e}")
            self.error_occurred.emit(str(e))

    def _extract_from_pdf(self):
        """Extract text from PDF file with LLM fallback - Claude Generated"""
        self.progress_updated.emit("PDF wird gelesen...")

        try:
            with open(self.source_data, "rb") as file:
                reader = PyPDF2.PdfReader(file)
                text_parts = []

                for i, page in enumerate(reader.pages):
                    self.progress_updated.emit(
                        f"Seite {i+1} von {len(reader.pages)} wird verarbeitet..."
                    )
                    page_text = page.extract_text()
                    text_parts.append(page_text)

                full_text = "\\n\\n".join(text_parts).strip()
                filename = os.path.basename(self.source_data)
                
                # Prüfe Text-Qualität
                text_quality = self._assess_text_quality(full_text)
                
                if text_quality['is_good']:
                    # Direkter Text ist brauchbar
                    source_info = f"PDF: {filename} ({len(reader.pages)} Seiten, Text extrahiert)"
                    self.text_extracted.emit(full_text, source_info)
                else:
                    # Text-Qualität schlecht, verwende LLM-OCR
                    self.progress_updated.emit("Text-Qualität unzureichend, starte OCR-Analyse...")
                    self._extract_pdf_with_llm(filename, len(reader.pages))

        except Exception as e:
            self.error_occurred.emit(f"PDF-Fehler: {str(e)}")

    def _assess_text_quality(self, text: str) -> Dict[str, Any]:
        """Assess quality of extracted PDF text - Claude Generated"""
        if not text or len(text.strip()) == 0:
            return {'is_good': False, 'reason': 'Kein Text gefunden'}
        
        # Grundlegende Qualitätsprüfungen
        char_count = len(text)
        word_count = len(text.split())
        
        # Prüfe auf Mindestlänge
        if char_count < 50:
            return {'is_good': False, 'reason': 'Text zu kurz'}
        
        # Prüfe Zeichen-zu-Wort-Verhältnis (durchschnittliche Wortlänge)
        if word_count > 0:
            avg_word_length = char_count / word_count
            if avg_word_length < 2 or avg_word_length > 20:
                return {'is_good': False, 'reason': 'Ungewöhnliche Wortlängen'}
        
        # Prüfe auf zu viele Sonderzeichen oder Fragmente  
        special_char_ratio = sum(1 for c in text if not c.isalnum() and c not in ' \n\t.,!?;:-()[]') / len(text)
        if special_char_ratio > 0.3:
            return {'is_good': False, 'reason': 'Zu viele Sonderzeichen'}
        
        # Prüfe auf zusammenhängenden Text (nicht nur einzelne Zeichen)
        lines_with_content = [line.strip() for line in text.split('\n') if len(line.strip()) > 5]
        if len(lines_with_content) < max(1, word_count // 20):
            return {'is_good': False, 'reason': 'Text fragmentiert'}
            
        return {'is_good': True, 'reason': 'Text-Qualität ausreichend'}

    def _extract_pdf_with_llm(self, filename: str, page_count: int):
        """Extract PDF using LLM OCR when text quality is poor - Claude Generated"""
        if not self.llm_service:
            self.error_occurred.emit("LLM-Service nicht verfügbar für PDF-OCR")
            return
        
        try:
            import uuid
            from ..llm.prompt_service import PromptService
            
            # Konvertiere PDF zu Bild für LLM-Analyse (erste Seite als Test)
            self.progress_updated.emit("Konvertiere PDF für OCR-Analyse...")
            
            # Verwende pdf2image für Konvertierung
            try:
                import pdf2image
                images = pdf2image.convert_from_path(
                    self.source_data, 
                    first_page=1, 
                    last_page=min(3, page_count),  # Max. erste 3 Seiten für OCR
                    dpi=200
                )
                
                if not images:
                    self.error_occurred.emit("PDF konnte nicht zu Bildern konvertiert werden")
                    return
                
                # Speichere erstes Bild temporär für LLM-Analyse
                import tempfile
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                    images[0].save(tmp_file.name, 'PNG')
                    temp_image_path = tmp_file.name
                
                # Verwende LLM für OCR
                self._extract_image_with_llm(
                    temp_image_path, 
                    f"PDF (OCR): {filename} ({page_count} Seiten, per LLM analysiert)",
                    cleanup_temp=True
                )
                
            except ImportError:
                self.error_occurred.emit("pdf2image-Bibliothek nicht verfügbar. Installieren Sie: pip install pdf2image")
            except Exception as e:
                self.error_occurred.emit(f"PDF-zu-Bild Konvertierung fehlgeschlagen: {str(e)}")
                
        except Exception as e:
            self.error_occurred.emit(f"PDF-LLM-Extraktion fehlgeschlagen: {str(e)}")

    def _extract_from_image(self):
        """Extract text from image using LLM - Claude Generated"""
        if not self.llm_service:
            self.error_occurred.emit("LLM-Service nicht verfügbar für Bilderkennung")
            return

        filename = os.path.basename(self.source_data)
        source_info = f"Bild: {filename}"
        
        self._extract_image_with_llm(self.source_data, source_info)

    def _extract_image_with_llm(self, image_path: str, source_info: str, cleanup_temp: bool = False):
        """Extract text from image using LLM with task preferences system - Claude Generated"""
        self.progress_updated.emit("Bild wird mit LLM analysiert...")

        try:
            if self.alima_manager:
                self.progress_updated.emit("Verwende neue Task-Präferenz-Logik...")
                
                # Create task context
                context = {'image_data': image_path}
                
                # Execute task using the refactored system
                extracted_text = self.alima_manager.execute_task(
                    task_name="image_text_extraction",
                    context=context,
                    stream_callback=None  # No streaming for images
                )
                
                # Clean output
                extracted_text = self._clean_ocr_output(extracted_text)
                
                # Cleanup temp file
                if cleanup_temp:
                    try:
                        os.unlink(image_path)
                    except:
                        pass
                
                if extracted_text.strip():
                    self.text_extracted.emit(extracted_text, source_info)
                    return
                else:
                    self.error_occurred.emit("LLM konnte keinen Text im Bild erkennen")
            else:
                self.error_occurred.emit("alima_manager not found")

        except Exception as e:
            # Cleanup temporäre Datei auch bei Fehlern
            if cleanup_temp:
                try:
                    os.unlink(image_path)
                except:
                    pass
            self.logger.error(f"Image LLM extraction error: {e}")
            self.error_occurred.emit(f"LLM-Bilderkennung fehlgeschlagen: {str(e)}")
    
    def _extract_image_with_llm_legacy(self, image_path: str, source_info: str, cleanup_temp: bool = False):
        """Legacy image extraction method for fallback - Claude Generated"""
        try:
            import uuid
            from ..llm.prompt_service import PromptService
            
            # Lade OCR-Prompt
            import os
            prompts_path = os.path.join(os.path.dirname(__file__), '..', '..', 'prompts.json')
            prompt_service = PromptService(prompts_path, self.logger)
            
            # Verwende image_text_extraction Task
            prompt_config_data = prompt_service.get_prompt_config(
                task="image_text_extraction",
                model="default"  # Wird automatisch den besten verfügbaren Provider wählen
            )
            
            if not prompt_config_data:
                self.error_occurred.emit("OCR-Prompt nicht gefunden. Bitte prüfen Sie prompts.json")
                return
            
            # Konvertiere PromptConfigData zu Dictionary für Kompatibilität
            prompt_config = {
                'prompt': prompt_config_data.prompt,
                'system': prompt_config_data.system or '',
                'temperature': prompt_config_data.temp,
                'top_p': prompt_config_data.p_value,
                'seed': prompt_config_data.seed
            }
            
            request_id = str(uuid.uuid4())
            
            # Bestimme besten verfügbaren Provider für Bilderkennung mit Task Preferences
            provider, model = self._get_best_vision_provider_with_task_preferences()
            
            if not provider:
                self.error_occurred.emit("Kein Provider mit Bilderkennung verfügbar")
                return
            
            self.progress_updated.emit(f"Verwende {provider} ({model}) für Bilderkennung...")
            
            # LLM-Aufruf für Bilderkennung
            response = self.llm_service.generate_response(
                provider=provider,
                model=model,
                prompt=prompt_config['prompt'],
                system=prompt_config.get('system', ''),
                request_id=request_id,
                temperature=float(prompt_config.get('temperature', 0.1)),
                p_value=float(prompt_config.get('top_p', 0.1)),
                seed=prompt_config.get('seed'),
                image=image_path,
                stream=False
            )

            # Handle verschiedene Response-Typen
            extracted_text = ""
            if hasattr(response, "__iter__") and not isinstance(response, str):
                # Generator response (z.B. von Ollama)
                text_parts = []
                for chunk in response:
                    if isinstance(chunk, str):
                        text_parts.append(chunk)
                    elif hasattr(chunk, 'text'):
                        text_parts.append(chunk.text)
                    elif hasattr(chunk, 'content'):
                        text_parts.append(chunk.content)
                    else:
                        text_parts.append(str(chunk))
                extracted_text = "".join(text_parts)
            else:
                extracted_text = str(response)

            # Bereinige Ausgabe von LLM-Metakommentaren
            extracted_text = self._clean_ocr_output(extracted_text)
            
            # Cleanup temporäre Datei wenn angefordert
            if cleanup_temp:
                try:
                    os.unlink(image_path)
                except:
                    pass
            
            if extracted_text.strip():
                self.text_extracted.emit(extracted_text, source_info)
            else:
                self.error_occurred.emit("LLM konnte keinen Text im Bild erkennen")

        except Exception as e:
            # Cleanup temporäre Datei auch bei Fehlern
            if cleanup_temp:
                try:
                    os.unlink(image_path)
                except:
                    pass
            self.logger.error(f"Legacy image LLM extraction error: {e}")
            self.error_occurred.emit(f"Legacy LLM-Bilderkennung fehlgeschlagen: {str(e)}")

    def _get_best_vision_provider(self) -> tuple:
        """Get best available provider for vision tasks - Claude Generated"""
        # Prioritätsliste für Vision-Provider
        vision_providers = [
            ("gemini", ["gemini-2.0-flash", "gemini-1.5-flash"]),
            ("openai", ["gpt-4o", "gpt-4-vision-preview"]),
            ("anthropic", ["claude-3-5-sonnet", "claude-3-opus"]),
            ("ollama", ["llava", "minicpm-v", "cogito:32b"])
        ]
        
        try:
            available_providers = self.llm_service.get_available_providers()
            
            for provider_name, preferred_models in vision_providers:
                if provider_name in available_providers:
                    try:
                        available_models = self.llm_service.get_available_models(provider_name)
                        
                        # Finde das beste verfügbare Modell
                        for preferred_model in preferred_models:
                            if preferred_model in available_models:
                                return provider_name, preferred_model
                        
                        # Falls kein bevorzugtes Modell, nimm das erste verfügbare
                        if available_models:
                            return provider_name, available_models[0]
                            
                    except Exception as e:
                        self.logger.warning(f"Error checking models for {provider_name}: {e}")
                        continue
            
            return None, None

        except Exception as e:
            self.logger.error(f"Error determining best vision provider: {e}")
            return None, None

    def _get_best_vision_provider_with_task_preferences(self) -> tuple:
        """Get best vision provider using task preferences for image_text_extraction - Claude Generated"""
        try:
            # 🔍 DEBUG: Start vision provider selection with task preferences - Claude Generated
            self.logger.critical(f"🔍 VISION_TASK_START: Selecting provider for image_text_extraction")

            # Get unified config for task preferences - Claude Generated

            # Try to get config manager from multiple sources - Claude Generated
            config_manager = getattr(self.llm_service, 'config_manager', None)
            if not config_manager:
                config_manager = getattr(self.alima_manager, 'config_manager', None)

            # 🔍 ROBUST FALLBACK: Try to create ConfigManager if no access via services - Claude Generated
            if not config_manager:
                self.logger.critical("🔍 CONFIG_MANAGER_FALLBACK: Attempting to create ConfigManager directly")
                try:
                    from ..utils.config_manager import ConfigManager
                    config_manager = ConfigManager()
                    self.logger.critical("🔍 CONFIG_MANAGER_CREATED: Successfully created ConfigManager directly")
                except Exception as e:
                    self.logger.critical(f"🔍 CONFIG_MANAGER_CREATION_FAILED: Failed to create ConfigManager: {e}")

            # 🔍 DEBUG: Log config manager availability - Claude Generated
            self.logger.critical(f"🔍 CONFIG_MANAGER: llm_service has config_manager={getattr(self.llm_service, 'config_manager', None) is not None}")
            self.logger.critical(f"🔍 CONFIG_MANAGER: alima_manager has config_manager={getattr(self.alima_manager, 'config_manager', None) is not None}")
            self.logger.critical(f"🔍 CONFIG_MANAGER: final config_manager={config_manager is not None}, type={type(config_manager).__name__ if config_manager else None}")

            if not config_manager:
                self.logger.critical("🔍 CONFIG_MANAGER_MISSING: No config_manager available, falling back to default vision provider")
                return self._get_best_vision_provider()

            unified_config = config_manager.get_unified_config()

            # 🔍 DEBUG: Log unified config loading and contents - Claude Generated
            self.logger.critical(f"🔍 UNIFIED_CONFIG: loaded={unified_config is not None}")
            if unified_config:
                self.logger.critical(f"🔍 UNIFIED_CONFIG_PROVIDERS: {len(unified_config.providers)} providers configured")
                self.logger.critical(f"🔍 UNIFIED_CONFIG_TASK_PREFS: {len(unified_config.task_preferences)} task preferences: {list(unified_config.task_preferences.keys())}")
                self.logger.critical(f"🔍 UNIFIED_CONFIG_PROVIDER_PRIORITY: {unified_config.provider_priority}")

            # Get model priority for image_text_extraction task
            model_priority = unified_config.get_model_priority_for_task("image_text_extraction") if unified_config else []

            # 🔍 DEBUG: Log detailed analysis of task preferences - Claude Generated
            if unified_config and hasattr(unified_config, 'task_preferences'):
                image_task_pref = unified_config.task_preferences.get("image_text_extraction")
                self.logger.critical(f"🔍 IMAGE_TASK_PREF_OBJECT: {image_task_pref}")
                if image_task_pref:
                    self.logger.critical(f"🔍 IMAGE_TASK_PREF_MODEL_PRIORITY: {getattr(image_task_pref, 'model_priority', None)}")
                    self.logger.critical(f"🔍 IMAGE_TASK_PREF_CHUNKED: {getattr(image_task_pref, 'chunked_model_priority', None)}")
            else:
                self.logger.critical("🔍 NO_TASK_PREFERENCES: unified_config has no task_preferences attribute")

            # 🔍 ROBUST FALLBACK: If no model priority from unified config, try direct config access - Claude Generated
            if not model_priority:
                self.logger.critical("🔍 FALLBACK_TO_DIRECT_CONFIG: Trying direct AlimaConfig access for task preferences")
                try:
                    # Load AlimaConfig directly
                    alima_config = config_manager.load_config()
                    if hasattr(alima_config, 'unified_config') and alima_config.unified_config.task_preferences:
                        task_prefs = alima_config.unified_config.task_preferences.get("image_text_extraction", {})
                        model_priority = task_prefs.get('model_priority', [])
                        self.logger.critical(f"🔍 DIRECT_CONFIG_TASK_PREFS: Found {len(model_priority) if model_priority else 0} providers in direct config")
                except Exception as e:
                    self.logger.critical(f"🔍 DIRECT_CONFIG_ERROR: Failed to access task preferences from direct config: {e}")

            # 🔍 DEBUG: Log task preferences - Claude Generated
            self.logger.critical(f"🔍 TASK_PREFERENCES: image_text_extraction model_priority={model_priority}")

            if model_priority:
                self.logger.critical(f"🔍 USING_TASK_PREFERENCES: {len(model_priority)} providers configured for image_text_extraction: {model_priority}")

                # Try each configured provider/model in priority order
                for i, priority_item in enumerate(model_priority):
                    provider_name = priority_item.get("provider_name", "")
                    model_name = priority_item.get("model_name", "")

                    self.logger.critical(f"🔍 TRYING_PROVIDER_{i+1}: {provider_name}/{model_name}")

                    if provider_name and model_name:
                        try:
                            # Check if provider is available
                            available_providers = self.llm_service.get_available_providers()
                            self.logger.critical(f"🔍 AVAILABLE_PROVIDERS: {available_providers}")

                            if provider_name in available_providers:
                                available_models = self.llm_service.get_available_models(provider_name)
                                self.logger.critical(f"🔍 AVAILABLE_MODELS_{provider_name}: {available_models}")

                                if model_name in available_models or model_name == "default":
                                    self.logger.critical(f"🔍 VISION_SUCCESS: Using configured vision provider: {provider_name}/{model_name}")
                                    return provider_name, model_name
                                else:
                                    self.logger.critical(f"🔍 MODEL_UNAVAILABLE: Configured model {model_name} not available for {provider_name} (available: {available_models})")
                            else:
                                self.logger.critical(f"🔍 PROVIDER_UNAVAILABLE: Configured provider {provider_name} not available (available: {available_providers})")
                        except Exception as e:
                            self.logger.critical(f"🔍 PROVIDER_CHECK_ERROR: Error checking configured provider {provider_name}: {e}")
                            continue

                self.logger.critical("🔍 NO_CONFIGURED_PROVIDERS: No configured and available providers found in task preferences, falling back to default vision provider")

            else:
                self.logger.critical("🔍 NO_TASK_PREFERENCES: No task preferences configured for image_text_extraction, using default")

            # Fallback to default vision provider selection
            fallback_result = self._get_best_vision_provider()
            self.logger.critical(f"🔍 FALLBACK_RESULT: Using fallback vision provider: {fallback_result}")
            return fallback_result

        except Exception as e:
            self.logger.critical(f"🔍 VISION_ERROR: Error getting vision provider with task preferences: {e}")
            import traceback
            self.logger.critical(f"🔍 VISION_TRACEBACK: {traceback.format_exc()}")
            return self._get_best_vision_provider()

    def _clean_ocr_output(self, text: str) -> str:
        """Clean OCR output from common LLM artifacts - Claude Generated"""
        if not text:
            return ""
        
        # Entferne häufige LLM-Metakommentare
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            # Überspringe typische LLM-Metakommentare
            if any(phrase in line.lower() for phrase in [
                'hier ist der text',
                'der text lautet',
                'ich kann folgenden text erkennen',
                'das bild enthält folgenden text',
                'extracted text:',
                'ocr result:',
                'text erkannt:',
                'gefundener text:'
            ]):
                continue
            
            if line:  # Nur nicht-leere Zeilen
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines).strip()

    def _extract_from_doi(self):
        """Extract metadata from DOI using unified resolver - Claude Generated"""
        self.progress_updated.emit("DOI wird aufgelöst...")

        try:
            success, text_content, error_msg = resolve_input_to_text(
                self.source_data, self.logger
            )

            if success:
                source_info = (
                    f"DOI: {self.source_data} (Länge: {len(text_content)} Zeichen)"
                )
                self.text_extracted.emit(text_content, source_info)
                self.logger.info(
                    f"DOI {self.source_data} successfully resolved to {len(text_content)} chars"
                )
            else:
                self.error_occurred.emit(f"DOI-Auflösung fehlgeschlagen: {error_msg}")

        except Exception as e:
            self.error_occurred.emit(f"DOI-Fehler: {str(e)}")

    def _extract_from_url(self):
        """Extract text from URL using unified resolver - Claude Generated"""
        self.progress_updated.emit("URL wird abgerufen...")

        try:
            success, text_content, error_msg = resolve_input_to_text(
                self.source_data, self.logger
            )

            if success:
                source_info = (
                    f"URL: {self.source_data} (Länge: {len(text_content)} Zeichen)"
                )
                self.text_extracted.emit(text_content, source_info)
                self.logger.info(
                    f"URL {self.source_data} successfully resolved to {len(text_content)} chars"
                )
            else:
                self.error_occurred.emit(f"URL-Auflösung fehlgeschlagen: {error_msg}")

        except Exception as e:
            self.error_occurred.emit(f"URL-Fehler: {str(e)}")


class UnifiedInputWidget(QWidget):
    """Einheitliches Input-Widget mit Drag-n-Drop und verschiedenen Quellen - Claude Generated"""

    # Signals
    text_ready = pyqtSignal(str, str)  # text, source_info
    input_cleared = pyqtSignal()

    def __init__(self, llm_service: Optional[LlmService] = None, alima_manager: Optional[AlimaManager] = None, parent=None):
        super().__init__(parent)
        self.llm_service = llm_service
        self.alima_manager = alima_manager
        self.logger = logging.getLogger(__name__)
        self.current_extraction_worker: Optional[TextExtractionWorker] = None

        # Enable drag and drop
        self.setAcceptDrops(True)

        self.setup_ui()

    def setup_ui(self):
        """Setup der UI - Claude Generated"""
        layout = QVBoxLayout(self)
        layout.setSpacing(15)

        # Header mit Titel
        header_layout = QHBoxLayout()
        title_label = QLabel("📥 INPUT")
        title_font = QFont()
        title_font.setPointSize(14)
        title_font.setBold(True)
        title_label.setFont(title_font)
        header_layout.addWidget(title_label)

        header_layout.addStretch()

        # Clear button
        clear_button = QPushButton("🗑️ Leeren")
        clear_button.clicked.connect(self.clear_input)
        header_layout.addWidget(clear_button)

        layout.addLayout(header_layout)

        # Main input area: Drop Zone + Input Methods side by side
        self.create_main_input_area(layout)

        # Text Display Area
        self.create_text_display(layout)

        # Progress Area
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_label = QLabel()
        self.progress_label.setVisible(False)
        layout.addWidget(self.progress_label)
        layout.addWidget(self.progress_bar)

    def create_main_input_area(self, layout):
        """Create main input area with drop zone and input methods side by side - Claude Generated"""
        main_input_layout = QHBoxLayout()
        main_input_layout.setSpacing(15)

        # Left side: Drop Zone
        self.create_drop_zone_compact(main_input_layout)

        # Right side: Input Methods (vertical layout)
        self.create_input_methods_vertical(main_input_layout)

        layout.addLayout(main_input_layout)

    def create_drop_zone_compact(self, layout):
        """Create compact drop zone - Claude Generated"""
        drop_zone_group = QGroupBox("📤 Drag & Drop")
        drop_zone_layout = QVBoxLayout(drop_zone_group)

        # Drop area
        self.drop_frame = QFrame()
        self.drop_frame.setFrameStyle(QFrame.Shape.Box)
        self.drop_frame.setLineWidth(2)
        self.drop_frame.setMinimumHeight(120)
        self.drop_frame.setStyleSheet(
            """
            QFrame {
                border: 2px dashed #ccc;
                border-radius: 8px;
                background-color: #f9f9f9;
                padding: 20px;
            }
            QFrame:hover {
                border-color: #2196f3;
                background-color: #e3f2fd;
            }
        """
        )

        frame_layout = QVBoxLayout(self.drop_frame)
        frame_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Drop instruction
        drop_label = QLabel("Dateien hier ablegen")
        drop_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        drop_label.setStyleSheet("color: #666; font-size: 14px; font-weight: bold;")
        frame_layout.addWidget(drop_label)

        supported_label = QLabel("PDF, Bilder, Text-Dateien")
        supported_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        supported_label.setStyleSheet("color: #999; font-size: 11px;")
        frame_layout.addWidget(supported_label)

        drop_zone_layout.addWidget(self.drop_frame)
        layout.addWidget(drop_zone_group)

    def create_input_methods_vertical(self, layout):
        """Create vertical input methods - Claude Generated"""
        methods_group = QGroupBox("🔧 Eingabemethoden")
        methods_layout = QVBoxLayout(methods_group)
        methods_layout.setSpacing(10)

        # File selection button
        file_button = QPushButton("📁 Datei auswählen")
        file_button.clicked.connect(self.select_file)
        methods_layout.addWidget(file_button)

        # DOI/URL input with auto-detection
        doi_url_layout = QHBoxLayout()
        self.doi_url_input = QLineEdit()
        self.doi_url_input.setPlaceholderText(
            "DOI oder URL eingeben (z.B. 10.1007/... oder https://...)"
        )
        self.doi_url_input.returnPressed.connect(self.process_doi_url_input)
        doi_url_layout.addWidget(self.doi_url_input)

        resolve_button = QPushButton("🔍 Auflösen")
        resolve_button.clicked.connect(self.process_doi_url_input)
        resolve_button.setMaximumWidth(80)
        doi_url_layout.addWidget(resolve_button)

        methods_layout.addLayout(doi_url_layout)

        # Paste button
        paste_button = QPushButton("📋 Aus Zwischenablage einfügen")
        paste_button.clicked.connect(self.paste_from_clipboard)
        methods_layout.addWidget(paste_button)

        # Add stretch to push everything to the top
        methods_layout.addStretch()

        layout.addWidget(methods_group)

    def create_drop_zone(self, layout):
        """Create drag and drop zone - Claude Generated"""
        self.drop_zone = QFrame()
        self.drop_zone.setFrameStyle(QFrame.Shape.Box)
        self.drop_zone.setLineWidth(2)
        self.drop_zone.setStyleSheet(
            """
            QFrame {
                border: 2px dashed #ccc;
                border-radius: 8px;
                background-color: #f9f9f9;
                min-height: 80px;
            }
            QFrame:hover {
                border-color: #2196f3;
                background-color: #e3f2fd;
            }
        """
        )

        drop_layout = QVBoxLayout(self.drop_zone)
        drop_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        drop_icon = QLabel("🎯")
        drop_icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        drop_icon.setStyleSheet("font-size: 32px; color: #666;")
        drop_layout.addWidget(drop_icon)

        drop_text = QLabel("Dateien hierher ziehen oder klicken zum Auswählen")
        drop_text.setAlignment(Qt.AlignmentFlag.AlignCenter)
        drop_text.setStyleSheet("color: #666; font-weight: bold;")
        drop_layout.addWidget(drop_text)

        drop_hint = QLabel("PDF, Bilder, oder Text kopieren und einfügen")
        drop_hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        drop_hint.setStyleSheet("color: #999; font-size: 11px;")
        drop_layout.addWidget(drop_hint)

        # Make drop zone clickable
        self.drop_zone.mousePressEvent = self.on_drop_zone_clicked

        layout.addWidget(self.drop_zone)

    def create_input_methods(self, layout):
        """Create input method tabs - Claude Generated"""
        methods_group = QGroupBox("Eingabemethoden")
        methods_layout = QVBoxLayout(methods_group)

        # Quick action buttons
        button_layout = QHBoxLayout()

        # File button
        file_button = QPushButton("📁 Datei auswählen")
        file_button.clicked.connect(self.select_file)
        button_layout.addWidget(file_button)

        # DOI button
        doi_button = QPushButton("🔗 DOI eingeben")
        doi_button.clicked.connect(self.enter_doi)
        button_layout.addWidget(doi_button)

        # URL button
        url_button = QPushButton("🌐 URL eingeben")
        url_button.clicked.connect(self.enter_url)
        button_layout.addWidget(url_button)

        # Paste button
        paste_button = QPushButton("📋 Einfügen")
        paste_button.clicked.connect(self.paste_from_clipboard)
        button_layout.addWidget(paste_button)

        methods_layout.addLayout(button_layout)
        layout.addWidget(methods_group)

    def create_text_display(self, layout):
        """Create text display area - Claude Generated"""
        display_group = QGroupBox("Extrahierter Text")
        display_layout = QVBoxLayout(display_group)

        # Source info
        self.source_info_label = QLabel("Keine Quelle ausgewählt")
        self.source_info_label.setStyleSheet(
            "font-weight: bold; color: #666; padding: 5px;"
        )
        display_layout.addWidget(self.source_info_label)

        # Text area
        self.text_display = QTextEdit()
        self.text_display.setPlaceholderText("Text wird hier angezeigt...")
        self.text_display.setMinimumHeight(200)

        # Enhanced styling
        font = QFont()
        font.setPointSize(11)
        self.text_display.setFont(font)

        self.text_display.setStyleSheet(
            """
            QTextEdit {
                border: 1px solid #ddd;
                border-radius: 4px;
                padding: 8px;
                background-color: white;
            }
            QTextEdit:focus {
                border-color: #2196f3;
            }
        """
        )

        display_layout.addWidget(self.text_display)

        # Action buttons for text
        text_actions = QHBoxLayout()

        use_button = QPushButton("✅ Text verwenden")
        use_button.setStyleSheet(
            """
            QPushButton {
                background-color: #4caf50;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """
        )
        use_button.clicked.connect(self.use_current_text)
        text_actions.addWidget(use_button)

        text_actions.addStretch()

        edit_button = QPushButton("✏️ Bearbeiten")
        edit_button.clicked.connect(self.enable_text_editing)
        text_actions.addWidget(edit_button)

        display_layout.addLayout(text_actions)
        layout.addWidget(display_group)

    def dragEnterEvent(self, event: QDragEnterEvent):
        """Handle drag enter event - Claude Generated"""
        if event.mimeData().hasUrls() or event.mimeData().hasText():
            event.acceptProposedAction()
            # Update the new drop_frame styling
            if hasattr(self, "drop_frame"):
                self.drop_frame.setStyleSheet(
                    """
                    QFrame {
                        border: 2px solid #4caf50;
                        border-radius: 8px;
                        background-color: #e8f5e8;
                        padding: 20px;
                    }
            """
                )
        else:
            event.ignore()

    def dragLeaveEvent(self, event):
        """Handle drag leave event - Claude Generated"""
        # Reset the new drop_frame styling
        if hasattr(self, "drop_frame"):
            self.drop_frame.setStyleSheet(
                """
                QFrame {
                    border: 2px dashed #ccc;
                    border-radius: 8px;
                    background-color: #f9f9f9;
                    padding: 20px;
                }
                QFrame:hover {
                    border-color: #2196f3;
                    background-color: #e3f2fd;
                }
            """
            )

    def dropEvent(self, event: QDropEvent):
        """Handle drop event - Claude Generated"""
        self.dragLeaveEvent(event)  # Reset styling

        mime_data = event.mimeData()

        if mime_data.hasUrls():
            # Handle file drops
            urls = mime_data.urls()
            if urls:
                file_path = urls[0].toLocalFile()
                if file_path:
                    self.process_file(file_path)
                    event.acceptProposedAction()
                    return

        if mime_data.hasText():
            # Handle text drops
            text = mime_data.text().strip()
            if text:
                self.set_text_directly(text, "Eingefügter Text")
                event.acceptProposedAction()
                return

        event.ignore()

    def on_drop_zone_clicked(self, event):
        """Handle drop zone click - Claude Generated"""
        self.select_file()

    def select_file(self):
        """Open file dialog - Claude Generated"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Datei auswählen",
            "",
            "Alle unterstützten Dateien (*.pdf *.png *.jpg *.jpeg *.txt);;PDF-Dateien (*.pdf);;Bilder (*.png *.jpg *.jpeg);;Textdateien (*.txt)",
        )

        if file_path:
            self.process_file(file_path)

    def process_file(self, file_path: str):
        """Process selected file - Claude Generated"""
        if not os.path.exists(file_path):
            QMessageBox.warning(self, "Fehler", "Datei nicht gefunden!")
            return

        file_ext = os.path.splitext(file_path)[1].lower()

        if file_ext == ".pdf":
            self.extract_text("pdf", file_path)
        elif file_ext in [".png", ".jpg", ".jpeg"]:
            self.extract_text("image", file_path)
        elif file_ext == ".txt":
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read()
                    filename = os.path.basename(file_path)
                    self.set_text_directly(text, f"Textdatei: {filename}")
            except Exception as e:
                QMessageBox.critical(
                    self, "Fehler", f"Fehler beim Lesen der Datei: {e}"
                )
        else:
            QMessageBox.warning(
                self,
                "Nicht unterstützt",
                f"Dateityp {file_ext} wird nicht unterstützt!",
            )

    def enter_doi(self):
        """Enter DOI for metadata extraction - Claude Generated"""
        from PyQt6.QtWidgets import QInputDialog

        doi, ok = QInputDialog.getText(self, "DOI eingeben", "DOI:")
        if ok and doi.strip():
            self.extract_text("doi", doi.strip())

    def enter_url(self):
        """Enter URL for text extraction - Claude Generated"""
        from PyQt6.QtWidgets import QInputDialog

        url, ok = QInputDialog.getText(self, "URL eingeben", "URL:")
        if ok and url.strip():
            self.extract_text("url", url.strip())

    def process_doi_url_input(self):
        """Process DOI or URL input with auto-detection - Claude Generated"""
        input_text = self.doi_url_input.text().strip()
        if not input_text:
            QMessageBox.warning(
                self, "Keine Eingabe", "Bitte geben Sie eine DOI oder URL ein."
            )
            return

        # Auto-detect type based on input
        if input_text.startswith(("http://", "https://")):
            # It's a URL
            self.extract_text("url", input_text)
        elif input_text.startswith("10.") and "/" in input_text:
            # It's likely a DOI (DOIs start with "10." and contain a slash)
            self.extract_text("doi", input_text)
        elif "doi.org/" in input_text:
            # Extract DOI from DOI URL (e.g., https://doi.org/10.1007/...)
            doi_part = input_text.split("doi.org/")[-1]
            self.extract_text("doi", doi_part)
        else:
            # Assume it's a DOI if it doesn't look like a URL
            self.extract_text("doi", input_text)

        # Clear the input after processing
        self.doi_url_input.clear()

    def paste_from_clipboard(self):
        """Paste text from clipboard - Claude Generated"""
        clipboard = QApplication.clipboard()
        text = clipboard.text()

        if text.strip():
            self.set_text_directly(text, "Zwischenablage")
        else:
            QMessageBox.information(
                self, "Zwischenablage leer", "Die Zwischenablage enthält keinen Text."
            )

    def extract_text(self, source_type: str, source_data: Any):
        """Start text extraction worker - Claude Generated"""
        if (
            self.current_extraction_worker
            and self.current_extraction_worker.isRunning()
        ):
            self.current_extraction_worker.terminate()
            self.current_extraction_worker.wait()

        self.current_extraction_worker = TextExtractionWorker(
            source_type=source_type,
            source_data=source_data,
            llm_service=self.llm_service,
            alima_manager=self.alima_manager,
        )

        self.current_extraction_worker.text_extracted.connect(self.on_text_extracted)
        self.current_extraction_worker.error_occurred.connect(self.on_extraction_error)
        self.current_extraction_worker.progress_updated.connect(
            self.on_progress_updated
        )

        # Show progress
        self.progress_bar.setVisible(True)
        self.progress_label.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate

        self.current_extraction_worker.start()

    @pyqtSlot(str, str)
    def on_text_extracted(self, text: str, source_info: str):
        """Handle extracted text - Claude Generated"""
        self.progress_bar.setVisible(False)
        self.progress_label.setVisible(False)

        self.set_text_directly(text, source_info)

    @pyqtSlot(str)
    def on_extraction_error(self, error_message: str):
        """Handle extraction error - Claude Generated"""
        self.progress_bar.setVisible(False)
        self.progress_label.setVisible(False)

        QMessageBox.critical(self, "Extraction Error", error_message)

    @pyqtSlot(str)
    def on_progress_updated(self, message: str):
        """Handle progress update - Claude Generated"""
        self.progress_label.setText(message)

    def set_text_directly(self, text: str, source_info: str):
        """Set text directly in display - Claude Generated"""
        self.text_display.setPlainText(text)
        self.source_info_label.setText(f"📄 {source_info} | {len(text)} Zeichen")

        # Enable editing
        self.text_display.setReadOnly(False)

    def use_current_text(self):
        """Use current text for pipeline - Claude Generated"""
        text = self.text_display.toPlainText().strip()
        source_info = self.source_info_label.text()

        if text:
            self.text_ready.emit(text, source_info)
        else:
            QMessageBox.warning(self, "Kein Text", "Kein Text zum Verwenden vorhanden!")

    def enable_text_editing(self):
        """Enable text editing - Claude Generated"""
        self.text_display.setReadOnly(False)
        self.text_display.setFocus()
        QMessageBox.information(
            self, "Bearbeitung aktiviert", "Sie können den Text jetzt bearbeiten."
        )

    def clear_input(self):
        """Clear all input - Claude Generated"""
        self.text_display.clear()
        self.source_info_label.setText("Keine Quelle ausgewählt")
        self.input_cleared.emit()

    def get_current_text(self) -> str:
        """Get current text - Claude Generated"""
        return self.text_display.toPlainText().strip()

    def get_source_info(self) -> str:
        """Get current source info - Claude Generated"""
        return self.source_info_label.text()
