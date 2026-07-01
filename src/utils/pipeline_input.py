"""Input extraction for the ALIMA pipeline - Claude Generated.

Text extraction from PDF / image / text / file sources, with optional
Vision-LLM OCR fallback. Split out of the former ``pipeline_utils`` god-module;
still re-exported from ``pipeline_utils`` for backward compatibility.
"""

from typing import Any, Dict, List, Optional, Tuple

def execute_input_extraction(
    llm_service,
    input_source: str,
    input_type: str = "auto",
    stream_callback: Optional[callable] = None,
    logger=None,
    **kwargs,
) -> Tuple[str, str, str]:
    """
    Extract text from various input sources (PDF, Image, Text) - Claude Generated
    
    Args:
        llm_service: LLM service instance for image OCR
        input_source: File path or text content
        input_type: "auto", "pdf", "image", "text", or "file"
        stream_callback: Callback for progress updates
        logger: Logger instance
        **kwargs: Additional parameters for LLM
        
    Returns:
        Tuple of (extracted_text, source_info, extraction_method)
    """
    import os
    import PyPDF2
    import tempfile
    from pathlib import Path
    
    if logger:
        logger.info(f"Starting input extraction: {input_source[:50]}... (type: {input_type})")
    
    # Auto-detect input type if not specified
    if input_type == "auto":
        if os.path.isfile(input_source):
            ext = Path(input_source).suffix.lower()
            if ext == ".pdf":
                input_type = "pdf" 
            elif ext in [".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff"]:
                input_type = "image"
            else:
                input_type = "file"
        else:
            input_type = "text"
    
    # Dispatch to the registered input source for this type (registry replaces the
    # former if/elif — Debt D-11). text/file/pdf/image behave byte-for-byte as
    # before (their sources call the same helpers below). - Claude Generated
    try:
        from .input_sources import INPUT_SOURCE_REGISTRY

        src_cls = INPUT_SOURCE_REGISTRY.get(input_type)
        if src_cls is None:
            raise Exception(f"Unbekannter Input-Typ: {input_type}")
        settings = _input_settings_for(input_type)
        try:
            source_obj = src_cls(**settings)
        except TypeError:
            source_obj = src_cls()
        return source_obj.extract(
            input_source,
            llm_service=llm_service,
            stream_callback=stream_callback,
            logger=logger,
            **kwargs,
        )
    except Exception as e:
        error_msg = f"Input-Extraktion fehlgeschlagen: {str(e)}"
        if logger:
            logger.error(error_msg)
        raise Exception(error_msg)


def _input_settings_for(input_type: str) -> dict:
    """Instance settings for a configurable input source (url_fetch/doi_*).

    The no-config built-ins (text/file/pdf/image) never touch the config, keeping
    the hot extraction path allocation-free. - Claude Generated
    """
    if input_type in ("text", "file", "pdf", "image"):
        return {}
    try:
        from .config_manager import ConfigManager

        cfg = ConfigManager().load_config()
        for inst in cfg.enabled_instances_for("input_source"):
            if inst.provider_id == input_type:
                return dict(inst.settings or {})
    except Exception:
        pass
    return {}


def _extract_from_pdf_pipeline(
    pdf_path: str, 
    llm_service,
    stream_callback: Optional[callable] = None,
    logger=None
) -> Tuple[str, str, str]:
    """Extract text from PDF with LLM fallback for pipeline - Claude Generated"""
    import os
    from pathlib import Path

    try:
        import PyPDF2
    except ImportError:
        raise ImportError("PyPDF2 ist nicht installiert. Bitte mit 'pip install PyPDF2' installieren.")

    filename = os.path.basename(pdf_path)

    if stream_callback:
        stream_callback(f"📄 PDF wird gelesen: {filename}")

    try:
        with open(pdf_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            text_parts = []
            
            for i, page in enumerate(reader.pages):
                if stream_callback:
                    stream_callback(f"📄 Seite {i+1} von {len(reader.pages)} wird verarbeitet...")
                page_text = page.extract_text()
                text_parts.append(page_text)
            
            full_text = "\\n\\n".join(text_parts).strip()
            
            # Text-Qualität prüfen
            quality_assessment = _assess_text_quality_pipeline(full_text)
            
            if quality_assessment['is_good']:
                # Direkter Text ist brauchbar
                source_info = f"PDF: {filename} ({len(reader.pages)} Seiten, Text extrahiert)"
                return full_text, source_info, "pdf_direct"
            else:
                # Text-Qualität schlecht, verwende LLM-OCR
                if stream_callback:
                    stream_callback(f"📄 Text-Qualität unzureichend ({quality_assessment['reason']}), starte OCR...")
                
                return _extract_pdf_with_llm_pipeline(pdf_path, filename, len(reader.pages), llm_service, stream_callback, logger)
                
    except Exception as e:
        raise Exception(f"PDF-Verarbeitung fehlgeschlagen: {str(e)}")


def _extract_pdf_with_llm_pipeline(
    pdf_path: str,
    filename: str, 
    page_count: int,
    llm_service,
    stream_callback: Optional[callable] = None,
    logger=None
) -> Tuple[str, str, str]:
    """Extract PDF using LLM OCR for pipeline - Claude Generated"""
    
    try:
        # Versuche pdf2image Import
        try:
            import pdf2image  # pyright: ignore[reportMissingImports]
        except ImportError:
            raise Exception("pdf2image-Bibliothek nicht verfügbar. Installieren Sie: pip install pdf2image")
        
        if stream_callback:
            stream_callback("📄 Konvertiere PDF für OCR-Analyse...")
        
        # Konvertiere PDF zu Bildern (max. erste 3 Seiten)
        images = pdf2image.convert_from_path(
            pdf_path,
            first_page=1,
            last_page=min(3, page_count),
            dpi=200
        )
        
        if not images:
            raise Exception("PDF konnte nicht zu Bildern konvertiert werden")
        
        # Speichere erstes Bild temporär
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            images[0].save(tmp_file.name, 'PNG')
            temp_image_path = tmp_file.name
        
        try:
            # Verwende LLM für OCR
            extracted_text, _, _ = _extract_from_image_pipeline(
                temp_image_path, 
                llm_service, 
                stream_callback, 
                logger
            )
            
            source_info = f"PDF (OCR): {filename} ({page_count} Seiten, per LLM analysiert)"
            return extracted_text, source_info, "pdf_llm_ocr"
            
        finally:
            # Cleanup temporäre Datei
            try:
                os.unlink(temp_image_path)
            except OSError:
                pass
                
    except Exception as e:
        raise Exception(f"PDF-LLM-OCR fehlgeschlagen: {str(e)}")


def _extract_from_image_pipeline(
    image_path: str,
    llm_service, 
    stream_callback: Optional[callable] = None,
    logger=None
) -> Tuple[str, str, str]:
    """Extract text from image using LLM for pipeline - Claude Generated"""
    import uuid
    import os
    from pathlib import Path
    from ..llm.prompt_service import PromptService
    
    filename = os.path.basename(image_path)
    
    if stream_callback:
        stream_callback(f"🖼️ Analysiere Bild mit LLM: {filename}")
    
    try:
        # Lade Konfiguration und bestimme zuerst das tatsächliche Vision-Modell.
        # So können modellspezifische OCR-Prompts und Parameter geladen werden.
        from ..utils.config_manager import ConfigManager
        config_manager = ConfigManager()
        config = config_manager.load_config()

        # Check if a vision-capable model is configured for OCR.
        # The PyQt settings UI stores this under "vision", while some older/manual configs
        # may use the prompt-specific key "image_text_extraction".
        has_providers = False
        configured_task = None
        for task_name in ["image_text_extraction", "vision", "initialisation", "keywords"]:
            task_prefs = config.unified_config.task_preferences.get(task_name)
            if task_prefs and bool(task_prefs.model_priority):
                has_providers = True
                configured_task = task_name
                break

        if not has_providers:
            error_msg = (
                "❌ Kein Vision-Modell für Bilderkennung konfiguriert!\n"
                "Bitte in der Config file unter 'unified_config.task_preferences.vision' "
                "oder 'unified_config.task_preferences.image_text_extraction' "
                "einen 'model_priority'-Eintrag hinzufügen.\n"
                "Beispiel: 'model_priority': [{'provider_name': 'openai_compatible', 'model_name': 'gpt-4o'}]"
            )
            if logger:
                logger.error(error_msg)
            raise Exception(error_msg)
        elif logger:
            logger.info(f"Vision/OCR task configuration found via '{configured_task}'")

        # Bestimme besten Provider für Bilderkennung
        provider, model = _get_best_vision_provider_pipeline(llm_service, logger)

        if not provider:
            raise Exception("Kein Provider mit Bilderkennung verfügbar")

        prompts_path = config.system_config.prompts_path
        prompt_service = PromptService(prompts_path, logger)

        # Load OCR prompt for the actual selected model, not the generic default.
        prompt_config_data = prompt_service.get_prompt_config(
            task="image_text_extraction",
            model=model or "default"
        )

        if not prompt_config_data:
            raise Exception("OCR-Prompt 'image_text_extraction' nicht gefunden in prompts.json")

        # Konvertiere PromptConfigData zu Dictionary für Kompatibilität
        prompt_config = {
            'prompt': prompt_config_data.prompt,
            'system': prompt_config_data.system or '',
            'temperature': prompt_config_data.temp,
            'top_p': prompt_config_data.p_value,
            'seed': prompt_config_data.seed
        }

        if stream_callback:
            stream_callback(f"🖼️ Verwende {provider} ({model}) für Bilderkennung...")
        
        request_id = str(uuid.uuid4())

        # LLM-Aufruf für Bilderkennung mit Streaming - Claude Generated
        response = llm_service.generate_response(
            provider=provider,
            model=model,
            prompt=prompt_config['prompt'],
            system=prompt_config.get('system', ''),
            request_id=request_id,
            temperature=float(prompt_config.get('temperature', 0.1)),
            p_value=float(prompt_config.get('top_p', 0.1)),
            seed=prompt_config.get('seed'),
            image=image_path,
            stream=True,  # Enable streaming for live feedback - Claude Generated
            output_format="xml",  # OCR expects raw text, not JSON-mode structured output.
        )

        # Handle streaming response with live callback - Claude Generated
        extracted_text = ""
        if hasattr(response, "__iter__") and not isinstance(response, str):
            # Generator response with live streaming
            text_parts = []
            for chunk in response:
                chunk_text = ""
                if isinstance(chunk, str):
                    chunk_text = chunk
                elif hasattr(chunk, 'text'):
                    chunk_text = chunk.text
                elif hasattr(chunk, 'content'):
                    chunk_text = chunk.content
                else:
                    chunk_text = str(chunk)

                # Send to live callback if available - Claude Generated
                if chunk_text and stream_callback:
                    stream_callback(chunk_text)

                text_parts.append(chunk_text)
            extracted_text = "".join(text_parts)
        else:
            extracted_text = str(response)

        error_markers = (
            "error with ",
            "error code:",
            "invalid_request_error",
            "unsupported_value",
            "unsupported value:",
        )
        lowered_response = extracted_text.strip().lower()
        if lowered_response and any(marker in lowered_response for marker in error_markers):
            raise Exception(extracted_text.strip())
        
        # Bereinige LLM-Output
        extracted_text = _clean_ocr_output_pipeline(extracted_text)
        
        if not extracted_text.strip():
            raise Exception("LLM konnte keinen Text im Bild erkennen")
        
        source_info = f"Bild (OCR): {filename}"
        return extracted_text, source_info, "image_llm_ocr"
        
    except Exception as e:
        raise Exception(f"Bild-LLM-OCR fehlgeschlagen: {str(e)}")


def _assess_text_quality_pipeline(text: str) -> Dict[str, Any]:
    """Assess quality of extracted PDF text for pipeline - Claude Generated"""
    if not text or len(text.strip()) == 0:
        return {'is_good': False, 'reason': 'Kein Text gefunden'}
    
    char_count = len(text)
    word_count = len(text.split())
    
    if char_count < 50:
        return {'is_good': False, 'reason': 'Text zu kurz'}
    
    if word_count > 0:
        avg_word_length = char_count / word_count
        if avg_word_length < 2 or avg_word_length > 20:
            return {'is_good': False, 'reason': 'Ungewöhnliche Wortlängen'}
    
    special_char_ratio = sum(1 for c in text if not c.isalnum() and c not in ' \n\t.,!?;:-()[]') / len(text)
    if special_char_ratio > 0.3:
        return {'is_good': False, 'reason': 'Zu viele Sonderzeichen'}
    
    lines_with_content = [line.strip() for line in text.split('\n') if len(line.strip()) > 5]
    if len(lines_with_content) < max(1, word_count // 20):
        return {'is_good': False, 'reason': 'Text fragmentiert'}
        
    return {'is_good': True, 'reason': 'Text-Qualität ausreichend'}


def _get_best_vision_provider_pipeline(llm_service, logger=None) -> Tuple[Optional[str], Optional[str]]:
    """Get best available provider for vision tasks using SmartProviderSelector - Claude Generated"""
    try:
        from .smart_provider_selector import SmartProviderSelector
        from .config_models import TaskType
        from .config_manager import ConfigManager

        # Initialize ConfigManager for task_preferences access - Claude Generated
        config_manager = ConfigManager()
        selector = SmartProviderSelector(config_manager)
        selection = selector.select_provider(
            task_type=TaskType.VISION,
            prefer_fast=False,
            task_name="image_text_extraction"  # Task preferences define explicit provider - no capability filtering needed - Claude Generated
        )

        if logger:
            logger.info(f"SmartProviderSelector chose {selection.provider} with {selection.model} for vision task (fallback_used: {selection.fallback_used})")
        
        return selection.provider, selection.model
        
    except Exception as e:
        if logger:
            logger.warning(f"SmartProviderSelector failed, falling back to legacy selection: {e}")
        
        # Legacy fallback for compatibility
        vision_providers = [
            ("gemini", ["gemini-2.0-flash", "gemini-1.5-flash"]),
            ("openai", ["gpt-4o", "gpt-4-vision-preview"]),
            ("anthropic", ["claude-3-5-sonnet", "claude-3-opus"]),
            ("ollama", ["llava", "minicpm-v", "cogito:32b"])
        ]
        
        try:
            available_providers = llm_service.get_available_providers()
            
            for provider_name, preferred_models in vision_providers:
                if provider_name in available_providers:
                    try:
                        available_models = llm_service.get_available_models(provider_name)
                        
                        for preferred_model in preferred_models:
                            if preferred_model in available_models:
                                return provider_name, preferred_model
                        
                        if available_models:
                            return provider_name, available_models[0]
                            
                    except Exception as e:
                        if logger:
                            logger.warning(f"Error checking models for {provider_name}: {e}")
                        continue
            
            return None, None
            
        except Exception as e:
            if logger:
                logger.error(f"Error determining best vision provider: {e}")
            return None, None


def _clean_ocr_output_pipeline(text: str) -> str:
    """Clean OCR output from common LLM artifacts for pipeline - Claude Generated"""
    if not text:
        return ""

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

        if line:
            cleaned_lines.append(line)

    return '\n'.join(cleaned_lines).strip()
