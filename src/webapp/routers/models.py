"""Model-list + queue-status endpoints for the webapp (F-6 split).

Claude Generated — moved verbatim from ``app.py``. Provides the override
dropdown's provider/model list, a refresh hook, and the global LLM+pipeline
queue badge data.
"""

import logging

from fastapi import APIRouter

from src.webapp.session_state import AppContext, sessions

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/api/models")
async def get_available_models() -> list:
    """Get available provider/model combinations for override dropdown - Claude Generated

    Live-detects models per provider via ProviderDetectionService (shared 300s TTL
    cache, same source as the Qt6 GUI), instead of reading the runtime-only
    ``provider.available_models`` field that is never populated on the webapp
    backend. Falls back to the persisted list / preferred model when detection
    yields nothing (e.g. provider unreachable) so the dropdown is never empty.
    """
    try:
        app_context = AppContext()
        services = app_context.get_services()
        config_manager = services['config_manager']
        detection = config_manager.get_provider_detection_service()
        unified_config = config_manager.get_unified_config()
        enabled_providers = unified_config.get_enabled_providers()

        models = []
        for provider in enabled_providers:
            provider_name = provider.name
            try:
                available = detection.get_available_models(provider_name) or []
            except Exception as e:
                logger.warning(f"Model detection failed for {provider_name}: {e}")
                available = []
            if not available:
                available = list(getattr(provider, 'available_models', []) or [])
            if not available and getattr(provider, 'preferred_model', None):
                available = [provider.preferred_model]
            for model in available:
                models.append({
                    "provider": provider_name,
                    "model": model,
                    "value": f"{provider_name}|{model}"
                })
        return models
    except Exception as e:
        logger.error(f"Error getting models: {e}")
        return []


@router.post("/api/models/refresh")
async def refresh_models() -> list:
    """Re-read config.json + re-detect provider models, then return the fresh list - Claude Generated

    Mirrors the Qt6 GUI's "Refresh Models" / config-change path
    (``MainWindow._refresh_components``): force-reload config from disk, rebuild
    the generation provider clients, and rebuild the detection service while
    clearing its model cache. Lets the webapp pick up providers/models added
    externally (Qt6 GUI or a config.json edit) without a server restart.
    """
    try:
        app_context = AppContext()
        services = app_context.get_services()
        config_manager = services['config_manager']
        config_manager.load_config(force_reload=True)
        try:
            services['llm_service'].reload_providers()
        except Exception as e:
            logger.warning(f"llm_service.reload_providers failed: {e}")
        config_manager.get_provider_detection_service().reload()
    except Exception as e:
        logger.error(f"Error refreshing models: {e}")
    return await get_available_models()


@router.get("/api/queue/status")
async def get_queue_status() -> dict:
    """Get combined LLM + Pipeline queue status - Claude Generated (2026-01-13)"""
    # Get LLM stats from AppContext.pipeline_manager
    app_context = AppContext()
    try:
        llm_stats = app_context.pipeline_manager.get_llm_queue_status()
    except Exception as e:
        logger.warning(f"Could not get LLM queue status: {e}")
        llm_stats = {
            "active_llm_requests": 0,
            "pending_llm_requests": 0,
            "max_concurrent": 3,
            "total_completed": 0,
            "avg_duration_seconds": 0.0
        }

    # Pipeline stats (simplified)
    pipeline_stats = {
        "active_pipelines": len([s for s in sessions.values() if s.status == "running"]),
        "queued_sessions": 0  # No longer queueing at pipeline level
    }

    # Determine overall status
    status = "healthy"
    if llm_stats["pending_llm_requests"] > 10:
        status = "busy"

    return {
        "llm": llm_stats,
        "pipeline": pipeline_stats,
        "status": status
    }
