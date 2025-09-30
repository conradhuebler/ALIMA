#!/usr/bin/env python3
"""
ProviderStatusService - Centralized Provider Status Management
Eliminates redundant provider tests and UI blocking by providing cached status with background updates.
Claude Generated
"""

from PyQt6.QtCore import QObject, pyqtSignal, QThread, QTimer
from typing import Dict, List, Optional, Any
import logging
import datetime
import time
from ..utils.config_manager import ProviderDetectionService


class ProviderTestWorker(QThread):
    """Worker thread for non-blocking provider tests - Claude Generated"""

    # Signals for communicating test results back to main thread
    test_completed = pyqtSignal(str, dict)  # provider_name, provider_info
    all_tests_completed = pyqtSignal()

    def __init__(self, provider_names: List[str], detection_service: ProviderDetectionService):
        super().__init__()
        self.provider_names = provider_names
        self.detection_service = detection_service
        self.logger = logging.getLogger(__name__)

    def run(self):
        """Execute provider tests sequentially to avoid gRPC threading conflicts - Claude Generated"""
        # No logging from thread to prevent reentrant issues - status reported via signals

        for provider_name in self.provider_names:
            try:
                # Special handling for gRPC-based providers (Gemini) to avoid thread conflicts
                if provider_name == 'gemini':
                    # Add delay to ensure no overlap with other gRPC operations
                    time.sleep(0.1)

                # Test provider comprehensively
                provider_info = self._test_single_provider(provider_name)
                self.test_completed.emit(provider_name, provider_info)

                # Small delay between provider tests to avoid resource conflicts
                time.sleep(0.05)

            except Exception as e:
                # Emit error info via signal (no direct logging from thread)
                error_info = {
                    'reachable': False,
                    'models': [],
                    'model_count': 0,
                    'last_checked': datetime.datetime.now(),
                    'status': 'error',
                    'error_message': str(e)
                }
                self.test_completed.emit(provider_name, error_info)

                # Continue with next provider even if one fails
                time.sleep(0.05)

        self.all_tests_completed.emit()

    def _test_single_provider(self, provider_name: str) -> Dict[str, Any]:
        """Test a single provider with gRPC-aware error handling - Claude Generated"""
        start_time = time.time()

        try:
            # Special handling for gRPC-based providers (Gemini)
            if provider_name == 'gemini':
                return self._test_gemini_safe(provider_name, start_time)
            else:
                return self._test_provider_standard(provider_name, start_time)

        except Exception as e:
            # Enhanced error handling for different provider types
            error_msg = str(e)
            if 'gpr_atm' in error_msg or 'epoll' in error_msg:
                error_msg = 'gRPC threading conflict (will retry later)'

            return {
                'reachable': False,
                'models': [],
                'model_count': 0,
                'last_checked': datetime.datetime.now(),
                'status': 'error',
                'error_message': error_msg
            }

    def _test_gemini_safe(self, provider_name: str, start_time: float) -> Dict[str, Any]:
        """Test Gemini provider with gRPC thread-safety measures - Claude Generated"""
        try:
            # Check basic availability with extra caution for gRPC
            available_providers = self.detection_service.get_available_providers()
            is_available = provider_name in available_providers

            if not is_available:
                return {
                    'reachable': False,
                    'models': [],
                    'model_count': 0,
                    'last_checked': datetime.datetime.now(),
                    'status': 'not_configured',
                    'error_message': 'Gemini provider not configured'
                }

            # Test reachability with gRPC error handling
            try:
                is_reachable = self.detection_service.is_provider_reachable(provider_name)
            except Exception as e:
                if 'gpr_atm' in str(e) or 'epoll' in str(e):
                    return {
                        'reachable': False,
                        'models': [],
                        'model_count': 0,
                        'last_checked': datetime.datetime.now(),
                        'status': 'gRPC_conflict',
                        'error_message': 'gRPC threading conflict - provider may be functional'
                    }
                raise

            if not is_reachable:
                return {
                    'reachable': False,
                    'models': [],
                    'model_count': 0,
                    'last_checked': datetime.datetime.now(),
                    'status': 'offline',
                    'error_message': 'Gemini provider not reachable'
                }

            # Get available models with gRPC protection
            try:
                models = self.detection_service.get_available_models(provider_name)
            except Exception as e:
                if 'gpr_atm' in str(e) or 'epoll' in str(e):
                    # gRPC conflict, but provider might be functional
                    return {
                        'reachable': True,  # Assume reachable since basic test passed
                        'models': [],  # Can't get models due to gRPC issue
                        'model_count': 0,
                        'last_checked': datetime.datetime.now(),
                        'status': 'gRPC_partial',
                        'error_message': 'gRPC conflict during model detection'
                    }
                raise

            test_duration = time.time() - start_time

            return {
                'reachable': True,
                'models': models,
                'model_count': len(models),
                'last_checked': datetime.datetime.now(),
                'status': 'available',
                'error_message': None,
                'test_duration': test_duration
            }

        except Exception as e:
            return {
                'reachable': False,
                'models': [],
                'model_count': 0,
                'last_checked': datetime.datetime.now(),
                'status': 'error',
                'error_message': f'Gemini test failed: {str(e)}'
            }

    def _test_provider_standard(self, provider_name: str, start_time: float) -> Dict[str, Any]:
        """Test standard (non-gRPC) provider - Claude Generated"""
        # Check basic availability
        available_providers = self.detection_service.get_available_providers()
        is_available = provider_name in available_providers

        if not is_available:
            return {
                'reachable': False,
                'models': [],
                'model_count': 0,
                'last_checked': datetime.datetime.now(),
                'status': 'not_configured',
                'error_message': 'Provider not configured'
            }

        # Test reachability
        is_reachable = self.detection_service.is_provider_reachable(provider_name)

        if not is_reachable:
            return {
                'reachable': False,
                'models': [],
                'model_count': 0,
                'last_checked': datetime.datetime.now(),
                'status': 'offline',
                'error_message': 'Provider not reachable'
            }

        # Get available models
        models = self.detection_service.get_available_models(provider_name)

        test_duration = time.time() - start_time

        return {
            'reachable': True,
            'models': models,
            'model_count': len(models),
            'last_checked': datetime.datetime.now(),
            'status': 'available',
            'error_message': None,
            'test_duration': test_duration
        }


class ProviderStatusService(QObject):
    """
    Central Provider Status Management Service - Claude Generated

    Provides cached provider status with background updates to eliminate UI blocking.
    Single source of truth for all provider information across the application.

    Features:
    - Non-blocking provider tests via QThread
    - Intelligent caching with configurable refresh intervals
    - Signal-based UI updates for reactive interfaces
    - Force refresh capability for manual updates
    """

    # Signals for UI integration
    status_updated = pyqtSignal()  # Cache was updated
    check_started = pyqtSignal()   # Background tests started
    provider_tested = pyqtSignal(str, dict)  # Single provider test completed

    def __init__(self, llm_service: 'LlmService'):
        super().__init__()
        self.llm_service = llm_service
        self.logger = logging.getLogger(__name__)

        # Initialize detection service (reuse existing infrastructure)
        try:
            self.detection_service = ProviderDetectionService()
            self.logger.info("ProviderDetectionService initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize ProviderDetectionService: {e}")
            self.detection_service = None

        # Status cache: provider_name -> provider_info
        self._status_cache: Dict[str, Dict[str, Any]] = {}

        # Control flags
        self._is_checking = False
        self._last_full_check: Optional[datetime.datetime] = None

        # Background worker
        self._test_worker: Optional[ProviderTestWorker] = None

        # Auto-refresh timer (optional - for very dynamic environments)
        self._auto_refresh_timer = QTimer()
        self._auto_refresh_timer.timeout.connect(lambda: self.refresh_all(force=False))
        # Start with 10 minute auto-refresh (can be configured)
        self._auto_refresh_interval = 10 * 60 * 1000  # 10 minutes in milliseconds

        self.logger.info("ProviderStatusService initialized")

    def get_all_provider_info(self) -> Dict[str, Dict[str, Any]]:
        """
        Get cached provider information for all providers - Claude Generated

        Returns immediately with cached data. UI should call this for fast access.
        If cache is empty, returns empty dict and triggers background refresh.

        Returns:
            Dict mapping provider names to their status information
        """
        if not self._status_cache:
            self.logger.info("Cache is empty, triggering background refresh")
            self.refresh_all(force=False)
            return {}

        self.logger.debug(f"Returning cached info for {len(self._status_cache)} providers")
        return self._status_cache.copy()

    def get_provider_status(self, provider_name: str) -> Optional[Dict[str, Any]]:
        """
        Get cached status for a specific provider - Claude Generated

        Args:
            provider_name: Name of the provider to get status for

        Returns:
            Provider status dict or None if not in cache
        """
        return self._status_cache.get(provider_name)

    def is_provider_reachable(self, provider_name: str) -> bool:
        """
        Quick check if provider is reachable based on cache - Claude Generated

        Args:
            provider_name: Name of the provider to check

        Returns:
            True if provider is cached as reachable, False otherwise
        """
        provider_info = self.get_provider_status(provider_name)
        return provider_info.get('reachable', False) if provider_info else False

    def get_available_models(self, provider_name: str) -> List[str]:
        """
        Get cached model list for a provider - Claude Generated

        Args:
            provider_name: Name of the provider

        Returns:
            List of available models (empty if provider not reachable or not cached)
        """
        provider_info = self.get_provider_status(provider_name)
        return provider_info.get('models', []) if provider_info else []

    def refresh_all(self, force: bool = False):
        """
        Refresh provider status for all providers - Claude Generated

        Args:
            force: If True, refresh even if cache is recent
                  If False, only refresh if cache is stale or empty
        """
        if not self.detection_service:
            self.logger.warning("Cannot refresh: ProviderDetectionService not available")
            return

        # Check if refresh is needed
        if not force and self._is_checking:
            self.logger.debug("Refresh already in progress, skipping")
            return

        if not force and self._last_full_check:
            time_since_last = datetime.datetime.now() - self._last_full_check
            if time_since_last.total_seconds() < 300:  # 5 minutes
                self.logger.debug("Cache is recent, skipping refresh")
                return

        # Get list of providers to test
        try:
            available_providers = self.detection_service.get_available_providers()
            if not available_providers:
                self.logger.warning("No providers available for testing")
                return

        except Exception as e:
            self.logger.error(f"Failed to get available providers: {e}")
            return

        self.logger.info(f"Starting provider refresh for {len(available_providers)} providers (force={force})")

        # Set checking flag and emit signal
        self._is_checking = True
        self.check_started.emit()

        # Clean up any existing worker
        if self._test_worker and self._test_worker.isRunning():
            self._test_worker.terminate()
            self._test_worker.wait()

        # Create and start background worker
        self._test_worker = ProviderTestWorker(available_providers, self.detection_service)
        self._test_worker.test_completed.connect(self._on_provider_tested)
        self._test_worker.all_tests_completed.connect(self._on_all_tests_completed)
        self._test_worker.start()

    def _on_provider_tested(self, provider_name: str, provider_info: Dict[str, Any]):
        """Handle completion of single provider test - Claude Generated"""
        # Update cache
        self._status_cache[provider_name] = provider_info

        # Log result safely from main thread
        status = provider_info.get('status', 'unknown')
        model_count = provider_info.get('model_count', 0)
        self.logger.info(f"Provider {provider_name}: {status} ({model_count} models)")

        # Emit signals
        self.provider_tested.emit(provider_name, provider_info)
        self.status_updated.emit()  # Trigger UI updates

    def _on_all_tests_completed(self):
        """Handle completion of all provider tests - Claude Generated"""
        self._is_checking = False
        self._last_full_check = datetime.datetime.now()

        # Final status update
        self.status_updated.emit()

        # Log summary safely from main thread
        self.logger.info(f"Provider refresh completed. Cache contains {len(self._status_cache)} providers")

        reachable_count = sum(1 for info in self._status_cache.values() if info.get('reachable', False))
        total_models = sum(info.get('model_count', 0) for info in self._status_cache.values())
        self.logger.info(f"Summary: {reachable_count}/{len(self._status_cache)} providers reachable, {total_models} total models")

    def start_auto_refresh(self, interval_minutes: int = 10):
        """
        Start automatic background refresh - Claude Generated

        Args:
            interval_minutes: Refresh interval in minutes
        """
        self._auto_refresh_interval = interval_minutes * 60 * 1000
        self._auto_refresh_timer.start(self._auto_refresh_interval)
        self.logger.info(f"Auto-refresh started with {interval_minutes} minute interval")

    def stop_auto_refresh(self):
        """Stop automatic background refresh - Claude Generated"""
        self._auto_refresh_timer.stop()
        self.logger.info("Auto-refresh stopped")

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the current cache - Claude Generated

        Returns:
            Dict with cache statistics for monitoring/debugging
        """
        if not self._status_cache:
            return {
                'provider_count': 0,
                'reachable_count': 0,
                'total_models': 0,
                'last_check': None,
                'is_checking': self._is_checking
            }

        reachable_count = sum(1 for info in self._status_cache.values() if info.get('reachable', False))
        total_models = sum(info.get('model_count', 0) for info in self._status_cache.values())

        return {
            'provider_count': len(self._status_cache),
            'reachable_count': reachable_count,
            'total_models': total_models,
            'last_check': self._last_full_check,
            'is_checking': self._is_checking
        }

    def cleanup(self):
        """Clean up resources when service is destroyed - Claude Generated"""
        if self._test_worker and self._test_worker.isRunning():
            self._test_worker.terminate()
            self._test_worker.wait()

        self.stop_auto_refresh()
        self.logger.info("ProviderStatusService cleaned up")