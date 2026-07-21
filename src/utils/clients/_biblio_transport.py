"""Transport reliability for BiblioClient - Claude Generated.

Split out of ``biblio_client.py`` (WP cleanup D, second axis). Verbatim mixin
extraction: the methods stay on ``BiblioClient`` via MRO, so no call site
changes.

The reliability layer around the Libero SOAP endpoint: rate limiting, a
circuit breaker (open/close on consecutive failures), session recycling, and
the performance counters. These methods read AND write ``self`` state
(``consecutive_failures``, ``circuit_breaker_open``, ``last_search_time`` …),
which is fine across a mixin — it is one instance; the attributes are still
initialised in ``BiblioClient.__init__``. They read no other method except
``log_performance_stats`` → ``get_performance_stats`` (both here).

``logger`` is the named ``biblio_extractor`` logger, re-declared with that exact
name so its lines keep landing where they did.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict

import requests

logger = logging.getLogger("biblio_extractor")


class BiblioTransportMixin:
    """Rate limit, circuit breaker, session, perf counters. Mixed into :class:`BiblioClient`."""

    def get_performance_stats(self) -> Dict[str, int]:
        """Get performance statistics for JSON cache vs SOAP - Claude Generated

        Returns:
            Dictionary with statistics about cache hits and SOAP calls
        """
        total_lookups = self._json_cache_hits + self._soap_calls
        json_hit_rate = (self._json_cache_hits / total_lookups * 100) if total_lookups > 0 else 0

        return {
            'json_cache_hits': self._json_cache_hits,
            'soap_calls': self._soap_calls,
            'total_lookups': total_lookups,
            'json_hit_rate': json_hit_rate,
            'cache_enabled': self.use_json_cache
        }

    def log_performance_stats(self):
        """Log performance statistics - Claude Generated"""
        stats = self.get_performance_stats()
        logger.debug(f"Performance: {stats['json_cache_hits']} cache hits, {stats['soap_calls']} SOAP calls, "
                     f"{stats['total_lookups']} total, {stats['json_hit_rate']:.1f}% hit rate")

    def _apply_rate_limit(self):
        """Apply rate limiting delay between searches - Claude Generated"""
        import time
        if self.last_search_time is not None:
            elapsed_ms = (time.time() - self.last_search_time) * 1000
            if elapsed_ms < self.rate_limit_delay_ms:
                sleep_ms = self.rate_limit_delay_ms - elapsed_ms
                time.sleep(sleep_ms / 1000.0)
                logger.debug(f"⏱️ Rate limit: slept {sleep_ms:.0f}ms")

        self.last_search_time = time.time()

    def _check_circuit_breaker(self) -> bool:
        """Check if circuit breaker is open - Claude Generated

        Returns:
            True if circuit is open (should skip search), False if closed
        """
        import time
        if not self.circuit_breaker_open:
            return False

        # Check if reset time has passed
        if self.circuit_breaker_reset_time and time.time() >= self.circuit_breaker_reset_time:
            logger.info("🔄 Circuit breaker reset - allowing searches")
            self.circuit_breaker_open = False
            self.consecutive_failures = 0
            return False

        remaining_time = self.circuit_breaker_reset_time - time.time() if self.circuit_breaker_reset_time else 0
        logger.warning(f"⚠️ Circuit breaker OPEN - skipping search (resets in {remaining_time:.0f}s)")
        return True

    def _record_search_failure(self):
        """Record search failure for circuit breaker - Claude Generated"""
        import time
        self.consecutive_failures += 1

        if self.consecutive_failures >= self.circuit_breaker_threshold:
            self.circuit_breaker_open = True
            self.circuit_breaker_reset_time = time.time() + 60  # 60 second reset
            logger.error(f"🔴 Circuit breaker OPENED after {self.consecutive_failures} consecutive failures")

    def _record_search_success(self):
        """Record search success - resets circuit breaker - Claude Generated"""
        if self.consecutive_failures > 0:
            logger.info(f"✅ Search success - resetting failure counter (was {self.consecutive_failures})")
        self.consecutive_failures = 0
        self.circuit_breaker_open = False

    def get_circuit_breaker_status(self) -> Dict[str, Any]:
        """Get circuit breaker status for UI feedback - Claude Generated

        Returns:
            Dictionary with circuit breaker state:
            - open: bool - True if circuit is open (blocking requests)
            - remaining_seconds: int - Seconds until reset (only if open)
            - consecutive_failures: int - Current failure count
        """
        if not self.circuit_breaker_open:
            return {
                "open": False,
                "consecutive_failures": self.consecutive_failures
            }

        remaining = 0
        if self.circuit_breaker_reset_time:
            remaining = max(0, int(self.circuit_breaker_reset_time - time.time()))

        return {
            "open": True,
            "remaining_seconds": remaining,
            "consecutive_failures": self.consecutive_failures
        }

    def _reset_session_if_needed(self):
        """Reset session after N requests to prevent staleness - Claude Generated"""
        self.session_request_count += 1

        if self.session_request_count >= self.session_max_requests:
            logger.debug(f"Resetting session after {self.session_request_count} requests")
            self.session.close()
            self.session = requests.Session()
            self.session_request_count = 0

    def _get_appropriate_delay(self, default_delay: float = 1.5) -> float:
        """
        Dynamically determine appropriate delay based on lookup pattern.

        If we're mostly using JSON lookups, use minimal delay.
        If we're mostly using SOAP lookups, use full delay to be nice to server.

        Args:
            default_delay: Base delay in seconds

        Returns:
            Appropriate delay in seconds
        """
        # If we haven't done any lookups yet, use default
        if not hasattr(self, '_json_lookups_used') and not hasattr(self, '_soap_lookups_needed'):
            return default_delay

        json_count = getattr(self, '_json_lookups_used', 0)
        soap_count = getattr(self, '_soap_lookups_needed', 0)
        total_lookups = json_count + soap_count

        # If we haven't done many lookups yet, use default
        if total_lookups < 3:
            return default_delay

        # Calculate ratio of JSON to total lookups
        json_ratio = json_count / total_lookups if total_lookups > 0 else 0

        # If mostly JSON lookups, use minimal delay
        if json_ratio > 0.7:
            return 0.01  # Nearly instant for JSON-dominated lookups
        # If mostly SOAP, use full delay
        elif json_ratio < 0.3:
            return default_delay
        # Mixed case, use reduced delay
        else:
            return default_delay * 0.5

    def set_disable_sql_cache(self, disable: bool):
        """Set whether to disable SQL database caching for testing."""
        self.disable_sql_cache = disable
        logger.info(f"SQL cache {'DISABLED' if disable else 'ENABLED'} for testing")
