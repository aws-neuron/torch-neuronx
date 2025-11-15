"""Cache management for compiled NEFF binaries."""

import logging

from torch_neuronx.utils import (
    is_neff_cache_disabled,
    log_neff_cache_hit,
    log_neff_cache_miss,
    log_neff_cache_store,
)

logger = logging.getLogger(__name__)

# Module-level global cache shared by all CompilationCache instances
_GLOBAL_CACHE: dict[str, tuple[str, bytes]] = {}
_GLOBAL_CACHE_DISABLED_LOGGED: bool = False


class CompilationCache:
    """Manages caching of compiled NEFF binaries (process-wide)."""

    def __init__(self):
        # No per-instance NEFF store; use global cache
        # Keep per-instance metadata for compatibility with callers
        self._metadata: dict[str, dict] = {}
        self._cache_disabled_logged = False

    def get_neff(self, cache_key: str) -> bytes | None:
        """Retrieve NEFF from cache if available.

        Args:
            cache_key: Cache key for the NEFF

        Returns:
            NEFF bytes if found, None otherwise
        """
        global _GLOBAL_CACHE_DISABLED_LOGGED
        if is_neff_cache_disabled():
            if not _GLOBAL_CACHE_DISABLED_LOGGED:
                logger.info("NEFF cache is disabled")
                _GLOBAL_CACHE_DISABLED_LOGGED = True
            return None

        if cache_key in _GLOBAL_CACHE:
            log_neff_cache_hit(cache_key)
            _, neff_bytes = _GLOBAL_CACHE[cache_key]
            return neff_bytes

        log_neff_cache_miss(cache_key)
        return None

    def store_neff(self, cache_key: str, neff_bytes: bytes, metadata: dict | None = None) -> None:
        """Store NEFF in cache.

        Args:
            cache_key: Cache key for the NEFF
            neff_bytes: Compiled NEFF bytes
            metadata: Optional metadata to store with NEFF
        """
        global _GLOBAL_CACHE_DISABLED_LOGGED
        if is_neff_cache_disabled():
            if not _GLOBAL_CACHE_DISABLED_LOGGED:
                logger.info("NEFF cache is disabled")
                _GLOBAL_CACHE_DISABLED_LOGGED = True
            return

        _GLOBAL_CACHE[cache_key] = ("xla.neff", neff_bytes)
        if metadata:
            self._metadata[cache_key] = metadata
        log_neff_cache_store(cache_key)
        logger.debug(f"Stored NEFF in cache with key: {cache_key}")

    def get_metadata(self, cache_key: str) -> dict | None:
        """Get metadata for cached NEFF.

        Args:
            cache_key: Cache key for the NEFF

        Returns:
            Metadata dict if found, None otherwise
        """
        return self._metadata.get(cache_key)

    def clear(self) -> None:
        """Clear all cached data."""
        _GLOBAL_CACHE.clear()
        self._metadata.clear()
        logger.info("Cleared compilation cache")

    def size(self) -> int:
        """Get number of cached NEFF entries."""
        return len(_GLOBAL_CACHE)

    def memory_usage(self) -> int:
        """Estimate memory usage of cached NEFFs in bytes."""
        total = 0
        for _, (_, neff_bytes) in _GLOBAL_CACHE.items():
            total += len(neff_bytes)
        return total
