"""Build-time configuration - auto-generated, do not edit."""

# This value is baked in at build time
BUILD_TYPE = "RELEASE"

# Build-time defaults based on release flag
if BUILD_TYPE == "RELEASE":
    # Release defaults - optimized for production
    DEFAULT_SYNC_MODE = "1"  # Async
    DEFAULT_FALLBACK_ONLY_FOR_UNIMPLEMENTED = "0"
else:
    # Development defaults - optimized for debugging
    DEFAULT_SYNC_MODE = "0"  # Sync
    DEFAULT_FALLBACK_ONLY_FOR_UNIMPLEMENTED = "1"
