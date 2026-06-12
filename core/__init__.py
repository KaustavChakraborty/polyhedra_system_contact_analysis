# core/__init__.py
# ==============================================================================
# Package: core
# Purpose: Shared utilities, constants, exceptions, and types for contact analysis
# 
# This module provides foundational components used across all domain modules:
#   - Constants: Physical constants, tolerances, defaults
#   - Exceptions: Custom exception hierarchy for debugging
#   - Types: Immutable data structures (dataclasses)
#   - Utilities: Helper functions (math, validation, logging)
#
# Author: Contact Analysis Team
# Version: 1.0.0
# ==============================================================================

from __future__ import annotations

import logging
from typing import List

# Set up logging for the core module
logger = logging.getLogger(__name__)

# Package metadata
__version__ = "1.0.0"
__author__ = "Contact Analysis Team"
__all__ = [
    # Constants
    "TOLERANCE_QUATERNION",
    "POLYGON_REORDER_TOL",
    "CONVEX_DECOMP_TOL",
    "MIN_OVERLAP_AREA",
    "RDF_DEFAULT_BINS",
    "RDF_DEFAULT_R_MAX",
    "DEFAULT_NEIGHBOR_R_MAX",
    "COMPONENTS",
    "AVAILABLE_METRICS",
    # Exceptions
    "ContactAnalysisError",
    "ConfigError",
    "GeometryError",
    "ParticleError",
    "ContactError",
    "MetricError",
    "TrajectoryError",
    "VisualizationError",
    "ValidationError",
    "DataTypeError",
    # Types
    "Box",
    "ComputationResult",
    # Utilities
    "normalize_vector",
    "rotation_matrix_from_quaternion",
    "apply_rotation",
    "distance_point_to_plane",
    "validate_vertices",
    "validate_faces",
    "validate_box",
    "setup_logging",
]

try:
    """
    Import all public symbols from submodules.
    
    Imports are wrapped in try-except to provide detailed debugging information
    if any submodule fails to load, helping identify import errors early.
    """
    from .constants import (
        TOLERANCE_QUATERNION,
        POLYGON_REORDER_TOL,
        CONVEX_DECOMP_TOL,
        MIN_OVERLAP_AREA,
        RDF_DEFAULT_BINS,
        RDF_DEFAULT_R_MAX,
        DEFAULT_NEIGHBOR_R_MAX,
        COMPONENTS,
        AVAILABLE_METRICS,
    )
    logger.debug("Successfully imported constants from core.constants")

except ImportError as e:
    logger.error(f"Failed to import constants: {e}", exc_info=True)
    raise ImportError(f"core.constants import failed: {e}") from e

try:
    from .exceptions import (
        ContactAnalysisError,
        ConfigError,
        GeometryError,
        ParticleError,
        ContactError,
        MetricError,
        TrajectoryError,
        VisualizationError,
        ValidationError,
        DataTypeError,
    )
    logger.debug("Successfully imported exceptions from core.exceptions")

except ImportError as e:
    logger.error(f"Failed to import exceptions: {e}", exc_info=True)
    raise ImportError(f"core.exceptions import failed: {e}") from e

try:
    from .types import (
        Box,
        ComputationResult,
    )
    logger.debug("Successfully imported types from core.types")

except ImportError as e:
    logger.error(f"Failed to import types: {e}", exc_info=True)
    raise ImportError(f"core.types import failed: {e}") from e

try:
    from .utilities import (
        normalize_vector,
        rotation_matrix_from_quaternion,
        apply_rotation,
        distance_point_to_plane,
        validate_vertices,
        validate_faces,
        validate_box,
        setup_logging,
    )
    logger.debug("Successfully imported utilities from core.utilities")

except ImportError as e:
    logger.error(f"Failed to import utilities: {e}", exc_info=True)
    raise ImportError(f"core.utilities import failed: {e}") from e


def validate_core_imports() -> bool:
    """
    Validate that all core submodules are properly imported.
    
    This function checks that all expected symbols are available in the core
    namespace, useful for debugging import issues.
    
    Returns
    -------
    bool
        True if all imports successful, raises exception otherwise.
        
    Raises
    ------
    ContactAnalysisError
        If any required symbol is missing from the core namespace.
        
    Examples
    --------
    >>> from core import validate_core_imports
    >>> if validate_core_imports():
    ...     print("Core module is ready!")
    """
    required_symbols = __all__
    
    missing_symbols = []
    for symbol in required_symbols:
        if not hasattr(__import__(__name__), symbol):
            missing_symbols.append(symbol)
    
    if missing_symbols:
        error_msg = f"Missing core symbols: {missing_symbols}"
        logger.error(error_msg)
        raise ContactAnalysisError(error_msg)
    
    logger.info(f"✓ Core module validation successful. {len(required_symbols)} symbols available.")
    return True


# Automatically validate imports when module is loaded
try:
    validate_core_imports()
    logger.info(f"Core module v{__version__} initialized successfully")
except ContactAnalysisError as e:
    logger.critical(f"Core module initialization failed: {e}", exc_info=True)
    raise


if __name__ == "__main__":
    """
    Test/demo when module is run directly.
    
    This section allows quick validation that the core module is properly
    configured and all submodules can be imported correctly.
    """
    import sys
    
    # Enable detailed logging for debugging
    setup_logging(level="DEBUG")
    
    print("\n" + "="*80)
    print("CORE MODULE VALIDATION TEST")
    print("="*80)
    
    print(f"\nVersion: {__version__}")
    print(f"Author: {__author__}")
    print(f"\nExported symbols ({len(__all__)}):")
    for i, symbol in enumerate(__all__, 1):
        print(f"  {i:2d}. {symbol}")
    
    print("\n" + "="*80)
    print(" Core module is ready for use!")
    print("="*80 + "\n")