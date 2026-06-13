# geometry/__init__.py
# ==============================================================================
# Package: geometry
# Purpose: Geometric computations for polygon and convex shape operations
#
# This module provides all geometric operations needed for contact analysis:
#   - Primitives: Polygon and ConvexShape data structures
#   - Calculators: Pure calculation functions (area, normal, centroid, etc.)
#   - Processor: Stateful operations with caching
#   - Convex: Convex hull and decomposition algorithms
#
# The geometry module is organized in layers:
#   primitives.py  → Data models (immutable)
#   calculator.py  → Pure functions (no side effects)
#   processor.py   → Stateful operations (caching, mutable)
#   convex.py      → Complex algorithms
#
# Author: Contact Analysis Team
# Version: 1.0.0
# ==============================================================================

from __future__ import annotations

import logging
from typing import List, Tuple

# Set up logging for the geometry module
logger = logging.getLogger(__name__)

# Package metadata
__version__ = "1.0.0"
__author__ = "Contact Analysis Team"

__all__ = [
    # Primitives (data models)
    "Polygon",
    "ConvexShape",
    # Calculators (pure functions)
    "polygon_area_3d",
    "polygon_normal",
    "polygon_centroid",
    "reorder_polygon_vertices",
    "polygon_intersection_area",
    "check_coplanar",
    # Processor (stateful)
    "GeometryProcessor",
    # Convex algorithms
    "convex_hull_3d",
    "convex_decomposition",
    "ConvexShapeValidator",
]

try:
    """
    Import all public symbols from submodules.
    Wrapped in try-except for detailed error reporting.
    """
    from .primitives import Polygon, ConvexShape
    logger.debug("Successfully imported primitives from geometry.primitives")

except ImportError as e:
    logger.error(f"Failed to import primitives: {e}", exc_info=True)
    raise ImportError(f"geometry.primitives import failed: {e}") from e

try:
    from .calculator import (
        polygon_area_3d,
        polygon_normal,
        polygon_centroid,
        reorder_polygon_vertices,
        polygon_intersection_area,
        check_coplanar,
    )
    logger.debug("Successfully imported calculator functions from geometry.calculator")

except ImportError as e:
    logger.error(f"Failed to import calculator: {e}", exc_info=True)
    raise ImportError(f"geometry.calculator import failed: {e}") from e

try:
    from .processor import GeometryProcessor
    logger.debug("Successfully imported GeometryProcessor from geometry.processor")

except ImportError as e:
    logger.error(f"Failed to import processor: {e}", exc_info=True)
    raise ImportError(f"geometry.processor import failed: {e}") from e

try:
    from .convex import (
        convex_hull_3d,
        convex_decomposition,
        ConvexShapeValidator,
    )
    logger.debug("Successfully imported convex algorithms from geometry.convex")

except ImportError as e:
    logger.error(f"Failed to import convex: {e}", exc_info=True)
    raise ImportError(f"geometry.convex import failed: {e}") from e


def validate_geometry_imports() -> bool:
    """
    Validate that all geometry submodules are properly imported.
    
    Returns
    -------
    bool
        True if all imports successful
        
    Raises
    ------
    ContactAnalysisError
        If any required symbol is missing
    """
    from core import ContactAnalysisError
    
    required_symbols = __all__
    missing_symbols = []
    
    for symbol in required_symbols:
        if not hasattr(__import__(__name__), symbol):
            missing_symbols.append(symbol)
    
    if missing_symbols:
        error_msg = f"Missing geometry symbols: {missing_symbols}"
        logger.error(error_msg)
        raise ContactAnalysisError(error_msg)
    
    logger.info(f"✓ Geometry module validation successful. {len(required_symbols)} symbols available.")
    return True


# Validate imports when module loads
try:
    validate_geometry_imports()
    logger.info(f"Geometry module v{__version__} initialized successfully")
except Exception as e:
    logger.critical(f"Geometry module initialization failed: {e}", exc_info=True)
    raise


if __name__ == "__main__":
    """
    Test/demo when module is run directly.
    """
    import sys
    from core.utilities import setup_logging
    
    # Enable detailed logging
    setup_logging(level="DEBUG")
    
    print("\n" + "="*80)
    print("GEOMETRY MODULE VALIDATION TEST")
    print("="*80)
    
    print(f"\nVersion: {__version__}")
    print(f"Author: {__author__}")
    print(f"\nExported symbols ({len(__all__)}):")
    for i, symbol in enumerate(__all__, 1):
        print(f"  {i:2d}. {symbol}")
    
    print("\n" + "="*80)
    print("✓ Geometry module is ready for use!")
    print("="*80 + "\n")
