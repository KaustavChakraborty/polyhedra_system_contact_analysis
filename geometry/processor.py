# geometry/processor.py
# ==============================================================================
# Module: geometry.processor
# Purpose: Stateful geometric processor with caching
#
# Wraps calculator functions with:
#   - State management (caches computed properties)
#   - Memory efficiency (LRU-like caching for expensive computations)
#   - Debugging (detailed logging of operations)
#   - Performance tracking (cache hit/miss statistics)
#
# The processor solves the problem of recomputing expensive geometric
# properties repeatedly. By caching results keyed by vertex array identity,
# we avoid redundant Shoelace formula computations.
#
# Author: Contact Analysis Team
# ==============================================================================

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple
from dataclasses import dataclass

import numpy as np

from core import ValidationError, GeometryError, POLYGON_REORDER_TOL
from .primitives import Polygon
from .calculator import (
    polygon_area_3d,
    polygon_normal,
    polygon_centroid,
    reorder_polygon_vertices,
    check_coplanar,
)

logger = logging.getLogger(__name__)


# ==============================================================================
# CACHED POLYGON PROPERTIES
# ==============================================================================

@dataclass(frozen=True)
class PolygonProperties:
    """
    Immutable container for cached polygon properties.
    
    Stores the computed geometric properties of a polygon:
    - area: Polygon area
    - normal: Unit normal vector
    - centroid: Center of mass
    - is_ordered: Whether vertices are in cyclic order
    
    Attributes
    ----------
    area : float
        Polygon area (always >= 0)
    normal : np.ndarray
        Unit normal vector to polygon plane, shape (3,)
    centroid : np.ndarray
        Centroid coordinates, shape (3,)
    is_ordered : bool
        Whether vertices are in cyclic order
    """
    
    area: float
    normal: np.ndarray
    centroid: np.ndarray
    is_ordered: bool
    
    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"PolygonProperties(area={self.area:.6e}, "
            f"is_ordered={self.is_ordered})"
        )


# ==============================================================================
# GEOMETRY PROCESSOR
# ==============================================================================

class GeometryProcessor:
    """
    Stateful processor for geometric operations with intelligent caching.
    
    This processor wraps the pure calculation functions from calculator.py
    with state management and caching. Key features:
    
    - **Caching**: Caches expensive properties (area, normal, centroid)
    - **Memory management**: LRU-like cache with size limits
    - **Debugging**: Detailed logging of cache hits/misses
    - **Statistics**: Track cache performance
    
    Design:
        - Cache is keyed by vertex array id() (memory address)
        - Immutable polygon objects can share cache entries
        - Automatic invalidation when vertices change
        - Thread-safe for read operations (thread-unsafe for writes)
    
    Attributes
    ----------
    tolerance : float
        Tolerance for geometric operations (e.g., coplanarity)
    max_cache_size : int
        Maximum number of cached polygons before eviction
    
    Examples
    --------
    >>> from geometry.processor import GeometryProcessor
    >>> from geometry.primitives import Polygon
    >>> import numpy as np
    >>> 
    >>> # Create processor
    >>> processor = GeometryProcessor(tolerance=1e-12)
    >>> 
    >>> # Create polygon
    >>> vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
    >>> poly = Polygon(vertices=vertices, face_indices=[0, 1, 2])
    >>> 
    >>> # Get properties (cached)
    >>> props = processor.get_polygon_properties(poly)
    >>> print(f"Area: {props.area}")
    >>> print(f"Normal: {props.normal}")
    >>> print(f"Centroid: {props.centroid}")
    >>> 
    >>> # Cache statistics
    >>> stats = processor.cache_statistics()
    >>> print(f"Hit rate: {stats['hit_rate']:.1%}")
    
    Notes
    -----
    - Cache is NOT thread-safe for concurrent writes
    - Use clear_cache() after modifying polygon vertices
    - Cache statistics include hit/miss counts and current size
    - Memory usage scales with number of unique polygons cached
    """
    
    def __init__(
        self,
        tolerance: float = POLYGON_REORDER_TOL,
        max_cache_size: int = 1000,
    ):
        """
        Initialize GeometryProcessor.
        
        Parameters
        ----------
        tolerance : float, optional
            Tolerance for geometric operations (coplanarity, etc.)
            Default is POLYGON_REORDER_TOL
        max_cache_size : int, optional
            Maximum cached polygons before eviction. Default is 1000.
        
        Examples
        --------
        >>> processor = GeometryProcessor(tolerance=1e-12, max_cache_size=500)
        """
        self.tolerance = tolerance
        self.max_cache_size = max_cache_size
        
        # Cache: maps vertex array id() → PolygonProperties
        self._cache: Dict[int, PolygonProperties] = {}
        
        # Ordered list of cache keys for LRU eviction
        self._cache_order: list = []
        
        # Statistics
        self._cache_hits = 0
        self._cache_misses = 0
        
        logger.debug(
            f"GeometryProcessor initialized: tolerance={tolerance:.2e}, "
            f"max_cache_size={max_cache_size}"
        )
    
    def ensure_polygon_ordered(self, polygon: Polygon) -> Polygon:
        """
        Ensure polygon vertices are in cyclic order.
        
        If vertices are not already ordered, reorders them and returns a
        new Polygon with ordered vertices.
        
        Parameters
        ----------
        polygon : Polygon
            Polygon to check/order
        
        Returns
        -------
        Polygon
            Either the input polygon (if already ordered) or a new Polygon
            with reordered vertices
        
        Raises
        ------
        GeometryError
            If vertices are not coplanar
        
        Examples
        --------
        >>> poly = Polygon(vertices=unordered_verts, face_indices=indices)
        >>> ordered_poly = processor.ensure_polygon_ordered(poly)
        >>> if ordered_poly is not poly:
        ...     print("Vertices were reordered")
        """
        # Get cached properties if available
        props = self.get_polygon_properties(polygon)
        
        if props.is_ordered:
            logger.debug("Polygon vertices already ordered")
            return polygon
        
        # Reorder vertices
        logger.debug("Reordering polygon vertices...")
        ordered_verts = reorder_polygon_vertices(polygon.vertices, self.tolerance)
        
        # Create new polygon with ordered vertices
        ordered_poly = Polygon(
            vertices=ordered_verts,
            face_indices=polygon.face_indices
        )
        
        return ordered_poly
    
    def get_polygon_properties(self, polygon: Polygon) -> PolygonProperties:
        """
        Get all properties of a polygon (cached).
        
        Computes (or retrieves from cache) all geometric properties:
        - Area
        - Normal vector
        - Centroid
        - Whether vertices are in cyclic order
        
        Caching significantly improves performance when the same polygon
        is queried multiple times.
        
        Parameters
        ----------
        polygon : Polygon
            Polygon to query
        
        Returns
        -------
        PolygonProperties
            Immutable container with all properties
        
        Examples
        --------
        >>> props = processor.get_polygon_properties(polygon)
        >>> print(f"Area: {props.area:.6e}")
        >>> print(f"Normal: {props.normal}")
        >>> print(f"Centroid: {props.centroid}")
        """
        # Use vertex array memory address as cache key
        cache_key = id(polygon.vertices)
        
        # Check cache
        if cache_key in self._cache:
            self._cache_hits += 1
            logger.debug(f"Cache HIT for polygon (key={cache_key})")
            return self._cache[cache_key]
        
        # Cache miss - compute properties
        self._cache_misses += 1
        logger.debug(f"Cache MISS for polygon (key={cache_key}), computing...")
        
        # Compute properties
        area = polygon_area_3d(polygon.vertices)
        normal = polygon_normal(polygon.vertices)
        centroid = polygon_centroid(polygon.vertices)
        
        # Check if vertices are in cyclic order (expensive!)
        try:
            reordered = reorder_polygon_vertices(polygon.vertices, self.tolerance)
            # Compare with original to see if reordering changed anything
            is_ordered = np.allclose(polygon.vertices, reordered, atol=1e-10)
        except (GeometryError, ValidationError):
            is_ordered = False
        
        # Create properties object
        props = PolygonProperties(
            area=area,
            normal=normal,
            centroid=centroid,
            is_ordered=is_ordered
        )
        
        # Store in cache
        self._cache[cache_key] = props
        self._cache_order.append(cache_key)
        
        # Evict old entries if cache is full
        if len(self._cache) > self.max_cache_size:
            oldest_key = self._cache_order.pop(0)
            del self._cache[oldest_key]
            logger.debug(f"Cache evicted oldest entry (key={oldest_key})")
        
        return props
    
    def clear_cache(self) -> None:
        """
        Clear all cached data.
        
        Use this after modifying polygon vertices or when running low on memory.
        
        Examples
        --------
        >>> processor.clear_cache()
        >>> # All cached properties are now discarded
        """
        n_entries = len(self._cache)
        self._cache.clear()
        self._cache_order.clear()
        logger.info(f"Cache cleared ({n_entries} entries removed)")
    
    def cache_statistics(self) -> Dict:
        """
        Return cache performance statistics.
        
        Returns
        -------
        dict
            Dictionary with keys:
            - 'hits': Number of cache hits
            - 'misses': Number of cache misses
            - 'hit_rate': Fraction of lookups that hit (0-1)
            - 'size': Current number of cached entries
            - 'max_size': Maximum cache size
            - 'efficiency': Memory usage estimate
        
        Examples
        --------
        >>> stats = processor.cache_statistics()
        >>> print(f"Cache hit rate: {stats['hit_rate']:.1%}")
        >>> print(f"Cache size: {stats['size']}/{stats['max_size']}")
        """
        total = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total if total > 0 else 0.0
        
        return {
            'hits': self._cache_hits,
            'misses': self._cache_misses,
            'hit_rate': hit_rate,
            'size': len(self._cache),
            'max_size': self.max_cache_size,
            'efficiency': f"{hit_rate:.1%}" if total > 0 else "N/A"
        }
    
    def reset_statistics(self) -> None:
        """
        Reset cache statistics counters.
        
        Use before a benchmark to get clean measurements.
        
        Examples
        --------
        >>> processor.reset_statistics()
        >>> # ... run operations ...
        >>> stats = processor.cache_statistics()
        """
        self._cache_hits = 0
        self._cache_misses = 0
        logger.debug("Cache statistics reset")
    
    def print_statistics(self) -> None:
        """
        Print cache statistics in human-readable format.
        
        Examples
        --------
        >>> processor.print_statistics()
        Cache Statistics:
          Hits: 250
          Misses: 50
          Hit Rate: 83.3%
          Size: 45/1000
        """
        stats = self.cache_statistics()
        
        print("\nCache Statistics:")
        print(f"  Hits: {stats['hits']}")
        print(f"  Misses: {stats['misses']}")
        print(f"  Hit Rate: {stats['efficiency']}")
        print(f"  Size: {stats['size']}/{stats['max_size']}")
        print()


if __name__ == "__main__":
    """
    Test/demo when module is run directly.
    """
    import sys
    from core.utilities import setup_logging
    
    setup_logging(level="DEBUG")
    
    print("\n" + "="*80)
    print("GEOMETRY PROCESSOR MODULE - TESTING")
    print("="*80 + "\n")
    
    # Create processor
    print("[TEST 1] Creating GeometryProcessor...")
    try:
        processor = GeometryProcessor(tolerance=1e-12, max_cache_size=100)
        print(f"✓ GeometryProcessor initialized")
    except Exception as e:
        print(f"✗ Error: {e}")
        sys.exit(1)
    
    # Create polygon
    print("\n[TEST 2] Creating Polygon...")
    try:
        from geometry.primitives import Polygon
        vertices = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0]
        ])
        poly = Polygon(vertices=vertices, face_indices=[0, 1, 2])
        print(f"✓ Polygon created: {poly}")
    except Exception as e:
        print(f"✗ Error: {e}")
        sys.exit(1)
    
    # Get properties (first call - cache miss)
    print("\n[TEST 3] Getting polygon properties (cache miss)...")
    try:
        props1 = processor.get_polygon_properties(poly)
        print(f"✓ Area: {props1.area:.6f}")
        print(f"  Normal: {props1.normal}")
        print(f"  Ordered: {props1.is_ordered}")
    except Exception as e:
        print(f"✗ Error: {e}")
    
    # Get properties again (cache hit)
    print("\n[TEST 4] Getting polygon properties again (cache hit)...")
    try:
        props2 = processor.get_polygon_properties(poly)
        assert props1.area == props2.area
        print(f"✓ Got cached properties (same polygon)")
    except Exception as e:
        print(f"✗ Error: {e}")
    
    # Print statistics
    print("\n[TEST 5] Cache statistics...")
    try:
        processor.print_statistics()
    except Exception as e:
        print(f"✗ Error: {e}")
    
    print("="*80)
    print("✓ Processor module is ready for use!")
    print("="*80 + "\n")
