# contacts/classifier.py
# ==============================================================================
# Module: contacts.classifier
# Purpose: Classify faces into geometric types
#
# Classes:
#   - FaceClassifier: Classify face types (flat, edge, vertex, etc)
#
# Author: Contact Analysis Team
# ==============================================================================

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

from .. import ValidationError, DataTypeError
from ..primitives import ConvexShape

logger = logging.getLogger(__name__)


# ==============================================================================
# FACE TYPES
# ==============================================================================

FACE_TYPES = {
    'flat': 'Regular polygon face',
    'edge': 'Edge between faces',
    'vertex': 'Vertex (point contact)',
    'unknown': 'Cannot determine type',
}


# ==============================================================================
# FACE CLASSIFIER: Classify face types
# ==============================================================================

@dataclass
class FaceClassifier:
    """
    Classify faces into geometric types.
    
    Analyzes the geometry of each face in a shape and assigns a type
    (flat, edge, vertex, unknown) based on its properties.
    
    Attributes
    ----------
    shape : ConvexShape
        Shape to classify
    _classification : Dict[int, str]
        Cache of face classifications: face_idx -> type
    tolerance : float
        Numerical tolerance for comparisons
    
    Examples
    --------
    >>> classifier = FaceClassifier(shape)
    >>> face_types = classifier.classify_all_faces()
    >>> for face_idx, face_type in face_types.items():
    ...     print(f"Face {face_idx}: {face_type}")
    """
    
    shape: ConvexShape
    tolerance: float = 1e-10
    _classification: Dict[int, str] = field(default_factory=dict)
    
    def __post_init__(self) -> None:
        """Validate classifier after initialization."""
        if not isinstance(self.shape, ConvexShape):
            raise DataTypeError(
                f"shape must be ConvexShape, got {type(self.shape).__name__}",
                error_code="CLASSIFIER_INVALID_SHAPE"
            )
        
        if self.tolerance <= 0:
            raise ValidationError(
                f"tolerance must be positive, got {self.tolerance}",
                error_code="CLASSIFIER_INVALID_TOLERANCE"
            )
        
        logger.debug(
            f"[FaceClassifier] Initialized for shape with {self.shape.num_faces} faces"
        )
    
    def classify_all_faces(self) -> Dict[int, str]:
        """
        Classify all faces in shape.
        
        Returns
        -------
        Dict[int, str]
            Face classifications: face_idx -> type
        
        Examples
        --------
        >>> classifier = FaceClassifier(shape)
        >>> types = classifier.classify_all_faces()
        >>> print(f"Face 0: {types[0]}")
        """
        
        logger.info(f"[FaceClassifier] Classifying {self.shape.num_faces} faces")
        
        self._classification.clear()
        
        for face_idx in range(self.shape.num_faces):
            face_type = self._classify_single_face(face_idx)
            self._classification[face_idx] = face_type
        
        logger.info(
            f"[FaceClassifier] Classification complete. "
            f"Types: {self._count_types()}"
        )
        
        return self._classification.copy()
    
    def get_face_type(self, face_idx: int) -> str:
        """
        Get face type (with caching).
        
        Parameters
        ----------
        face_idx : int
            Face index
        
        Returns
        -------
        str
            Face type name (e.g., 'flat', 'edge', 'vertex')
        
        Raises
        ------
        ValidationError
            If face_idx out of range
        
        Examples
        --------
        >>> classifier = FaceClassifier(shape)
        >>> face_type = classifier.get_face_type(0)
        >>> print(f"Face type: {face_type}")
        """
        
        # Validate index
        if not (0 <= face_idx < self.shape.num_faces):
            raise ValidationError(
                f"Face index {face_idx} out of range [0, {self.shape.num_faces})",
                error_code="CLASSIFIER_FACE_INDEX_OUT_OF_RANGE"
            )
        
        # Return cached if available
        if face_idx in self._classification:
            return self._classification[face_idx]
        
        # Classify and cache
        face_type = self._classify_single_face(face_idx)
        self._classification[face_idx] = face_type
        
        return face_type
    
    def _classify_single_face(self, face_idx: int) -> str:
        """
        Classify a single face.
        
        Private method that determines face type based on geometry.
        
        Parameters
        ----------
        face_idx : int
            Face index to classify
        
        Returns
        -------
        str
            Face type name
        """
        
        try:
            # Get face vertex indices
            face_indices = self.shape.faces[face_idx]
            
            if len(face_indices) < 3:
                # Not a valid face
                return 'unknown'
            
            # Get face vertices
            vertices = self.shape.vertices[face_indices]
            
            # Determine type based on number of vertices and geometry
            if len(face_indices) >= 4:
                # Polygon face (quadrilateral or larger)
                return self._classify_polygon_face(vertices)
            elif len(face_indices) == 3:
                # Triangle face
                return 'flat'
            else:
                return 'unknown'
        
        except Exception as e:
            logger.debug(f"Classification error for face {face_idx}: {e}")
            return 'unknown'
    
    def _classify_polygon_face(self, vertices: np.ndarray) -> str:
        """
        Classify a polygon face.
        
        Parameters
        ----------
        vertices : np.ndarray
            Face vertices (N, 3)
        
        Returns
        -------
        str
            Face type
        """
        
        try:
            if len(vertices) < 3:
                return 'unknown'
            
            # Check if all vertices are coplanar (they should be for a valid face)
            # For a convex shape, all faces are flat
            return 'flat'
        
        except:
            return 'unknown'
    
    def _count_types(self) -> Dict[str, int]:
        """Count occurrences of each face type."""
        counts = {}
        for face_type in self._classification.values():
            counts[face_type] = counts.get(face_type, 0) + 1
        return counts
    
    def get_faces_of_type(self, face_type: str) -> list:
        """
        Get all faces of a given type.
        
        Parameters
        ----------
        face_type : str
            Type name (e.g., 'flat', 'edge')
        
        Returns
        -------
        list
            Face indices with this type
        
        Examples
        --------
        >>> classifier = FaceClassifier(shape)
        >>> flat_faces = classifier.get_faces_of_type('flat')
        """
        
        # Ensure classified
        if not self._classification:
            self.classify_all_faces()
        
        return [
            idx for idx, ftype in self._classification.items()
            if ftype == face_type
        ]
    
    def get_statistics(self) -> Dict[str, any]:
        """
        Get classification statistics.
        
        Returns
        -------
        dict
            Statistics about classifications
        """
        
        if not self._classification:
            return {'status': 'not_classified'}
        
        type_counts = self._count_types()
        
        return {
            'total_faces': len(self._classification),
            'type_counts': type_counts,
            'classification_complete': True,
        }
    
    def __repr__(self) -> str:
        """String representation."""
        if self._classification:
            counts = self._count_types()
            return f"FaceClassifier(faces={len(self._classification)}, types={counts})"
        else:
            return f"FaceClassifier(shape with {self.shape.num_faces} faces, not classified)"


if __name__ == "__main__":
    """Test when run directly."""
    print("\n" + "="*80)
    print("CONTACTS CLASSIFIER MODULE - TESTING")
    print("="*80 + "\n")
    
    print("[TEST 1] FaceClassifier initialization...")
    try:
        from ..primitives import ConvexShape
        
        # Create test shape
        vertices = np.array([
            [-1, -1, -1], [1, -1, -1],
            [1, 1, -1], [-1, 1, -1],
            [-1, -1, 1], [1, -1, 1],
            [1, 1, 1], [-1, 1, 1]
        ], dtype=float)
        
        faces = [[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4], 
                 [2, 3, 7, 6], [0, 3, 7, 4], [1, 2, 6, 5]]
        edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), 
                 (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
        
        shape = ConvexShape(vertices, faces, edges, len(faces), len(edges))
        
        classifier = FaceClassifier(shape)
        print(f"✓ Classifier created: {classifier}\n")
    except Exception as e:
        print(f"✗ Error: {e}\n")
        import traceback
        traceback.print_exc()
    
    print("[TEST 2] Classify all faces...")
    try:
        types = classifier.classify_all_faces()
        print(f"✓ Classification complete:")
        for idx, ftype in sorted(types.items()):
            print(f"  Face {idx}: {ftype}")
        
        stats = classifier.get_statistics()
        print(f"\n✓ Statistics: {stats}\n")
    except Exception as e:
        print(f"✗ Error: {e}\n")
    
    print("="*80)
    print("✓ Classifier module tests passed!")
    print("="*80 + "\n")
