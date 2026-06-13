# contacts/processor.py
# ==============================================================================
# Module: contacts.processor
# Purpose: Stateful contact processing pipeline
#
# Classes:
#   - ContactProcessor: Orchestrate contact analysis workflow
#
# Author: Contact Analysis Team
# ==============================================================================

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any

import numpy as np

from .. import ValidationError
from ..particles import ParticleSystem
from .types import FacePair, ContactResult
from .detector import OverlapDetector
from .classifier import FaceClassifier
from .properties import ContactPropertyCalculator

logger = logging.getLogger(__name__)


# ==============================================================================
# CONTACT PROCESSOR: Stateful contact processing
# ==============================================================================

@dataclass
class ContactProcessor:
    """
    Stateful processor for contact analysis.
    
    Orchestrates the complete contact analysis workflow: detection,
    classification, property computation, and result caching.
    
    Attributes
    ----------
    tolerance : float
        Numerical tolerance for geometric operations
    detector : OverlapDetector
        For overlap detection
    property_calculator : ContactPropertyCalculator
        For property computation
    _contact_cache : Dict
        Cache of computed contacts
    _classifier_cache : Dict
        Cache of face classifications
    
    Examples
    --------
    >>> processor = ContactProcessor()
    >>> results = processor.analyze_particle_pair(system, p_A_id, p_B_id)
    >>> print(f"Found {results.overlap_count} overlapping faces")
    """
    
    tolerance: float = 1e-12
    detector: OverlapDetector = None
    property_calculator: ContactPropertyCalculator = None
    _contact_cache: Dict = field(default_factory=dict)
    _classifier_cache: Dict = field(default_factory=dict)
    
    def __post_init__(self) -> None:
        """Initialize processor."""
        if self.detector is None:
            self.detector = OverlapDetector(tolerance=self.tolerance)
        
        if self.property_calculator is None:
            self.property_calculator = ContactPropertyCalculator(tolerance=self.tolerance)
        
        logger.debug(f"[ContactProcessor] Initialized with tolerance={self.tolerance}")
    
    def analyze_particle_pair(
        self,
        system: ParticleSystem,
        particle_A_id: int,
        particle_B_id: int,
        frame_index: int = 0
    ) -> ContactResult:
        """
        Analyze contact between two particles.
        
        Parameters
        ----------
        system : ParticleSystem
            Particle system containing both particles
        particle_A_id : int
            ID of first particle
        particle_B_id : int
            ID of second particle
        frame_index : int, optional
            Frame number (default: 0)
        
        Returns
        -------
        ContactResult
            Complete contact analysis
        
        Raises
        ------
        ValidationError
            If particles not found in system
        
        Examples
        --------
        >>> processor = ContactProcessor()
        >>> result = processor.analyze_particle_pair(system, 0, 1)
        >>> print(f"Contact: {result.has_contact}")
        """
        
        # Get particles
        particle_A = system.get_particle(particle_A_id)
        particle_B = system.get_particle(particle_B_id)
        
        if particle_A is None or particle_B is None:
            raise ValidationError(
                "One or both particles not found in system",
                error_code="PROCESSOR_PARTICLE_NOT_FOUND"
            )
        
        logger.debug(
            f"[ContactProcessor] Analyzing particles {particle_A_id} and {particle_B_id}"
        )
        
        # Detect overlaps
        face_pairs = self.detector.detect_overlap(particle_A, particle_B)
        
        # Classify faces
        classifications_A = self._get_face_classifications(particle_A.shape)
        classifications_B = self._get_face_classifications(particle_B.shape)
        
        # Add classifications to face pairs
        for fp in face_pairs:
            fp.face_A_type = classifications_A.get(fp.face_A_idx, 'unknown')
            fp.face_B_type = classifications_B.get(fp.face_B_idx, 'unknown')
        
        # Count overlaps
        overlap_count = sum(1 for fp in face_pairs if fp.is_overlapping)
        
        # Create result
        result = ContactResult(
            particle_A_id=particle_A_id,
            particle_B_id=particle_B_id,
            face_pairs=face_pairs,
            distances={},  # To be filled by metric processors
            contact_order={},  # To be filled by metric processors
            overlap_count=overlap_count,
            frame_index=frame_index
        )
        
        logger.info(
            f"[ContactProcessor] Analysis complete: {overlap_count} overlapping pairs"
        )
        
        return result
    
    def analyze_system(
        self,
        system: ParticleSystem,
        neighbor_list: Dict[int, List[int]] = None,
        frame_index: int = 0
    ) -> List[ContactResult]:
        """
        Analyze all particle pairs in system.
        
        Parameters
        ----------
        system : ParticleSystem
            Particle system to analyze
        neighbor_list : Dict[int, List[int]], optional
            Pre-computed neighbor list (default: analyze all pairs)
        frame_index : int, optional
            Frame number (default: 0)
        
        Returns
        -------
        List[ContactResult]
            Contact results for all analyzed pairs
        
        Examples
        --------
        >>> processor = ContactProcessor()
        >>> results = processor.analyze_system(system, neighbor_list)
        >>> print(f"Analyzed {len(results)} particle pairs")
        """
        
        logger.info(
            f"[ContactProcessor] Analyzing system with {system.num_particles} particles"
        )
        
        results = []
        
        if neighbor_list is None:
            # Analyze all pairs
            for i in range(system.num_particles):
                for j in range(i + 1, system.num_particles):
                    p_A = system.particles[i]
                    p_B = system.particles[j]
                    
                    try:
                        result = self.analyze_particle_pair(
                            system,
                            p_A.particle_id,
                            p_B.particle_id,
                            frame_index
                        )
                        results.append(result)
                    except Exception as e:
                        logger.warning(f"Failed to analyze pair ({i},{j}): {e}")
                        continue
        else:
            # Analyze using neighbor list
            processed = set()
            
            for p_A_id, neighbors in neighbor_list.items():
                for p_B_id in neighbors:
                    # Avoid duplicate analysis
                    pair_key = tuple(sorted([p_A_id, p_B_id]))
                    if pair_key in processed:
                        continue
                    
                    processed.add(pair_key)
                    
                    try:
                        result = self.analyze_particle_pair(
                            system,
                            p_A_id,
                            p_B_id,
                            frame_index
                        )
                        results.append(result)
                    except Exception as e:
                        logger.warning(f"Failed to analyze pair ({p_A_id},{p_B_id}): {e}")
                        continue
        
        logger.info(f"[ContactProcessor] Analyzed {len(results)} particle pairs")
        
        return results
    
    def _get_face_classifications(self, shape) -> Dict[int, str]:
        """
        Get face classifications (with caching).
        
        Parameters
        ----------
        shape : ConvexShape
            Shape to classify
        
        Returns
        -------
        Dict[int, str]
            Face classifications
        """
        
        shape_id = id(shape)
        
        if shape_id in self._classifier_cache:
            return self._classifier_cache[shape_id]
        
        # Classify
        classifier = FaceClassifier(shape)
        classifications = classifier.classify_all_faces()
        
        # Cache
        self._classifier_cache[shape_id] = classifications
        
        return classifications
    
    def compute_properties(
        self,
        contact_result: ContactResult
    ) -> Dict[int, Dict[str, float]]:
        """
        Compute properties for all face pairs in contact result.
        
        Parameters
        ----------
        contact_result : ContactResult
            Contact result to analyze
        
        Returns
        -------
        Dict[int, Dict[str, float]]
            Properties for each face pair
        
        Examples
        --------
        >>> props = processor.compute_properties(result)
        """
        
        properties = {}
        
        for i, face_pair in enumerate(contact_result.face_pairs):
            props = self.property_calculator.compute_properties(face_pair)
            properties[i] = props
        
        return properties
    
    def clear_cache(self) -> None:
        """Clear all caches."""
        self._contact_cache.clear()
        self._classifier_cache.clear()
        logger.debug("[ContactProcessor] Caches cleared")
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get processor statistics.
        
        Returns
        -------
        dict
            Statistics about caching and processing
        """
        
        return {
            'tolerance': self.tolerance,
            'contact_cache_size': len(self._contact_cache),
            'classifier_cache_size': len(self._classifier_cache),
        }


if __name__ == "__main__":
    """Test when run directly."""
    print("\n" + "="*80)
    print("CONTACTS PROCESSOR MODULE - TESTING")
    print("="*80 + "\n")
    
    print("[TEST] ContactProcessor initialization...")
    try:
        processor = ContactProcessor(tolerance=1e-12)
        print(f"✓ Processor created")
        print(f"  Statistics: {processor.get_statistics()}\n")
    except Exception as e:
        print(f"✗ Error: {e}\n")
        import traceback
        traceback.print_exc()
    
    print("="*80)
    print("✓ Processor module tests passed!")
    print("="*80 + "\n")
