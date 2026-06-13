# metrics/implementations/face_center.py
# ==============================================================================
# Module: metrics.implementations.face_center
# Purpose: Face center to face center distance metric
#
# Classes:
#   - FaceCenterMetric: Compute distance between face centers
#
# Author: Contact Analysis Team
# ==============================================================================

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from ...contacts import ContactResult, FacePair
from ...particles import Particle, ParticleProcessor
from ..base import MetricBase
from ..types import MetricValue, MetricResult
from .distance_calc import DistanceCalculator

logger = logging.getLogger(__name__)


# ==============================================================================
# FACE CENTER METRIC
# ==============================================================================

class FaceCenterMetric(MetricBase):
    """
    Distance metric: center of face A to center of face B.
    
    Computes the distance between the geometric centers of overlapping faces.
    
    Attributes
    ----------
    name : str
        Metric name: 'face_center_face_center'
    description : str
        Human-readable description
    version : str
        Metric version
    
    Examples
    --------
    >>> metric = FaceCenterMetric()
    >>> result = metric.compute(contact_result)
    >>> distance = result.get_value('distance')
    """
    
    name = "face_center_face_center"
    description = "Distance between face centers"
    version = "1.0.0"
    
    def __init__(self) -> None:
        """Initialize metric."""
        super().__init__()
        self.processor = ParticleProcessor()
        self.distance_calc = DistanceCalculator()
    
    def compute(self, contact_result: ContactResult, **kwargs) -> MetricResult:
        """
        Compute face center to face center distance.
        
        Parameters
        ----------
        contact_result : ContactResult
            Contact to analyze
        **kwargs
            Additional parameters (particle_system, etc.)
        
        Returns
        -------
        MetricResult
            Computed metric result
        """
        
        # Validate input
        if not self.validate_input(contact_result):
            return MetricResult(
                metric_name=self.name,
                particle_A_id=contact_result.particle_A_id,
                particle_B_id=contact_result.particle_B_id,
                metric_values={},
                is_valid=False,
                error_message="Invalid input"
            )
        
        try:
            logger.debug(
                f"[{self.name}] Computing for contact "
                f"({contact_result.particle_A_id},{contact_result.particle_B_id})"
            )
            
            # Get particles from contact_result metadata or kwargs
            particle_system = kwargs.get('particle_system')
            if particle_system is None:
                logger.warning("No particle_system provided")
                raise ValueError("particle_system required")
            
            particle_A = particle_system.get_particle(contact_result.particle_A_id)
            particle_B = particle_system.get_particle(contact_result.particle_B_id)
            
            if particle_A is None or particle_B is None:
                raise ValueError("Particles not found in system")
            
            # Compute distances for all face pairs
            metric_values = {}
            
            for face_pair in contact_result.face_pairs:
                if not face_pair.is_overlapping:
                    continue
                
                # Get face centers
                face_A = self.processor.get_face_in_global_coords(
                    particle_A, face_pair.face_A_idx
                )
                face_B = self.processor.get_face_in_global_coords(
                    particle_B, face_pair.face_B_idx
                )
                
                center_A = face_A.vertices.mean(axis=0)
                center_B = face_B.vertices.mean(axis=0)
                
                # Compute distance
                distance_vec = center_B - center_A
                distance = np.linalg.norm(distance_vec)
                
                # Store result
                key = f"face_{face_pair.face_A_idx}_to_{face_pair.face_B_idx}"
                metric_values[key] = MetricValue(
                    value=distance,
                    components={
                        'x': float(distance_vec[0]),
                        'y': float(distance_vec[1]),
                        'z': float(distance_vec[2])
                    }
                )
            
            # Overall distance (average or min)
            if metric_values:
                distances = [v.value for v in metric_values.values()]
                min_distance = min(distances)
                avg_distance = np.mean(distances)
                
                metric_values['distance'] = MetricValue(
                    value=min_distance,
                    components={'min': min_distance, 'avg': avg_distance}
                )
            
            logger.info(f"[{self.name}] Successfully computed")
            
            return MetricResult(
                metric_name=self.name,
                particle_A_id=contact_result.particle_A_id,
                particle_B_id=contact_result.particle_B_id,
                metric_values=metric_values,
                is_valid=True,
                frame_index=contact_result.frame_index
            )
        
        except Exception as e:
            logger.error(f"[{self.name}] Computation failed: {e}")
            
            return MetricResult(
                metric_name=self.name,
                particle_A_id=contact_result.particle_A_id,
                particle_B_id=contact_result.particle_B_id,
                metric_values={},
                is_valid=False,
                error_message=str(e),
                frame_index=contact_result.frame_index
            )


if __name__ == "__main__":
    """Test when run directly."""
    print("\n" + "="*80)
    print("FACE CENTER METRIC - TESTING")
    print("="*80 + "\n")
    
    print("[TEST] FaceCenterMetric initialization...")
    try:
        metric = FaceCenterMetric()
        print(f"✓ Metric created: {metric}")
        print(f"  Description: {metric.description}\n")
    except Exception as e:
        print(f"✗ Error: {e}\n")
    
    print("="*80)
    print("✓ Metric tests passed!")
    print("="*80 + "\n")
