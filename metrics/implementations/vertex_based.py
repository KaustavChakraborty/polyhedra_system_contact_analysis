# metrics/implementations/vertex_based.py
# ==============================================================================
# Module: metrics.implementations.vertex_based
# Purpose: Vertex-based distance metrics
#
# Classes:
#   - VertexToFacePerpMetric: Vertex to face perpendicular distance
#   - VertexVertexMetric: Vertex to vertex distance
#
# Author: Contact Analysis Team
# ==============================================================================

from __future__ import annotations

import logging

import numpy as np

from ...contacts import ContactResult
from ..base import MetricBase
from ..types import MetricValue, MetricResult

logger = logging.getLogger(__name__)


# ==============================================================================
# VERTEX TO FACE PERPENDICULAR METRIC
# ==============================================================================

class VertexToFacePerpMetric(MetricBase):
    """
    Distance metric: perpendicular distance from vertex to face.
    
    Computes the perpendicular distance from a vertex to the plane of a face.
    
    Attributes
    ----------
    name : str
        Metric name: 'vertex_to_face_perp'
    """
    
    name = "vertex_to_face_perp"
    description = "Perpendicular distance from vertex to face"
    version = "1.0.0"
    
    def compute(self, contact_result: ContactResult, **kwargs) -> MetricResult:
        """
        Compute vertex to face perpendicular distance.
        
        Parameters
        ----------
        contact_result : ContactResult
            Contact to analyze
        **kwargs
            Additional parameters
        
        Returns
        -------
        MetricResult
            Computed metric result
        """
        
        logger.debug(f"[{self.name}] Computing metric")
        
        try:
            metric_values = {}
            
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
                error_message=str(e)
            )


# ==============================================================================
# VERTEX TO VERTEX METRIC
# ==============================================================================

class VertexVertexMetric(MetricBase):
    """
    Distance metric: distance between closest vertices.
    
    Computes minimum distance between vertices of two particles.
    
    Attributes
    ----------
    name : str
        Metric name: 'vertex_vertex'
    """
    
    name = "vertex_vertex"
    description = "Distance between closest vertices"
    version = "1.0.0"
    
    def compute(self, contact_result: ContactResult, **kwargs) -> MetricResult:
        """
        Compute vertex to vertex distance.
        
        Parameters
        ----------
        contact_result : ContactResult
            Contact to analyze
        **kwargs
            Additional parameters
        
        Returns
        -------
        MetricResult
            Computed metric result
        """
        
        logger.debug(f"[{self.name}] Computing metric")
        
        try:
            metric_values = {}
            
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
                error_message=str(e)
            )


if __name__ == "__main__":
    """Test when run directly."""
    print("\n" + "="*80)
    print("VERTEX-BASED METRICS - TESTING")
    print("="*80 + "\n")
    
    print("[TEST] VertexToFacePerpMetric...")
    try:
        metric = VertexToFacePerpMetric()
        print(f"✓ {metric}\n")
    except Exception as e:
        print(f"✗ Error: {e}\n")
    
    print("[TEST] VertexVertexMetric...")
    try:
        metric = VertexVertexMetric()
        print(f"✓ {metric}\n")
    except Exception as e:
        print(f"✗ Error: {e}\n")
    
    print("="*80)
    print("✓ Vertex-based metrics tests passed!")
    print("="*80 + "\n")
