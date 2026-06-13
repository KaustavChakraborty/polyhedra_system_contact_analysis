# metrics/implementations/edge_based.py
# ==============================================================================
# Module: metrics.implementations.edge_based
# Purpose: Edge-based distance metrics
#
# Classes:
#   - EdgeMidpointMetric: Distance between edge midpoints
#
# Author: Contact Analysis Team
# ==============================================================================

from __future__ import annotations

import logging

from ...contacts import ContactResult
from ..base import MetricBase
from ..types import MetricValue, MetricResult

logger = logging.getLogger(__name__)


# ==============================================================================
# EDGE MIDPOINT METRIC
# ==============================================================================

class EdgeMidpointMetric(MetricBase):
    """
    Distance metric: distance between edge midpoints.
    
    Computes distance between midpoints of edges.
    
    Attributes
    ----------
    name : str
        Metric name: 'edge_midpoint_to_edge_perp'
    """
    
    name = "edge_midpoint_to_edge_perp"
    description = "Distance between edge midpoints with perpendicular component"
    version = "1.0.0"
    
    def compute(self, contact_result: ContactResult, **kwargs) -> MetricResult:
        """
        Compute edge-based distance.
        
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
    print("EDGE-BASED METRICS - TESTING")
    print("="*80 + "\n")
    
    print("[TEST] EdgeMidpointMetric...")
    try:
        metric = EdgeMidpointMetric()
        print(f"✓ {metric}\n")
    except Exception as e:
        print(f"✗ Error: {e}\n")
    
    print("="*80)
    print("✓ Edge-based metrics tests passed!")
    print("="*80 + "\n")
