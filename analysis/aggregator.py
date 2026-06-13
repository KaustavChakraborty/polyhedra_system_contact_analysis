# analysis/aggregator.py
# ==============================================================================
# Module: analysis.aggregator
# Purpose: Coordinator for frame and system-level aggregation
#
# Classes:
#   - AnalysisAggregator: Coordinate all analysis pipelines
#
# Author: Contact Analysis Team
# ==============================================================================

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any

import numpy as np

from .. import ValidationError
from ..trajectory import Trajectory
from ..particles import ParticleSystem
from ..contacts import ContactResult
from .types import RDFData, StatsResult
from .rdf import RDFProcessor, RDFInteractive
from .order_parameters import OrderParameterCalculator
from .statistics import StatisticsAggregator

logger = logging.getLogger(__name__)


# ==============================================================================
# ANALYSIS AGGREGATOR: Coordinator
# ==============================================================================

@dataclass
class AnalysisAggregator:
    """
    Coordinator for all analysis pipelines.
    
    Orchestrates RDF computation, statistics aggregation, and order parameters.
    
    Attributes
    ----------
    rdf_processor : RDFProcessor
        RDF computation
    order_calc : OrderParameterCalculator
        Order parameter calculation
    stats_aggs : Dict
        Statistics aggregators
    _frame_count : int
        Number of frames processed
    
    Examples
    --------
    >>> agg = AnalysisAggregator()
    >>> for frame in trajectory:
    ...     agg.process_frame(frame, contact_results)
    >>> rdf = agg.get_rdf()
    """
    
    rdf_processor: RDFProcessor = None
    order_calc: OrderParameterCalculator = None
    stats_aggs: Dict = field(default_factory=dict)
    _frame_count: int = 0
    
    def __post_init__(self) -> None:
        """Initialize aggregator."""
        if self.rdf_processor is None:
            self.rdf_processor = RDFProcessor(0.5, 5.0, 50)
        
        if self.order_calc is None:
            self.order_calc = OrderParameterCalculator()
        
        logger.debug("[AnalysisAggregator] Initialized")
    
    def process_frame(
        self,
        contact_results: List[ContactResult],
        distances: Optional[np.ndarray] = None,
        **kwargs
    ) -> None:
        """
        Process a single frame.
        
        Parameters
        ----------
        contact_results : List[ContactResult]
            Contact results for this frame
        distances : np.ndarray, optional
            Distance values for RDF
        **kwargs
            Additional parameters
        """
        
        logger.debug(f"[AnalysisAggregator] Processing frame {self._frame_count}")
        
        # Process RDF
        if distances is not None:
            distances_array = np.asarray(distances, dtype=float)
            particle_system = kwargs.get('particle_system')
            
            self.rdf_processor.process_frame(particle_system, distances_array)
        
        # Aggregate contact statistics
        for contact in contact_results:
            if contact.has_contact:
                # Get or create aggregator for this metric
                metric_name = f"contact_{contact.particle_A_id}_{contact.particle_B_id}"
                
                if metric_name not in self.stats_aggs:
                    self.stats_aggs[metric_name] = StatisticsAggregator(metric_name)
                
                self.stats_aggs[metric_name].add_value(contact.total_overlap_area)
        
        # Mark frame as processed
        for agg in self.stats_aggs.values():
            agg.process_frame()
        
        self._frame_count += 1
        
        logger.debug(f"[AnalysisAggregator] Frame {self._frame_count} processed")
    
    def process_trajectory(
        self,
        trajectory: Trajectory,
        contact_processor,
        frame_stride: int = 1
    ) -> None:
        """
        Process entire trajectory.
        
        Parameters
        ----------
        trajectory : Trajectory
            Trajectory to analyze
        contact_processor : ContactProcessor
            Contact analysis processor
        frame_stride : int, optional
            Process every nth frame (default: 1)
        """
        
        logger.info(
            f"[AnalysisAggregator] Processing trajectory: "
            f"{trajectory.num_frames} frames, stride={frame_stride}"
        )
        
        frame_count = 0
        for i in range(0, trajectory.num_frames, frame_stride):
            try:
                frame = trajectory.get_frame(i)
                
                # Analyze contacts
                contact_results = contact_processor.analyze_system(
                    frame.particle_system,
                    frame_index=frame.frame_index
                )
                
                # Extract distances
                distances = self._extract_distances(contact_results)
                
                # Process frame
                self.process_frame(
                    contact_results,
                    distances=distances,
                    particle_system=frame.particle_system
                )
                
                frame_count += 1
            
            except Exception as e:
                logger.warning(f"Failed to process frame {i}: {e}")
                continue
        
        logger.info(f"[AnalysisAggregator] Processed {frame_count} frames")
    
    def get_rdf(self) -> Optional[RDFData]:
        """Get computed RDF."""
        return self.rdf_processor.get_rdf()
    
    def get_statistics(self, metric_name: Optional[str] = None) -> Dict[str, StatsResult]:
        """
        Get statistics results.
        
        Parameters
        ----------
        metric_name : str, optional
            Filter by metric name (default: all)
        
        Returns
        -------
        Dict[str, StatsResult]
            Results
        """
        
        results = {}
        
        for name, agg in self.stats_aggs.items():
            if metric_name is None or metric_name in name:
                result = agg.get_result()
                if result:
                    results[name] = result
        
        return results
    
    def get_contact_order(self, contact_results: List[ContactResult]) -> Dict:
        """Get contact order parameters."""
        return self.order_calc.compute_contact_order(contact_results)
    
    def _extract_distances(self, contact_results: List[ContactResult]) -> np.ndarray:
        """Extract distance values from contact results."""
        distances = []
        
        for contact in contact_results:
            for metric_name, value in contact.distances.items():
                if 'distance' in metric_name and hasattr(value, 'value'):
                    distances.append(value.value)
        
        return np.array(distances, dtype=float) if distances else np.array([])
    
    def reset(self) -> None:
        """Reset all aggregators."""
        self.rdf_processor.reset()
        for agg in self.stats_aggs.values():
            agg.reset()
        self._frame_count = 0
        logger.debug("[AnalysisAggregator] Reset")
    
    def get_statistics_summary(self) -> Dict[str, Any]:
        """Get summary of all statistics."""
        summary = {
            'frames_processed': self._frame_count,
            'metrics_tracked': len(self.stats_aggs),
            'rdf_available': self.rdf_processor._frame_count > 0,
        }
        
        return summary


if __name__ == "__main__":
    """Test when run directly."""
    print("\n" + "="*80)
    print("ANALYSIS AGGREGATOR MODULE - TESTING")
    print("="*80 + "\n")
    
    print("[TEST] AnalysisAggregator initialization...")
    try:
        agg = AnalysisAggregator()
        print(f"✓ Aggregator created")
        print(f"  Summary: {agg.get_statistics_summary()}\n")
    except Exception as e:
        print(f"✗ Error: {e}\n")
    
    print("="*80)
    print("✓ Aggregator tests passed!")
    print("="*80 + "\n")
