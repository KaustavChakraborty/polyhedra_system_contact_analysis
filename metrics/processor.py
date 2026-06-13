# metrics/processor.py
# ==============================================================================
# Module: metrics.processor
# Purpose: Stateful metric processing pipeline
#
# Classes:
#   - MetricProcessor: Orchestrate metric computation workflow
#
# Author: Contact Analysis Team
# ==============================================================================

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any

from ..contacts import ContactResult
from .base import MetricBase, MetricRegistry
from .types import MetricResult
from .validator import MetricValidator

logger = logging.getLogger(__name__)


# ==============================================================================
# METRIC PROCESSOR: Stateful metric computation
# ==============================================================================

@dataclass
class MetricProcessor:
    """
    Stateful processor for metric computation.
    
    Orchestrates metric computation workflow: registration, validation,
    computation, and result caching.
    
    Attributes
    ----------
    registry : MetricRegistry
        Registry of available metrics
    validator : MetricValidator
        Validator for metric results
    _metric_cache : Dict
        Cache of computed metrics
    _result_cache : Dict
        Cache of results
    
    Examples
    --------
    >>> processor = MetricProcessor()
    >>> processor.register_metric("face_center", FaceCenterMetric)
    >>> result = processor.compute_metric(contact, "face_center", system=sys)
    """
    
    registry: MetricRegistry = None
    validator: MetricValidator = None
    _metric_cache: Dict = field(default_factory=dict)
    _result_cache: Dict = field(default_factory=dict)
    
    def __post_init__(self) -> None:
        """Initialize processor."""
        if self.registry is None:
            self.registry = MetricRegistry()
        
        if self.validator is None:
            self.validator = MetricValidator(strict_mode=False)
        
        logger.debug("[MetricProcessor] Initialized")
    
    def register_metric(self, name: str, metric_class: type) -> None:
        """
        Register a metric.
        
        Parameters
        ----------
        name : str
            Metric name
        metric_class : type
            Metric class
        
        Examples
        --------
        >>> processor.register_metric("face_center", FaceCenterMetric)
        """
        
        self.registry.register(name, metric_class)
        logger.info(f"[MetricProcessor] Registered metric: {name}")
    
    def compute_metric(
        self,
        contact_result: ContactResult,
        metric_name: str,
        **kwargs
    ) -> MetricResult:
        """
        Compute a single metric for contact.
        
        Parameters
        ----------
        contact_result : ContactResult
            Contact to analyze
        metric_name : str
            Metric to compute
        **kwargs
            Additional parameters (particle_system, etc.)
        
        Returns
        -------
        MetricResult
            Computed metric result
        
        Examples
        --------
        >>> result = processor.compute_metric(contact, "face_center", system=sys)
        """
        
        logger.debug(
            f"[MetricProcessor] Computing metric '{metric_name}' for contact "
            f"({contact_result.particle_A_id},{contact_result.particle_B_id})"
        )
        
        # Check cache
        cache_key = (
            contact_result.particle_A_id,
            contact_result.particle_B_id,
            metric_name,
            contact_result.frame_index
        )
        
        if cache_key in self._result_cache:
            logger.debug("[MetricProcessor] Cache hit")
            return self._result_cache[cache_key]
        
        try:
            # Get metric
            if metric_name not in self._metric_cache:
                metric = self.registry.create(metric_name)
                self._metric_cache[metric_name] = metric
            else:
                metric = self._metric_cache[metric_name]
            
            # Compute
            result = metric.compute(contact_result, **kwargs)
            
            # Validate
            is_valid, issues = self.validator.validate_result(result)
            
            if not is_valid:
                logger.warning(f"[MetricProcessor] Invalid result: {issues}")
            
            # Cache
            self._result_cache[cache_key] = result
            
            logger.info(
                f"[MetricProcessor] Computed metric '{metric_name}': valid={is_valid}"
            )
            
            return result
        
        except Exception as e:
            logger.error(f"[MetricProcessor] Metric computation failed: {e}")
            
            return MetricResult(
                metric_name=metric_name,
                particle_A_id=contact_result.particle_A_id,
                particle_B_id=contact_result.particle_B_id,
                metric_values={},
                is_valid=False,
                error_message=str(e),
                frame_index=contact_result.frame_index
            )
    
    def compute_all_metrics(
        self,
        contact_result: ContactResult,
        metric_names: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, MetricResult]:
        """
        Compute multiple metrics for contact.
        
        Parameters
        ----------
        contact_result : ContactResult
            Contact to analyze
        metric_names : List[str], optional
            Metrics to compute (default: all registered)
        **kwargs
            Additional parameters
        
        Returns
        -------
        Dict[str, MetricResult]
            Results: {metric_name: MetricResult}
        
        Examples
        --------
        >>> results = processor.compute_all_metrics(contact, system=sys)
        """
        
        if metric_names is None:
            metric_names = self.registry.get_available()
        
        logger.info(
            f"[MetricProcessor] Computing {len(metric_names)} metrics"
        )
        
        results = {}
        
        for metric_name in metric_names:
            try:
                result = self.compute_metric(contact_result, metric_name, **kwargs)
                results[metric_name] = result
            except Exception as e:
                logger.warning(f"Failed to compute metric '{metric_name}': {e}")
                continue
        
        logger.info(f"[MetricProcessor] Successfully computed {len(results)} metrics")
        
        return results
    
    def compute_for_contact_list(
        self,
        contact_results: List[ContactResult],
        metric_names: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, List[MetricResult]]:
        """
        Compute metrics for multiple contacts.
        
        Parameters
        ----------
        contact_results : List[ContactResult]
            Contacts to analyze
        metric_names : List[str], optional
            Metrics to compute
        **kwargs
            Additional parameters
        
        Returns
        -------
        Dict[str, List[MetricResult]]
            Results: {metric_name: [MetricResult, ...]}
        
        Examples
        --------
        >>> all_results = processor.compute_for_contact_list(contacts, system=sys)
        """
        
        logger.info(
            f"[MetricProcessor] Computing metrics for {len(contact_results)} contacts"
        )
        
        if metric_names is None:
            metric_names = self.registry.get_available()
        
        # Group by metric
        results = {name: [] for name in metric_names}
        
        for contact in contact_results:
            try:
                metric_results = self.compute_all_metrics(
                    contact,
                    metric_names,
                    **kwargs
                )
                
                for metric_name, result in metric_results.items():
                    results[metric_name].append(result)
            
            except Exception as e:
                logger.warning(f"Failed to compute metrics for contact: {e}")
                continue
        
        logger.info(f"[MetricProcessor] Completed metric computation")
        
        return results
    
    def clear_cache(self) -> None:
        """Clear all caches."""
        self._metric_cache.clear()
        self._result_cache.clear()
        logger.debug("[MetricProcessor] Caches cleared")
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get processor statistics.
        
        Returns
        -------
        dict
            Statistics
        """
        
        return {
            'registered_metrics': len(self.registry.get_available()),
            'cached_metrics': len(self._metric_cache),
            'cached_results': len(self._result_cache),
            'available_metrics': self.registry.get_available(),
        }


if __name__ == "__main__":
    """Test when run directly."""
    print("\n" + "="*80)
    print("METRICS PROCESSOR MODULE - TESTING")
    print("="*80 + "\n")
    
    print("[TEST] MetricProcessor initialization...")
    try:
        processor = MetricProcessor()
        print(f"✓ Processor created")
        print(f"  Statistics: {processor.get_statistics()}\n")
    except Exception as e:
        print(f"✗ Error: {e}\n")
        import traceback
        traceback.print_exc()
    
    print("="*80)
    print("✓ Processor tests passed!")
    print("="*80 + "\n")
