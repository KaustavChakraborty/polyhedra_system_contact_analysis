# metrics/base.py
# ==============================================================================
# Module: metrics.base
# Purpose: Base class and registry for distance metrics
#
# Classes:
#   - MetricBase: Abstract base class for all metrics
#   - MetricRegistry: Registry of available metrics
#
# Author: Contact Analysis Team
# ==============================================================================

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Type, Optional, Any

from .. import ValidationError
from ..particles import Particle
from ..contacts import ContactResult
from .types import MetricValue, MetricResult

logger = logging.getLogger(__name__)


# ==============================================================================
# METRIC BASE CLASS
# ==============================================================================

class MetricBase(ABC):
    """
    Abstract base class for distance metrics.
    
    Defines interface that all metric implementations must follow.
    
    Attributes
    ----------
    name : str
        Metric name (e.g., 'face_center_face_center')
    description : str
        Human-readable description
    version : str
        Metric version
    
    Examples
    --------
    >>> class CustomMetric(MetricBase):
    ...     name = "custom_metric"
    ...     def compute(self, ...):
    ...         return MetricResult(...)
    """
    
    name: str = "base_metric"
    description: str = "Base metric class"
    version: str = "1.0.0"
    
    def __init__(self) -> None:
        """Initialize metric."""
        logger.debug(f"[{self.name}] Initialized")
    
    @abstractmethod
    def compute(self, contact_result: ContactResult) -> MetricResult:
        """
        Compute metric for contact.
        
        Parameters
        ----------
        contact_result : ContactResult
            Contact to analyze
        
        Returns
        -------
        MetricResult
            Computed metric
        
        Notes
        -----
        Subclasses must implement this method.
        """
        pass
    
    def validate_input(self, contact_result: ContactResult) -> bool:
        """
        Validate input for metric computation.
        
        Parameters
        ----------
        contact_result : ContactResult
            Input to validate
        
        Returns
        -------
        bool
            Whether input is valid
        """
        
        if not isinstance(contact_result, ContactResult):
            logger.warning(f"[{self.name}] Invalid input type")
            return False
        
        return True
    
    def __repr__(self) -> str:
        """String representation."""
        return f"{self.name} v{self.version}"


# ==============================================================================
# METRIC REGISTRY
# ==============================================================================

@dataclass
class MetricRegistry:
    """
    Registry of available metrics.
    
    Maintains registry of metric implementations and provides factory methods.
    
    Attributes
    ----------
    metrics : Dict[str, Type[MetricBase]]
        Registered metrics: name -> class
    
    Examples
    --------
    >>> registry = MetricRegistry()
    >>> registry.register("face_center_face_center", FaceCenterMetric)
    >>> metric = registry.create("face_center_face_center")
    """
    
    metrics: Dict[str, Type[MetricBase]] = None
    
    def __post_init__(self) -> None:
        """Initialize registry."""
        if self.metrics is None:
            self.metrics = {}
        
        logger.debug("[MetricRegistry] Initialized")
    
    def register(self, name: str, metric_class: Type[MetricBase]) -> None:
        """
        Register a metric.
        
        Parameters
        ----------
        name : str
            Metric name
        metric_class : Type[MetricBase]
            Metric class (must inherit from MetricBase)
        
        Raises
        ------
        ValidationError
            If metric_class invalid
        
        Examples
        --------
        >>> registry.register("custom", CustomMetric)
        """
        
        # Validate class
        if not issubclass(metric_class, MetricBase):
            raise ValidationError(
                f"Metric class must inherit from MetricBase",
                error_code="REGISTRY_INVALID_METRIC_CLASS"
            )
        
        # Register
        self.metrics[name] = metric_class
        logger.info(f"[MetricRegistry] Registered metric: {name}")
    
    def unregister(self, name: str) -> None:
        """
        Unregister a metric.
        
        Parameters
        ----------
        name : str
            Metric name
        """
        
        if name in self.metrics:
            del self.metrics[name]
            logger.info(f"[MetricRegistry] Unregistered metric: {name}")
    
    def create(self, name: str) -> MetricBase:
        """
        Create metric instance.
        
        Parameters
        ----------
        name : str
            Metric name
        
        Returns
        -------
        MetricBase
            Metric instance
        
        Raises
        ------
        ValidationError
            If metric not registered
        
        Examples
        --------
        >>> metric = registry.create("face_center_face_center")
        """
        
        if name not in self.metrics:
            raise ValidationError(
                f"Metric '{name}' not registered",
                error_code="REGISTRY_METRIC_NOT_FOUND",
                context={"available": list(self.metrics.keys())}
            )
        
        metric_class = self.metrics[name]
        
        try:
            metric = metric_class()
            logger.debug(f"[MetricRegistry] Created metric: {name}")
            return metric
        except Exception as e:
            raise ValidationError(
                f"Failed to create metric '{name}': {e}",
                error_code="REGISTRY_METRIC_CREATION_ERROR"
            ) from e
    
    def is_registered(self, name: str) -> bool:
        """Check if metric is registered."""
        return name in self.metrics
    
    def get_available(self) -> list:
        """Get list of available metric names."""
        return list(self.metrics.keys())
    
    def get_info(self, name: str) -> Dict[str, Any]:
        """
        Get metric information.
        
        Parameters
        ----------
        name : str
            Metric name
        
        Returns
        -------
        dict
            Metric information
        """
        
        if name not in self.metrics:
            raise ValidationError(
                f"Metric '{name}' not found",
                error_code="REGISTRY_METRIC_NOT_FOUND"
            )
        
        metric_class = self.metrics[name]
        
        return {
            'name': metric_class.name,
            'description': metric_class.description,
            'version': metric_class.version,
        }
    
    def __repr__(self) -> str:
        """String representation."""
        return f"MetricRegistry({len(self.metrics)} metrics)"


if __name__ == "__main__":
    """Test when run directly."""
    print("\n" + "="*80)
    print("METRICS BASE MODULE - TESTING")
    print("="*80 + "\n")
    
    print("[TEST 1] MetricRegistry...")
    try:
        registry = MetricRegistry()
        print(f"✓ Registry created: {registry}\n")
    except Exception as e:
        print(f"✗ Error: {e}\n")
    
    print("[TEST 2] Invalid metric class...")
    try:
        class BadMetric:
            pass
        
        registry.register("bad", BadMetric)
    except ValidationError as e:
        print(f"✓ Correctly caught: {e.error_code}\n")
    
    print("[TEST 3] Metric not found...")
    try:
        metric = registry.create("nonexistent")
    except ValidationError as e:
        print(f"✓ Correctly caught: {e.error_code}\n")
    
    print("="*80)
    print("✓ Base module tests passed!")
    print("="*80 + "\n")
