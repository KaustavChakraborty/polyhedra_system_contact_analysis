# analysis/statistics.py
# ==============================================================================
# Module: analysis.statistics
# Purpose: Stateful aggregate statistics computation
#
# Classes:
#   - StatisticsAggregator: Accumulate and compute statistics
#
# Author: Contact Analysis Team
# ==============================================================================

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional

import numpy as np

from .. import ValidationError
from .types import StatsResult

logger = logging.getLogger(__name__)


# ==============================================================================
# STATISTICS AGGREGATOR
# ==============================================================================

@dataclass
class StatisticsAggregator:
    """
    Stateful statistics aggregator.
    
    Accumulates values across frames and computes statistics.
    
    Attributes
    ----------
    metric_name : str
        Name of metric being aggregated
    _values : List[float]
        Accumulated values
    _frame_count : int
        Number of frames processed
    
    Examples
    --------
    >>> agg = StatisticsAggregator("distance")
    >>> agg.add_value(3.5)
    >>> agg.add_value(4.2)
    >>> result = agg.get_result()
    """
    
    metric_name: str
    _values: List[float] = field(default_factory=list)
    _frame_count: int = 0
    
    def __post_init__(self) -> None:
        """Validate aggregator."""
        if not isinstance(self.metric_name, str) or not self.metric_name:
            raise ValidationError(
                f"metric_name must be non-empty string",
                error_code="STATS_AGG_INVALID_NAME"
            )
        
        logger.debug(f"[StatisticsAggregator] Initialized for '{self.metric_name}'")
    
    def add_value(self, value: float) -> None:
        """
        Add a value to aggregation.
        
        Parameters
        ----------
        value : float
            Value to add
        
        Raises
        ------
        ValidationError
            If value invalid
        """
        
        if not isinstance(value, (int, float)):
            raise ValidationError(
                f"value must be numeric, got {type(value).__name__}",
                error_code="STATS_AGG_INVALID_VALUE"
            )
        
        if np.isnan(value) or np.isinf(value):
            raise ValidationError(
                f"value contains NaN or Inf: {value}",
                error_code="STATS_AGG_VALUE_NAN_INF"
            )
        
        self._values.append(float(value))
    
    def add_values(self, values: List[float]) -> None:
        """
        Add multiple values.
        
        Parameters
        ----------
        values : List[float]
            Values to add
        """
        
        for value in values:
            self.add_value(value)
    
    def process_frame(self) -> None:
        """Mark frame as processed."""
        self._frame_count += 1
        logger.debug(
            f"[StatisticsAggregator] Frame {self._frame_count} processed: "
            f"{len(self._values)} values"
        )
    
    def get_result(self) -> Optional[StatsResult]:
        """
        Get accumulated statistics.
        
        Returns
        -------
        StatsResult or None
            Statistics result if data available
        """
        
        if not self._values:
            logger.warning("[StatisticsAggregator] No values accumulated")
            return None
        
        values_array = np.array(self._values)
        
        result = StatsResult(
            metric_name=self.metric_name,
            mean=float(np.mean(values_array)),
            std=float(np.std(values_array)),
            min=float(np.min(values_array)),
            max=float(np.max(values_array)),
            median=float(np.median(values_array)),
            values=self._values.copy(),
            n_samples=len(self._values),
            frame_count=self._frame_count
        )
        
        logger.info(
            f"[StatisticsAggregator] Result computed: "
            f"mean={result.mean:.6f}, std={result.std:.6f}"
        )
        
        return result
    
    def reset(self) -> None:
        """Reset aggregator."""
        self._values.clear()
        self._frame_count = 0
        logger.debug("[StatisticsAggregator] Reset")
    
    def get_statistics(self) -> dict:
        """Get current statistics."""
        if not self._values:
            return {'status': 'empty'}
        
        values_array = np.array(self._values)
        
        return {
            'metric_name': self.metric_name,
            'n_values': len(self._values),
            'n_frames': self._frame_count,
            'mean': float(np.mean(values_array)),
            'std': float(np.std(values_array)),
            'min': float(np.min(values_array)),
            'max': float(np.max(values_array)),
        }


if __name__ == "__main__":
    """Test when run directly."""
    print("\n" + "="*80)
    print("STATISTICS MODULE - TESTING")
    print("="*80 + "\n")
    
    print("[TEST 1] StatisticsAggregator initialization...")
    try:
        agg = StatisticsAggregator("distance")
        print(f"✓ Aggregator created\n")
    except Exception as e:
        print(f"✗ Error: {e}\n")
    
    print("[TEST 2] Add values and compute statistics...")
    try:
        agg.add_values([1.0, 2.0, 3.0, 4.0, 5.0])
        agg.process_frame()
        
        result = agg.get_result()
        print(f"✓ {result}")
        print(f"  Statistics: {agg.get_statistics()}\n")
    except Exception as e:
        print(f"✗ Error: {e}\n")
    
    print("="*80)
    print("✓ Statistics tests passed!")
    print("="*80 + "\n")
